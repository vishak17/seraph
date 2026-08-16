"""C8 — Reconciliation Layer (FR25-FR27).

Implements the S6 `ReconciliationLayer` contract twice:

    KalmanReconciliationLayer       mode = "kalman"        production (FR25/26)
    ForwardFillReconciliationLayer  mode = "forward_fill"  ablation only (FR27)

Both consume S5 `PillarEmission`s and produce S6 `ReconciledState`s, so C10 can
run byte-identical folds through either and isolate the reconciliation layer's
marginal contribution (FR36). Production reads the Kalman one; forward-fill
exists so O6's claim can be *measured* rather than asserted.

What this layer is actually for (SPEC O6): the three pillars arrive on
different clocks — Hawkes at bar frequency, RMT and Hamilton daily — and one of
them does not exist at all before ~2015 (SPEC §4). Forward-fill answers "what
is Hawkes worth in 2008?" with "whatever it was last", which is nothing, stated
confidently. The Kalman layer answers with the prior and an honest, *bounded*
variance, and the D2 mask keeps that dimension out of the CSRS entirely.

Five behaviours worth stating outright, because each is load-bearing and silent
when wrong:

1.  **Absence is never zero.** An `unavailable` emission runs no update. The
    state keeps its prior (or its last observed value); `P[p][p]` grows.
    Imputing 0 would assert calm — every pillar's zero means "no stress".
2.  **`P` is bounded (D4).** The predict step applies the covariance ceiling
    `R_max^(p)`, so 200 ticks of structural absence saturate rather than
    diverge. Without it FR29's confidence interval is infinite, which is the
    argument that decided D4's functional form in the first place.
3.  **Only a genuinely new `tau` updates the filter (FR26).** Pillar engines
    legitimately re-serve the same estimate: C7's `emit(ts)` returns the last
    close's value with `tau < ts` whenever it is polled between updates. Each
    such re-delivery is the *same* observation, and folding it in again would
    shrink `P` as though independent evidence had arrived — a slow, invisible
    over-confidence exactly where FR29's interval is supposed to widen.
4.  **`coverage()` decides structural absence (D2/S5).** A pillar outside its
    declared coverage window is masked out whether or not it says anything.
    Without this, "no emission arrived" and "this pillar cannot exist yet" are
    the same state, and 2008 quietly scores as if Hawkes were merely quiet.
5.  **Nothing raises across the seam.** A `tau` after its own `ts`, a
    non-finite value, an out-of-order emission — each returns a typed
    `CONTRACT_VIOLATION`, per AGENTS.md §5.

The recursion core is synchronous (`_ingest`, `_advance`); the S6 methods are
thin `async` wrappers over it. That split is what lets `fitting.py` replay tens
of thousands of emissions per likelihood evaluation without an event loop, and
it guarantees the fitted parameters come from the same code path production
runs — not a reimplementation of it.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from seraph.reconciliation.config import SECONDS_PER_DAY, ReconciliationConfig
from seraph.reconciliation.kalman import (
    apply_ceiling,
    predict,
    scalar_update,
    to_mat3,
    to_vec3,
)
from seraph.reconciliation.noise_model import (
    PillarStatus,
    R0Tracker,
    ceiling,
    epoch_seconds,
    is_saturated,
    mask_bit,
    observation_noise,
    staleness_seconds,
)
from seraph.shared_types import (
    PILLAR_INDEX,
    PILLARS,
    AvailabilityMask,
    ContractViolation,
    Err,
    ISOTimestamp,
    ObservedEmission,
    PillarCoverage,
    PillarEmission,
    PillarId,
    ReconciledState,
    ReconciliationMode,
    Result,
    SeraphWarning,
    UnavailableEmission,
    err,
    ok,
)

__all__ = ["ForwardFillReconciliationLayer", "KalmanReconciliationLayer"]


@dataclass
class _PillarRecord:
    """Everything C8 remembers about one pillar."""

    tracker: R0Tracker
    status: PillarStatus = "never"
    last_tau: ISOTimestamp | None = None
    last_value: float | None = None
    coverage: PillarCoverage | None = None
    updates: int = 0
    redeliveries: int = 0


def _ts_of(emission: PillarEmission) -> ISOTimestamp:
    return emission.obs.ts if isinstance(emission, ObservedEmission) else emission.ts


class _BaseLayer:
    """Emission validation, per-pillar bookkeeping and the D2 mask.

    Shared by both modes so that the Kalman path and the forward-fill baseline
    cannot disagree about *which* pillars are available — only about what is
    done with them. An ablation whose two arms differed on the mask would be
    measuring two things at once.
    """

    mode: ReconciliationMode

    def __init__(self, cfg: ReconciliationConfig | None = None) -> None:
        self.cfg = cfg or ReconciliationConfig()
        self._records = tuple(
            _PillarRecord(
                tracker=R0Tracker(
                    prior=self.cfg.r0_prior[i],
                    floor=self.cfg.r0_floor,
                    window=self.cfg.r0_window,
                )
            )
            for i in range(len(PILLARS))
        )
        self._ts: ISOTimestamp | None = None
        self._history: OrderedDict[ISOTimestamp, ReconciledState] = OrderedDict()

    # -- S5 coverage (D2) -----------------------------------------------------

    def declare_coverage(self, pillar: PillarId, coverage: PillarCoverage) -> None:
        """Register `PillarEngine.coverage()` — S5's D2 declaration.

        A pillar is *structurally* absent outside this window, whatever it does
        or does not emit there. This is the difference between "Hawkes is quiet
        today" and "Hawkes cannot exist in 2008" (SPEC §4's coverage matrix),
        and only the pillar itself can tell C8 which one it is.

        Optional: with no declaration C8 falls back to what the emissions say,
        which is correct as long as the pillar reports `structural` absence
        itself. `ReconciliationRunner` calls this for every engine it drives.
        """
        self._records[PILLAR_INDEX[pillar]].coverage = coverage

    def _outside_coverage(self, index: int, ts: ISOTimestamp) -> bool:
        coverage = self._records[index].coverage
        if coverage is None:
            return False
        t = epoch_seconds(ts)
        return t < epoch_seconds(coverage.from_ts) or t > epoch_seconds(coverage.to_ts)

    # -- S6 -------------------------------------------------------------------

    async def state_at(self, ts: ISOTimestamp) -> Result[ReconciledState]:
        """A state the recursion has already produced — never a new one.

        Deliberately not "predict backwards to `ts`": the Kalman recursion is
        not time-reversible, and a state reconstructed by rewinding would not
        be the state C9 actually scored. States older than `history_limit`
        ticks are C1's job to hold (FR38), not this layer's.
        """
        found = self._history.get(ts)
        if found is not None:
            return ok(found)
        return err(
            ContractViolation(
                field="state_at.ts",
                detail=(
                    f"no reconciled state produced at {ts}; C8 serves states it "
                    f"has produced, it does not run the recursion backwards"
                ),
            )
        )

    def states(self) -> tuple[ReconciledState, ...]:
        """Every state still in the bounded history, oldest first (FR38)."""
        return tuple(self._history.values())

    # -- validation -----------------------------------------------------------

    def _validate(self, emission: PillarEmission) -> Err | None:
        """Reject what the S5 shape deliberately permits but C8 cannot accept.

        `PillarObservation` allows `tau > ts` on purpose (see shared_types) so
        that C8 can be the thing that catches it — a pillar claiming a value
        was computed in the future would otherwise silently produce a negative
        `Delta` and a noise variance below its own floor.
        """
        if isinstance(emission, ObservedEmission):
            obs = emission.obs
            if not math.isfinite(obs.value):
                return err(
                    ContractViolation(
                        field="PillarObservation.value",
                        detail=f"{obs.pillar} emitted a non-finite value at {obs.ts}",
                    )
                )
            if epoch_seconds(obs.tau) > epoch_seconds(obs.ts):
                return err(
                    ContractViolation(
                        field="PillarObservation.tau",
                        detail=(
                            f"{obs.pillar} reports tau={obs.tau} after ts={obs.ts}; "
                            f"R^(p)(ts - tau) is only defined for tau <= ts"
                        ),
                    )
                )
            variance = obs.estimation_variance
            if variance is not None and (not math.isfinite(variance) or variance < 0.0):
                return err(
                    ContractViolation(
                        field="PillarObservation.estimation_variance",
                        detail=(
                            f"{obs.pillar} reported estimation_variance={variance!r}; "
                            f"a variance must be finite and non-negative"
                        ),
                    )
                )
        ts = _ts_of(emission)
        if self._ts is not None and epoch_seconds(ts) < epoch_seconds(self._ts):
            return err(
                ContractViolation(
                    field="PillarEmission.ts",
                    detail=(
                        f"emission at {ts} precedes the current state at "
                        f"{self._ts}; C8 only moves forward"
                    ),
                )
            )
        return None

    def _validate_all(self, emissions: tuple[PillarEmission, ...]) -> Err | None:
        """Validate the whole batch before applying any of it.

        All-or-nothing on purpose: C10 replays these streams, and a batch that
        half-applied before failing would leave a state no replay reproduces.
        """
        if not emissions:
            return err(
                ContractViolation(
                    field="update.emissions",
                    detail="update() requires at least one emission; use predict()",
                )
            )
        for emission in emissions:
            violation = self._validate(emission)
            if violation is not None:
                return violation
        return None

    # -- FR26: is this actually new information? ------------------------------

    def _is_new_information(self, emission: ObservedEmission) -> bool:
        """True when `tau` is strictly newer than the last one applied.

        FR26 runs an update "whenever a pillar emits a *new* value". A pillar
        polled twice between its own updates emits the same estimate twice,
        with the same `tau`; treating that as two observations is double
        counting, and its whole effect is on `P` — the state barely moves while
        the variance halves. See the module docstring, point 3.
        """
        record = self._records[PILLAR_INDEX[emission.obs.pillar]]
        if record.last_tau is None:
            return True
        return epoch_seconds(emission.obs.tau) > epoch_seconds(record.last_tau)

    # -- D2 -------------------------------------------------------------------

    def _r0(self, index: int, reported: float | None = None) -> float:
        return self._records[index].tracker.r0(reported)

    def _noise_now(self, index: int, ts: ISOTimestamp) -> tuple[float, float]:
        """`(R^(p)(Delta), R_0^(p))` at the current staleness of pillar `index`."""
        cfg = self.cfg
        r0 = self._r0(index)
        delta = staleness_seconds(ts, self._records[index].last_tau)
        r = observation_noise(
            delta, r0, cfg.h_seconds[index], cfg.r_max_ratio, cfg.noise_form
        )
        return r, r0

    def _mask_and_saturation(
        self, ts: ISOTimestamp
    ) -> tuple[AvailabilityMask, tuple[bool, bool, bool]]:
        bits: list[bool] = []
        saturated: list[bool] = []
        for i in range(len(PILLARS)):
            if self.mode == "forward_fill":
                # The baseline has no age model at all (FR27), so nothing can
                # saturate. Its mask turns on structural absence alone — which
                # is the point: forward-fill cannot tell a fresh pillar from a
                # six-month-old one.
                sat = False
            else:
                r, r0 = self._noise_now(i, ts)
                sat = is_saturated(r, r0, self.cfg)
            status = self._records[i].status
            if self._outside_coverage(i, ts):
                status = "structural"
            saturated.append(sat)
            bits.append(mask_bit(status, sat))
        return (
            (bits[0], bits[1], bits[2]),
            (saturated[0], saturated[1], saturated[2]),
        )

    def _tau_triplet(
        self,
    ) -> tuple[ISOTimestamp | None, ISOTimestamp | None, ISOTimestamp | None]:
        r = self._records
        return (r[0].last_tau, r[1].last_tau, r[2].last_tau)

    def _record_state(
        self, ts: ISOTimestamp, state: ReconciledState
    ) -> ReconciledState:
        self._history[ts] = state
        self._history.move_to_end(ts)
        while len(self._history) > self.cfg.history_limit:
            self._history.popitem(last=False)
        return state

    def _saturation_warnings(
        self, ts: ISOTimestamp, saturated: tuple[bool, bool, bool]
    ) -> tuple[SeraphWarning, ...]:
        return tuple(
            SeraphWarning(
                code="NOISE_SATURATED",
                message=(
                    f"{PILLARS[i]} is at the D4 noise ceiling and is excluded "
                    f"from the D2 mask; any CSRS scored here is degraded"
                ),
                context={"pillar": PILLARS[i], "ts": ts},
            )
            for i, sat in enumerate(saturated)
            if sat
        )

    def _coverage_warning(self, emission: ObservedEmission) -> SeraphWarning | None:
        """A pillar emitting outside the window it declared is a contradiction.

        Warned, not rejected: the emission is real data and the declaration is
        the pillar's own summary of itself, so the honest move is to take the
        observation and flag the disagreement for C10 rather than to discard
        evidence on the strength of metadata.
        """
        index = PILLAR_INDEX[emission.obs.pillar]
        if not self._outside_coverage(index, emission.obs.ts):
            return None
        coverage = self._records[index].coverage
        assert coverage is not None
        return SeraphWarning(
            code="PARTIAL_COVERAGE",
            message=(
                f"{emission.obs.pillar} emitted outside its own declared "
                f"coverage window; applied, but D2 masks it out"
            ),
            context={
                "pillar": emission.obs.pillar,
                "ts": emission.obs.ts,
                "coverage_from": coverage.from_ts,
                "coverage_to": coverage.to_ts,
            },
        )

    def _apply_absence(self, emission: UnavailableEmission) -> None:
        """Record an `unavailable` emission. No update runs — that is the point."""
        index = PILLAR_INDEX[emission.pillar]
        self._records[index].status = emission.absence  # "structural"|"transient"

    # -- diagnostics ----------------------------------------------------------

    def update_counts(self) -> tuple[int, int, int]:
        """Genuine Kalman updates applied per pillar, in `PILLAR_ORDER`."""
        return (
            self._records[0].updates,
            self._records[1].updates,
            self._records[2].updates,
        )

    def redelivery_counts(self) -> tuple[int, int, int]:
        """Re-served estimates skipped per pillar (FR26's "new value")."""
        return (
            self._records[0].redeliveries,
            self._records[1].redeliveries,
            self._records[2].redeliveries,
        )


class KalmanReconciliationLayer(_BaseLayer):
    """S6 `ReconciliationLayer`, production path (FR25, FR26).

    Args:
        cfg: see `config.ReconciliationConfig`.

    Cost is a handful of 3x3 operations per tick — NFR3's "negligible relative
    to pillar computation" is not in doubt here; a single Hawkes MLE window is
    several orders of magnitude more expensive than the entire filter run.
    """

    mode: ReconciliationMode = "kalman"

    def __init__(self, cfg: ReconciliationConfig | None = None) -> None:
        super().__init__(cfg)
        c = self.cfg
        self._x = np.array(c.prior_mean, dtype=float)
        self._p = np.diag(
            [
                c.r_max_ratio * c.r0_prior[i]
                if c.prior_variance_at_ceiling
                else c.r0_prior[i]
                for i in range(len(PILLARS))
            ]
        ).astype(float)
        self._q = np.array(c.q_proc_per_day, dtype=float)
        self._loglik = 0.0

    # -- S6 -------------------------------------------------------------------

    async def predict(self, to_ts: ISOTimestamp) -> Result[ReconciledState]:
        """FR25 — advance to `to_ts` with no new information."""
        return self.advance(to_ts)

    async def update(
        self, emissions: tuple[PillarEmission, ...]
    ) -> Result[ReconciledState]:
        """FR26 — fold emissions into the state, oldest first."""
        return self.ingest(emissions)

    # -- synchronous core -----------------------------------------------------

    def advance(self, to_ts: ISOTimestamp) -> Result[ReconciledState]:
        """`predict()` without the coroutine (see the module docstring).

        `x_hat` is unchanged (random walk); `P` grows by `Q_proc * dt` and is
        then bounded by the D4 ceiling. `trace(P)` is therefore non-decreasing
        across this call, which is half of SPEC O6's acceptance criterion.
        """
        if self._ts is not None and epoch_seconds(to_ts) < epoch_seconds(self._ts):
            return err(
                ContractViolation(
                    field="predict.to_ts",
                    detail=f"cannot predict backwards from {self._ts} to {to_ts}",
                )
            )
        self._advance_to(to_ts)
        return self._emit(to_ts)

    def ingest(self, emissions: tuple[PillarEmission, ...]) -> Result[ReconciledState]:
        """`update()` without the coroutine.

        Emissions may span several timestamps and arrive in any order; they are
        sorted and each is preceded by a predict to its own `ts`, so a batch
        replayed from the store gives the same state as the same emissions
        delivered live, one at a time. That equality is what makes C10's
        "byte-identical folds" (O6 acceptance) achievable at all.
        """
        invalid = self._validate_all(emissions)
        if invalid is not None:
            return invalid

        ordered = sorted(emissions, key=lambda e: epoch_seconds(_ts_of(e)))
        warnings: list[SeraphWarning] = []
        seen: set[tuple[str, str]] = set()

        def warn(w: SeraphWarning | None) -> None:
            if w is None:
                return
            key = (w.code, w.message)
            if key not in seen:
                seen.add(key)
                warnings.append(w)

        for emission in ordered:
            self._advance_to(_ts_of(emission))
            if isinstance(emission, ObservedEmission):
                warn(self._coverage_warning(emission))
                for w in self._apply_observation(emission):
                    warn(w)
            else:
                self._apply_absence(emission)

        return self._emit(_ts_of(ordered[-1]), extra=tuple(warnings))

    def replay(self, ordered: tuple[PillarEmission, ...]) -> None:
        """Apply an already-validated, already-sorted stream, emitting nothing.

        The fitting loop in `fitting.py` runs this hundreds of times over the
        same history and only ever reads `log_likelihood` back, so paying for
        re-validation, re-sorting and a pydantic `ReconciledState` per call is
        pure waste — those costs are per *call*, not per emission, and they
        dominate at this call count.

        **Caller's obligation:** `ordered` must have passed `_validate_all` and
        be sorted by `ts`. Nothing else about this path differs from `ingest`;
        it is the same recursion, so a fitted parameter set is fitted on the
        filter that will run in production.
        """
        for emission in ordered:
            self._advance_to(_ts_of(emission))
            if isinstance(emission, ObservedEmission):
                self._apply_observation(emission)
            else:
                self._apply_absence(emission)

    def prepare(
        self, emissions: Sequence[PillarEmission]
    ) -> tuple[PillarEmission, ...] | Err:
        """Validate and sort a stream once, for repeated `replay` calls."""
        stream = tuple(emissions)
        invalid = self._validate_all(stream)
        if invalid is not None:
            return invalid
        return tuple(sorted(stream, key=lambda e: epoch_seconds(_ts_of(e))))

    # -- recursion ------------------------------------------------------------

    @property
    def log_likelihood(self) -> float:
        """Sum of Gaussian innovation log-densities over every genuine update.

        This is the objective `fitting.py` maximises for D4's `h_p`,
        `R_max/R_0` and `Q_proc` (SPEC OQ10: "remain MLE-fitted"). It is also a
        standalone diagnostic — a noise model that is badly wrong shows up here
        long before it shows up in an AUC.
        """
        return self._loglik

    def _advance_to(self, ts: ISOTimestamp) -> None:
        """FR25 predict + the D4 covariance ceiling."""
        if self._ts is None:
            self._ts = ts
            self._p = apply_ceiling(self._p, self._ceilings())
            return
        dt_days = (epoch_seconds(ts) - epoch_seconds(self._ts)) / SECONDS_PER_DAY
        self._x, self._p = predict(self._x, self._p, self._q, dt_days)
        self._p = apply_ceiling(self._p, self._ceilings())
        self._ts = ts

    def _ceilings(self) -> np.ndarray:
        cfg = self.cfg
        return np.array(
            [
                ceiling(self._r0(i), cfg.r_max_ratio, cfg.noise_form)
                for i in range(len(PILLARS))
            ]
        )

    def _apply_observation(
        self, emission: ObservedEmission
    ) -> tuple[SeraphWarning, ...]:
        """FR26 — one scalar update at `R^(p)(ts - tau)`."""
        obs = emission.obs
        index = PILLAR_INDEX[obs.pillar]
        cfg = self.cfg
        record = self._records[index]

        if not self._is_new_information(emission):
            record.redeliveries += 1
            return ()

        # R_0 is read BEFORE this value enters the rolling window: the noise on
        # an observation must not be informed by the observation itself.
        r0 = record.tracker.r0(obs.estimation_variance)
        delta = staleness_seconds(obs.ts, obs.tau)
        r = observation_noise(
            delta, r0, cfg.h_seconds[index], cfg.r_max_ratio, cfg.noise_form
        )

        innovation = obs.value - float(self._x[index])
        innovation_var = float(self._p[index, index]) + r
        if innovation_var > 0.0:
            self._loglik -= 0.5 * (
                math.log(2.0 * math.pi * innovation_var)
                + innovation * innovation / innovation_var
            )

        self._x, self._p, _gain = scalar_update(self._x, self._p, index, obs.value, r)

        record.tracker.observe(obs.value)
        record.status = "observed"
        record.last_tau = obs.tau
        record.last_value = obs.value
        record.updates += 1

        warnings: list[SeraphWarning] = []
        if delta > cfg.h_seconds[index]:
            warnings.append(
                SeraphWarning(
                    code="STALE_OBSERVATION",
                    message=(
                        f"{obs.pillar} emission is older than its own update "
                        f"cadence; D4 age-inflation applied"
                    ),
                    context={
                        "pillar": obs.pillar,
                        "delta_seconds": delta,
                        "h_seconds": cfg.h_seconds[index],
                    },
                )
            )
        if is_saturated(r, r0, cfg):
            warnings.append(
                SeraphWarning(
                    code="NOISE_SATURATED",
                    message=(
                        f"{obs.pillar} emission arrived at the D4 noise ceiling; "
                        f"its Kalman gain is ~0"
                    ),
                    context={"pillar": obs.pillar, "ts": obs.ts},
                )
            )
        return tuple(warnings)

    def _emit(
        self, ts: ISOTimestamp, extra: tuple[SeraphWarning, ...] = ()
    ) -> Result[ReconciledState]:
        mask, saturated = self._mask_and_saturation(ts)
        state = ReconciledState(
            ts=ts,
            x_hat=to_vec3(self._x),
            p_t=to_mat3(self._p),
            tau_last_update=self._tau_triplet(),
            mask=mask,
            noise_saturated=saturated,
            mode=self.mode,
        )
        already = {(w.code, w.message) for w in extra}
        warnings = extra + tuple(
            w
            for w in self._saturation_warnings(ts, saturated)
            if (w.code, w.message) not in already
        )
        return ok(self._record_state(ts, state), warnings=warnings)


class ForwardFillReconciliationLayer(_BaseLayer):
    """FR27's ablation baseline. **Not a production path.**

    `x_hat[p]` is the last value pillar `p` emitted, carried forward
    indefinitely; `P` is diagonal, holding each pillar's `R_0` with **no age
    term at all**. `R_0` is the same rolling quantity the Kalman arm uses — so
    it does move as a pillar's own dispersion moves — but nothing here responds
    to *staleness*: a six-month-old value is reported at exactly the confidence
    a fresh one would be. That is the "naive forward-fill propagation" SPEC O6
    exists to beat, implemented honestly so the comparison means something: no
    age inflation, no ceiling, no way to tell a fresh observation from a stale
    one.

    It keeps the same D2 mask rules and the same FR26 new-information rule as
    the Kalman path, so the two arms of the FR36 ablation differ in the
    reconciliation recursion and nothing else.
    """

    mode: ReconciliationMode = "forward_fill"

    async def predict(self, to_ts: ISOTimestamp) -> Result[ReconciledState]:
        """Time passing changes nothing here — that is the baseline's defect."""
        return self.advance(to_ts)

    async def update(
        self, emissions: tuple[PillarEmission, ...]
    ) -> Result[ReconciledState]:
        return self.ingest(emissions)

    def advance(self, to_ts: ISOTimestamp) -> Result[ReconciledState]:
        if self._ts is not None and epoch_seconds(to_ts) < epoch_seconds(self._ts):
            return err(
                ContractViolation(
                    field="predict.to_ts",
                    detail=f"cannot predict backwards from {self._ts} to {to_ts}",
                )
            )
        self._ts = to_ts
        return self._emit(to_ts)

    def ingest(self, emissions: tuple[PillarEmission, ...]) -> Result[ReconciledState]:
        invalid = self._validate_all(emissions)
        if invalid is not None:
            return invalid

        ordered = sorted(emissions, key=lambda e: epoch_seconds(_ts_of(e)))
        for emission in ordered:
            self._ts = _ts_of(emission)
            if isinstance(emission, ObservedEmission):
                record = self._records[PILLAR_INDEX[emission.obs.pillar]]
                if not self._is_new_information(emission):
                    record.redeliveries += 1
                    continue
                record.tracker.observe(emission.obs.value)
                record.status = "observed"
                record.last_tau = emission.obs.tau
                record.last_value = emission.obs.value
                record.updates += 1
            else:
                self._apply_absence(emission)

        return self._emit(_ts_of(ordered[-1]))

    def _emit(self, ts: ISOTimestamp) -> Result[ReconciledState]:
        cfg = self.cfg
        carried = [
            record.last_value if record.last_value is not None else cfg.prior_mean[i]
            for i, record in enumerate(self._records)
        ]
        variance = [
            self._r0(i) * cfg.forward_fill_variance_ratio for i in range(len(PILLARS))
        ]
        mask, saturated = self._mask_and_saturation(ts)
        state = ReconciledState(
            ts=ts,
            x_hat=(carried[0], carried[1], carried[2]),
            p_t=(
                (variance[0], 0.0, 0.0),
                (0.0, variance[1], 0.0),
                (0.0, 0.0, variance[2]),
            ),
            tau_last_update=self._tau_triplet(),
            mask=mask,
            noise_saturated=saturated,
            mode=self.mode,
        )
        return ok(self._record_state(ts, state))
