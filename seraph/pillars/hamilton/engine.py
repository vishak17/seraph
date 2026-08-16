"""C7 — Hamilton Engine (FR20, FR21, FR23, FR24).

Implements the S5 `PillarEngine` contract for `pillar = "hamilton"`:

    emit / emit_range -> Result[PillarEmission]     value = LSD_t (FR23)
    coverage          -> Result[PillarCoverage]     D2 structural window
    detail            -> Result[HamiltonDetail]     xi, p_jj, half-lives (FR24)
    details_range     -> Result[(ts, HamiltonDetail)]  the xi series C9 scores
    outputs           -> Result[HamiltonOutputRow]  E8 rows for C1

Five properties this file is built around, all of them silent if broken:

1.  **Emissions come from FILTERED probabilities, never smoothed ones.** The
    Kim smoother runs inside EM only. A smoothed xi_t conditions on the future
    and would walk straight into C9's CSRS and C10's AUC.
2.  **Parameters are fitted on an expanding window ending at the emission
    date.** Standardisation statistics and the covariate set are frozen at fit
    time, so nothing about t+1 reaches t.
3.  **`tau` is honest.** It is the NSE close of the trading date whose data
    produced the estimate — not the query time. `emit(ts)` for a ts after that
    close returns the same value with `tau < ts`, which is exactly the
    staleness `Delta` that D4's `R^(p)(Delta)` is defined on.
4.  **Absence is `Ok`, not `Err`.** Before `min_history_days` the pillar
    cannot exist -> `structural`/`insufficient_history`. A failed EM ->
    `transient`/`estimation_failed`, which keeps Hamilton in the D2 mask.
    `Err` is reserved for a failure of the *call*: a missing C4 dependency, a
    source that raises, or output that violates the S4 shape.
5.  **Nothing raises past the seam.** Every numerical failure path — a
    non-PD regime covariance, a diverging M-step, a `nan` filtered vector —
    degrades to a transient absence. AGENTS.md §5: return an `Err` variant
    (or an absence), never an exception across a component boundary.

Cost, measured on a 5,000-day corpus with the default config: a cold call
(one EM fit) ~7 s; a subsequent call on the cached fit ~0.1 s, which is the
daily production path NFR1 cares about. A full expanding-window backtest pays
one EM fit per `refit_every_days` — roughly 10 minutes over the whole corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp, softmax

from seraph.pillars.hamilton.config import (
    CRISIS,
    IST_OFFSET,
    MARKET_CLOSE,
    STRESSED,
    HamiltonConfig,
)
from seraph.pillars.hamilton.em import (
    HamiltonEstimationError,
    HamiltonParams,
    fit_em,
)
from seraph.pillars.hamilton.observations import (
    HamiltonFeatureSource,
    ObservationPanel,
    build_panel,
    design_matrix,
    panel_fingerprint,
    standardise,
    usable_covariates,
    validate_observations,
)
from seraph.pillars.hamilton.output import HamiltonOutputRow
from seraph.pillars.hamilton.tvtp import half_life
from seraph.shared_types import (
    HamiltonDetail,
    ISODate,
    ISOTimestamp,
    MissingDependency,
    ObservedEmission,
    PillarCoverage,
    PillarEmission,
    PillarId,
    PillarObservation,
    Result,
    SeraphWarning,
    Simplex3,
    UnavailableEmission,
    err,
    ok,
)
from seraph.shared_types.common import (
    ContractViolation,
    Err,
    InsufficientHistory,
)

__all__ = ["HamiltonEngine", "close_ts", "date_of"]


def close_ts(date: ISODate) -> ISOTimestamp:
    """NSE close of `date`, IST-explicit (AGENTS.md §5)."""
    return f"{date}T{MARKET_CLOSE}{IST_OFFSET}"


def date_of(ts: ISOTimestamp) -> ISODate:
    return ts[:10]


def _as_simplex(v: np.ndarray) -> Simplex3:
    """Renormalise a filtered probability vector into an exact Simplex3."""
    p = np.clip(np.asarray(v, dtype=float), 0.0, None)
    total = p.sum()
    p = p / total if total > 0 else np.full(3, 1.0 / 3.0)
    # Absorb float error into the largest component so the sum is exactly 1.
    k = int(np.argmax(p))
    p[k] += 1.0 - p.sum()
    return (float(p[0]), float(p[1]), float(p[2]))


@dataclass(frozen=True, eq=False)
class _State:
    """Filtered state at one trading date."""

    index: int
    date: ISODate
    xi: np.ndarray  # (3,) filtered regime probabilities
    trans: np.ndarray  # (3, 3) transition INTO this date, given z_t
    params: HamiltonParams


@dataclass
class _RunCache:
    """Fit and filter state carried across calls. See `HamiltonEngine._run`."""

    fingerprint: bytes = b""
    n_rows: int = 0
    params: HamiltonParams | None = None
    used_cols: tuple[int, ...] | None = None
    fit_end: int = -1
    last_index: int = -1
    last_filtered: np.ndarray | None = None
    states: dict[int, _State] = field(default_factory=dict)
    missing: dict[int, tuple[str, str]] = field(default_factory=dict)
    warns: list[SeraphWarning] = field(default_factory=list)
    seen: set[tuple[str, str]] = field(default_factory=set)


class _Prefix:
    """Incremental forward filter under one frozen parameter set.

    Kept separate from `filter.hamilton_filter_log` (which is batch and is what
    the statsmodels cross-check runs against) so that walking the backtest
    forward stays O(T) overall instead of O(T^2).

    `resume_at`/`resume_filtered` restore the recursion's position from a cached
    run. Only ever valid when the panel prefix has been content-verified — the
    filtered vector at t is a sufficient statistic for everything before it, so
    resuming from it is exact, not an approximation.
    """

    def __init__(
        self,
        params: HamiltonParams,
        y_std: np.ndarray,
        z_design: np.ndarray,
        resume_at: int = -1,
        resume_filtered: np.ndarray | None = None,
    ):
        self.params = params
        self.y = y_std
        self.z = z_design
        self.chol = [np.linalg.cholesky(params.sigma[j]) for j in range(3)]
        self.log_det = [2.0 * np.log(np.diag(c)).sum() for c in self.chol]
        self.const = y_std.shape[1] * np.log(2.0 * np.pi)
        self.last = resume_at if resume_filtered is not None else -1
        self.prev: np.ndarray | None = resume_filtered
        self.trans: np.ndarray | None = None

    def _log_dens(self, t: int) -> np.ndarray:
        out = np.empty(3)
        for j in range(3):
            diff = self.y[t] - self.params.mu[j]
            sol = solve_triangular(self.chol[j], diff, lower=True)
            out[j] = -0.5 * (self.const + self.log_det[j] + float(sol @ sol))
        return out

    def _transition_at(self, t: int) -> np.ndarray:
        eta = np.einsum("p,ijp->ij", self.z[t], self.params.gamma)
        return softmax(eta, axis=1)

    def advance_to(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Filter forward to `index`; returns (filtered xi, transition into it)."""
        if index < self.last:
            raise ValueError("_Prefix only moves forward")
        if index == self.last and self.prev is not None:
            # Resumed exactly at the requested date: the filtered vector is
            # already correct, only the transition matrix needs rebuilding.
            return self.prev, self._transition_at(index)
        for t in range(self.last + 1, index + 1):
            trans = self._transition_at(t)
            if self.prev is None:
                # Stationary distribution of the t=0 transition matrix.
                a = np.vstack([trans.T - np.eye(3), np.ones((1, 3))])
                b = np.array([0.0, 0.0, 0.0, 1.0])
                pi, *_ = np.linalg.lstsq(a, b, rcond=None)
                pred = np.clip(pi, 1e-12, None)
                pred = pred / pred.sum()
            else:
                pred = np.clip(self.prev @ trans, 1e-300, None)
            log_joint = np.log(pred) + self._log_dens(t)
            self.prev = np.exp(log_joint - logsumexp(log_joint))
            self.trans = trans
            self.last = t
        if self.prev is None or self.trans is None:
            raise ValueError("filter advanced over an empty range")
        return self.prev, self.trans


class HamiltonEngine:
    """S5 `PillarEngine` for Pillar 3 (FR20, FR21, FR23, FR24).

    Args:
        source: the S4 slice C4 exposes (`observation_vector`,
            `covariate_vector`). C7 never reads the store directly.
        cfg: see `config.HamiltonConfig`.
    """

    pillar: PillarId = "hamilton"

    def __init__(
        self, source: HamiltonFeatureSource, cfg: HamiltonConfig | None = None
    ) -> None:
        self.source = source
        self.cfg = cfg or HamiltonConfig()
        # Fit + filter state, continued across calls. Content-keyed, so it is a
        # cache in the strict sense: dropping it changes cost, never answers.
        # One engine instance is single-threaded by construction (NFR5 puts
        # each pillar in its own process, not its own thread).
        self._cache: _RunCache | None = None

    def reset_cache(self) -> None:
        """Drop cached parameters and filter state. Answers are unaffected."""
        self._cache = None

    # -- S5 ------------------------------------------------------------------

    async def coverage(self) -> Result[PillarCoverage]:
        """D2 — the window over which Hamilton can exist at all.

        Starts at the first date with `min_history_days` of observations behind
        it. Outside this window absence is `structural`, which is what removes
        Hamilton from the C8 mask rather than letting a drifting Kalman prior
        dilute the CSRS.
        """
        loaded = await self._load(self.cfg.corpus_start, "9999-12-31")
        if isinstance(loaded, Err):
            return loaded
        panel = loaded
        need = self.cfg.min_history_days
        if len(panel) < need:
            return err(
                InsufficientHistory(
                    required=need,
                    available=len(panel),
                    as_of=panel.dates[-1] if len(panel) else self.cfg.corpus_start,
                )
            )
        return ok(
            PillarCoverage(
                from_ts=close_ts(panel.dates[need - 1]),
                to_ts=close_ts(panel.dates[-1]),
            ),
            warnings=panel.warnings,
        )

    async def emit(self, ts: ISOTimestamp) -> Result[PillarEmission]:
        """Latest Hamilton estimate valid as of `ts` (FR23).

        `ts` may fall mid-session or on a holiday; the emission then carries
        the previous close as `tau`, so C8 ages it correctly instead of being
        told a stale value is fresh.
        """
        as_of = date_of(ts)
        loaded = await self._load(self.cfg.corpus_start, as_of)
        if isinstance(loaded, Err):
            return loaded
        panel = loaded

        idx = self._last_index_on_or_before(panel, as_of)
        if idx is None:
            return ok(
                UnavailableEmission(
                    pillar=self.pillar,
                    ts=ts,
                    absence="structural",
                    reason="no_data_coverage",
                ),
                warnings=panel.warnings,
            )

        states, missing, warns = self._run(panel, [idx])
        if idx in missing:
            absence, reason = missing[idx]
            return ok(
                UnavailableEmission(
                    pillar=self.pillar, ts=ts, absence=absence, reason=reason
                ),
                warnings=panel.warnings + tuple(warns),
            )

        state = states[idx]
        emission = ObservedEmission(
            obs=PillarObservation(
                pillar=self.pillar,
                ts=ts,
                tau=close_ts(state.date),
                value=self._lsd(state.xi),
                estimation_variance=self._lsd_variance(state.xi),
            )
        )
        if state.date != as_of:
            warns.append(
                SeraphWarning(
                    code="STALE_OBSERVATION",
                    message="no Hamilton update on the queried date; last close used",
                    context={"queried": as_of, "last_update": state.date},
                )
            )
        return ok(emission, warnings=panel.warnings + tuple(warns))

    async def emit_range(
        self, from_ts: ISOTimestamp, to_ts: ISOTimestamp
    ) -> Result[tuple[PillarEmission, ...]]:
        """One emission per trading date in [from_ts, to_ts] (NFR4: daily)."""
        from_date, to_date = date_of(from_ts), date_of(to_ts)
        loaded = await self._load(self.cfg.corpus_start, to_date)
        if isinstance(loaded, Err):
            return loaded
        panel = loaded

        targets = [i for i, d in enumerate(panel.dates) if from_date <= d <= to_date]
        states, missing, warns = self._run(panel, targets)

        emissions: list[PillarEmission] = []
        for i in targets:
            ts_i = close_ts(panel.dates[i])
            if i in missing:
                absence, reason = missing[i]
                emissions.append(
                    UnavailableEmission(
                        pillar=self.pillar, ts=ts_i, absence=absence, reason=reason
                    )
                )
                continue
            state = states[i]
            emissions.append(
                ObservedEmission(
                    obs=PillarObservation(
                        pillar=self.pillar,
                        ts=ts_i,
                        tau=ts_i,  # computed from data through this close
                        value=self._lsd(state.xi),
                        estimation_variance=self._lsd_variance(state.xi),
                    )
                )
            )
        return ok(tuple(emissions), warnings=panel.warnings + tuple(warns))

    # -- C7 detail (rides alongside the emission, never inside it) -----------

    async def detail(self, ts: ISOTimestamp) -> Result[HamiltonDetail]:
        """FR23/FR24 detail for the estimate valid as of `ts`."""
        as_of = date_of(ts)
        loaded = await self._load(self.cfg.corpus_start, as_of)
        if isinstance(loaded, Err):
            return loaded
        panel = loaded
        idx = self._last_index_on_or_before(panel, as_of)
        if idx is None:
            return err(MissingDependency(entity="E8:hamilton_output", as_of=ts))
        states, missing, warns = self._run(panel, [idx])
        if idx in missing:
            return err(MissingDependency(entity="E8:hamilton_output", as_of=ts))
        return ok(self._detail(states[idx]), warnings=panel.warnings + tuple(warns))

    async def details_range(
        self, from_ts: ISOTimestamp, to_ts: ISOTimestamp
    ) -> Result[tuple[tuple[ISOTimestamp, HamiltonDetail], ...]]:
        """FR23/FR24 detail per trading date — the xi series C9 scores with.

        One pass over the range, so a backtest does not pay a full refit sweep
        per date the way repeated `detail()` calls would. Dates with no
        estimate (structural or transient absence) are simply absent from the
        result; pair this with `emit_range`, which reports them explicitly.
        """
        from_date, to_date = date_of(from_ts), date_of(to_ts)
        loaded = await self._load(self.cfg.corpus_start, to_date)
        if isinstance(loaded, Err):
            return loaded
        panel = loaded
        targets = [i for i, d in enumerate(panel.dates) if from_date <= d <= to_date]
        states, _missing, warns = self._run(panel, targets)

        out = tuple(
            (close_ts(states[i].date), self._detail(states[i]))
            for i in targets
            if i in states
        )
        return ok(out, warnings=panel.warnings + tuple(warns))

    async def outputs(
        self, from_ts: ISOTimestamp, to_ts: ISOTimestamp
    ) -> Result[tuple[HamiltonOutputRow, ...]]:
        """E8 rows over a date range, for C1 to persist (FR38)."""
        from_date, to_date = date_of(from_ts), date_of(to_ts)
        loaded = await self._load(self.cfg.corpus_start, to_date)
        if isinstance(loaded, Err):
            return loaded
        panel = loaded
        targets = [i for i, d in enumerate(panel.dates) if from_date <= d <= to_date]
        states, missing, warns = self._run(panel, targets)

        out: list[HamiltonOutputRow] = []
        for i in targets:
            if i in missing:
                continue
            state = states[i]
            src = panel.rows[i]  # same fetch that produced y_t — cannot disagree
            xi = _as_simplex(state.xi)
            detail = self._detail(state)
            out.append(
                HamiltonOutputRow(
                    date=state.date,
                    rv_t_yz=src.rv_yang_zhang,
                    rv_t_5min=src.rv_5min,
                    ar_t=src.ar_t,
                    bas_t=src.bas_t,
                    xi=xi,
                    lsd_t=self._lsd(state.xi),
                    estimation_uncertainty=self._lsd_variance(state.xi),
                    p_hat_22=detail.p_hat_22,
                    p_hat_33=detail.p_hat_33,
                    tau_half_stressed=detail.tau_half_stressed,
                    tau_half_crisis=detail.tau_half_crisis,
                    gamma_ij=tuple(
                        tuple(tuple(float(v) for v in row) for row in mat)
                        for mat in state.params.gamma
                    ),
                    covariates=state.params.covariates,
                    fitted_through=state.params.fitted_through,
                    fit_converged=state.params.converged,
                    fit_loglik=state.params.loglik,
                )
            )
        return ok(tuple(out), warnings=panel.warnings + tuple(warns))

    # -- FR23 scalars ---------------------------------------------------------

    @staticmethod
    def _lsd(xi: np.ndarray) -> float:
        """FR23 — LSD_t = xi_2 + 2*xi_3, i.e. 0 tranquil -> 2 crisis.

        Note SPEC E8 annotates `LSD_t float in [0,3]`, which the FR23 formula
        cannot reach: its maximum is 2 at xi_3 = 1. FR23 is implemented; the E8
        range annotation is a documentation slip, flagged rather than coded to.
        """
        return float(xi[STRESSED] + 2.0 * xi[CRISIS])

    @staticmethod
    def _lsd_variance(xi: np.ndarray) -> float:
        """FR23's "per-update estimation uncertainty", feeding D4's `R_0`.

        LSD_t is a deterministic function of the regime: it takes values
        (0, 1, 2) with probabilities xi. Its variance under the filtered
        posterior is therefore

            Var(LSD_t) = (xi_2 + 4*xi_3) - LSD_t^2

        which is 0 when the filter is certain and maximal when it is torn
        between tranquil and crisis — precisely the uncertainty C8 should widen
        on. It captures *regime* uncertainty only: parameter uncertainty in
        (mu, Sigma, gamma) is not propagated, since that needs the observed
        information matrix of the EM fit. Stated as a limitation, not hidden.
        """
        lsd = float(xi[STRESSED] + 2.0 * xi[CRISIS])
        second = float(xi[STRESSED] + 4.0 * xi[CRISIS])
        return max(second - lsd * lsd, 0.0)

    def _detail(self, state: _State) -> HamiltonDetail:
        p22 = float(state.trans[STRESSED, STRESSED])
        p33 = float(state.trans[CRISIS, CRISIS])
        return HamiltonDetail(
            xi=_as_simplex(state.xi),
            p_hat_22=p22,
            p_hat_33=p33,
            tau_half_stressed=half_life(p22, self.cfg.p_jj_cap),
            tau_half_crisis=half_life(p33, self.cfg.p_jj_cap),
        )

    # -- internals ------------------------------------------------------------

    async def _load(
        self, from_date: ISODate, to_date: ISODate
    ) -> ObservationPanel | Err:
        """Fetch y_t and z_t through `to_date` — never past it.

        Every failure mode of the upstream call is converted into a typed
        `Err` here: an `Err` result, a raised exception from a source that does
        not honour the Result convention, or output that violates the S4 shape
        (duplicate dates, non-finite levels). C7 raises nothing at its seam.
        """
        as_of = close_ts(to_date[:10])
        try:
            obs = await self.source.observation_vector(from_date, to_date)
        except Exception as exc:  # a source that raises instead of returning Err
            return err(
                ContractViolation(
                    field="C4:observation_vector",
                    detail=f"source raised instead of returning a Result: {exc!r}",
                )
            )
        if isinstance(obs, Err):
            return err(MissingDependency(entity="C4:observation_vector", as_of=as_of))

        try:
            macro = await self.source.covariate_vector(from_date, to_date)
        except Exception as exc:
            return err(
                ContractViolation(
                    field="C4:covariate_vector",
                    detail=f"source raised instead of returning a Result: {exc!r}",
                )
            )
        if isinstance(macro, Err):
            return err(MissingDependency(entity="C4:covariate_vector", as_of=as_of))

        rows = tuple(obs.value)
        violation = validate_observations(rows)
        if violation is not None:
            field_name, detail = violation
            return err(ContractViolation(field=field_name, detail=detail))

        return build_panel(rows, tuple(macro.value), self.cfg, apply_drop=False)

    @staticmethod
    def _last_index_on_or_before(panel: ObservationPanel, date: ISODate) -> int | None:
        for i in range(len(panel) - 1, -1, -1):
            if panel.dates[i] <= date:
                return i
        return None

    def _fit(
        self,
        panel: ObservationPanel,
        upto: int,
        cols: tuple[int, ...],
        warm_start: HamiltonParams | None,
    ) -> HamiltonParams:
        """EM on the expanding window [0, upto]. Nothing after `upto` is read."""
        cfg = self.cfg
        y_win = panel.y[: upto + 1]
        z_win = panel.z[: upto + 1][:, list(cols)] if cols else np.empty((upto + 1, 0))

        if cfg.standardize_y:
            y_std, y_center, y_scale = standardise(y_win)
        else:
            y_std = y_win
            y_center = np.zeros(y_win.shape[1])
            y_scale = np.ones(y_win.shape[1])
        z_std, z_center, z_scale = standardise(z_win)

        return fit_em(
            y=y_std,
            z=design_matrix(z_std),
            cfg=cfg,
            covariates=tuple(panel.covariates[c] for c in cols),
            y_center=y_center,
            y_scale=y_scale,
            z_center=z_center,
            z_scale=z_scale,
            fitted_through=panel.dates[upto],
            warm_start=warm_start,
        )

    def _prefix_inputs(
        self, panel: ObservationPanel, params: HamiltonParams, cols: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the fit's frozen standardisation to the whole loaded panel."""
        y_std, _, _ = standardise(panel.y, params.y_center, params.y_scale)
        z_raw = panel.z[:, list(cols)] if cols else np.empty((len(panel), 0))
        z_std, _, _ = standardise(z_raw, params.z_center, params.z_scale)
        return y_std, design_matrix(z_std)

    def _reusable(
        self, panel: ObservationPanel, targets: list[int]
    ) -> _RunCache | None:
        """Decide whether the cached fit and filter state may be continued.

        Three conditions, all necessary:
          * the cached prefix is a *content* match for the same prefix of the
            new panel — a source that restates history invalidates the cache
            instead of being trusted;
          * the new panel is not shorter than that prefix;
          * every requested target is either already resolved or lies ahead of
            the filter's position (the recursion only moves forward).
        """
        cache = self._cache
        if cache is None or len(panel) < cache.n_rows:
            return None
        if panel_fingerprint(panel, cache.n_rows) != cache.fingerprint:
            return None
        for i in targets:
            resolved = i in cache.states or i in cache.missing
            if not resolved and i <= cache.last_index:
                return None
        return cache

    def _run(
        self, panel: ObservationPanel, targets: list[int]
    ) -> tuple[
        dict[int, _State],
        dict[int, tuple[str, str]],
        list[SeraphWarning],
    ]:
        """Walk the target dates forward, refitting on cadence.

        Returns `(states, missing, warnings)` where `missing[i]` is the
        `(absence, reason)` pair for a date that cannot be emitted.

        Results are cached on the instance and continued across calls, so a
        daily production run costs one filter step rather than a full corpus
        sweep, and `emit` followed by `detail` for the same date does not refit
        anything. The cache is keyed on panel content (see `_reusable`), never
        on call order.
        """
        cfg = self.cfg
        cache = self._reusable(panel, targets)
        if cache is None:
            cache = _RunCache()
        self._cache = cache

        states = cache.states
        missing = cache.missing
        warns = cache.warns
        seen = cache.seen

        params = cache.params
        used_cols = cache.used_cols
        fit_end = cache.fit_end
        prefix: _Prefix | None = None
        if params is not None and cache.last_filtered is not None:
            y_std, z_design = self._prefix_inputs(panel, params, used_cols or ())
            prefix = _Prefix(
                params,
                y_std,
                z_design,
                resume_at=cache.last_index,
                resume_filtered=cache.last_filtered,
            )

        def warn(w: SeraphWarning) -> None:
            """One warning per distinct (code, message) — a backtest walks
            thousands of dates and would otherwise emit thousands of copies."""
            key = (w.code, w.message)
            if key not in seen:
                seen.add(key)
                warns.append(w)

        for i in sorted(targets):
            if i in states or i in missing:
                continue
            if i + 1 < cfg.min_history_days:
                missing[i] = ("structural", "insufficient_history")
                continue

            cols = usable_covariates(panel.z, panel.covariates, i)
            stale = params is None or (i - fit_end) >= cfg.refit_every_days
            if params is not None and cols != used_cols:
                stale = True
                warn(
                    SeraphWarning(
                        code="PARTIAL_COVERAGE",
                        message="z_t covariate set changed; TVTP refitted",
                        context={
                            "date": panel.dates[i],
                            "covariates": ",".join(panel.covariates[c] for c in cols),
                        },
                    )
                )

            if stale:
                try:
                    params = self._fit(panel, i, cols, warm_start=params)
                    used_cols = cols
                    fit_end = i
                    prefix = None
                    if len(cols) < len(panel.covariates):
                        dropped = [
                            panel.covariates[c]
                            for c in range(len(panel.covariates))
                            if c not in cols
                        ]
                        warn(
                            SeraphWarning(
                                code="PARTIAL_COVERAGE",
                                message=(
                                    "covariate dropped from z_t — still null over "
                                    "this fit window (SPEC OQ5 open for India VIX "
                                    "pre-Nov-2007)"
                                ),
                                context={
                                    "dropped": ",".join(dropped),
                                    "fitted_through": panel.dates[i],
                                },
                            )
                        )
                except HamiltonEstimationError as exc:
                    fit_end = i  # retry on the next cadence, not every day
                    if params is None or used_cols is None:
                        missing[i] = ("transient", "estimation_failed")
                        continue
                    warn(
                        SeraphWarning(
                            code="ESTIMATOR_FALLBACK",
                            message=(
                                "EM refit failed; previous parameter set retained "
                                f"({exc})"
                            ),
                            context={"date": panel.dates[i]},
                        )
                    )
                    cols = used_cols

            assert params is not None and used_cols is not None
            if not params.converged:
                warn(
                    SeraphWarning(
                        code="ESTIMATOR_FALLBACK",
                        message="EM hit em_max_iter without meeting em_tol",
                        context={
                            "fitted_through": params.fitted_through,
                            "n_iter": params.n_iter,
                        },
                    )
                )

            # Constructing the prefix factorises the regime covariances, which
            # is itself a place a degenerate fit can fail — so it lives inside
            # the same guard as the recursion. Nothing numerical may raise past
            # this method: the seam returns absence, never an exception.
            try:
                if prefix is None:
                    y_std, z_design = self._prefix_inputs(panel, params, used_cols)
                    prefix = _Prefix(params, y_std, z_design)
                xi, trans = prefix.advance_to(i)
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                missing[i] = ("transient", "estimation_failed")
                prefix = None
                continue

            if not (np.isfinite(xi).all() and np.isfinite(trans).all()):
                missing[i] = ("transient", "estimation_failed")
                prefix = None
                continue

            states[i] = _State(
                index=i,
                date=panel.dates[i],
                xi=xi.copy(),
                trans=trans.copy(),
                params=params,
            )

        cache.params = params
        cache.used_cols = used_cols
        cache.fit_end = fit_end
        if prefix is not None and prefix.prev is not None:
            cache.last_index = prefix.last
            cache.last_filtered = prefix.prev.copy()
        # The fingerprinted prefix must span every index the cache has an answer
        # for, not just the filter's position. `missing` can hold verdicts past
        # that position — a first fit that failed leaves the filter at -1 while
        # recording a transient absence — and fingerprinting only up to the
        # filter would let a source restate exactly the rows that produced the
        # failure and still be served the cached absence forever.
        resolved = max([cache.last_index, *states, *missing], default=-1)
        cache.n_rows = max(resolved + 1, 0)
        cache.fingerprint = panel_fingerprint(panel, cache.n_rows)

        return states, missing, warns
