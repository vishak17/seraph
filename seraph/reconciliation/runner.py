"""C8 — the S5 driver: `C8 -> C5 / C6 / C7` (ARCHITECTURE §3).

The dependency graph points *from* C8 to the three pillar engines: C8 pulls,
the pillars do not push. `ReconciliationRunner` is that arrow — it holds
`PillarEngine` implementations, primes their D2 coverage windows, polls them
across a timestamp grid and steps the filter.

Everything here is policy, not mathematics; the recursion lives in `layer.py`.
Three policies, each of which would otherwise be improvised differently at
every call site:

1.  **A pillar's `Err` is a transient absence, not a dead run.** When C7's EM
    fails or a store read times out, that pillar's `Result` is an `Err`. The
    filter's answer to "this pillar has nothing for me right now" is already
    defined by D2/D4 — so the runner converts it to
    `unavailable/transient/estimation_failed` and keeps going, surfacing the
    original error as a warning. A failed Hawkes fit must not stop the RMT and
    Hamilton signals from reaching the CSRS. `transient`, never `structural`:
    a failure is not a statement about whether the pillar can exist.
2.  **Coverage is primed once, up front.** `PillarEngine.coverage()` is the
    only way C8 can distinguish "silent today" from "cannot exist this decade"
    (SPEC §4). An engine that cannot report coverage runs without one, and the
    layer falls back to what its emissions say.
3.  **The grid is the caller's.** C8 owns no trading calendar — C1/C3 do.
    The caller supplies timestamps; the runner does not invent them.

The runner never touches the store and imports no component package: it is
typed against S5's `PillarEngine` protocol, so C5/C6/C7 plug in unchanged and
a test stub plugs in identically.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from seraph.reconciliation.layer import (
    ForwardFillReconciliationLayer,
    KalmanReconciliationLayer,
)
from seraph.reconciliation.noise_model import epoch_seconds
from seraph.shared_types import (
    Err,
    ISOTimestamp,
    ObservedEmission,
    PillarEmission,
    PillarEngine,
    PillarId,
    ReconciledState,
    Result,
    SeraphWarning,
    UnavailableEmission,
    ok,
)

__all__ = ["ReconciliationRunner", "RunReport"]

type AnyLayer = KalmanReconciliationLayer | ForwardFillReconciliationLayer


def _emission_ts(emission: PillarEmission) -> ISOTimestamp:
    return emission.obs.ts if isinstance(emission, ObservedEmission) else emission.ts


@dataclass(frozen=True)
class RunReport:
    """What a run did, for C10's provenance and for debugging a flat CSRS."""

    states: tuple[ReconciledState, ...]
    updates: tuple[int, int, int]
    redeliveries: tuple[int, int, int]
    engine_errors: tuple[int, int, int]


class ReconciliationRunner:
    """Drives S5 `PillarEngine`s into a reconciliation layer.

    Args:
        engines: one per pillar, in any order. A pillar with no engine is
            simply never observed — the D2 mask excludes it, which is the
            correct behaviour for an ablation subset that leaves a pillar out
            (FR35's 7 subsets are expressed exactly this way).
        layer: the Kalman layer in production, the forward-fill layer for the
            FR36 baseline arm. Both satisfy S6.
    """

    def __init__(self, engines: Sequence[PillarEngine], layer: AnyLayer) -> None:
        self.layer = layer
        self.engines: dict[PillarId, PillarEngine] = {}
        for engine in engines:
            if engine.pillar in self.engines:
                raise ValueError(f"two engines supplied for pillar {engine.pillar!r}")
            self.engines[engine.pillar] = engine
        self._errors: dict[PillarId, int] = {p: 0 for p in self.engines}

    async def prime_coverage(self) -> tuple[SeraphWarning, ...]:
        """Ask every engine for its D2 window (S5 `coverage()`).

        Call once before stepping. An engine that returns `Err` here is not
        fatal: the layer then relies on that pillar's own `structural`
        emissions, which is what CT-4's mock stream does.
        """
        warnings: list[SeraphWarning] = []
        for pillar, engine in self.engines.items():
            try:
                result = await engine.coverage()
            except Exception as exc:  # an engine that raises instead of returning
                warnings.append(self._engine_warning(pillar, f"raised: {exc!r}"))
                continue
            if isinstance(result, Err):
                warnings.append(
                    self._engine_warning(pillar, f"coverage: {result.error.kind}")
                )
                continue
            self.layer.declare_coverage(pillar, result.value)
        return tuple(warnings)

    async def step(self, ts: ISOTimestamp) -> Result[ReconciledState]:
        """Poll every engine at `ts` and fold the results into the filter.

        With no engines at all this degrades to a pure FR25 predict, which is
        the right answer rather than an error: time passed, nothing was said.
        """
        emissions: list[PillarEmission] = []
        warnings: list[SeraphWarning] = []

        for pillar, engine in self.engines.items():
            emission, warning = await self._poll(pillar, engine, ts)
            emissions.append(emission)
            if warning is not None:
                warnings.append(warning)

        if not emissions:
            result = self.layer.advance(ts)
        else:
            result = self.layer.ingest(tuple(emissions))

        if isinstance(result, Err):
            return result
        return ok(result.value, warnings=result.warnings + tuple(warnings))

    async def run(self, grid: Sequence[ISOTimestamp]) -> Result[RunReport]:
        """Step the whole grid, oldest first, stopping at the first `Err`.

        Stopping is deliberate: an `Err` from the layer is a contract
        violation, not a data gap (data gaps are emissions), and continuing
        past one would produce states whose provenance nobody can reconstruct.
        """
        states: list[ReconciledState] = []
        for ts in grid:
            result = await self.step(ts)
            if isinstance(result, Err):
                return result
            states.append(result.value)
        return ok(self.report(tuple(states)))

    async def backfill(
        self, from_ts: ISOTimestamp, to_ts: ISOTimestamp
    ) -> Result[RunReport]:
        """Historical run: one `emit_range` per engine, then one pass of ticks.

        `run()` polls every engine on every tick, which is right for live
        operation and wrong for a twenty-year corpus — it is one round trip per
        pillar per tick, and S5 defines `emit_range` precisely so a backfill
        does not have to pay that. The tick grid is then whatever timestamps
        the pillars actually emitted on, unioned: C8 owns no calendar and will
        not invent dates the pillars did not speak on.

        The recursion is identical either way, so a backfilled state and a
        live-stepped state for the same emissions are the same state — which is
        what makes C10's "byte-identical folds" (O6) hold across both.
        """
        emissions: list[PillarEmission] = []
        warnings: list[SeraphWarning] = []

        for pillar, engine in self.engines.items():
            try:
                result = await engine.emit_range(from_ts, to_ts)
            except Exception as exc:  # an engine that raises instead of returning
                self._errors[pillar] = self._errors.get(pillar, 0) + 1
                warnings.append(self._engine_warning(pillar, f"raised: {exc!r}"))
                continue
            if isinstance(result, Err):
                self._errors[pillar] = self._errors.get(pillar, 0) + 1
                warnings.append(
                    self._engine_warning(pillar, f"emit_range: {result.error.kind}")
                )
                continue
            emissions.extend(result.value)

        grouped: dict[ISOTimestamp, list[PillarEmission]] = {}
        for emission in emissions:
            grouped.setdefault(_emission_ts(emission), []).append(emission)

        states: list[ReconciledState] = []
        for ts in sorted(grouped, key=epoch_seconds):
            stepped = self.layer.ingest(tuple(grouped[ts]))
            if isinstance(stepped, Err):
                return stepped
            states.append(stepped.value)
            warnings.extend(stepped.warnings)

        return ok(self.report(tuple(states)), warnings=tuple(warnings))

    def report(self, states: tuple[ReconciledState, ...]) -> RunReport:
        return RunReport(
            states=states,
            updates=self.layer.update_counts(),
            redeliveries=self.layer.redelivery_counts(),
            engine_errors=(
                self._errors.get("hawkes", 0),
                self._errors.get("rmt", 0),
                self._errors.get("hamilton", 0),
            ),
        )

    # -- internals ------------------------------------------------------------

    async def _poll(
        self, pillar: PillarId, engine: PillarEngine, ts: ISOTimestamp
    ) -> tuple[PillarEmission, SeraphWarning | None]:
        try:
            result = await engine.emit(ts)
        except Exception as exc:  # a pillar that raises instead of returning Err
            return self._degrade(pillar, ts, f"raised: {exc!r}")
        if isinstance(result, Err):
            return self._degrade(pillar, ts, result.error.kind)
        return result.value, None

    def _degrade(
        self, pillar: PillarId, ts: ISOTimestamp, detail: str
    ) -> tuple[PillarEmission, SeraphWarning]:
        """Policy 1: an engine failure becomes a transient absence."""
        self._errors[pillar] = self._errors.get(pillar, 0) + 1
        emission = UnavailableEmission(
            pillar=pillar,
            ts=ts,
            absence="transient",
            reason="estimation_failed",
        )
        return emission, self._engine_warning(pillar, detail, ts)

    @staticmethod
    def _engine_warning(
        pillar: PillarId, detail: str, ts: ISOTimestamp | None = None
    ) -> SeraphWarning:
        context: dict[str, str | float] = {"pillar": pillar, "detail": detail}
        if ts is not None:
            context["ts"] = ts
        return SeraphWarning(
            code="ESTIMATOR_FALLBACK",
            message=(
                f"{pillar} engine did not return a usable emission; treated as "
                f"transient absence (D2 keeps it in the mask, D4 inflates its noise)"
            ),
            context=context,
        )
