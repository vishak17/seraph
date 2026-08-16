"""C7 -> C8 wiring: the composition root for the reconciliation run.

ARCHITECTURE draws two arrows into C9, not one:

    C9 --S6--> C8      the reconciled state (x_hat, P, mask, mode)
    C9 --xi--> C7      the regime probabilities that weight it (FR28)

A run that produces only the first is not scoreable. `CSRS_t = sum_j xi_{j,t}
w_j' x_hat_t` needs `xi_t` from the *same* timestamp as `x_hat_t`, and R5 is
explicit that Hamilton is load-bearing twice — it is both an element of `x_hat`
and the weighting over `x_hat`. So this module emits `ReconciledPoint`s that
carry both, aligned, and says so when `xi` is missing rather than letting C9
discover it.

What it composes:

    C4 (S4)  PanelFeatureSource / any HamiltonFeatureSource
      -> C7  HamiltonEngine, the S5 PillarEngine
      -> C8  ReconciliationRunner + a layer (kalman or forward_fill)
      -> S6  ReconciledState + xi, ready for C9
      -> E8/E9 rows, ready for C1 (FR38)

Both FR36 arms run through here by construction: `mode="kalman"` is production,
`mode="forward_fill"` is the ablation baseline, and nothing else about the run
differs between them — same engine, same emissions, same grid. That is what
makes the comparison attributable to the reconciliation layer rather than to
two different pipelines.

Owner note (SPEC §10): C7 and C8 are both owner C's, so this file wires two of
one owner's components. It imports C7 in exactly one place — the
`from_hamilton_source` constructor — and is otherwise typed against S5's
`PillarEngine`, so C5 and C6 slot in without touching it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from seraph.reconciliation.config import ReconciliationConfig
from seraph.reconciliation.fitting import NoiseFit, fit_noise_parameters
from seraph.reconciliation.layer import (
    ForwardFillReconciliationLayer,
    KalmanReconciliationLayer,
)
from seraph.reconciliation.output import e9_rows
from seraph.reconciliation.runner import ReconciliationRunner, RunReport
from seraph.shared_types import (
    Err,
    HamiltonDetail,
    ISODate,
    ISOTimestamp,
    PillarEmission,
    PillarEngine,
    ReconciledState,
    ReconciliationMode,
    Result,
    SeraphWarning,
    Simplex3,
    ok,
)

__all__ = [
    "PipelineRun",
    "ReconciledPoint",
    "ReconciliationPipeline",
    "RegimeDetailSource",
]


class RegimeDetailSource(Protocol):
    """Whatever can supply `xi_t` — in practice C7 (S5's detail sidecar).

    Structural, not nominal: C8 depends on the *shape* of C7's detail feed, not
    on the class. Substituting a recorded xi series for a backtest, or C10's
    `xiMode="uniform"` control, needs no change here.
    """

    async def details_range(
        self, from_ts: ISOTimestamp, to_ts: ISOTimestamp
    ) -> Result[tuple[tuple[ISOTimestamp, HamiltonDetail], ...]]: ...


@dataclass(frozen=True)
class ReconciledPoint:
    """One scoreable instant: everything C9's `score()` takes, at one `ts`."""

    ts: ISOTimestamp
    state: ReconciledState
    xi: Simplex3 | None
    """`None` when Hamilton produced no estimate at this tick.

    Not defaulted to uniform here: C9's `xiMode` is a *declared* control
    variant (S8, R5), and silently substituting uniform weights would make an
    unlabelled control row look like a Hamilton-weighted one. The caller picks,
    and the choice is recorded on the `CsrsPoint`.
    """

    @property
    def scoreable(self) -> bool:
        """True when C9 can score this point without an `xiMode` fallback."""
        return self.xi is not None and any(self.state.mask)


@dataclass(frozen=True)
class PipelineRun:
    """One end-to-end reconciliation run."""

    points: tuple[ReconciledPoint, ...]
    report: RunReport
    mode: ReconciliationMode
    cfg: ReconciliationConfig
    noise_fit: NoiseFit | None
    warnings: tuple[SeraphWarning, ...]

    @property
    def states(self) -> tuple[ReconciledState, ...]:
        return tuple(p.state for p in self.points)

    def e9(self) -> tuple[dict[str, object], ...]:
        """E9 rows for C1's `writeBatch` (FR38). See `output.py`."""
        return e9_rows(self.states)


class ReconciliationPipeline:
    """Builds the layer, drives the pillars, aligns `xi`, returns S6 + E9.

    Args:
        engines: S5 pillar engines. One, two or three of them — a pillar with
            no engine is masked out by D2, which is exactly how FR35's ablation
            subsets are expressed.
        detail_source: supplies `xi_t` (C7). Optional: without it every point
            carries `xi=None` and C9 must run its `uniform` control variant.
        cfg: C8 configuration.
        mode: `"kalman"` for production, `"forward_fill"` for FR27's baseline.
    """

    def __init__(
        self,
        engines: Sequence[PillarEngine],
        detail_source: RegimeDetailSource | None = None,
        cfg: ReconciliationConfig | None = None,
        mode: ReconciliationMode = "kalman",
    ) -> None:
        self.engines = list(engines)
        self.detail_source = detail_source
        self.cfg = cfg or ReconciliationConfig()
        self.mode: ReconciliationMode = mode

    @classmethod
    def from_hamilton_source(
        cls,
        source: object,
        hamilton_cfg: object | None = None,
        cfg: ReconciliationConfig | None = None,
        mode: ReconciliationMode = "kalman",
    ) -> ReconciliationPipeline:
        """Convenience wiring for the Hamilton-only run (C4 -> C7 -> C8).

        `source` is any `HamiltonFeatureSource` (S4). Imported lazily so that
        C8's package does not pull C7 in for callers that never ask for it —
        `pipeline.py` is the only place the two meet, and this is the only line
        that makes them meet.
        """
        from seraph.pillars.hamilton import HamiltonConfig, HamiltonEngine

        engine = HamiltonEngine(
            source,  # type: ignore[arg-type]
            hamilton_cfg if isinstance(hamilton_cfg, HamiltonConfig) else None,
        )
        return cls([engine], detail_source=engine, cfg=cfg, mode=mode)

    # -- the run --------------------------------------------------------------

    async def run(
        self,
        from_ts: ISOTimestamp,
        to_ts: ISOTimestamp,
        *,
        fit_noise_through: ISOTimestamp | None = None,
    ) -> Result[PipelineRun]:
        """Reconcile `[from_ts, to_ts]` and return scoreable points.

        Args:
            fit_noise_through: when given, D4's `h_p`, `R_max/R_0` and `Q_proc`
                are MLE-fitted on emissions up to this timestamp and the fitted
                configuration is used for the whole run. **It must be the end
                of the training fold.** Fitting through the evaluation window is
                leakage, and C10's `leakageCheck` cannot see into this call —
                it inspects `FusionWeights`, not C8's noise parameters.
        """
        warnings: list[SeraphWarning] = []
        cfg = self.cfg
        noise_fit: NoiseFit | None = None

        if fit_noise_through is not None:
            fitted = await self._fit_noise(from_ts, fit_noise_through)
            if isinstance(fitted, Err):
                return fitted
            noise_fit = fitted.value
            cfg = noise_fit.cfg
            warnings.extend(fitted.warnings)

        layer: KalmanReconciliationLayer | ForwardFillReconciliationLayer = (
            KalmanReconciliationLayer(cfg)
            if self.mode == "kalman"
            else ForwardFillReconciliationLayer(cfg)
        )
        runner = ReconciliationRunner(self.engines, layer)

        warnings.extend(await runner.prime_coverage())

        backfilled = await runner.backfill(from_ts, to_ts)
        if isinstance(backfilled, Err):
            return backfilled
        report = backfilled.value
        warnings.extend(backfilled.warnings)

        xi_by_ts = await self._xi_series(from_ts, to_ts, warnings)
        points = tuple(
            ReconciledPoint(ts=state.ts, state=state, xi=xi_by_ts.get(state.ts))
            for state in report.states
        )

        if points and not any(p.scoreable for p in points):
            warnings.append(
                SeraphWarning(
                    code="MASK_DEGRADED",
                    message=(
                        "no point in this run is scoreable without an xiMode "
                        "fallback — either Hamilton produced no regime estimate "
                        "or every mask is empty"
                    ),
                    context={"from": from_ts, "to": to_ts, "points": len(points)},
                )
            )

        return ok(
            PipelineRun(
                points=points,
                report=report,
                mode=self.mode,
                cfg=cfg,
                noise_fit=noise_fit,
                warnings=tuple(warnings),
            ),
            warnings=tuple(warnings),
        )

    # -- internals ------------------------------------------------------------

    async def _fit_noise(
        self, from_ts: ISOTimestamp, through_ts: ISOTimestamp
    ) -> Result[NoiseFit]:
        """Collect the training-fold emissions and run the D4 MLE on them."""
        emissions: list[PillarEmission] = []
        for engine in self.engines:
            result = await engine.emit_range(from_ts, through_ts)
            if isinstance(result, Err):
                return result
            emissions.extend(result.value)
        return fit_noise_parameters(emissions, self.cfg)

    async def _xi_series(
        self,
        from_ts: ISOTimestamp,
        to_ts: ISOTimestamp,
        warnings: list[SeraphWarning],
    ) -> dict[ISOTimestamp, Simplex3]:
        if self.detail_source is None:
            return {}
        result = await self.detail_source.details_range(from_ts, to_ts)
        if isinstance(result, Err):
            warnings.append(
                SeraphWarning(
                    code="PARTIAL_COVERAGE",
                    message=(
                        "regime detail unavailable for this range; points carry "
                        "xi=None and C9 must use its uniform control variant"
                    ),
                    context={"error": result.error.kind},
                )
            )
            return {}
        return {ts: detail.xi for ts, detail in result.value}


def trading_grid(dates: Sequence[ISODate], close: str = "15:30:00") -> tuple[str, ...]:
    """ISO dates -> NSE-close timestamps, IST-explicit (AGENTS.md §5).

    A convenience for callers holding a date list; C8 still owns no calendar.
    """
    return tuple(f"{d}T{close}+05:30" for d in dates)
