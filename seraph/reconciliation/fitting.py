"""C8 — maximum-likelihood fit of `h_p`, `R_max/R_0` and `Q_proc` (D4).

D4 fixed the *form* of `R^(p)(.)` and closed SPEC OQ10, but explicitly left the
parameters to estimation: "`h_p` and the `R_max/R_0` ratio, together with
`Q_proc`, are fitted by maximum likelihood on the same rolling windows as the
pillars". This module is that fit. Without it the initialisations in
`config.py` — native cadence, 100x, and an OQ9 placeholder for `Q_proc` — are
the whole model, and the D4 sweep CT-4 runs has nothing to compare against.

**The objective.** A Kalman filter's likelihood is the product of its one-step
innovation densities. `KalmanReconciliationLayer` accumulates exactly that in
`log_likelihood`, over genuine updates only, so the objective here is a replay:
build a layer at the candidate parameters, push the emission history through
it, read the number back. The fit therefore optimises the code that runs in
production rather than a second implementation of it — the usual way a fitted
parameter set ends up subtly mismatched with the filter that consumes it.

**What is and is not identified.** Three honest limits, reported rather than
hidden:

* A pillar with no updates in the history identifies nothing; its parameters
  are held at their initialisation. Hawkes before ~2015 is precisely this case
  (SPEC §4), and it is why the fit is per-pillar rather than global.
* `h_p` is identified only by *variation in staleness*. A pillar that is always
  fresh (`Delta = 0` at every arrival) never exercises the ageing curve, so
  `h_p` is held fixed and `HELD` is reported for it.
* `Q_proc` and `R_max` trade off against each other — both inflate `P` between
  arrivals. They are jointly identified only because they enter on different
  time profiles (linear in `dt` versus saturating in `Delta`). With a short
  history that separation is weak; the per-pillar update counts are returned so
  a caller can judge, and FR34's sensitivity sweep is the intended check.

The optimiser is deterministic L-BFGS-B from a fixed start, so a run is
reproducible from the emission stream alone (NFR21).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

from seraph.reconciliation.config import SECONDS_PER_DAY, ReconciliationConfig
from seraph.reconciliation.layer import KalmanReconciliationLayer
from seraph.reconciliation.noise_model import epoch_seconds
from seraph.shared_types import (
    PILLAR_INDEX,
    PILLARS,
    Err,
    EstimationDiverged,
    InsufficientHistory,
    ObservedEmission,
    PillarEmission,
    Result,
    SeraphWarning,
    Vec3,
    err,
    ok,
)

__all__ = ["NoiseFit", "fit_noise_parameters"]

# Bounds, in natural units. Wide enough not to bind in practice, tight enough
# that the optimiser cannot wander into a region where the filter is degenerate.
H_BOUNDS_SECONDS = (60.0, 30.0 * SECONDS_PER_DAY)
R_MAX_RATIO_BOUNDS = (2.0, 1.0e4)
Q_BOUNDS_PER_DAY = (1.0e-8, 1.0e2)

# Below this many genuine updates a pillar's parameters are held at their
# initialisation. Roughly a quarter of daily emissions — enough for the ageing
# curve to have been exercised more than incidentally.
MIN_UPDATES_PER_PILLAR = 30

# Staleness spread (in units of the initial `h_p`) below which `h_p` is treated
# as unidentified: every arrival was equally fresh, so nothing in the data
# distinguishes one ageing scale from another.
MIN_STALENESS_SPREAD = 1.0e-3

# Restart the optimiser from its own answer until a pass gains less than this
# fraction of the objective. Three passes is empirically enough for the fit to
# become idempotent on the streams C8 sees; the loop exits early when it is.
_MAX_RESTARTS = 3
_RESTART_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class NoiseFit:
    """Result of the D4 parameter fit."""

    cfg: ReconciliationConfig
    """`base` with the fitted parameters substituted — pass straight to a layer."""

    log_likelihood: float
    initial_log_likelihood: float
    updates: tuple[int, int, int]
    """Genuine updates per pillar, `PILLAR_ORDER`. The fit's actual sample size."""
    fitted_pillars: tuple[bool, bool, bool]
    h_identified: tuple[bool, bool, bool]
    converged: bool
    n_evaluations: int

    @property
    def improvement(self) -> float:
        """Log-likelihood gained over the D4 initialisation. Never negative."""
        return self.log_likelihood - self.initial_log_likelihood


def _replay(
    ordered: tuple[PillarEmission, ...], cfg: ReconciliationConfig
) -> tuple[float, tuple[int, int, int]]:
    """Push the whole history through a fresh layer; read the likelihood back.

    `ordered` must already have been through `prepare()` — validation happens
    once per fit, not once per likelihood evaluation.
    """
    layer = KalmanReconciliationLayer(cfg)
    layer.replay(ordered)
    return layer.log_likelihood, layer.update_counts()


def _staleness_spread(emissions: tuple[PillarEmission, ...]) -> Vec3:
    """Per-pillar spread of `Delta = ts - tau`, in seconds.

    Only genuinely new observations count — a re-served estimate carries a
    larger `Delta` but no new information, and the filter skips it (FR26), so
    counting it here would claim identification the fit does not have.
    """
    deltas: list[list[float]] = [[], [], []]
    last_tau: list[str | None] = [None, None, None]
    for emission in emissions:
        if not isinstance(emission, ObservedEmission):
            continue
        obs = emission.obs
        index = PILLAR_INDEX[obs.pillar]
        if last_tau[index] is not None and epoch_seconds(obs.tau) <= epoch_seconds(
            last_tau[index]  # type: ignore[arg-type]
        ):
            continue
        last_tau[index] = obs.tau
        deltas[index].append(epoch_seconds(obs.ts) - epoch_seconds(obs.tau))
    spread = [max(d) - min(d) if d else 0.0 for d in deltas]
    return (spread[0], spread[1], spread[2])


def fit_noise_parameters(
    emissions: Sequence[PillarEmission],
    base: ReconciliationConfig | None = None,
    *,
    min_updates: int = MIN_UPDATES_PER_PILLAR,
    max_iterations: int = 200,
) -> Result[NoiseFit]:
    """Fit D4's free parameters on an emission history.

    Args:
        emissions: the S5 stream to fit on — typically one pillar-covered
            training window. Must be the *training* fold only: fitting on an
            epoch that is later evaluated is leakage, and C10's `leakageCheck`
            has no visibility into this call.
        base: starting configuration; its `noise_form` is respected and never
            fitted (D4 fixed the form — only its parameters are free).
        min_updates: per-pillar floor below which parameters are held.

    Returns `Err(INSUFFICIENT_HISTORY)` when no pillar clears `min_updates`,
    and `Err(ESTIMATION_DIVERGED)` when the optimiser cannot improve on a
    finite likelihood.
    """
    cfg = base or ReconciliationConfig()
    stream = tuple(emissions)
    if not stream:
        return err(
            InsufficientHistory(required=min_updates, available=0, as_of="1970-01-01")
        )

    # Validate and sort once; every likelihood evaluation replays this same
    # prepared stream (see `KalmanReconciliationLayer.replay`).
    prepared = KalmanReconciliationLayer(cfg).prepare(stream)
    if isinstance(prepared, Err):
        return prepared
    initial_loglik, updates = _replay(prepared, cfg)

    fitted = (
        updates[0] >= min_updates,
        updates[1] >= min_updates,
        updates[2] >= min_updates,
    )
    if not any(fitted):
        return err(
            InsufficientHistory(
                required=min_updates,
                available=max(updates),
                as_of=_last_ts(stream)[:10],
            )
        )

    spread = _staleness_spread(stream)
    identified = [
        fitted[i] and spread[i] > MIN_STALENESS_SPREAD * cfg.h_seconds[i]
        for i in range(len(PILLARS))
    ]
    h_identified = (identified[0], identified[1], identified[2])

    # `R_max` only exists for the bounded form; for the sweep's linear/power
    # variants the ratio is not a parameter of the model at all.
    fit_ratio = cfg.noise_form == "saturating_exponential"

    theta0: list[float] = []
    bounds: list[tuple[float, float]] = []
    slots: list[tuple[str, int]] = []

    for i in range(len(PILLARS)):
        if h_identified[i]:
            theta0.append(math.log(cfg.h_seconds[i]))
            bounds.append(
                (math.log(H_BOUNDS_SECONDS[0]), math.log(H_BOUNDS_SECONDS[1]))
            )
            slots.append(("h", i))
    for i in range(len(PILLARS)):
        if fitted[i]:
            theta0.append(math.log(cfg.q_proc_per_day[i]))
            bounds.append(
                (math.log(Q_BOUNDS_PER_DAY[0]), math.log(Q_BOUNDS_PER_DAY[1]))
            )
            slots.append(("q", i))
    if fit_ratio:
        theta0.append(math.log(cfg.r_max_ratio))
        bounds.append(
            (math.log(R_MAX_RATIO_BOUNDS[0]), math.log(R_MAX_RATIO_BOUNDS[1]))
        )
        slots.append(("ratio", 0))

    if not theta0:
        # Nothing free to move: report the initialisation honestly instead of
        # dressing it up as a fit.
        return ok(
            NoiseFit(
                cfg=cfg,
                log_likelihood=initial_loglik,
                initial_log_likelihood=initial_loglik,
                updates=updates,
                fitted_pillars=fitted,
                h_identified=h_identified,
                converged=True,
                n_evaluations=1,
            ),
            warnings=(_held_warning(fitted, h_identified),),
        )

    evaluations = 0

    def unpack(theta: np.ndarray) -> ReconciliationConfig:
        h = list(cfg.h_seconds)
        q = list(cfg.q_proc_per_day)
        ratio = cfg.r_max_ratio
        for value, (kind, index) in zip(theta, slots, strict=True):
            if kind == "h":
                h[index] = math.exp(value)
            elif kind == "q":
                q[index] = math.exp(value)
            else:
                ratio = math.exp(value)
        return replace(
            cfg,
            h_seconds=(h[0], h[1], h[2]),
            q_proc_per_day=(q[0], q[1], q[2]),
            r_max_ratio=ratio,
        )

    def objective(theta: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        try:
            candidate = unpack(theta)
        except ValueError:
            return float(np.inf)  # a bound-violating config; steer away
        loglik, _ = _replay(prepared, candidate)
        if not math.isfinite(loglik):
            return float(np.inf)
        return -loglik

    # L-BFGS-B differentiates this objective numerically, and a replayed
    # likelihood is flat in some directions (`h_p` barely moves the answer when
    # arrivals are nearly fresh). One pass therefore stops well short of the
    # optimum — measurably: re-fitting from its own answer used to gain another
    # ~1% of the log-likelihood. Restarting from the previous point until it
    # stops paying makes the fit idempotent, which is what lets a caller trust
    # `improvement` as a real quantity rather than an artefact of where the
    # optimiser happened to stall. `eps` is widened from the 1e-8 default for
    # the same reason: in log-space that step is inside the objective's own
    # numerical noise.
    theta = np.array(theta0, dtype=float)
    best_objective = objective(theta)
    result = None
    for _ in range(_MAX_RESTARTS):
        result = minimize(
            objective,
            theta,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations, "eps": 1.0e-6},
        )
        gain = best_objective - float(result.fun)
        if gain <= _RESTART_TOLERANCE * max(1.0, abs(best_objective)):
            break
        best_objective = float(result.fun)
        theta = np.asarray(result.x, dtype=float)
    assert result is not None

    best_cfg = unpack(
        theta if float(result.fun) > best_objective else np.asarray(result.x, float)
    )
    final_loglik, final_updates = _replay(prepared, best_cfg)

    if not math.isfinite(final_loglik):
        return err(
            EstimationDiverged(
                estimator="C8:D4-noise-mle",
                iterations=evaluations,
                last_objective=float("nan"),
            )
        )

    # L-BFGS-B can return a point marginally worse than the start on a flat
    # surface. Keeping the better of the two makes `improvement >= 0` a real
    # guarantee rather than a hope.
    if final_loglik < initial_loglik:
        best_cfg, final_loglik, final_updates = cfg, initial_loglik, updates

    warnings: list[SeraphWarning] = []
    if not all(fitted) or not all(h_identified):
        warnings.append(_held_warning(fitted, h_identified))
    pinned = _pinned_parameters(_theta_of(best_cfg, cfg, slots), bounds, slots)
    if pinned:
        warnings.append(
            SeraphWarning(
                code="ESTIMATOR_FALLBACK",
                message=(
                    "D4 parameter sat on its bound rather than at an interior "
                    "optimum — the history does not pin it down, so treat it as "
                    "a floor/ceiling artefact, not an estimate"
                ),
                context={"pinned": ",".join(pinned)},
            )
        )
    if not bool(result.success):
        warnings.append(
            SeraphWarning(
                code="ESTIMATOR_FALLBACK",
                message=(
                    "D4 noise MLE did not converge; best point found is returned "
                    "and is never worse than the D4 initialisation"
                ),
                context={"message": str(result.message), "evaluations": evaluations},
            )
        )

    return ok(
        NoiseFit(
            cfg=best_cfg,
            log_likelihood=final_loglik,
            initial_log_likelihood=initial_loglik,
            updates=final_updates,
            fitted_pillars=fitted,
            h_identified=h_identified,
            converged=bool(result.success),
            n_evaluations=evaluations,
        ),
        warnings=tuple(warnings),
    )


def _theta_of(
    fitted: ReconciliationConfig,
    base: ReconciliationConfig,
    slots: list[tuple[str, int]],
) -> np.ndarray:
    """Re-derive the optimiser's coordinates from the configuration returned.

    Read from `best_cfg` rather than from `result.x`, so the bound check
    describes the parameters actually being shipped — including the case where
    the fit fell back to the D4 initialisation.
    """
    values: list[float] = []
    for kind, index in slots:
        if kind == "h":
            values.append(math.log(fitted.h_seconds[index]))
        elif kind == "q":
            values.append(math.log(fitted.q_proc_per_day[index]))
        else:
            values.append(math.log(fitted.r_max_ratio))
    return np.array(values, dtype=float)


def _pinned_parameters(
    theta: np.ndarray,
    bounds: list[tuple[float, float]],
    slots: list[tuple[str, int]],
) -> list[str]:
    """Names of parameters the optimiser drove onto a bound.

    Worth reporting rather than silently accepting: a bound-pinned `h_p` means
    the likelihood is monotone in it over the admissible range, i.e. the data
    do not identify the ageing scale at all — the same situation as
    `h_identified = False`, just detected after the fact instead of before it.
    """
    pinned: list[str] = []
    for value, (low, high), (kind, index) in zip(theta, bounds, slots, strict=True):
        name = f"{kind}[{PILLARS[index]}]" if kind != "ratio" else "r_max_ratio"
        if abs(value - low) < 1e-8 or abs(value - high) < 1e-8:
            pinned.append(name)
    return pinned


def _held_warning(
    fitted: tuple[bool, ...], h_identified: tuple[bool, ...]
) -> SeraphWarning:
    held = [
        PILLARS[i] for i in range(len(PILLARS)) if not fitted[i] or not h_identified[i]
    ]
    return SeraphWarning(
        code="PARTIAL_COVERAGE",
        message=(
            "D4 parameters held at their initialisation for pillars the history "
            "does not identify (too few updates, or no variation in staleness)"
        ),
        context={"held": ",".join(held)},
    )


def _last_ts(emissions: tuple[PillarEmission, ...]) -> str:
    last = emissions[-1]
    return last.obs.ts if isinstance(last, ObservedEmission) else last.ts
