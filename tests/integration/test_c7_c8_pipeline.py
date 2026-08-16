"""C4 -> C7 -> C8, end to end on generated market data.

The unit suites prove each seam in isolation against doubles. This one runs the
real chain: OHLC bars -> `y_t`/`z_t` (C4) -> EM + TVTP filter (C7) -> S5
emissions -> Kalman recursion (C8) -> S6 states paired with `xi`, ready for C9.

What it is *not*: a validation of C7's statistical accuracy. Regime recovery on
generated data is a property of the generator as much as the estimator, and
FR35's ablation over six labelled epochs is where that question is settled
(C10, owner D). The assertions here are integration properties — alignment,
honesty of `tau`, D2 masking through the warm-up, both FR36 arms on one grid,
and the fact that the regime signal survives the seam at all.

C7 is used read-only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from functools import lru_cache
from typing import Any

import numpy as np
import pytest

from fixtures.synthetic_market import SyntheticMarket, generate_market
from seraph.features import FeatureConfig, PanelFeatureSource
from seraph.pillars.hamilton import HamiltonConfig
from seraph.reconciliation import (
    KalmanReconciliationLayer,
    PipelineRun,
    ReconciliationPipeline,
    ReconciliationRunner,
)
from seraph.reconciliation.output import E9_NATURAL_KEY
from seraph.shared_types import SIMPLEX_TOLERANCE
from tests.reconciliation.helpers import unwrap

pytestmark = pytest.mark.slow

FROM_TS = "2019-01-01T15:30:00+05:30"
TO_TS = "2099-12-31T15:30:00+05:30"

# Small enough to keep one EM fit per run in the low seconds, long enough to
# clear C7's 250-day history floor plus C4's 63-day absorption warm-up.
N_DAYS = 560
SEED = 5

HAMILTON_CFG = HamiltonConfig(
    corpus_start="2019-01-01",
    min_history_days=250,
    refit_every_days=250,
    em_max_iter=60,
    logit_max_iter=40,
)


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


@lru_cache(maxsize=1)
def market() -> SyntheticMarket:
    return generate_market(n_days=N_DAYS, n_symbols=30, seed=SEED)


def source() -> PanelFeatureSource:
    m = market()
    return PanelFeatureSource(
        panel=m.panel, macro=m.macro, events=m.events, cfg=FeatureConfig()
    )


@lru_cache(maxsize=2)
def pipeline_run(mode: str = "kalman") -> PipelineRun:
    pipe = ReconciliationPipeline.from_hamilton_source(
        source(),
        HAMILTON_CFG,
        mode=mode,  # type: ignore[arg-type]
    )
    return unwrap(_run(pipe.run(FROM_TS, TO_TS)))


# -- the seam ----------------------------------------------------------------


def test_the_chain_produces_scoreable_points() -> None:
    """One `ReconciledPoint` per trading day C7 spoke on, and enough of them
    carrying both halves C9's `score()` needs."""
    run = pipeline_run()

    assert run.points, "C4 -> C7 -> C8 produced nothing"
    assert all(p.ts == p.state.ts for p in run.points)
    assert [p.ts for p in run.points] == sorted(p.ts for p in run.points)

    scoreable = [p for p in run.points if p.scoreable]
    assert len(scoreable) > 100, "too few scoreable points to be a real run"
    assert run.report.engine_errors == (0, 0, 0)


def test_xi_is_a_simplex_and_lines_up_with_its_own_state() -> None:
    """FR28 multiplies `xi_t` by `w_j' x_hat_t`; a mismatched pair would be a
    silent off-by-one between two components, which is the exact failure the
    pipeline exists to make impossible."""
    for point in pipeline_run().points:
        if point.xi is None:
            continue
        assert abs(sum(point.xi) - 1.0) <= SIMPLEX_TOLERANCE
        assert all(v >= 0.0 for v in point.xi)
        # Hamilton is in the mask exactly when it produced this xi.
        assert point.state.mask[2] is True
        assert point.state.tau_last_update[2] is not None


def test_the_warmup_is_structurally_absent_not_zero() -> None:
    """C4 needs 63 days for `AR_t` and C7 needs 250 for a TVTP fit. Those dates
    genuinely have no Hamilton estimate — D2 must exclude them rather than let
    a drifting prior into the CSRS."""
    run = pipeline_run()
    early = run.points[0]

    assert early.xi is None
    assert early.state.mask[2] is False
    assert early.state.tau_last_update[2] is None
    # ...and the other two pillars are absent for the whole run: no engine.
    assert all(
        p.state.mask[0] is False and p.state.mask[1] is False for p in run.points
    )


def test_tau_is_the_close_that_produced_the_estimate() -> None:
    """S5's central honesty requirement: `tau` is when C7 computed the value,
    which is what makes C8's `R^(p)(ts - tau)` mean anything."""
    for point in pipeline_run().points:
        tau = point.state.tau_last_update[2]
        if tau is None:
            continue
        assert tau.endswith("T15:30:00+05:30")
        assert tau <= point.ts


def test_no_redeliveries_when_each_day_is_emitted_once() -> None:
    """`emit_range` yields one estimate per trading date, each with its own
    `tau`, so FR26's new-information rule should never fire here. If it does,
    either C7 is repeating itself or C8 is mis-reading `tau`."""
    run = pipeline_run()
    assert run.report.redeliveries == (0, 0, 0)
    assert run.report.updates[2] == sum(1 for p in run.points if p.xi is not None)


# -- FR36: both arms, one grid ------------------------------------------------


def test_both_reconciliation_modes_run_on_an_identical_grid() -> None:
    """FR36 isolates the reconciliation layer's contribution, which is only
    meaningful if everything else about the two runs is the same."""
    kalman = pipeline_run("kalman")
    forward_fill = pipeline_run("forward_fill")

    assert [p.ts for p in kalman.points] == [p.ts for p in forward_fill.points]
    assert [p.state.mask for p in kalman.points] == [
        p.state.mask for p in forward_fill.points
    ]
    assert [p.xi for p in kalman.points] == [p.xi for p in forward_fill.points]
    assert kalman.points[-1].state.mode == "kalman"
    assert forward_fill.points[-1].state.mode == "forward_fill"


def test_the_two_arms_give_genuinely_different_trajectories() -> None:
    """If the FR36 comparison is to mean anything, the arms must differ in the
    state itself and not merely in a label.

    Both `P` series move, because both use the same rolling `R_0` — the
    difference is that forward-fill's has no staleness term in it at all. That
    age-response property is pinned directly in
    `tests/reconciliation/test_layer.py`; what belongs here is the end-to-end
    consequence, which is that the two arms hand C9 different numbers.
    """
    kalman = pipeline_run("kalman").points
    forward_fill = pipeline_run("forward_fill").points

    kalman_x = np.array([p.state.x_hat[2] for p in kalman])
    ff_x = np.array([p.state.x_hat[2] for p in forward_fill])
    kalman_p = np.array([p.state.p_t[2][2] for p in kalman])
    ff_p = np.array([p.state.p_t[2][2] for p in forward_fill])

    assert not np.allclose(kalman_x, ff_x)
    assert not np.allclose(kalman_p, ff_p)

    # And a finding worth recording rather than hiding: on THIS run the two
    # estimates correlate above 0.99. That is correct, not a bug. One pillar
    # emitting every trading day is never stale, so `R^(p)(0) = R_0` is small
    # against `P` and the Kalman gain sits near 1 — the filter reproduces the
    # observation because reproducing it is the right answer. O6's advantage
    # lives in the gaps (a missing pillar, a stale one, a pre-2015 epoch), and
    # CT-4 is where that is measured. A dense single-pillar run is the case
    # where forward-fill looks best, and the ablation should say so.
    assert float(np.corrcoef(kalman_x, ff_x)[0, 1]) > 0.9


# -- O6: identical folds ------------------------------------------------------


def test_backfill_and_tick_by_tick_produce_the_same_states() -> None:
    """O6's acceptance leans on "byte-identical folds". `backfill()` exists for
    corpus runs and `run()` for live operation; if they disagreed, a backtest
    would not describe the system that runs in production."""
    from seraph.pillars.hamilton import HamiltonEngine

    grid_engine = HamiltonEngine(source(), HAMILTON_CFG)
    backfilled = unwrap(
        _run(
            ReconciliationRunner([grid_engine], KalmanReconciliationLayer()).backfill(
                FROM_TS, TO_TS
            )
        )
    )
    grid = [s.ts for s in backfilled.states]

    stepped = unwrap(
        _run(
            ReconciliationRunner(
                [HamiltonEngine(source(), HAMILTON_CFG)], KalmanReconciliationLayer()
            ).run(grid)
        )
    )

    assert len(stepped.states) == len(backfilled.states)
    for a, b in zip(backfilled.states, stepped.states, strict=True):
        assert a.model_dump() == b.model_dump()


# -- what reaches C9 ----------------------------------------------------------


def test_the_regime_signal_survives_the_reconciliation_layer() -> None:
    """`x_hat[2]` is C7's `LSD_t` after C8 has filtered it. It should still
    track the regime the data was generated from — the filter is there to
    smooth arrival noise, not to erase the signal.

    Correlation, not classification accuracy: `y_t` is built from 21- and
    63-day rolling windows, so a regime shorter than one window is invisible in
    principle and a hit-rate assertion would be testing the generator.
    """
    run = pipeline_run()
    m = market()
    lsd, truth = [], []
    for point in run.points:
        if point.xi is None:
            continue
        lsd.append(point.state.x_hat[2])
        truth.append(m.regimes[m.panel.dates.index(point.ts[:10])])

    assert len(lsd) > 100
    correlation = float(np.corrcoef(np.asarray(lsd), np.asarray(truth))[0, 1])
    assert (
        correlation > 0.3
    ), f"reconciled LSD_t barely tracks the regime: {correlation}"


def test_e9_rows_are_archivable_and_uniquely_keyed() -> None:
    """FR38: C8 owns E9. C1 upserts on `(ts, mode)`, so the run must not
    produce two rows for one key."""
    rows = pipeline_run().e9()
    assert len(rows) == len(pipeline_run().points)

    keys = {tuple(str(row[k]) for k in E9_NATURAL_KEY) for row in rows}
    assert len(keys) == len(rows)

    first = rows[0]
    assert set(first) == {
        "ts",
        "x_hat",
        "P_t",
        "tau_last_update",
        "mode",
        "mask",
        "noise_saturated",
    }
    assert len(first["x_hat"]) == 3  # type: ignore[arg-type]
    assert len(first["P_t"]) == 3  # type: ignore[arg-type]


def test_the_d4_noise_fit_runs_on_the_training_fold_only() -> None:
    """SPEC OQ10's MLE, wired into the pipeline. The cut-off is the caller's
    training boundary — fitting through the evaluation window is leakage C10
    cannot detect from the `FusionWeights` artefact."""
    pipe = ReconciliationPipeline.from_hamilton_source(source(), HAMILTON_CFG)
    train_end = market().panel.dates[int(0.7 * N_DAYS)]
    run = unwrap(
        _run(pipe.run(FROM_TS, TO_TS, fit_noise_through=f"{train_end}T15:30:00+05:30"))
    )

    assert run.noise_fit is not None
    assert run.noise_fit.fitted_pillars == (False, False, True)
    assert run.noise_fit.improvement >= 0.0
    assert run.cfg == run.noise_fit.cfg
    assert run.points
