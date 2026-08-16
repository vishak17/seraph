"""C7 — HamiltonEngine, the S5 seam (FR23, FR24, D2, D4).

ARCHITECTURE §5 defines no CT-N for C7 in isolation — the Hamilton engine is
exercised through CT-4 (pillars <-> C8), which belongs to C8's session and runs
against T0 mocks. These are C7's own seam tests: they assert the properties
CT-4 will later depend on, so a defect surfaces here rather than inside
somebody else's Kalman filter.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import numpy as np
import pytest

from seraph.pillars.hamilton import HamiltonConfig, HamiltonEngine, close_ts
from seraph.pillars.hamilton.em import HamiltonEstimationError, HamiltonParams
from seraph.shared_types import (
    Err,
    ObservedEmission,
    UnavailableEmission,
)
from tests.pillars.synthetic import (
    MockFeatureSource,
    SyntheticPanel,
    as_ok,
    make_panel,
    observed,
    unwrap,
)

CFG = HamiltonConfig(
    corpus_start="2015-01-01",
    min_history_days=250,
    refit_every_days=120,
    em_max_iter=25,
    logit_max_iter=40,
)


def _engine(
    n_days: int = 400, seed: int = 11, vix_missing_first: int = 0
) -> tuple[HamiltonEngine, SyntheticPanel]:
    panel = make_panel(n_days=n_days, seed=seed, vix_missing_first=vix_missing_first)
    source = MockFeatureSource(panel)
    return HamiltonEngine(source, CFG), panel


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


# -- absence is a value, not an error ---------------------------------------


def test_pre_history_absence_is_ok_structural() -> None:
    """Before min_history_days Hamilton cannot exist -> D2 mask exclusion."""
    engine, panel = _engine()
    early = panel.observations[10].date
    res = _run(engine.emit(close_ts(early)))

    emission = unwrap(res)
    assert isinstance(emission, UnavailableEmission)
    assert emission.absence == "structural"
    assert emission.reason == "insufficient_history"
    assert emission.pillar == "hamilton"


def test_before_corpus_start_is_structural_no_data_coverage() -> None:
    engine, _ = _engine()
    emission = unwrap(_run(engine.emit("2010-06-01T15:30:00+05:30")))
    assert isinstance(emission, UnavailableEmission)
    assert emission.absence == "structural"
    assert emission.reason == "no_data_coverage"


def test_failed_estimation_is_transient_not_structural(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed EM must keep Hamilton IN the mask (D2) — transient, not gone."""
    engine, panel = _engine()

    def boom(*args: object, **kwargs: object) -> HamiltonParams:
        raise HamiltonEstimationError("forced", 3, -1.0)

    monkeypatch.setattr("seraph.pillars.hamilton.engine.fit_em", boom)
    emission = unwrap(_run(engine.emit(close_ts(panel.observations[-1].date))))
    assert isinstance(emission, UnavailableEmission)
    assert emission.absence == "transient"
    assert emission.reason == "estimation_failed"


def test_missing_upstream_dependency_is_an_error() -> None:
    """A broken C4 call IS an Err — that is a failure of the call, not absence."""
    panel = make_panel(n_days=300)
    engine = HamiltonEngine(MockFeatureSource(panel, fail=True), CFG)
    res = _run(engine.emit(close_ts(panel.observations[-1].date)))

    assert isinstance(res, Err)
    assert res.error.kind == "MISSING_DEPENDENCY"
    assert res.error.retryable is True


# -- the observed emission ---------------------------------------------------


@pytest.mark.slow
def test_observed_emission_shape_and_bounds() -> None:
    engine, panel = _engine()
    last = panel.observations[-1].date
    obs = observed(_run(engine.emit(close_ts(last))))
    assert obs.pillar == "hamilton"
    assert obs.tau <= obs.ts  # CT-4's contract check
    assert obs.tau == close_ts(last)
    assert 0.0 <= obs.value <= 2.0  # FR23: LSD_t = xi_2 + 2*xi_3
    assert obs.estimation_variance is not None
    assert obs.estimation_variance >= 0.0


@pytest.mark.slow
def test_query_after_the_last_close_ages_rather_than_lies() -> None:
    """tau stays at the real computation date, so C8's Delta is honest (D4)."""
    engine, panel = _engine()
    last = panel.observations[-1].date
    fresh = observed(_run(engine.emit(f"{last}T15:30:00+05:30")))

    # Query a week later: same estimate, older tau, STALE_OBSERVATION flagged.
    later = as_ok(_run(engine.emit("2099-01-01T15:30:00+05:30")))
    stale = observed(later)
    assert stale.value == fresh.value
    assert stale.tau == fresh.tau
    assert stale.tau < stale.ts
    assert "STALE_OBSERVATION" in {w.code for w in later.warnings}


@pytest.mark.slow
def test_emit_range_is_daily_and_deterministic() -> None:
    engine, panel = _engine()
    from_ts = close_ts(panel.observations[300].date)
    to_ts = close_ts(panel.observations[-1].date)

    first = unwrap(_run(engine.emit_range(from_ts, to_ts)))
    second = unwrap(_run(engine.emit_range(from_ts, to_ts)))

    assert len(first) == len(panel.observations) - 300
    assert [e.model_dump() for e in first] == [e.model_dump() for e in second]
    for emission in first:
        assert isinstance(emission, ObservedEmission)
        assert emission.obs.tau == emission.obs.ts  # computed at that close


@pytest.mark.slow
def test_emissions_do_not_use_future_data() -> None:
    """The single most damaging silent bug available to C7.

    Re-running the same date with more future rows in the store must not change
    the answer. If EM refits, standardisation stats, or the smoother ever leak
    backwards, this is what catches it.
    """
    panel = make_panel(n_days=400, seed=11)
    truncated = SyntheticPanel(
        observations=panel.observations[:330],
        macro=panel.macro[:330],
        states=panel.states[:330],
    )

    target = close_ts(panel.observations[329].date)
    full = observed(_run(HamiltonEngine(MockFeatureSource(panel), CFG).emit(target)))
    short = observed(
        _run(HamiltonEngine(MockFeatureSource(truncated), CFG).emit(target))
    )

    assert full.model_dump() == short.model_dump()


# -- FR24 detail and E8 ------------------------------------------------------


@pytest.mark.slow
def test_detail_reports_a_simplex_and_finite_half_lives() -> None:
    engine, panel = _engine()
    detail = unwrap(_run(engine.detail(close_ts(panel.observations[-1].date))))
    assert abs(sum(detail.xi) - 1.0) < 1e-9
    assert all(x >= 0.0 for x in detail.xi)
    assert 0.0 < detail.p_hat_22 < 1.0
    assert 0.0 < detail.p_hat_33 < 1.0
    assert np.isfinite(detail.tau_half_stressed)
    assert np.isfinite(detail.tau_half_crisis)
    assert detail.tau_half_stressed > 0.0


@pytest.mark.slow
def test_lsd_matches_the_detail_xi() -> None:
    """FR23's LSD_t and the xi C9 consumes must describe the same posterior."""
    engine, panel = _engine()
    ts = close_ts(panel.observations[-1].date)
    obs = observed(_run(engine.emit(ts)))
    detail = unwrap(_run(engine.detail(ts)))

    expected = detail.xi[1] + 2.0 * detail.xi[2]
    assert obs.value == pytest.approx(expected, abs=1e-12)

    expected_var = (detail.xi[1] + 4.0 * detail.xi[2]) - expected**2
    assert obs.estimation_variance == pytest.approx(max(expected_var, 0.0), abs=1e-12)


@pytest.mark.slow
def test_e8_rows_carry_gamma_and_fit_provenance() -> None:
    engine, panel = _engine()
    res = _run(
        engine.outputs(
            close_ts(panel.observations[350].date),
            close_ts(panel.observations[-1].date),
        )
    )
    rows = unwrap(res)
    assert rows

    row = rows[0]
    assert len(row.gamma_ij) == 3
    assert all(len(mat) == 3 for mat in row.gamma_ij)
    assert len(row.gamma_ij[0][0]) == 1 + len(row.covariates)  # intercept + z
    assert row.fitted_through <= row.date  # never fitted on the future
    assert row.lsd_t == pytest.approx(row.xi[1] + 2.0 * row.xi[2])
    assert row.rv_t_yz > 0.0


@pytest.mark.slow
def test_coverage_starts_after_the_minimum_history() -> None:
    engine, panel = _engine()
    coverage = unwrap(_run(engine.coverage()))
    first_day = panel.observations[CFG.min_history_days - 1].date
    assert coverage.from_ts == close_ts(first_day)
    assert coverage.to_ts == close_ts(panel.observations[-1].date)


# -- z_t handling ------------------------------------------------------------


@pytest.mark.slow
def test_incomplete_covariate_is_dropped_with_a_warning_not_zero_filled() -> None:
    """SPEC OQ5 (India VIX pre-Nov-2007) is open — degrade loudly, never guess."""
    engine, panel = _engine(vix_missing_first=280)
    res = as_ok(_run(engine.emit(close_ts(panel.observations[299].date))))

    assert isinstance(res.value, ObservedEmission)
    warnings = {w.code for w in res.warnings}
    assert "PARTIAL_COVERAGE" in warnings
    messages = " ".join(w.message for w in res.warnings)
    assert "covariate dropped" in messages
