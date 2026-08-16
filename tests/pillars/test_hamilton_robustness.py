"""C7 — robustness at the seam.

C7 sits downstream of a component (C4) that does not exist yet, and upstream of
one (C8) whose contract says a pillar returns absence rather than throwing. So
the properties under test here are: nothing raises past the seam, malformed
upstream output is rejected as a typed error, and the cross-call cache is a
cost optimisation that can never change an answer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import numpy as np
import pytest

import seraph.pillars.hamilton.engine as engine_module
from seraph.pillars.hamilton import (
    HamiltonConfig,
    HamiltonEngine,
    HamiltonEstimationError,
    close_ts,
)
from seraph.pillars.hamilton.engine import _Prefix
from seraph.shared_types import (
    Err,
    MacroRow,
    ObservationRow,
    ObservedEmission,
    PillarEngine,
    Result,
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


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def _engine(panel: SyntheticPanel) -> HamiltonEngine:
    return HamiltonEngine(MockFeatureSource(panel), CFG)


def _with_rows(
    panel: SyntheticPanel, rows: tuple[ObservationRow, ...]
) -> SyntheticPanel:
    return SyntheticPanel(observations=rows, macro=panel.macro, states=panel.states)


# -- the engine satisfies the S5 contract statically -------------------------


def test_engine_is_a_pillar_engine() -> None:
    """Structural conformance to S5. mypy enforces this; the runtime check
    here keeps it honest for anyone who runs the tests without a type check."""
    engine: PillarEngine = _engine(make_panel(n_days=260))
    assert engine.pillar == "hamilton"
    for name in ("emit", "emit_range", "coverage"):
        assert callable(getattr(engine, name))


# -- malformed upstream output ------------------------------------------------


def test_source_that_raises_becomes_a_contract_violation() -> None:
    """A C4 that throws instead of returning Err must not throw through C7."""

    class ExplodingSource:
        async def observation_vector(
            self, from_date: str, to_date: str
        ) -> Result[tuple[ObservationRow, ...]]:
            raise RuntimeError("connection reset")

        async def covariate_vector(
            self, from_date: str, to_date: str
        ) -> Result[tuple[MacroRow, ...]]:
            raise RuntimeError("connection reset")

    res = _run(HamiltonEngine(ExplodingSource(), CFG).emit(close_ts("2020-01-02")))
    assert isinstance(res, Err)
    assert res.error.kind == "CONTRACT_VIOLATION"
    assert "connection reset" in res.error.detail


def test_duplicate_observation_dates_are_rejected() -> None:
    panel = make_panel(n_days=300)
    rows = panel.observations + (panel.observations[-1],)
    res = _run(_engine(_with_rows(panel, rows)).emit(close_ts(rows[-1].date)))

    assert isinstance(res, Err)
    assert res.error.kind == "CONTRACT_VIOLATION"
    assert res.error.field == "ObservationRow.date"


@pytest.mark.parametrize("field_name", ["rv_yang_zhang", "ar_t", "bas_t"])
def test_non_finite_observations_are_rejected(field_name: str) -> None:
    """A broken estimator upstream must not become a nan regime probability."""
    panel = make_panel(n_days=300)
    rows = list(panel.observations)
    rows[100] = panel.observations[100].model_copy(update={field_name: float("nan")})
    res = _run(_engine(_with_rows(panel, tuple(rows))).emit(close_ts(rows[-1].date)))

    assert isinstance(res, Err)
    assert res.error.kind == "CONTRACT_VIOLATION"
    assert res.error.field == f"ObservationRow.{field_name}"


def test_empty_corpus_is_structural_absence_not_an_error() -> None:
    panel = make_panel(n_days=260)
    empty = SyntheticPanel(observations=(), macro=(), states=panel.states[:0])
    emission = unwrap(_run(_engine(empty).emit(close_ts("2020-01-02"))))
    assert isinstance(emission, UnavailableEmission)
    assert emission.absence == "structural"
    assert emission.reason == "no_data_coverage"


# -- numerical failure never escapes -----------------------------------------


@pytest.mark.slow
def test_non_finite_filter_output_degrades_to_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the recursion ever produces nan, the seam reports absence.

    Constructing a `Simplex3` from nan would raise a pydantic error straight
    out of `emit`, which is exactly what S5 forbids.
    """
    panel = make_panel(n_days=300)

    def nan_advance(self: _Prefix, index: int) -> tuple[np.ndarray, np.ndarray]:
        return np.full(3, np.nan), np.full((3, 3), np.nan)

    monkeypatch.setattr(_Prefix, "advance_to", nan_advance)
    emission = unwrap(_run(_engine(panel).emit(close_ts(panel.observations[-1].date))))

    assert isinstance(emission, UnavailableEmission)
    assert emission.absence == "transient"
    assert emission.reason == "estimation_failed"


@pytest.mark.slow
def test_degenerate_regime_covariance_degrades_to_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-PD covariance is caught where the prefix is built, not later."""
    panel = make_panel(n_days=300)

    def bad_cholesky(_a: np.ndarray) -> np.ndarray:
        raise np.linalg.LinAlgError("Matrix is not positive definite")

    monkeypatch.setattr(np.linalg, "cholesky", bad_cholesky)
    emission = unwrap(_run(_engine(panel).emit(close_ts(panel.observations[-1].date))))

    assert isinstance(emission, UnavailableEmission)
    assert emission.absence == "transient"


# -- the cache is a cost optimisation, never an answer change -----------------


@pytest.mark.slow
def test_incremental_calls_match_a_cold_engine() -> None:
    """Walking forward day by day must equal one cold batch run."""
    panel = make_panel(n_days=330)
    dates = [r.date for r in panel.observations]

    warm = _engine(panel)
    incremental = [observed(_run(warm.emit(close_ts(d)))).value for d in dates[300:310]]

    cold = _engine(panel)
    batch = unwrap(_run(cold.emit_range(close_ts(dates[300]), close_ts(dates[309]))))
    batch_values = [e.obs.value for e in batch if isinstance(e, ObservedEmission)]

    assert incremental == pytest.approx(batch_values, abs=0.0)


@pytest.mark.slow
def test_cache_is_invalidated_when_history_is_restated() -> None:
    """A source that rewrites the past must not be served a stale answer."""
    panel = make_panel(n_days=300)
    engine = _engine(panel)
    target = close_ts(panel.observations[-1].date)
    before = observed(_run(engine.emit(target))).value

    rows = list(panel.observations)
    rows[50] = rows[50].model_copy(update={"rv_yang_zhang": rows[50].rv_yang_zhang * 8})
    engine.source = MockFeatureSource(_with_rows(panel, tuple(rows)))
    after = observed(_run(engine.emit(target))).value

    assert after != before


@pytest.mark.slow
def test_a_cached_transient_absence_is_invalidated_by_a_restatement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed *first* fit must not pin the pillar absent forever.

    When the very first EM fit fails there is nothing filtered yet, so the
    filter's position stays at -1 while the transient verdict is cached at the
    target index. Fingerprinting only up to the filter's position would leave
    that verdict outside the content check — the source could then correct the
    exact rows that caused the failure and still be served the cached absence,
    which C8 ages to the D4 ceiling and D2 eventually masks out. Hamilton would
    silently drop out of the CSRS on corrected data.
    """
    panel = make_panel(n_days=300)
    engine = _engine(panel)
    target = close_ts(panel.observations[-1].date)

    calls = {"n": 0}
    real_fit = engine_module.fit_em

    def failing_first_fit(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise HamiltonEstimationError("simulated first-fit failure", 3, -1.0)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(engine_module, "fit_em", failing_first_fit)

    first = unwrap(_run(engine.emit(target)))
    assert isinstance(first, UnavailableEmission)
    assert first.absence == "transient"

    # Same dates, restated values. The fit would now succeed.
    rows = tuple(
        r.model_copy(update={"rv_yang_zhang": r.rv_yang_zhang * 1.5})
        for r in panel.observations
    )
    engine.source = MockFeatureSource(_with_rows(panel, rows))

    after = unwrap(_run(engine.emit(target)))
    cold = unwrap(_run(_engine(_with_rows(panel, rows)).emit(target)))

    assert isinstance(after, ObservedEmission), "stale absence served after restatement"
    assert isinstance(cold, ObservedEmission)
    assert after.obs.value == pytest.approx(cold.obs.value, abs=0.0)


@pytest.mark.slow
def test_reset_cache_does_not_change_answers() -> None:
    panel = make_panel(n_days=300)
    engine = _engine(panel)
    target = close_ts(panel.observations[-1].date)
    first = observed(_run(engine.emit(target))).model_dump()
    engine.reset_cache()
    second = observed(_run(engine.emit(target))).model_dump()
    assert first == second


@pytest.mark.slow
def test_backwards_query_after_a_forward_one_is_still_correct() -> None:
    """The filter only moves forward; asking for an earlier date must refit."""
    panel = make_panel(n_days=330)
    dates = [r.date for r in panel.observations]

    engine = _engine(panel)
    _run(engine.emit(close_ts(dates[320])))
    later_then_earlier = observed(_run(engine.emit(close_ts(dates[300])))).value

    fresh = observed(_run(_engine(panel).emit(close_ts(dates[300])))).value
    assert later_then_earlier == pytest.approx(fresh, abs=0.0)


# -- details_range ------------------------------------------------------------


@pytest.mark.slow
def test_details_range_matches_pointwise_detail() -> None:
    panel = make_panel(n_days=310)
    dates = [r.date for r in panel.observations]
    engine = _engine(panel)

    ranged_res = _run(engine.details_range(close_ts(dates[300]), close_ts(dates[304])))
    ranged = as_ok(ranged_res).value
    assert len(ranged) == 5

    pointwise = _engine(panel)
    for ts, detail in ranged:
        one = unwrap(_run(pointwise.detail(ts)))
        assert one.model_dump() == detail.model_dump()


@pytest.mark.slow
def test_details_range_and_emit_range_agree_on_lsd() -> None:
    """FR23: the emitted value and the published xi describe one posterior."""
    panel = make_panel(n_days=310)
    dates = [r.date for r in panel.observations]
    engine = _engine(panel)
    from_ts, to_ts = close_ts(dates[295]), close_ts(dates[-1])

    emissions = unwrap(_run(engine.emit_range(from_ts, to_ts)))
    detail_pairs = as_ok(_run(engine.details_range(from_ts, to_ts))).value
    details = dict(detail_pairs)

    observed_emissions = [e for e in emissions if isinstance(e, ObservedEmission)]
    assert observed_emissions
    for e in observed_emissions:
        xi = details[e.obs.ts].xi
        assert e.obs.value == pytest.approx(xi[1] + 2.0 * xi[2], abs=1e-12)


# -- configuration ------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_history_days": 10},
        {"refit_every_days": 0},
        {"em_max_iter": 0},
        {"em_tol": 0.0},
        {"cov_ridge": -1.0},
        {"p_jj_cap": 1.0},
        {"covariates": ()},
        {"covariates": ("g_t", "g_t")},
        {"corpus_start": "2005/01/01"},
    ],
)
def test_invalid_config_is_rejected_at_construction(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        HamiltonConfig(**kwargs)
