"""C4 — the estimators behind `y_t` and `z_t`, and the S4 seam itself.

Each estimator is checked against a case where the right answer is known
analytically rather than against itself: a constant-volatility series for
Yang-Zhang, a one-factor panel for the absorption ratio, a pure bid-ask bounce
for Abdi-Ranaldo. A feature that is quietly wrong here reaches C7 as a regime
and C8 as a state, and nothing downstream can tell.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import numpy as np
import pytest

from seraph.features import (
    DailyPanel,
    FeatureConfig,
    PanelFeatureSource,
    TariffEvent,
    abdi_ranaldo_spread,
    absorption_ratio,
    tariff_covariate,
    yang_zhang_variance,
)
from seraph.shared_types import MacroRow
from tests.reconciliation.helpers import as_err, unwrap


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def _dates(n: int) -> tuple[str, ...]:
    from datetime import date, timedelta

    start = date(2020, 1, 1)
    return tuple((start + timedelta(days=i)).isoformat() for i in range(n))


# -- RV_t (FR20) --------------------------------------------------------------


def test_yang_zhang_recovers_a_known_constant_volatility() -> None:
    """A geometric random walk with no drift and no gaps: YZ should land near
    the variance it was generated with."""
    rng = np.random.default_rng(3)
    n, sigma = 4000, 0.02
    steps = rng.normal(0.0, sigma, n)
    log_close = np.cumsum(steps)
    log_open = log_close - steps * 0.5  # open halfway through the day's move
    log_high = np.maximum(log_open, log_close) + 0.25 * sigma
    log_low = np.minimum(log_open, log_close) - 0.25 * sigma

    out = yang_zhang_variance(log_open, log_high, log_low, log_close, window=250)
    estimate = float(np.nanmedian(out))
    assert 0.5 * sigma**2 < estimate < 2.0 * sigma**2


def test_yang_zhang_has_no_value_before_a_full_window() -> None:
    """A variance computed on a partial window is a different quantity, and a
    regime filter reads the difference as a regime."""
    n, window = 40, 21
    x = np.cumsum(np.random.default_rng(1).normal(0, 0.01, n))
    out = yang_zhang_variance(x, x + 0.01, x - 0.01, x, window=window)
    assert np.isnan(out[:window]).all()
    assert np.isfinite(out[window:]).all()


def test_yang_zhang_is_floored_so_c7_can_log_it() -> None:
    flat = np.zeros(60)
    out = yang_zhang_variance(flat, flat, flat, flat, window=21)
    finite = out[np.isfinite(out)]
    assert len(finite) > 0
    assert (finite > 0.0).all()  # log(0) would poison the regime likelihood


# -- AR_t (FR20, and it lives in C4) -----------------------------------------


def test_absorption_ratio_is_near_one_for_a_one_factor_panel() -> None:
    rng = np.random.default_rng(5)
    n_days, n_symbols = 400, 30
    factor = rng.normal(0.0, 0.02, (n_days, 1))
    returns = factor @ np.ones((1, n_symbols)) + rng.normal(
        0.0, 0.0005, (n_days, n_symbols)
    )
    ar = absorption_ratio(returns, window=120)
    assert float(np.nanmedian(ar)) > 0.99


def test_absorption_ratio_is_low_for_an_uncorrelated_panel() -> None:
    rng = np.random.default_rng(6)
    returns = rng.normal(0.0, 0.01, (400, 60))
    ar = absorption_ratio(returns, window=200, n_components=10)
    # 10 of 60 independent components explain ~1/6 of the variance, plus the
    # upward bias of picking the largest ten.
    assert 0.1 < float(np.nanmedian(ar)) < 0.5


def test_absorption_ratio_refuses_a_panel_too_narrow_to_mean_anything() -> None:
    """With N <= n_components the ratio is identically 1, which would read as
    total absorption rather than as "not enough symbols"."""
    rng = np.random.default_rng(7)
    ar = absorption_ratio(rng.normal(0, 0.01, (200, 8)), window=60, n_components=10)
    assert np.isnan(ar).all()


# -- BAS_t (FR6) --------------------------------------------------------------


def test_abdi_ranaldo_recovers_a_known_spread() -> None:
    """Efficient price plus an independent half-spread bounce — the model the
    estimator is derived under."""
    rng = np.random.default_rng(8)
    n, spread = 3000, 0.01
    efficient = np.cumsum(rng.normal(0.0, 0.008, n))
    bounce = rng.choice((-1.0, 1.0), n) * spread / 2.0
    log_close = efficient + bounce
    log_high = efficient + 0.004
    log_low = efficient - 0.004

    out = abdi_ranaldo_spread(log_high, log_low, log_close, window=500)
    estimate = float(np.nanmedian(out[np.isfinite(out)]))
    assert 0.5 * spread < estimate < 1.6 * spread


def test_abdi_ranaldo_never_returns_a_negative_spread() -> None:
    """The estimator's small-sample failure mode is a negative covariance; FR6
    reports an undefined spread, never a negative one."""
    rng = np.random.default_rng(9)
    efficient = np.cumsum(rng.normal(0.0, 0.01, 400))
    out = abdi_ranaldo_spread(
        efficient + 0.002, efficient - 0.002, efficient, window=21
    )
    finite = out[np.isfinite(out)]
    assert (finite >= 0.0).all()


# -- G_t (FR22) ---------------------------------------------------------------


def test_tariff_covariate_decays_from_each_announcement() -> None:
    dates = _dates(200)
    events = (TariffEvent(tau_k=dates[50], delta_r_k=0.25),)
    g = tariff_covariate(dates, events, eta_days=60.0)

    assert g[:50] == (0.0,) * 50  # zero before the event is a value, not a gap
    assert g[50] == pytest.approx(0.25)
    assert g[51] < g[50]
    assert g[-1] < g[100] < g[50]
    assert all(v >= 0.0 for v in g)


def test_tariff_covariate_superposes_events_and_uses_absolute_magnitude() -> None:
    """FR22 sums `|delta_r_k|`: a tariff being *cut* is still a shock."""
    dates = _dates(120)
    events = (
        TariffEvent(tau_k=dates[10], delta_r_k=0.25),
        TariffEvent(tau_k=dates[10], delta_r_k=-0.25),
    )
    g = tariff_covariate(dates, events, eta_days=60.0)
    assert g[10] == pytest.approx(0.5)


def test_tariff_covariate_rejects_a_non_positive_decay() -> None:
    with pytest.raises(ValueError, match="eta_days"):
        tariff_covariate(_dates(5), (), eta_days=0.0)


# -- the S4 seam --------------------------------------------------------------


def _panel(n_days: int = 200, n_symbols: int = 20, seed: int = 4) -> DailyPanel:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.01, (n_days, n_symbols))
    close = 100.0 * np.exp(np.cumsum(steps, axis=0))
    open_ = close * np.exp(-steps * 0.5)
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    return DailyPanel(
        dates=_dates(n_days),
        symbols=tuple(f"S{i}" for i in range(n_symbols)),
        open_=open_,
        high=high,
        low=low,
        close=close,
    )


def test_the_source_emits_rows_only_where_every_window_is_complete() -> None:
    cfg = FeatureConfig(rv_window=10, ar_window=40, bas_window=10, min_symbols=12)
    source = PanelFeatureSource(panel=_panel(), cfg=cfg)
    rows = unwrap(_run(source.observation_vector("1900-01-01", "2999-01-01")))

    assert rows
    assert rows[0].date == _panel().dates[cfg.warmup()]
    assert all(np.isfinite([r.rv_yang_zhang, r.ar_t, r.bas_t]).all() for r in rows)


def test_rv_5min_is_none_rather_than_a_proxy() -> None:
    """E1 intraday coverage is C2's; inventing `rv_5min` would let FR53's
    agreement check pass against a number nobody measured."""
    source = PanelFeatureSource(panel=_panel())
    rows = unwrap(_run(source.observation_vector("1900-01-01", "2999-01-01")))
    assert all(r.rv_5min is None for r in rows)


def test_c4_owns_g_t_and_overwrites_what_ingestion_supplied() -> None:
    """FR22 assigns `G_t` to C4. A stale value carried in from E4 must not win."""
    panel = _panel(n_days=60)
    macro = tuple(MacroRow(date=d, india_vix=15.0, g_t=999.0) for d in panel.dates)
    events = (TariffEvent(tau_k=panel.dates[10], delta_r_k=0.4),)
    source = PanelFeatureSource(panel=panel, macro=macro, events=events)

    rows = unwrap(_run(source.covariate_vector("1900-01-01", "2999-01-01")))
    assert all(r.g_t != 999.0 for r in rows)
    assert rows[10].g_t == pytest.approx(0.4)
    assert rows[0].india_vix == 15.0  # everything else is passed through


def test_a_date_with_no_macro_row_still_gets_one() -> None:
    """C7 aligns `z_t` to the trading-day grid; a missing date would shorten
    the panel rather than be reported as a partial covariate."""
    panel = _panel(n_days=30)
    source = PanelFeatureSource(panel=panel, macro=())
    rows = unwrap(_run(source.covariate_vector("1900-01-01", "2999-01-01")))
    assert len(rows) == len(panel.dates)
    assert all(r.india_vix is None for r in rows)


def test_a_malformed_panel_is_a_contract_violation_not_a_crash() -> None:
    good = _panel(n_days=40)
    broken = DailyPanel(
        dates=good.dates,
        symbols=good.symbols,
        open_=good.open_,
        high=good.low,  # high below low
        low=good.high,
        close=good.close,
    )
    error = as_err(
        _run(
            PanelFeatureSource(panel=broken).observation_vector(
                "1900-01-01", "2999-01-01"
            )
        )
    ).error
    assert error.kind == "CONTRACT_VIOLATION"


def test_an_empty_range_warns_rather_than_returning_a_bare_empty_tuple() -> None:
    source = PanelFeatureSource(panel=_panel(n_days=80))
    result = _run(source.observation_vector("1800-01-01", "1800-12-31"))
    assert result.status == "ok"
    assert result.value == ()
    assert any(w.code == "PARTIAL_COVERAGE" for w in result.warnings)
