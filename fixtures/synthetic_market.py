"""Synthetic daily market data for running C4 -> C7 -> C8 without real feeds.

**Not T0.** `mock_generator.py` (owner D, still unwritten) fakes the *outputs* —
`PillarEmission`, `ReconciledState`, `CsrsPoint` — so that C9-C12 can be built
before any pillar exists. This module fakes the *inputs*: OHLC bars and macro
rows, the E3/E4 slice C4 reads, so the real C4 -> C7 -> C8 chain can be run end
to end before C1 and C2 exist. The two are complementary and neither replaces
the other.

The generating process is a genuine three-state Markov chain with
regime-dependent market volatility, a market factor plus idiosyncratic noise
per symbol, an intraday path (so high/low are real extremes rather than a
fudge), and a bid-ask bounce (so Abdi-Ranaldo has something to estimate). That
matters: C7's EM has to find regimes that are actually there, and `BAS_t` has
to move for `y_t` to carry three distinguishable columns rather than two.

Deterministic given `seed` — every artefact downstream of it is reproducible
(NFR21).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from seraph.features import DailyPanel, TariffEvent
from seraph.shared_types import ISODate, MacroRow

__all__ = ["SyntheticMarket", "generate_market"]

# Regime-conditional daily volatility of the market factor: tranquil, stressed,
# crisis. Roughly 12% / 24% / 48% annualised.
REGIME_VOL = (0.0075, 0.0150, 0.0300)
# Regime-conditional relative spread, in return units. Liquidity dries up
# exactly when volatility spikes, which is the co-movement BAS_t exists to see.
REGIME_SPREAD = (0.0010, 0.0025, 0.0060)

# Expected sojourns of ~100 / ~33 / ~25 trading days. Persistence matters more
# than it looks: C4's `y_t` is built from rolling windows (21-day Yang-Zhang,
# 63-day absorption), so a regime that lasts less than one window is invisible
# in the observables no matter how large its volatility jump. A generator with
# week-long crises would therefore be testing whether C7 can see through a
# low-pass filter, which is not a property anyone wants it to have.
TRANSITION = np.array(
    [
        [0.990, 0.008, 0.002],
        [0.020, 0.970, 0.010],
        [0.005, 0.035, 0.960],
    ]
)

INTRADAY_STEPS = 24  # ~15-minute steps across a session; drives high/low


@dataclass(frozen=True, eq=False)
class SyntheticMarket:
    panel: DailyPanel
    macro: tuple[MacroRow, ...]
    events: tuple[TariffEvent, ...]
    regimes: np.ndarray  # (T,) true regime index — for diagnostics, never input


def _trading_days(n: int, start: date) -> list[ISODate]:
    out: list[ISODate] = []
    day = start
    while len(out) < n:
        if day.weekday() < 5:  # no exchange holiday calendar here; C1/C3 own that
            out.append(day.isoformat())
        day += timedelta(days=1)
    return out


def generate_market(
    n_days: int = 900,
    n_symbols: int = 40,
    start: date = date(2019, 1, 1),
    seed: int = 20260816,
) -> SyntheticMarket:
    """A reproducible daily panel with regime structure worth detecting."""
    rng = np.random.default_rng(seed)
    dates = _trading_days(n_days, start)

    # --- latent regime path -------------------------------------------------
    regimes = np.empty(n_days, dtype=int)
    state = 0
    for t in range(n_days):
        state = int(rng.choice(3, p=TRANSITION[state]))
        regimes[t] = state

    vol = np.array([REGIME_VOL[r] for r in regimes])
    spread = np.array([REGIME_SPREAD[r] for r in regimes])

    # --- per-symbol structure ----------------------------------------------
    beta = rng.uniform(0.6, 1.4, n_symbols)
    idio_vol = rng.uniform(0.006, 0.014, n_symbols)
    level = rng.uniform(80.0, 900.0, n_symbols)

    open_ = np.empty((n_days, n_symbols))
    high = np.empty((n_days, n_symbols))
    low = np.empty((n_days, n_symbols))
    close = np.empty((n_days, n_symbols))

    prev_close = level.copy()
    step = 1.0 / np.sqrt(INTRADAY_STEPS)

    for t in range(n_days):
        # Overnight gap: one market factor plus idiosyncratic noise, at ~35% of
        # a day's volatility. Yang-Zhang exists to handle exactly this term, so
        # a generator that omitted it would flatter the estimator.
        overnight_market = rng.normal(0.0, 0.35 * vol[t])
        overnight = beta * overnight_market + rng.normal(
            0.0, 0.35 * idio_vol, n_symbols
        )
        day_open = prev_close * np.exp(overnight)

        total_vol = np.sqrt((beta * vol[t]) ** 2 + idio_vol**2)
        increments = rng.normal(0.0, total_vol * step, size=(INTRADAY_STEPS, n_symbols))
        # A common intraday factor so symbols co-move within the day too.
        increments += rng.normal(
            0.0, vol[t] * step, size=(INTRADAY_STEPS, 1)
        ) * beta.reshape(1, -1)
        efficient = day_open * np.exp(np.cumsum(increments, axis=0))

        # Bid-ask bounce on EVERY trade, not just the close. This is the
        # microstructure model Abdi-Ranaldo actually assumes: recorded prices
        # are the efficient price plus an independent half-spread of random
        # sign, so (log high + log low) / 2 remains an unbiased proxy for the
        # efficient price while the close does not. Bouncing only the close —
        # and then widening high/low to contain it — makes the high-low
        # midpoint track the bounce, which destroys the very covariance the
        # estimator reads and leaves BAS_t anti-correlated with liquidity.
        side = rng.choice((-1.0, 1.0), size=(INTRADAY_STEPS, n_symbols))
        traded = efficient * np.exp(side * spread[t] / 2.0)

        open_[t] = day_open
        high[t] = np.maximum(traded.max(axis=0), day_open)
        low[t] = np.minimum(traded.min(axis=0), day_open)
        close[t] = traded[-1]
        prev_close = close[t]

    panel = DailyPanel(
        dates=tuple(dates),
        symbols=tuple(f"SYM{i:03d}" for i in range(n_symbols)),
        open_=open_,
        high=high,
        low=low,
        close=close,
    )

    # --- macro (E4) ---------------------------------------------------------
    # India VIX proxied from the regime's own volatility, which is the honest
    # description of what it is here: a covariate that genuinely carries regime
    # information, so the TVTP logit has something to fit.
    repo = 6.5 + np.cumsum(rng.normal(0.0, 0.004, n_days))
    credit = 12.0 + np.cumsum(rng.normal(0.0, 0.01, n_days))
    vix = 100.0 * vol * np.sqrt(252.0) + rng.normal(0.0, 0.8, n_days)
    inr = 74.0 + np.cumsum(rng.normal(0.0, 0.03, n_days))
    brent = 70.0 + np.cumsum(rng.normal(0.0, 0.4, n_days))

    macro = tuple(
        MacroRow(
            date=d,
            rbi_repo_rate=float(repo[i]),
            bank_credit_growth_yoy=float(credit[i]),
            india_vix=float(max(vix[i], 5.0)),
            vix_available=True,
            inr_twi=float(inr[i]),
            brent_price=float(brent[i]),
        )
        for i, d in enumerate(dates)
    )

    # --- tariff events (E5) -------------------------------------------------
    # Shaped after SPEC E5's known 2025-26 sequence, placed inside whatever
    # window the caller asked for. Illustrative dates, not the curated table —
    # FR4's real one is hand-built and is C2's deliverable.
    anchors = (0.30, 0.55, 0.75)
    deltas = (0.25, 0.25, -0.32)
    events = tuple(
        TariffEvent(
            tau_k=dates[int(a * (n_days - 1))],
            delta_r_k=d,
            event_id=f"synthetic-{i}",
            source_ref="fixtures.synthetic_market",
        )
        for i, (a, d) in enumerate(zip(anchors, deltas, strict=True))
    )

    return SyntheticMarket(panel=panel, macro=macro, events=events, regimes=regimes)
