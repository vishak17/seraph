"""C4 — Feature Deriver: the daily OHLCV slice the derivations read.

**Minimal implementation.** C4 proper (FR5, FR6, FR22, FR50, FR53) is owner
A/B's and spans jump extraction, the `S_t` quintile grid and the full estimator
ladder. What lives in this package is only the part C7 consumes across S4 —
`observationVector` and `covariateVector` — because without it C7 has no input
and the C7 -> C8 seam cannot be run on anything but test doubles.

`DailyPanel` is the E3 (`daily_prices`) slice those derivations actually need,
carried as dense NumPy rather than fetched from C1: the store does not exist
yet, and AGENTS.md §2 puts the Polars/NumPy boundary exactly here anyway — the
panel is (trading days x symbols), which is small and dense once it reaches the
feature layer. When C1 lands, `StoreReader.dailyBars()` fills this shape and
nothing downstream changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seraph.shared_types import ISODate, Symbol

__all__ = ["DailyPanel"]


@dataclass(frozen=True, eq=False)
class DailyPanel:
    """Aligned daily OHLC for a set of symbols on a common trading-day grid.

    Args:
        dates: trading days, strictly increasing. This *is* the calendar as far
            as C4/C7/C8 are concerned — none of them owns one.
        symbols: column order for every array.
        open_, high, low, close: `(T, N)` arrays of raw prices. `nan` marks a
            symbol not yet listed (or suspended) on that day, which is normal
            in a point-in-time panel and is handled per-derivation, never
            forward-filled here.
    """

    dates: tuple[ISODate, ...]
    symbols: tuple[Symbol, ...]
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    def __len__(self) -> int:
        return len(self.dates)

    def validate(self) -> str | None:
        """Structural check. Returns a `CONTRACT_VIOLATION` detail, or None."""
        n_days, n_symbols = len(self.dates), len(self.symbols)
        for name, array in (
            ("open", self.open_),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
        ):
            if array.shape != (n_days, n_symbols):
                return f"{name} has shape {array.shape}, expected {(n_days, n_symbols)}"
        if n_days == 0 or n_symbols == 0:
            return "panel is empty"
        if list(self.dates) != sorted(self.dates):
            return "dates are not in ascending order"
        if len(set(self.dates)) != n_days:
            return "duplicate trading date in the panel"
        with np.errstate(invalid="ignore"):
            if np.any(self.high < self.low):
                return "high < low on at least one bar"
            if np.any(self.close <= 0.0, where=~np.isnan(self.close)):
                return "non-positive close price"
        return None

    def log_close(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(self.close)

    def log_returns(self) -> np.ndarray:
        """`(T-1, N)` close-to-close log returns. Row `i` spans dates[i]->[i+1]."""
        logc = self.log_close()
        return logc[1:] - logc[:-1]

    def market_ohlc(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Equal-weighted composite OHLC, in log space, ignoring missing names.

        A stand-in for the index series the real C4 would read (SPEC FR20 says
        `RV_t` is market realised volatility, not a cross-section of it). Equal
        weighting is the assumption to revisit when C3's universe and turnover
        weights exist; it is stated here rather than buried in the estimator.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            parts = [
                np.nanmean(np.log(a), axis=1)
                for a in (self.open_, self.high, self.low, self.close)
            ]
        return parts[0], parts[1], parts[2], parts[3]
