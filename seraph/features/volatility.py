"""C4 — `RV_t`, the Yang-Zhang OHLC volatility estimator (FR20).

FR20 specifies Yang-Zhang across the full corpus, with true 5-minute realised
volatility where intraday data exists. Only the OHLC half is implemented here:
`rv_5min` needs E1 intraday bars (C2/Kite, ~2015+), and FR53's agreement check
is what would license substituting one for the other. C7's default config
(`rv_source="yz"`) already refuses to splice them mid-sample, so a single
consistent YZ series is exactly what it wants.

The estimator, over a rolling window of `n` days:

    sigma^2_YZ = sigma^2_overnight + k * sigma^2_open_to_close
                 + (1 - k) * sigma^2_rogers_satchell
    k          = 0.34 / (1.34 + (n + 1) / (n - 1))

Its point is drift-independence and open-jump handling — the two properties
close-to-close variance lacks and which matter most on exactly the gap days a
structural-break detector is looking at.
"""

from __future__ import annotations

import numpy as np

__all__ = ["yang_zhang_variance"]


def yang_zhang_variance(
    log_open: np.ndarray,
    log_high: np.ndarray,
    log_low: np.ndarray,
    log_close: np.ndarray,
    window: int,
    floor: float = 1.0e-12,
) -> np.ndarray:
    """Rolling Yang-Zhang variance per day, in daily units.

    Args:
        log_open/high/low/close: `(T,)` log prices of one series.
        window: `n`, in trading days.
        floor: smallest returned variance. C7 log-transforms `RV_t`, so a zero
            (a window of identical prices, which synthetic data produces easily)
            would become `-inf` and poison the regime likelihood.

    Returns `(T,)` with `nan` for the first `window` days, which have no
    complete window behind them — never a partial-window value, since a
    variance computed on three days and one computed on twenty are not the same
    quantity and a regime filter would read the difference as a regime.
    """
    n_days = log_close.shape[0]
    out = np.full(n_days, np.nan)
    if window < 2 or n_days <= window:
        return out

    # Overnight, open-to-close, and the two Rogers-Satchell legs.
    overnight = log_open[1:] - log_close[:-1]
    open_to_close = log_close[1:] - log_open[1:]
    u = log_high[1:] - log_open[1:]
    d = log_low[1:] - log_open[1:]
    rogers_satchell = u * (u - open_to_close) + d * (d - open_to_close)

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    for i in range(window, n_days):
        # Window of daily observations ending at day i (indices into the
        # difference arrays are shifted by one).
        sl = slice(i - window, i)
        o_win, c_win, rs_win = overnight[sl], open_to_close[sl], rogers_satchell[sl]
        if not (
            np.isfinite(o_win).all()
            and np.isfinite(c_win).all()
            and np.isfinite(rs_win).all()
        ):
            continue
        var_o = float(np.var(o_win, ddof=1))
        var_c = float(np.var(c_win, ddof=1))
        var_rs = float(np.mean(rs_win))
        out[i] = max(var_o + k * var_c + (1.0 - k) * var_rs, floor)
    return out
