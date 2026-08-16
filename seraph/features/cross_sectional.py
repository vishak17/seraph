"""C4 — the cross-sectional features `AR_t` and `BAS_t` (FR6, FR20).

**`AR_t` lives here, not in C6.** ARCHITECTURE calls this out twice and
AGENTS.md §5 repeats it: computing the absorption ratio inside the RMT engine
would make Hamilton (Pillar 3) secretly depend on RMT (Pillar 2), and FR35's
ablation could then never evaluate Hamilton standalone. Duplicating a
covariance computation between C4 and C6 is the correct price, paid on purpose.

`AR_t` — absorption ratio: the share of total return variance explained by the
leading 10 principal components of the rolling cross-sectional covariance
matrix. Rises when the cross-section starts moving as one thing, which is the
"coupling" half of the regime signal.

`BAS_t` — cross-sectional median Abdi-Ranaldo effective spread (FR6's primary
estimator):

    S = 2 * sqrt(max(E[(c_t - eta_t)(c_t - eta_{t+1})], 0))
    eta_t = (log high_t + log low_t) / 2

Corwin-Schultz (fallback) and Roll (sanity check) are the rest of FR6 and
belong to C4's full implementation; the primary estimator is what C7's `y_t`
consumes, so it is what exists here.
"""

from __future__ import annotations

import numpy as np

__all__ = ["absorption_ratio", "abdi_ranaldo_spread", "cross_sectional_spread"]


def absorption_ratio(
    log_returns: np.ndarray,
    window: int,
    n_components: int = 10,
    min_symbols: int = 12,
) -> np.ndarray:
    """Rolling `AR_t` in (0, 1].

    Args:
        log_returns: `(T-1, N)` close-to-close log returns.
        window: covariance window, in trading days.
        n_components: FR20's leading 10 PCs.
        min_symbols: below this many usable columns the ratio is meaningless
            (with N <= n_components it is identically 1), so `nan` is returned
            rather than a number that looks like maximum absorption.

    Returned on the *price* grid — index `i` is the ratio as of `dates[i]`,
    using returns up to and including that day, so nothing after `dates[i]` is
    read. Leading entries with no complete window are `nan`.
    """
    n_returns, _ = log_returns.shape
    out = np.full(n_returns + 1, np.nan)
    if window < 3:
        return out

    for i in range(window, n_returns + 1):
        block = log_returns[i - window : i]
        usable = np.isfinite(block).all(axis=0)
        if int(usable.sum()) < max(min_symbols, n_components + 1):
            continue
        cov = np.cov(block[:, usable], rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov)  # ascending, symmetric input
        total = float(eigenvalues.sum())
        if not np.isfinite(total) or total <= 0.0:
            continue
        top = float(eigenvalues[-n_components:].sum())
        out[i] = min(max(top / total, 0.0), 1.0)
    return out


def abdi_ranaldo_spread(
    log_high: np.ndarray, log_low: np.ndarray, log_close: np.ndarray, window: int
) -> np.ndarray:
    """Rolling Abdi-Ranaldo effective spread for one symbol (FR6).

    Negative covariance estimates are clipped to zero — they are the estimator's
    known small-sample failure mode, not evidence of a negative spread. FR6's
    contract is that an undefined spread is reported as such upstream
    (`spread_ar: null` on E2 plus a warning), never as a negative number.
    """
    n_days = log_close.shape[0]
    out = np.full(n_days, np.nan)
    if window < 2 or n_days < window + 1:
        return out

    eta = 0.5 * (log_high + log_low)
    # (c_t - eta_t) * (c_t - eta_{t+1}); the last day has no eta_{t+1}.
    product = (log_close[:-1] - eta[:-1]) * (log_close[:-1] - eta[1:])

    for i in range(window, n_days):
        block = product[i - window : i]
        if not np.isfinite(block).all():
            continue
        out[i] = 2.0 * np.sqrt(max(float(np.mean(block)), 0.0))
    return out


def cross_sectional_spread(
    log_high: np.ndarray,
    log_low: np.ndarray,
    log_close: np.ndarray,
    window: int,
    floor: float = 1.0e-8,
) -> np.ndarray:
    """`BAS_t` — the cross-sectional median Abdi-Ranaldo spread.

    Args are `(T, N)`. Median, not mean, because the estimator's tail is heavy
    on illiquid names and one clipped-to-zero symbol should not drag the market
    number. Floored for the same reason `RV_t` is: C7 log-transforms it.
    """
    n_days, n_symbols = log_close.shape
    per_symbol = np.full((n_days, n_symbols), np.nan)
    for j in range(n_symbols):
        per_symbol[:, j] = abdi_ranaldo_spread(
            log_high[:, j], log_low[:, j], log_close[:, j], window
        )
    # Median only over rows that have at least one usable symbol: an all-nan
    # row is the warm-up, not a data problem, and `nanmedian` would warn on it.
    usable = np.isfinite(per_symbol).any(axis=1)
    median = np.full(n_days, np.nan)
    if usable.any():
        median[usable] = np.nanmedian(per_symbol[usable], axis=1)
    return np.where(np.isfinite(median), np.maximum(median, floor), np.nan)
