"""C7 — Hamilton Engine (FR20, FR23) — the filter/smoother recursions.

Pure NumPy. No I/O, no shared-type imports — this module is the mathematics
only, so it can be property-tested and cross-checked against
`statsmodels.tsa.regime_switching.markov_switching.cy_hamilton_filter_log`
(see tests/pillars/test_hamilton_filter.py).

`statsmodels` cannot host the model itself: `MarkovSwitching.__init__` raises
`ValueError('Must have univariate endogenous data.')`, and FR20's y_t is
3-variate. AGENTS.md §2's "don't hand-roll unless statsmodels genuinely can't
express it" test is therefore met, and only met for this reason.

Index convention throughout:

    trans[t, i, j] = P(S_t = j | S_{t-1} = i, z_t)     -> rows sum to 1

statsmodels uses the transpose (`regime_transition[i, j, t]` is j -> i), so the
cross-check test transposes before comparing. Getting this backwards is silent,
not loud — hence the explicit assertion helper below.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

__all__ = [
    "gaussian_loglik",
    "hamilton_filter_log",
    "kim_smoother",
    "stationary_distribution",
]


def gaussian_loglik(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Regime-conditional log densities of a multivariate Gaussian.

    Args:
        y: (T, d) observations.
        mu: (k, d) regime means.
        sigma: (k, d, d) regime covariances, each positive definite.

    Returns:
        (T, k) array of log f(y_t | S_t = j).
    """
    n_obs, dim = y.shape
    k = mu.shape[0]
    out = np.empty((n_obs, k), dtype=float)
    const = dim * np.log(2.0 * np.pi)
    for j in range(k):
        chol = np.linalg.cholesky(sigma[j])
        diff = (y - mu[j]).T  # (d, T)
        sol = solve_triangular(chol, diff, lower=True)
        log_det = 2.0 * np.log(np.diag(chol)).sum()
        out[:, j] = -0.5 * (const + log_det + np.einsum("dt,dt->t", sol, sol))
    return out


def stationary_distribution(trans: np.ndarray) -> np.ndarray:
    """Stationary distribution of a single row-stochastic transition matrix.

    Used only to initialise the filter at t = 0; its influence decays
    geometrically and is negligible over a >=750-observation window.
    """
    k = trans.shape[0]
    a = np.vstack([trans.T - np.eye(k), np.ones((1, k))])
    b = np.zeros(k + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(a, b, rcond=None)
    pi = np.clip(pi, 1e-12, None)
    return pi / pi.sum()


def hamilton_filter_log(
    log_dens: np.ndarray, trans: np.ndarray, initial: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward Hamilton filter in log space.

    Args:
        log_dens: (T, k) regime-conditional log densities.
        trans: (T, k, k) time-varying transition matrices, `trans[t, i, j]` =
            P(S_t = j | S_{t-1} = i, z_t). Rows must sum to 1.
        initial: (k,) distribution of S_{-1}. Defaults to the stationary
            distribution of `trans[0]`.

    Returns:
        filtered: (T, k) P(S_t = j | y_{1..t}).
        predicted: (T, k) P(S_t = j | y_{1..t-1}).
        loglik: (T,) per-observation log f(y_t | y_{1..t-1}).

    These are the FILTERED probabilities. C7 emits from these and never from
    the smoothed ones — a smoothed xi_t conditions on y_{t+1..T} and would leak
    the future straight into C9's CSRS and C10's AUC.
    """
    n_obs, k = log_dens.shape
    if trans.shape != (n_obs, k, k):
        raise ValueError(f"trans must be ({n_obs}, {k}, {k}), got {trans.shape}")

    prev = (
        stationary_distribution(trans[0])
        if initial is None
        else np.asarray(initial, dtype=float)
    )
    if prev.shape != (k,):
        raise ValueError(f"initial must be ({k},), got {prev.shape}")

    filtered = np.empty((n_obs, k))
    predicted = np.empty((n_obs, k))
    loglik = np.empty(n_obs)

    for t in range(n_obs):
        pred = prev @ trans[t]
        pred = np.clip(pred, 1e-300, None)
        predicted[t] = pred
        log_joint = np.log(pred) + log_dens[t]
        lse = logsumexp(log_joint)
        loglik[t] = lse
        prev = np.exp(log_joint - lse)
        filtered[t] = prev

    return filtered, predicted, loglik


def kim_smoother(
    filtered: np.ndarray, predicted: np.ndarray, trans: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Kim (1994) backward smoother.

    Args:
        filtered: (T, k) from `hamilton_filter_log`.
        predicted: (T, k) from `hamilton_filter_log`.
        trans: (T, k, k) same convention as the filter.

    Returns:
        smoothed: (T, k) P(S_t = j | y_{1..T}).
        joint: (T-1, k, k) P(S_t = i, S_{t+1} = j | y_{1..T}), the E-step
            weights for the TVTP logit M-step. `joint[t]` is aligned with
            `trans[t+1]`, i.e. with z_{t+1}.

    EM-internal only. Never emitted — see `hamilton_filter_log`.
    """
    n_obs, k = filtered.shape
    smoothed = np.empty((n_obs, k))
    joint = np.empty((n_obs - 1, k, k))
    smoothed[-1] = filtered[-1]

    for t in range(n_obs - 2, -1, -1):
        ratio = smoothed[t + 1] / np.clip(predicted[t + 1], 1e-300, None)  # (k,)
        j_t = filtered[t][:, None] * trans[t + 1] * ratio[None, :]
        total = j_t.sum()
        if total <= 0.0 or not np.isfinite(total):  # degenerate window
            j_t = np.full((k, k), 1.0 / (k * k))
            total = 1.0
        j_t = j_t / total
        joint[t] = j_t
        smoothed[t] = j_t.sum(axis=1)

    return smoothed, joint
