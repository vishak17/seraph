"""C7 — Hamilton Engine (FR21, FR24) — time-varying transition probabilities.

FR21: transition probabilities are a multinomial logit on z_t,

    P(S_t = j | S_{t-1} = i, z_t) = exp(gamma_ij' z~_t) / sum_k exp(gamma_ik' z~_t)

with z~_t = [1, z_t] and `gamma_ii = 0` as the identifying restriction (the
staying-put category is the reference). Free parameters: k*(k-1)*(1 + dim z),
which is what SPEC E8's `gamma_ij float[3][3][dim z]` stores once the fixed
zero rows are materialised.

FR24: tau_half = ln(2) / (1 - p_hat_jj) at the *current* z_t, for the stressed
and crisis regimes.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

__all__ = [
    "half_life",
    "initial_gamma",
    "transition_matrices",
    "fit_logit_mstep",
]


def transition_matrices(gamma: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Build (T, k, k) row-stochastic transition matrices from the logit.

    Args:
        gamma: (k, k, p) coefficients; `gamma[i, i]` is required to be zero.
        z: (T, p) design matrix INCLUDING the intercept column.

    Returns:
        (T, k, k) with `trans[t, i, j]` = P(S_t = j | S_{t-1} = i, z_t).
    """
    eta = np.einsum("tp,ijp->tij", z, gamma)  # (T, k, k)
    return softmax(eta, axis=2)


def initial_gamma(k: int, p: int, p_stay: float = 0.9) -> np.ndarray:
    """Persistent starting point: intercepts giving `p_stay` on the diagonal.

    Regime-switching likelihoods are multimodal; starting from near-uniform
    transitions is the reliable way to land in the label-switching swamp.
    """
    if not 0.0 < p_stay < 1.0:
        raise ValueError("p_stay must lie in (0, 1)")
    off = np.log((1.0 - p_stay) / (k - 1)) - np.log(p_stay)
    gamma = np.zeros((k, k, p))
    for i in range(k):
        for j in range(k):
            if i != j:
                gamma[i, j, 0] = off
    return gamma


def _neg_q_and_grad(
    flat: np.ndarray, joint: np.ndarray, z: np.ndarray, k: int, p: int
) -> tuple[float, np.ndarray]:
    """Negative expected complete-data log-likelihood of the transitions.

    Q(gamma) = sum_t sum_{i,j} zeta_t[i, j] * log p_ij(z_t)

    with analytic gradient (AGENTS.md §2 — analytic, not finite-difference):

        dQ/dgamma_ij = sum_t z~_t * (zeta_t[i, j] - rowsum_i(zeta_t) * p_ij(z_t))
    """
    gamma = np.zeros((k, k, p))
    free = flat.reshape(k, k - 1, p)
    for i in range(k):
        cols = [j for j in range(k) if j != i]
        gamma[i, cols, :] = free[i]

    trans = transition_matrices(gamma, z)  # (T, k, k)
    log_trans = np.log(np.clip(trans, 1e-300, None))
    q = float(np.einsum("tij,tij->", joint, log_trans))

    row_mass = joint.sum(axis=2)  # (T, k)
    resid = joint - row_mass[:, :, None] * trans  # (T, k, k)
    grad_full = np.einsum("tij,tp->ijp", resid, z)  # (k, k, p)

    grad_free = np.empty_like(free)
    for i in range(k):
        cols = [j for j in range(k) if j != i]
        grad_free[i] = grad_full[i, cols, :]

    return -q, -grad_free.reshape(-1)


def fit_logit_mstep(
    joint: np.ndarray,
    z: np.ndarray,
    gamma0: np.ndarray,
    max_iter: int = 200,
) -> np.ndarray:
    """EM M-step for the TVTP coefficients — a weighted multinomial logit MLE.

    Args:
        joint: (T-1, k, k) smoothed pair probabilities from the Kim smoother,
            `joint[t]` aligned with z[t + 1].
        z: (T, p) design matrix including the intercept.
        gamma0: (k, k, p) warm start.

    Returns:
        (k, k, p) coefficients with the `gamma_ii = 0` restriction imposed.

    The objective is concave in gamma, so L-BFGS-B lands on the global M-step
    optimum; no restart logic is needed at this level.
    """
    k = joint.shape[1]
    p = z.shape[1]
    z_aligned = z[1:]  # transition into t uses z_t

    free0 = np.empty((k, k - 1, p))
    for i in range(k):
        cols = [j for j in range(k) if j != i]
        free0[i] = gamma0[i, cols, :]

    res = minimize(
        _neg_q_and_grad,
        free0.reshape(-1),
        args=(joint, z_aligned, k, p),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter},
    )

    gamma = np.zeros((k, k, p))
    free = res.x.reshape(k, k - 1, p)
    for i in range(k):
        cols = [j for j in range(k) if j != i]
        gamma[i, cols, :] = free[i]
    return gamma


def half_life(p_jj: float, cap: float) -> float:
    """FR24 — tau_1/2 = ln(2) / (1 - p_hat_jj), in trading days.

    `p_jj` is capped (config `p_jj_cap`) so a near-absorbing regime publishes a
    large finite half-life rather than an infinity that poisons every downstream
    aggregate.
    """
    p = float(np.clip(p_jj, 0.0, cap))
    return float(np.log(2.0) / (1.0 - p))
