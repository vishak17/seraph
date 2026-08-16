"""C7 — Hamilton Engine (FR20, FR21) — EM estimation.

FR20 asks for a *multivariate* three-state TVTP filter estimated by EM.
`statsmodels.tsa.regime_switching` cannot host it (univariate endog only —
verified: `MarkovSwitching.__init__` raises `ValueError('Must have univariate
endogenous data.')`), so the EM loop is hand-rolled here per AGENTS.md §2's
escape clause, and the filter it rests on is cross-checked against
statsmodels' own Cython kernel in the test suite.

E-step: Hamilton filter + Kim smoother (filter.py).
M-step: closed-form Gaussian moments, plus a concave weighted multinomial
        logit for the TVTP coefficients (tvtp.py, analytic gradient).

One honest caveat on monotonicity: the M-step maximises Q holding the filter's
initial distribution fixed, but that distribution is the stationary vector of
`trans[0]`, which itself moves with gamma. Ascent is therefore exact up to that
single boundary term — in practice dips of order 1e-2 against a log-likelihood
of order 1e3, shrinking as the fit settles. `loglik_trace` is retained so this
stays observable rather than folklore (see test_hamilton_em.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seraph.pillars.hamilton.config import N_REGIMES, HamiltonConfig
from seraph.pillars.hamilton.filter import (
    gaussian_loglik,
    hamilton_filter_log,
    kim_smoother,
)
from seraph.pillars.hamilton.tvtp import (
    fit_logit_mstep,
    initial_gamma,
    transition_matrices,
)

__all__ = ["HamiltonEstimationError", "HamiltonParams", "fit_em", "identify_regimes"]


class HamiltonEstimationError(RuntimeError):
    """EM failed. Carries what `ESTIMATION_DIVERGED` would need.

    C7 does not surface this across its seam: per S5 a failed update is an
    `unavailable`/`transient`/`estimation_failed` emission wrapped in `Ok`,
    never an `Err`. The exception exists so the engine can distinguish a
    failed *fit* from a failed *call*.
    """

    def __init__(self, message: str, iterations: int, last_objective: float) -> None:
        super().__init__(message)
        self.iterations = iterations
        self.last_objective = last_objective


@dataclass(frozen=True, eq=False)
class HamiltonParams:
    """Fitted parameters of the three-state TVTP filter (SPEC E8's gamma_ij).

    Regime indices are already identified: 0 tranquil, 1 stressed, 2 crisis.
    """

    mu: np.ndarray  # (k, d) regime means, in TRANSFORMED/STANDARDISED y space
    sigma: np.ndarray  # (k, d, d) regime covariances, same space
    gamma: np.ndarray  # (k, k, p) TVTP logit coefficients, gamma[i, i] == 0
    covariates: tuple[str, ...]  # z_t columns actually used (excl. intercept)
    y_center: np.ndarray  # (d,) y standardisation, frozen at fit time
    y_scale: np.ndarray  # (d,)
    z_center: np.ndarray  # (p-1,) z standardisation, frozen at fit time
    z_scale: np.ndarray  # (p-1,)
    loglik: float
    loglik_trace: tuple[float, ...]  # per-iteration; EM ascent is testable
    n_iter: int
    converged: bool
    n_obs: int
    fitted_through: str  # ISODate of the last observation in the fit window

    @property
    def n_regimes(self) -> int:
        return int(self.mu.shape[0])


def identify_regimes(
    mu: np.ndarray, sigma: np.ndarray, gamma: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve label switching: order regimes by mean RV ascending.

    EM is invariant to relabelling, so without this the emitted `xi` vector
    would mean something different on every refit and `LSD_t = xi_2 + 2*xi_3`
    (FR23) would be noise. RV is y_t's first component (FR20) and is the
    canonical stress ordering.

    Returns `(mu, sigma, gamma, perm)` with `perm[a]` the old index now at `a`.
    """
    perm = np.argsort(mu[:, 0], kind="stable")
    return mu[perm], sigma[perm], gamma[np.ix_(perm, perm)], perm


def _gaussian_mstep(
    y: np.ndarray, weights: np.ndarray, ridge: float
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted regime means and covariances."""
    k = weights.shape[1]
    dim = y.shape[1]
    mu = np.empty((k, dim))
    sigma = np.empty((k, dim, dim))
    for j in range(k):
        w = weights[:, j]
        mass = w.sum()
        if mass <= 1e-8:
            raise HamiltonEstimationError(
                f"regime {j} collapsed (posterior mass {mass:.2e})", 0, float("nan")
            )
        mu[j] = (w[:, None] * y).sum(axis=0) / mass
        diff = y - mu[j]
        cov = (w[:, None] * diff).T @ diff / mass
        cov = 0.5 * (cov + cov.T) + ridge * np.eye(dim)
        sigma[j] = cov
    return mu, sigma


def fit_em(
    y: np.ndarray,
    z: np.ndarray,
    cfg: HamiltonConfig,
    covariates: tuple[str, ...],
    y_center: np.ndarray,
    y_scale: np.ndarray,
    z_center: np.ndarray,
    z_scale: np.ndarray,
    fitted_through: str,
    warm_start: HamiltonParams | None = None,
) -> HamiltonParams:
    """Run EM to convergence on a fit window.

    Args:
        y: (T, d) transformed/standardised observations.
        z: (T, p) design matrix INCLUDING the leading intercept column.
        covariates: names of z's non-intercept columns, for provenance.
        warm_start: previous fit, used to seed the next expanding-window refit.

    Raises:
        HamiltonEstimationError: on a collapsed regime, a non-PD covariance, or
            a non-finite likelihood. Non-convergence within `em_max_iter` is
            NOT raised — it is returned with `converged=False` so the caller
            can decide (the engine treats it as a transient absence).
    """
    n_obs, dim = y.shape
    k = N_REGIMES
    if n_obs < cfg.min_history_days:
        raise HamiltonEstimationError(
            f"fit window has {n_obs} observations, need {cfg.min_history_days}",
            0,
            float("nan"),
        )

    rng = np.random.default_rng(cfg.seed)
    if warm_start is not None and warm_start.gamma.shape[2] == z.shape[1]:
        mu, sigma, gamma = (
            warm_start.mu.copy(),
            warm_start.sigma.copy(),
            warm_start.gamma.copy(),
        )
    else:
        # Deterministic init: split on RV terciles, which is also the ordering
        # `identify_regimes` will enforce at the end.
        order = np.argsort(y[:, 0], kind="stable")
        chunks = np.array_split(order, k)
        mu = np.stack([y[c].mean(axis=0) for c in chunks])
        base = np.cov(y.T) + cfg.cov_ridge * np.eye(dim)
        sigma = np.stack([base.copy() for _ in range(k)])
        gamma = initial_gamma(k, z.shape[1])
        # A whisker of jitter breaks exact ties between identical chunks.
        mu = mu + 1e-6 * rng.standard_normal(mu.shape)

    prev_ll = -np.inf
    loglik = -np.inf
    converged = False
    it = 0
    trace: list[float] = []

    for it in range(1, cfg.em_max_iter + 1):
        # ---- E step ---------------------------------------------------------
        trans = transition_matrices(gamma, z)
        try:
            log_dens = gaussian_loglik(y, mu, sigma)
        except np.linalg.LinAlgError as exc:  # non-PD covariance
            raise HamiltonEstimationError(
                f"regime covariance lost positive definiteness: {exc}", it, prev_ll
            ) from exc
        filtered, predicted, ll_t = hamilton_filter_log(log_dens, trans)
        loglik = float(ll_t.sum())
        if not np.isfinite(loglik):
            raise HamiltonEstimationError("non-finite log-likelihood", it, prev_ll)
        trace.append(loglik)

        smoothed, joint = kim_smoother(filtered, predicted, trans)

        # ---- M step ---------------------------------------------------------
        mu, sigma = _gaussian_mstep(y, smoothed, cfg.cov_ridge)
        gamma = fit_logit_mstep(joint, z, gamma, max_iter=cfg.logit_max_iter)
        # A diverging inner optimiser must not be allowed to leave the fit in a
        # state that only shows up later as a `nan` regime probability.
        if not (
            np.isfinite(mu).all()
            and np.isfinite(sigma).all()
            and np.isfinite(gamma).all()
        ):
            raise HamiltonEstimationError(
                "M-step produced non-finite parameters", it, loglik
            )

        # ---- convergence ----------------------------------------------------
        denom = max(abs(prev_ll), 1.0)
        if np.isfinite(prev_ll) and abs(loglik - prev_ll) / denom < cfg.em_tol:
            converged = True
            prev_ll = loglik
            break
        prev_ll = loglik

    mu, sigma, gamma, _ = identify_regimes(mu, sigma, gamma)

    return HamiltonParams(
        mu=mu,
        sigma=sigma,
        gamma=gamma,
        covariates=covariates,
        y_center=y_center,
        y_scale=y_scale,
        z_center=z_center,
        z_scale=z_scale,
        loglik=loglik,
        loglik_trace=tuple(trace),
        n_iter=it,
        converged=converged,
        n_obs=n_obs,
        fitted_through=fitted_through,
    )
