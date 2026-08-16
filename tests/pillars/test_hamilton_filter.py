"""C7 — filter/smoother correctness.

The headline test cross-checks the hand-rolled Hamilton filter against
statsmodels' own Cython kernel. statsmodels cannot host the C7 *model* (it is
univariate-only), but its filter kernel takes arbitrary conditional densities
and arbitrary time-varying transitions, so it is a valid independent oracle for
the recursion itself — including the transposed transition convention, which is
the single easiest thing to get silently wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.tsa.regime_switching.markov_switching import cy_hamilton_filter_log

from seraph.pillars.hamilton.filter import (
    gaussian_loglik,
    hamilton_filter_log,
    kim_smoother,
    stationary_distribution,
)
from seraph.pillars.hamilton.tvtp import (
    half_life,
    initial_gamma,
    transition_matrices,
)


def _random_transitions(rng: np.random.Generator, n_obs: int, k: int) -> np.ndarray:
    raw = rng.random((n_obs, k, k)) + 0.05
    return raw / raw.sum(axis=2, keepdims=True)


def test_filter_matches_statsmodels_kernel() -> None:
    rng = np.random.default_rng(4)
    n_obs, k = 120, 3
    log_dens = rng.normal(-2.0, 1.0, (n_obs, k))
    trans = _random_transitions(rng, n_obs, k)
    initial = np.array([0.5, 0.3, 0.2])

    filtered, predicted, loglik = hamilton_filter_log(log_dens, trans, initial)

    # statsmodels indexes [i, j, t] as P(j at t-1 -> i at t): our transpose.
    sm_trans = trans.transpose(2, 1, 0)
    sm_filtered, sm_predicted, sm_loglik = cy_hamilton_filter_log(
        initial, sm_trans, log_dens.T.copy(order="C"), 0
    )[:3]

    np.testing.assert_allclose(filtered, np.asarray(sm_filtered).T, atol=1e-12)
    np.testing.assert_allclose(predicted, np.asarray(sm_predicted).T, atol=1e-12)
    np.testing.assert_allclose(loglik, np.asarray(sm_loglik), atol=1e-10)


def test_filter_rows_are_probability_vectors() -> None:
    rng = np.random.default_rng(5)
    log_dens = rng.normal(0.0, 2.0, (200, 3))
    trans = _random_transitions(rng, 200, 3)
    filtered, predicted, _ = hamilton_filter_log(log_dens, trans)
    np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(predicted.sum(axis=1), 1.0, atol=1e-12)
    assert (filtered >= 0).all()


def test_smoother_is_consistent_with_its_own_joints() -> None:
    """P(S_t | T) must be both marginal of joint[t] and of joint[t-1]."""
    rng = np.random.default_rng(6)
    n_obs = 150
    log_dens = rng.normal(0.0, 1.5, (n_obs, 3))
    trans = _random_transitions(rng, n_obs, 3)
    filtered, predicted, _ = hamilton_filter_log(log_dens, trans)
    smoothed, joint = kim_smoother(filtered, predicted, trans)

    np.testing.assert_allclose(smoothed.sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(smoothed[-1], filtered[-1], atol=1e-12)
    np.testing.assert_allclose(joint.sum(axis=(1, 2)), 1.0, atol=1e-10)
    np.testing.assert_allclose(joint.sum(axis=2), smoothed[:-1], atol=1e-10)
    np.testing.assert_allclose(joint.sum(axis=1), smoothed[1:], atol=1e-10)


def test_gaussian_loglik_matches_scipy() -> None:
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(7)
    y = rng.normal(size=(40, 3))
    mu = rng.normal(size=(2, 3))
    a = rng.normal(size=(2, 3, 3))
    sigma = np.stack([m @ m.T + np.eye(3) for m in a])
    got = gaussian_loglik(y, mu, sigma)
    for j in range(2):
        expected = multivariate_normal(mean=mu[j], cov=sigma[j]).logpdf(y)
        np.testing.assert_allclose(got[:, j], expected, atol=1e-10)


def test_stationary_distribution_is_fixed_point() -> None:
    p = np.array([[0.9, 0.08, 0.02], [0.1, 0.85, 0.05], [0.05, 0.15, 0.8]])
    pi = stationary_distribution(p)
    np.testing.assert_allclose(pi @ p, pi, atol=1e-10)
    assert abs(pi.sum() - 1.0) < 1e-12


def test_transition_rows_sum_to_one_and_honour_gamma_ii_zero() -> None:
    rng = np.random.default_rng(8)
    gamma = initial_gamma(3, 4, p_stay=0.9)
    assert np.allclose(np.einsum("iip->ip", gamma), 0.0)
    z = np.column_stack([np.ones(50), rng.normal(size=(50, 3))])
    trans = transition_matrices(gamma, z)
    np.testing.assert_allclose(trans.sum(axis=2), 1.0, atol=1e-12)
    # With covariate coefficients still zero, only the intercepts act.
    np.testing.assert_allclose(np.einsum("tii->ti", trans), 0.9, atol=1e-12)


@pytest.mark.parametrize(
    ("p_jj", "expected"), [(0.5, np.log(2.0) / 0.5), (0.9, np.log(2.0) / 0.1)]
)
def test_half_life_formula(p_jj: float, expected: float) -> None:
    assert half_life(p_jj, cap=1 - 1e-4) == pytest.approx(expected)


def test_half_life_is_capped_not_infinite() -> None:
    """FR24 diverges as p_jj -> 1; the published number must stay finite."""
    value = half_life(1.0, cap=1 - 1e-4)
    assert np.isfinite(value)
    assert value == pytest.approx(np.log(2.0) / 1e-4)
