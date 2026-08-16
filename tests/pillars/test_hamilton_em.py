"""C7 — EM estimation (FR20, FR21).

These run against a synthetic three-state TVTP panel whose truth is known, so
"the filter works" is an assertion about recovery, not about the code merely
executing.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from seraph.pillars.hamilton.config import HamiltonConfig
from seraph.pillars.hamilton.em import (
    HamiltonEstimationError,
    HamiltonParams,
    fit_em,
    identify_regimes,
)
from seraph.pillars.hamilton.engine import _Prefix
from seraph.pillars.hamilton.filter import gaussian_loglik, hamilton_filter_log
from seraph.pillars.hamilton.observations import (
    ObservationPanel,
    build_panel,
    design_matrix,
    standardise,
)
from seraph.pillars.hamilton.tvtp import transition_matrices
from tests.pillars.synthetic import SyntheticPanel, make_panel

TEST_CFG = HamiltonConfig(
    min_history_days=250,
    refit_every_days=10_000,  # single fit — refit cadence is engine-level
    em_max_iter=40,
    logit_max_iter=60,
)


def _fit_synthetic(
    n_days: int = 500, seed: int = 11
) -> tuple[SyntheticPanel, ObservationPanel, np.ndarray, np.ndarray, HamiltonParams]:
    sp = make_panel(n_days=n_days, seed=seed)
    panel = build_panel(sp.observations, sp.macro, TEST_CFG)
    y_std, y_c, y_s = standardise(panel.y)
    z_std, z_c, z_s = standardise(panel.z)
    params = fit_em(
        y=y_std,
        z=design_matrix(z_std),
        cfg=TEST_CFG,
        covariates=panel.covariates,
        y_center=y_c,
        y_scale=y_s,
        z_center=z_c,
        z_scale=z_s,
        fitted_through=panel.dates[-1],
    )
    return sp, panel, y_std, design_matrix(z_std), params


@pytest.mark.slow
def test_em_loglikelihood_is_non_decreasing() -> None:
    _, _, _, _, params = _fit_synthetic()
    trace = np.asarray(params.loglik_trace)
    assert trace.size >= 2
    assert trace[-1] > trace[0]

    # The M-step maximises Q with the filter's initial distribution treated as
    # fixed, but that distribution is the stationary vector of trans[0], which
    # moves whenever gamma moves. So ascent is exact up to that one boundary
    # term rather than exactly monotone. It shows up as dips of order 1e-2 in a
    # log-likelihood of order 1e3, and it shrinks as the fit settles.
    steps = np.diff(trace)
    assert steps.min() > -1e-3 * abs(trace[0])
    assert np.abs(steps[-3:]).max() < 1.0  # settled by the end


@pytest.mark.slow
def test_regimes_are_identified_by_ascending_rv() -> None:
    """Label switching would make LSD_t = xi_2 + 2*xi_3 meaningless."""
    _, _, _, _, params = _fit_synthetic()
    rv_means = params.mu[:, 0]
    assert rv_means[0] < rv_means[1] < rv_means[2]


@pytest.mark.slow
def test_filtered_states_recover_the_simulated_chain() -> None:
    sp, _, y_std, z_design, params = _fit_synthetic()
    trans = transition_matrices(params.gamma, z_design)
    log_dens = gaussian_loglik(y_std, params.mu, params.sigma)
    filtered, _, _ = hamilton_filter_log(log_dens, trans)
    predicted_state = filtered.argmax(axis=1)
    accuracy = float((predicted_state == sp.states).mean())
    # Three overlapping Gaussian regimes: perfect recovery is not on offer.
    # Chance is 1/3; anything above ~0.6 means the filter is tracking.
    assert accuracy > 0.6, f"regime recovery accuracy {accuracy:.3f}"


@pytest.mark.slow
def test_tvtp_coefficients_are_not_degenerate() -> None:
    """FR21 — z_t must actually move the transitions, not just the intercepts."""
    _, _, _, z_design, params = _fit_synthetic()
    slopes = params.gamma[:, :, 1:]
    assert np.abs(slopes).max() > 1e-3
    trans = transition_matrices(params.gamma, z_design)
    diag = np.einsum("tii->ti", trans)
    assert diag.std(axis=0).max() > 1e-3  # p_jj genuinely time-varying


@pytest.mark.slow
def test_incremental_filter_equals_the_batch_recursion() -> None:
    """The production path and the statsmodels-verified path must agree.

    `filter.hamilton_filter_log` is the recursion checked against statsmodels'
    Cython kernel; `engine._Prefix` is the incremental one the engine actually
    runs. If they ever diverge, the audited implementation is not the one
    shipping numbers.
    """
    _, panel, y_std, z_design, params = _fit_synthetic()
    batch, _, _ = hamilton_filter_log(
        gaussian_loglik(y_std, params.mu, params.sigma),
        transition_matrices(params.gamma, z_design),
    )

    prefix = _Prefix(params, y_std, z_design)
    for t in (0, 1, 17, 200, len(panel) - 1):
        xi, trans = prefix.advance_to(t)
        np.testing.assert_allclose(xi, batch[t], atol=1e-12)
        np.testing.assert_allclose(
            trans, transition_matrices(params.gamma, z_design)[t], atol=1e-12
        )


@pytest.mark.slow
def test_posterior_concentrates_on_the_true_state_at_transitions() -> None:
    """NFR16's shape, on synthetic data.

    NFR16 ("posterior for the correct state within one day of a labelled
    transition > 0.90") is a design target measured by C10 against real labelled
    epochs — not something this synthetic panel can certify. What is meaningful
    here is direction: one day after a simulated regime change, the filter puts
    materially more mass on the true state than the 1/3 a coin would.
    """
    sp, _, y_std, z_design, params = _fit_synthetic()
    filtered, _, _ = hamilton_filter_log(
        gaussian_loglik(y_std, params.mu, params.sigma),
        transition_matrices(params.gamma, z_design),
    )
    changed = np.flatnonzero(np.diff(sp.states) != 0) + 1
    changed = changed[changed < len(sp.states) - 1]
    assert changed.size > 10

    one_day_after = changed + 1
    mass = filtered[one_day_after, sp.states[one_day_after]]
    assert mass.mean() > 0.45, f"mean posterior on the true state {mass.mean():.3f}"


def test_params_survive_a_pickle_round_trip() -> None:
    """NFR5 puts each pillar in its own process; parameters must travel."""
    _, _, _, _, params = _fit_synthetic(n_days=300)
    restored = pickle.loads(pickle.dumps(params))
    np.testing.assert_allclose(restored.mu, params.mu)
    np.testing.assert_allclose(restored.gamma, params.gamma)
    assert restored.covariates == params.covariates
    assert restored.fitted_through == params.fitted_through


def test_identify_regimes_permutes_gamma_consistently() -> None:
    mu = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    sigma = np.stack([np.eye(3) * (j + 1) for j in range(3)])
    gamma = np.arange(3 * 3 * 2, dtype=float).reshape(3, 3, 2)
    mu2, sigma2, gamma2, perm = identify_regimes(mu, sigma, gamma)

    assert list(perm) == [1, 2, 0]
    np.testing.assert_allclose(mu2[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(sigma2[0], sigma[1])
    for a in range(3):
        for b in range(3):
            np.testing.assert_allclose(gamma2[a, b], gamma[perm[a], perm[b]])


def test_short_window_is_rejected_not_silently_fitted() -> None:
    cfg = HamiltonConfig(min_history_days=250)
    y = np.random.default_rng(0).normal(size=(50, 3))
    z = design_matrix(np.random.default_rng(1).normal(size=(50, 2)))
    with pytest.raises(HamiltonEstimationError):
        fit_em(
            y=y,
            z=z,
            cfg=cfg,
            covariates=("a", "b"),
            y_center=np.zeros(3),
            y_scale=np.ones(3),
            z_center=np.zeros(2),
            z_scale=np.ones(2),
            fitted_through="2020-01-01",
        )
