"""C7 — invariant property tests (Hypothesis).

AGENTS.md §2 names Hypothesis for exactly this class of check: the invariants
that must hold for *every* input, not the ones a hand-picked fixture happens to
exercise. For C7 those are: xi stays on the simplex, transition rows stay
stochastic, LSD_t stays in its stated range, and the published half-life stays
finite.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from seraph.pillars.hamilton.engine import HamiltonEngine, _as_simplex
from seraph.pillars.hamilton.filter import hamilton_filter_log, kim_smoother
from seraph.pillars.hamilton.observations import standardise, usable_covariates
from seraph.pillars.hamilton.tvtp import half_life, transition_matrices
from seraph.shared_types import HamiltonDetail

FAST = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

finite = st.floats(
    min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False
)
non_negative = st.floats(
    min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False
)


def _simplex(draw_values: list[float]) -> np.ndarray:
    v = np.clip(np.asarray(draw_values, dtype=float), 0.0, None)
    total = v.sum()
    return v / total if total > 0 else np.full(3, 1.0 / 3.0)


@FAST
@given(st.lists(non_negative, min_size=3, max_size=3))
def test_as_simplex_always_constructs_a_valid_simplex3(values: list[float]) -> None:
    """Simplex3 validates sum == 1 +/- 1e-9; float renormalisation must clear it."""
    xi = _as_simplex(np.asarray(values, dtype=float))
    detail = HamiltonDetail(
        xi=xi,
        p_hat_22=0.5,
        p_hat_33=0.5,
        tau_half_stressed=1.0,
        tau_half_crisis=1.0,
    )
    assert abs(sum(detail.xi) - 1.0) <= 1e-9
    assert all(x >= 0.0 for x in detail.xi)


@FAST
@given(st.lists(non_negative, min_size=3, max_size=3))
def test_lsd_and_its_variance_stay_in_range(values: list[float]) -> None:
    """FR23: LSD_t in [0, 2]; its posterior variance in [0, 1]."""
    xi = _simplex(values)
    lsd = HamiltonEngine._lsd(xi)
    var = HamiltonEngine._lsd_variance(xi)
    assert 0.0 <= lsd <= 2.0 + 1e-12
    # Var of a {0,1,2}-valued variable is maximised at xi = (0.5, 0, 0.5).
    assert 0.0 <= var <= 1.0 + 1e-12


@FAST
@given(
    gamma=hnp.arrays(
        np.float64, (3, 3, 4), elements=st.floats(-10.0, 10.0, allow_nan=False)
    ),
    z=hnp.arrays(np.float64, (7, 4), elements=st.floats(-5.0, 5.0, allow_nan=False)),
)
def test_transition_rows_are_always_stochastic(
    gamma: np.ndarray, z: np.ndarray
) -> None:
    """FR21 — whatever the logit coefficients, P(.|S_{t-1}=i, z_t) is a pmf."""
    trans = transition_matrices(gamma, z)
    assert np.isfinite(trans).all()
    np.testing.assert_allclose(trans.sum(axis=2), 1.0, atol=1e-12)
    assert (trans >= 0.0).all()


@FAST
@given(
    p=st.floats(0.0, 1.0, allow_nan=False),
    q=st.floats(0.0, 1.0, allow_nan=False),
)
def test_half_life_is_finite_and_monotone(p: float, q: float) -> None:
    """FR24 — tau_1/2 grows with persistence and never reaches infinity."""
    cap = 1.0 - 1e-4
    lo, hi = (p, q) if p <= q else (q, p)
    h_lo, h_hi = half_life(lo, cap), half_life(hi, cap)
    assert np.isfinite(h_lo) and np.isfinite(h_hi)
    assert h_lo >= np.log(2.0) - 1e-12
    assert h_hi >= h_lo - 1e-12


@FAST
@given(
    log_dens=hnp.arrays(
        np.float64, (12, 3), elements=st.floats(-30.0, 5.0, allow_nan=False)
    ),
    raw=hnp.arrays(
        np.float64, (12, 3, 3), elements=st.floats(0.01, 1.0, allow_nan=False)
    ),
)
def test_filter_and_smoother_stay_probabilities(
    log_dens: np.ndarray, raw: np.ndarray
) -> None:
    trans = raw / raw.sum(axis=2, keepdims=True)
    filtered, predicted, loglik = hamilton_filter_log(log_dens, trans)
    assert np.isfinite(filtered).all()
    assert np.isfinite(loglik).all()
    np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-9)

    smoothed, joint = kim_smoother(filtered, predicted, trans)
    np.testing.assert_allclose(smoothed.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(joint.sum(axis=2), smoothed[:-1], atol=1e-8)
    np.testing.assert_allclose(joint.sum(axis=1), smoothed[1:], atol=1e-8)


@FAST
@given(
    a=hnp.arrays(np.float64, (20, 3), elements=st.floats(-1e3, 1e3, allow_nan=False))
)
def test_frozen_standardisation_is_reproducible(a: np.ndarray) -> None:
    """Refitting must not silently rescale history (the leakage path)."""
    scaled, center, scale = standardise(a)
    again, _, _ = standardise(a, center, scale)
    np.testing.assert_allclose(scaled, again, atol=0.0)
    assert np.isfinite(scaled).all()
    assert (scale > 0.0).all()


@FAST
@given(
    z=hnp.arrays(
        np.float64,
        (10, 3),
        elements=st.one_of(st.just(np.nan), st.floats(-5.0, 5.0, allow_nan=False)),
    ),
    upto=st.integers(min_value=0, max_value=9),
)
def test_usable_covariates_selects_exactly_the_complete_columns(
    z: np.ndarray, upto: int
) -> None:
    """SPEC OQ5's mechanism: a column enters z_t only once it is complete."""
    cols = usable_covariates(z, ("a", "b", "c"), upto)
    for c in range(z.shape[1]):
        complete = not np.isnan(z[: upto + 1, c]).any()
        assert (c in cols) == complete
