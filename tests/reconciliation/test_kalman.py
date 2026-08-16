"""C8 — the recursion itself (FR25, FR26).

The interesting assertions here are the ones with an independent oracle:

* a sequence of scalar updates must equal the textbook simultaneous vector
  update when the observation noises are independent — that is the algebraic
  fact the sequential implementation rests on;
* `P` must stay symmetric PSD under arbitrary update sequences (Hypothesis),
  because `ReconciledState` refuses to be constructed otherwise and a filter
  that only *usually* produces a valid covariance fails in production, not in
  CI.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from seraph.reconciliation.kalman import (
    apply_ceiling,
    predict,
    scalar_update,
    symmetrise,
    to_mat3,
)

Q = np.array([1e-3, 2e-3, 5e-4])


def _psd(p: np.ndarray, tol: float = 1e-12) -> bool:
    return bool(np.all(np.linalg.eigvalsh(symmetrise(p)) >= -tol))


def _corr(m: np.ndarray) -> float:
    return float(m[0, 1] / np.sqrt(m[0, 0] * m[1, 1]))


def test_predict_is_a_pure_random_walk() -> None:
    x = np.array([0.4, 0.2, 1.1])
    p = np.diag([0.01, 0.02, 0.03])
    x_new, p_new = predict(x, p, Q, dt_days=2.0)
    np.testing.assert_array_equal(x_new, x)  # FR25: the mean does not move
    np.testing.assert_allclose(np.diag(p_new), np.diag(p) + 2.0 * Q)


def test_predict_scales_with_elapsed_time_not_call_count() -> None:
    """Two half-steps must equal one whole step, or the filter's answer would
    depend on how often it happens to be polled."""
    p = np.diag([0.01, 0.02, 0.03])
    x = np.zeros(3)
    once = predict(x, p, Q, 1.0)[1]
    twice = predict(*predict(x, p, Q, 0.5), Q, 0.5)[1]
    np.testing.assert_allclose(once, twice, atol=1e-15)


def test_predict_ignores_non_positive_dt() -> None:
    p = np.diag([0.01, 0.02, 0.03])
    for dt in (0.0, -1.0):
        np.testing.assert_allclose(predict(np.zeros(3), p, Q, dt)[1], p)


def test_scalar_update_reduces_variance_and_moves_toward_the_observation() -> None:
    x = np.array([0.0, 0.0, 0.0])
    p = np.diag([1.0, 1.0, 1.0])
    x_new, p_new, gain = scalar_update(x, p, index=1, z=2.0, r=1.0)
    assert gain[1] == 0.5
    assert x_new[1] == 0.5 * 2.0
    assert p_new[1, 1] < p[1, 1]
    assert np.trace(p_new) < np.trace(p)


def test_a_huge_r_leaves_the_state_essentially_untouched() -> None:
    """D4's saturation behaviour at the recursion level: gain ~ 0."""
    x = np.array([0.4, 0.0, 0.0])
    p = np.diag([0.01, 0.01, 0.01])
    x_new, p_new, gain = scalar_update(x, p, index=0, z=99.0, r=1e6)
    assert gain[0] < 1e-6
    np.testing.assert_allclose(x_new, x, atol=1e-3)
    np.testing.assert_allclose(p_new, p, atol=1e-6)


def test_sequential_scalar_updates_equal_the_simultaneous_vector_update() -> None:
    """The oracle for doing three scalar updates instead of one 3x3 update."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(3, 3))
    p = a @ a.T + np.eye(3)
    x = rng.normal(size=3)
    z = rng.normal(size=3)
    r = np.array([0.2, 0.5, 1.3])

    x_seq, p_seq = x, p
    for i in range(3):
        x_seq, p_seq, _ = scalar_update(x_seq, p_seq, i, float(z[i]), float(r[i]))

    # Textbook joint update with H = I.
    s = p + np.diag(r)
    k = p @ np.linalg.inv(s)
    x_joint = x + k @ (z - x)
    p_joint = (np.eye(3) - k) @ p

    np.testing.assert_allclose(x_seq, x_joint, atol=1e-10)
    np.testing.assert_allclose(p_seq, p_joint, atol=1e-10)


def test_apply_ceiling_bounds_the_diagonal_and_keeps_p_psd() -> None:
    p = np.array([[4.0, 1.9, 0.0], [1.9, 1.0, 0.0], [0.0, 0.0, 0.25]])
    bounded = apply_ceiling(p, np.array([1.0, np.inf, np.inf]))
    assert bounded[0, 0] == 1.0
    assert bounded[1, 1] == 1.0  # untouched
    assert bounded[2, 2] == 0.25
    assert _psd(bounded)
    # Correlation is preserved by the congruence, which is the reason for
    # rescaling rather than clipping the diagonal in place: clipping P[0][0]
    # from 4.0 to 1.0 alone would push this correlation from 0.95 to 1.9.
    assert _corr(bounded) == pytest.approx(_corr(p))


def test_apply_ceiling_is_a_no_op_below_the_bound() -> None:
    p = np.diag([0.1, 0.2, 0.3])
    np.testing.assert_array_equal(apply_ceiling(p, np.array([1.0, 1.0, 1.0])), p)


def test_to_mat3_is_exactly_symmetric() -> None:
    """The S6 validator compares raw floats — near-symmetry is not enough."""
    p = np.array([[1.0, 0.3, 0.2], [0.3 + 1e-17, 1.0, 0.1], [0.2, 0.1, 1.0]])
    m = to_mat3(p)
    assert m[0][1] == m[1][0] and m[0][2] == m[2][0] and m[1][2] == m[2][1]


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(
    values=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=2),
            st.floats(min_value=-10.0, max_value=10.0),
            st.floats(min_value=1e-8, max_value=1e6),
            st.floats(min_value=0.0, max_value=5.0),
        ),
        min_size=1,
        max_size=25,
    )
)
def test_p_stays_symmetric_psd_under_arbitrary_update_sequences(
    values: list[tuple[int, float, float, float]],
) -> None:
    """AGENTS.md §5's standing Hypothesis invariant, for C8's half of it."""
    x = np.zeros(3)
    p = np.diag([1e-2, 1e-2, 1e-2])
    for index, z, r, dt in values:
        x, p = predict(x, p, Q, dt)
        p = apply_ceiling(p, np.array([1.0, 1.0, 1.0]))
        x, p, _ = scalar_update(x, p, index, z, r)
        assert np.allclose(p, p.T, atol=0.0)
        assert _psd(p)
        assert np.isfinite(p).all() and np.isfinite(x).all()
