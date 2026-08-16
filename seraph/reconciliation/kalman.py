"""C8 — Reconciliation Layer (FR25, FR26) — the 3-dimensional recursion.

Hand-rolled NumPy, per AGENTS.md §2: the state is fixed at 3 dimensions in
`PILLAR_ORDER`, so a filtering framework buys nothing and would make D4's
`R^(p)(Delta)` awkward to inline.

Design points that are not arbitrary:

* **Predict is a pure random walk (FR25).** `x_hat` is unchanged; only `P`
  grows, by `Q_proc * dt` with `dt` in trading days. Scaling by elapsed time is
  what makes the recursion invariant to how often it is polled — Hawkes ticks
  at 5 minutes and Hamilton once a day into the same filter.
* **Updates are sequential scalar updates, one per emission.** Each pillar
  observes exactly one component of the state (`H = e_p`), so a 3x3 inversion
  is never needed; sequential scalar updates are algebraically identical to a
  simultaneous vector update when the observation noises are independent,
  which they are — the pillars are estimated from disjoint pipelines.
* **Joseph form, not the short form.** `P = (I - KH) P (I - KH)' + K R K'` is
  algebraically equal to `(I - KH) P` but stays symmetric PSD under float
  error. `ReconciledState` asserts PSD on construction; the short form
  eventually fails that assertion after enough near-zero-`R` updates, and the
  cost here is a 3x3 matmul.
* **The covariance ceiling is applied on predict, never on update.** See
  `noise_model.ceiling` for why it exists.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "apply_ceiling",
    "predict",
    "scalar_update",
    "symmetrise",
    "to_mat3",
    "to_vec3",
]


def symmetrise(p: np.ndarray) -> np.ndarray:
    """Fold float asymmetry back out. Cheap, and keeps the S6 assertion true."""
    return 0.5 * (p + p.T)


def predict(
    x: np.ndarray, p: np.ndarray, q_per_day: np.ndarray, dt_days: float
) -> tuple[np.ndarray, np.ndarray]:
    """FR25 — random-walk predict over `dt_days` trading days.

    A zero or negative `dt_days` is a no-op rather than an error: `update()`
    predicts to each emission's `ts` in turn, and several emissions commonly
    share one timestamp.
    """
    if dt_days <= 0.0:
        return x, symmetrise(p)
    return x, symmetrise(p + np.diag(q_per_day) * dt_days)


def apply_ceiling(p: np.ndarray, ceilings: np.ndarray) -> np.ndarray:
    """Bound each `P[i][i]` at its D4 ceiling, preserving symmetry and PSD.

    Implemented as a congruence transform `P -> S P S` with diagonal
    `S[i] = sqrt(ceiling_i / P[i][i]) <= 1`, which is PSD-preserving for any
    real `S` — as opposed to clipping the diagonal in place, which is not:
    shrinking a variance while leaving its covariances alone can push a
    correlation past 1 and make `P` indefinite.
    """
    diag = np.diag(p).copy()
    scale = np.ones(3)
    for i in range(3):
        c = ceilings[i]
        if math.isfinite(c) and diag[i] > c > 0.0:
            scale[i] = math.sqrt(c / diag[i])
    if np.all(scale == 1.0):
        return p
    return symmetrise(p * np.outer(scale, scale))


def scalar_update(
    x: np.ndarray, p: np.ndarray, index: int, z: float, r: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FR26 — Kalman update on one pillar's scalar observation.

    Args:
        index: pillar position in `PILLAR_ORDER`.
        z: the emitted sub-score.
        r: `R^(p)(Delta)` from `noise_model.observation_noise`.

    Returns `(x, P, K)`. `trace(P)` strictly decreases here whenever `r` is
    finite and `P[index][index] > 0` — the arrival half of SPEC O6's
    acceptance criterion.
    """
    innovation_var = p[index, index] + r
    if innovation_var <= 0.0:
        # Degenerate only if both P and R are zero, which the R_0 floor rules
        # out. Skipping beats dividing by zero and poisoning the state.
        return x, symmetrise(p), np.zeros(3)

    gain = p[:, index] / innovation_var
    x_new = x + gain * (z - x[index])

    a = np.eye(3)
    a[:, index] -= gain
    p_new = a @ p @ a.T + r * np.outer(gain, gain)
    return x_new, symmetrise(p_new), gain


def to_vec3(x: np.ndarray) -> tuple[float, float, float]:
    return (float(x[0]), float(x[1]), float(x[2]))


def to_mat3(
    p: np.ndarray,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Exactly symmetric on the way out — the S6 validator compares raw floats.

    `symmetrise` leaves `P[i][j]` and `P[j][i]` equal to within one ulp; this
    makes them the same float, so a downstream equality check on the shape
    cannot fail for a reason that has nothing to do with the filter.
    """
    q = symmetrise(np.asarray(p, dtype=float))
    return (
        (float(q[0, 0]), float(q[0, 1]), float(q[0, 2])),
        (float(q[0, 1]), float(q[1, 1]), float(q[1, 2])),
        (float(q[0, 2]), float(q[1, 2]), float(q[2, 2])),
    )
