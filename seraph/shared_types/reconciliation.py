"""Shared types — S6, the C8 -> C9 seam (docs/ARCHITECTURE.md §2).

`ReconciledState` is the *only* thing C9 ever sees of the pillars: three
numbers, their covariance, and the provenance needed to know what those
numbers mean (which pillars are real, how stale they are, which recursion
produced them).

Two shape-level guarantees the rest of the system leans on:

* **`p_t` is symmetric PSD — asserted here, not assumed downstream.** FR29's
  `Var(CSRS_t) = Sum_j xi_j^2 w_j' P_t w_j` is a variance; a `P_t` that has
  drifted non-PSD through a badly-conditioned update produces a negative
  variance and a nonsensical confidence interval. CT-5's negative case
  (`CONSTRAINT_VIOLATED` on non-PSD `P`) is C9 rejecting what this validator
  would already have refused to construct.
* **Every vector/matrix axis is `PILLAR_ORDER`** — `x_hat`, `p_t`,
  `tau_last_update`, `mask` and `noise_saturated` alike.

Field names follow SPEC §6 E9 (`x_hat`, `p_t`, `tau_last_update`, `mode`);
`mask` and `noise_saturated` are ARCHITECTURE §2 additions carrying D2/D4
provenance that E9 predates.
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import model_validator

from seraph.shared_types.common import (
    AvailabilityMask,
    FrozenModel,
    ISOTimestamp,
    Mat3,
    Result,
    Vec3,
)
from seraph.shared_types.pillars import PillarEmission

# Relative slack for the symmetry/PSD assertions. Kalman updates in float64
# leave errors around 1e-16 * scale; 1e-9 * scale catches a genuinely broken
# covariance without tripping on rounding.
PSD_TOLERANCE = 1e-9

type ReconciliationMode = Literal["kalman", "forward_fill"]


def _psd_violation(p: Mat3, scale: float) -> str | None:
    """All seven principal minors of a 3x3 — the exact PSD criterion.

    Leading minors alone (Sylvester) test positive *definiteness*; a legitimate
    `P_t` can be singular (a pillar with zero variance right after an update
    with `R = 0`), so every principal minor is checked, not just the leading
    ones. Pure Python on purpose: `shared_types` stays import-light.
    """
    tol_1 = PSD_TOLERANCE * scale
    tol_2 = PSD_TOLERANCE * scale * scale
    tol_3 = PSD_TOLERANCE * scale * scale * scale

    for i in range(3):
        if p[i][i] < -tol_1:
            return f"P[{i}][{i}] = {p[i][i]!r} is negative"

    for i, j in ((0, 1), (0, 2), (1, 2)):
        minor = p[i][i] * p[j][j] - p[i][j] * p[j][i]
        if minor < -tol_2:
            return f"2x2 principal minor on ({i},{j}) = {minor!r} is negative"

    det = (
        p[0][0] * (p[1][1] * p[2][2] - p[1][2] * p[2][1])
        - p[0][1] * (p[1][0] * p[2][2] - p[1][2] * p[2][0])
        + p[0][2] * (p[1][0] * p[2][1] - p[1][1] * p[2][0])
    )
    if det < -tol_3:
        return f"det(P) = {det!r} is negative"
    return None


class ReconciledState(FrozenModel):
    """S6 `ReconciledState` / SPEC E9 `reconciled_state`.

    `tau_last_update[p]` is `None` until pillar `p` has ever been observed —
    not the epoch, not the state's own `ts`. CT-4 asserts exactly that for a
    structurally-absent Hawkes.
    """

    ts: ISOTimestamp
    x_hat: Vec3
    p_t: Mat3
    tau_last_update: tuple[
        ISOTimestamp | None, ISOTimestamp | None, ISOTimestamp | None
    ]
    mask: AvailabilityMask
    noise_saturated: tuple[bool, bool, bool]
    mode: ReconciliationMode

    @model_validator(mode="after")
    def _check_covariance(self) -> ReconciledState:
        p = self.p_t
        scale = max(1.0, p[0][0], p[1][1], p[2][2])

        for i in range(3):
            for j in range(3):
                if p[i][j] != p[i][j]:  # NaN
                    raise ValueError(f"P[{i}][{j}] is not finite")
        for i, j in ((0, 1), (0, 2), (1, 2)):
            if abs(p[i][j] - p[j][i]) > PSD_TOLERANCE * scale:
                raise ValueError(
                    f"P is not symmetric: P[{i}][{j}]={p[i][j]!r} vs "
                    f"P[{j}][{i}]={p[j][i]!r}"
                )

        violation = _psd_violation(p, scale)
        if violation is not None:
            raise ValueError(f"P is not positive semi-definite: {violation}")
        return self


class ReconciliationLayer(Protocol):
    """S6 `ReconciliationLayer`. Implemented by C8.

    `predict` advances time only (FR25). `update` folds in emissions (FR26).
    `state_at` is a read of an already-produced state — it never invents one
    by running the recursion backwards, since the Kalman recursion is not
    time-reversible.
    """

    mode: ReconciliationMode

    async def predict(self, to_ts: ISOTimestamp) -> Result[ReconciledState]: ...

    async def update(
        self, emissions: tuple[PillarEmission, ...]
    ) -> Result[ReconciledState]: ...

    async def state_at(self, ts: ISOTimestamp) -> Result[ReconciledState]: ...
