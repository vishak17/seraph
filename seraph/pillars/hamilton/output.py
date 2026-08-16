"""C7 — Hamilton Engine — E8 `hamilton_output` (SPEC §6).

C7 owns E8 exclusively (ARCHITECTURE §1). The row model lives here rather than
in `shared_types/` because no other component reads it: C8 consumes the S5
emission, C9 consumes `HamiltonDetail.xi`. C1's session will map this shape to
a hypertable; if that session prefers all E1-E14 models in `shared_types/`,
move it there and re-export — do not duplicate it.
"""

from __future__ import annotations

from seraph.shared_types import ISODate, Simplex3
from seraph.shared_types.common import FrozenModel


class HamiltonOutputRow(FrozenModel):
    """One daily E8 row (FR20, FR21, FR23, FR24)."""

    date: ISODate

    # y_t inputs, retained so a stored row is self-describing (SPEC E8)
    rv_t_yz: float
    rv_t_5min: float | None
    ar_t: float
    bas_t: float

    # FR23
    xi: Simplex3  # [tranquil, stressed, crisis] == xi_1, xi_2, xi_3
    lsd_t: float  # xi_2 + 2*xi_3, in [0, 2]
    estimation_uncertainty: float  # Var(LSD_t) under the filtered posterior

    # FR21 / FR24
    p_hat_22: float
    p_hat_33: float
    tau_half_stressed: float
    tau_half_crisis: float
    gamma_ij: tuple[tuple[tuple[float, ...], ...], ...]  # (3, 3, 1 + dim z)
    covariates: tuple[str, ...]  # z_t columns behind gamma_ij, in order

    # provenance — which fit produced this row
    fitted_through: ISODate
    fit_converged: bool
    fit_loglik: float
