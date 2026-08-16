"""Shared types — the C4 -> pillars slice of S4/S2 (docs/ARCHITECTURE.md §2).

Only the two rows C7 consumes are defined here (`ObservationRow` = y_t,
`MacroRow` = z_t). `DailyBar`, `IntradayBar`, `MicroState`, `JumpEvent` and the
rest of E1-E14 are C1/C4 session work and are deliberately absent.
"""

from __future__ import annotations

from seraph.shared_types.common import FrozenModel, ISODate


class ObservationRow(FrozenModel):
    """S4 `ObservationRow` — the Hamilton observation vector y_t (FR20).

    y_t = (RV_t, AR_t, BAS_t)^T.

    `ar_t` is computed in C4, never in C6 — putting it in the RMT engine would
    make Hamilton secretly depend on RMT and break FR35.
    """

    date: ISODate
    rv_yang_zhang: float
    rv_5min: float | None = None  # null before intraday coverage (~2015)
    ar_t: float
    bas_t: float


class MacroRow(FrozenModel):
    """S2/S4 `MacroRow` — the TVTP covariate vector z_t (FR21, FR22)."""

    date: ISODate
    rbi_repo_rate: float | None = None
    bank_credit_growth_yoy: float | None = None
    india_vix: float | None = None  # null before Nov 2007
    vix_available: bool = True
    inr_twi: float | None = None
    brent_price: float | None = None
    g_t: float | None = None  # FR22, written by C4
    g_t_sector_weighted: float | None = None
