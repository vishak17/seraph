"""C4 — Feature Deriver (FR5, FR6, FR20, FR22, FR50, FR53).

**Partial by design — the S4 slice C7 consumes, and nothing else.** C4 in full
is owner A/B's session and covers jump extraction (FR50), the `S_t` quintile
grid (FR6), signed volume (FR5) and FR53's estimator-agreement check. What is
implemented here is what the C7 -> C8 seam needs to run on real inputs:

    daily_panel.py       the E3 daily-OHLC slice the derivations read
    volatility.py        RV_t, Yang-Zhang (FR20)
    cross_sectional.py   AR_t and BAS_t — computed HERE, never in C6 (FR6, FR20)
    tariff_covariate.py  G_t (FR22)
    hamilton_source.py   the S4 seam: observation_vector / covariate_vector

Everything absent is absent honestly: `rv_5min` is `None` rather than a proxy,
and no `JumpEvent` or `MicroState` is invented for C5.
"""

from seraph.features.cross_sectional import (
    abdi_ranaldo_spread,
    absorption_ratio,
    cross_sectional_spread,
)
from seraph.features.daily_panel import DailyPanel
from seraph.features.hamilton_source import FeatureConfig, PanelFeatureSource
from seraph.features.tariff_covariate import TariffEvent, tariff_covariate
from seraph.features.volatility import yang_zhang_variance

__all__ = [
    "DailyPanel",
    "FeatureConfig",
    "PanelFeatureSource",
    "TariffEvent",
    "abdi_ranaldo_spread",
    "absorption_ratio",
    "cross_sectional_spread",
    "tariff_covariate",
    "yang_zhang_variance",
]
