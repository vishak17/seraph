"""C7 — Hamilton Engine (FR20, FR21, FR23, FR24).

Owns E8: the three-state TVTP filter, the EM procedure, transition
coefficients gamma_ij, regime moments, and liquidity-recovery half-lives.

Layout:
    config.py        HamiltonConfig — SPEC-fixed values vs documented ops defaults
    observations.py  y_t / z_t assembly from the C4 (S4) interface
    filter.py        Hamilton filter + Kim smoother (pure NumPy)
    tvtp.py          multinomial-logit transitions (FR21) and half-lives (FR24)
    em.py            EM estimation and regime identification
    engine.py        HamiltonEngine — the S5 PillarEngine implementation
    output.py        E8 `hamilton_output` row

Why EM is hand-rolled: `statsmodels.tsa.regime_switching` supports TVTP via
`exog_tvtp` but is univariate-only (`MarkovSwitching.__init__` raises
`ValueError('Must have univariate endogenous data.')`), and FR20's
y_t = (RV_t, AR_t, BAS_t) is 3-variate. AGENTS.md §2 permits hand-rolling only
in that case; the filter is cross-checked against statsmodels' own Cython
kernel in tests/pillars/test_hamilton_filter.py.
"""

from seraph.pillars.hamilton.config import HamiltonConfig
from seraph.pillars.hamilton.em import HamiltonEstimationError, HamiltonParams, fit_em
from seraph.pillars.hamilton.engine import HamiltonEngine, close_ts, date_of
from seraph.pillars.hamilton.observations import HamiltonFeatureSource
from seraph.pillars.hamilton.output import HamiltonOutputRow

__all__ = [
    "HamiltonConfig",
    "HamiltonEngine",
    "HamiltonEstimationError",
    "HamiltonFeatureSource",
    "HamiltonOutputRow",
    "HamiltonParams",
    "close_ts",
    "date_of",
    "fit_em",
]
