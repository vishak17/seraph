"""C7 — Hamilton Engine (FR20, FR21, FR23, FR24) — configuration.

Every knob here is either fixed by SPEC or is an operational default chosen in
this session and documented as such. SPEC-undefined *constants* (AGENTS.md §9)
are NOT invented here — the tariff decay `eta` lives in C4 (FR22), and the
covariate set is spec-listed (E4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# SPEC FR20 fixes three regimes. A fourth (recovery) regime is SPEC §7
# out-of-scope item 3 — do not parameterise it in.
N_REGIMES = 3

# Index semantics after label identification (see em.identify_regimes):
#   0 = tranquil, 1 = stressed, 2 = crisis   (SPEC E8's xi_1/xi_2/xi_3)
TRANQUIL, STRESSED, CRISIS = 0, 1, 2

# Every C7 timestamp is the NSE close of a trading date, IST-explicit.
MARKET_CLOSE = "15:30:00"
IST_OFFSET = "+05:30"

type CovariateName = Literal[
    "rbi_repo_rate",
    "bank_credit_growth_yoy",
    "india_vix",
    "inr_twi",
    "brent_price",
    "g_t",
    "g_t_sector_weighted",
]


@dataclass(frozen=True)
class HamiltonConfig:
    """C7 configuration.

    Defaults marked [SPEC] are requirements; [OPS] are this session's
    documented operational choices, reviewable without touching the spec.
    """

    # ---- z_t composition (FR21, FR22; SPEC E4) ------------------------------
    # [SPEC] The full E4 covariate set plus G_t. `g_t_sector_weighted` is
    # available but off by default — it is collinear with `g_t` by construction
    # and FR22 asks for it as a *variant*, not as an additional column.
    covariates: tuple[CovariateName, ...] = (
        "rbi_repo_rate",
        "bank_credit_growth_yoy",
        "india_vix",
        "inr_twi",
        "brent_price",
        "g_t",
    )

    # [OPS] SPEC E4 carries a `[GAP]`: no alignment rule for mixing publication
    # frequencies inside z_t. Policy: forward-fill each covariate onto the
    # trading-day grid (repo rate and credit growth are step series, so this is
    # the standard alignment), and emit PARTIAL_COVERAGE when it fires.
    covariate_fill: Literal["ffill", "none"] = "ffill"

    # [OPS] SPEC OQ5 is OPEN: India VIX starts Nov 2007, so the 2008 epoch has
    # leading nulls. Until OQ5 is decided, a covariate that is still null after
    # fill anywhere in the fit window is DROPPED for that window, with
    # PARTIAL_COVERAGE naming it. Never silently zero-filled.
    drop_incomplete_covariates: bool = True

    # ---- y_t construction (FR20) -------------------------------------------
    # [OPS] RV_t and BAS_t are positive and right-skewed; a Gaussian regime
    # likelihood on the raw level is badly misspecified. AR_t is already a
    # bounded ratio and is left alone. FR20 does not specify a transform.
    rv_transform: Literal["log", "identity"] = "log"
    ar_transform: Literal["logit", "identity"] = "identity"
    bas_transform: Literal["log", "identity"] = "log"

    # [OPS] FR20 says RV_t uses Yang-Zhang "across the full corpus (and true
    # 5-minute RV where intraday data exists)". Splicing estimators mid-sample
    # would inject a variance break at ~2015 straight into a *structural-break*
    # detector. Default is therefore the single consistent YZ series; FR53's
    # agreement check (owned by C4) is what licenses the substitution.
    rv_source: Literal["yz", "rv5min_where_available"] = "yz"

    standardize_y: bool = True  # scale-free regime means; stats frozen at fit

    # ---- estimation (FR20, FR21) -------------------------------------------
    # [OPS] 3-state, 3-variate, TVTP: 3*(3 + 6) Gaussian params + 3*2*(1+|z|)
    # logit params. ~750 trading days (3 years) is the floor for identifying
    # that; below it the engine reports structural absence rather than fitting.
    min_history_days: int = 750
    # [OPS] Expanding-window refit cadence. Between refits the filter runs
    # forward on frozen parameters — no lookahead, ever (see engine).
    refit_every_days: int = 63  # ~quarterly
    em_max_iter: int = 300
    em_tol: float = 1e-6  # relative log-likelihood change
    cov_ridge: float = 1e-8  # keeps every Sigma_j positive definite
    logit_max_iter: int = 200

    # ---- emission (FR23, FR24) ---------------------------------------------
    # [OPS] tau_half = ln2/(1 - p_jj) diverges as p_jj -> 1. Cap p_jj so the
    # published half-life stays finite and comparable.
    p_jj_cap: float = 1.0 - 1e-4  # => tau_half <= ~6931 trading days

    seed: int = 20260816

    # ---- provenance ---------------------------------------------------------
    corpus_start: str = "2005-01-01"  # SPEC E3 committed corpus
    _tag: str = field(default="C7", repr=False)

    def __post_init__(self) -> None:
        if self.min_history_days < 250:
            raise ValueError("min_history_days below 250 cannot identify a TVTP fit")
        if self.refit_every_days < 1:
            raise ValueError("refit_every_days must be at least 1 trading day")
        if self.em_max_iter < 1 or self.logit_max_iter < 1:
            raise ValueError("iteration caps must be positive")
        if self.em_tol <= 0.0:
            raise ValueError("em_tol must be positive")
        if self.cov_ridge < 0.0:
            raise ValueError("cov_ridge must be non-negative")
        if not 0.0 < self.p_jj_cap < 1.0:
            raise ValueError("p_jj_cap must lie in (0, 1)")
        if not self.covariates:
            raise ValueError("z_t needs at least one covariate (FR21)")
        if len(set(self.covariates)) != len(self.covariates):
            raise ValueError("duplicate covariate in z_t")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.corpus_start):
            raise ValueError("corpus_start must be an ISO date, YYYY-MM-DD")
