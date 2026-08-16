"""C8 — Reconciliation Layer (FR25-FR27) — configuration.

Defaults marked [SPEC] or [D4] are fixed by the documents; [OPS] are this
session's operational choices. SPEC OQ9 leaves `Q_proc`'s *magnitude*
undefined (D4 fixed the form of `R^(p)(.)`, not the process noise) — per
AGENTS.md §9 / roadmap §4 that is an "undefined constant" for FR34's
sensitivity sweep, so a starting value is chosen here and labelled, not
silently invented as if the spec had given it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from seraph.shared_types import PILLARS, Vec3

# One trading day in seconds. Used to express `q_proc_per_day` and the RMT /
# Hamilton staleness scales in the same unit as the timestamps themselves.
SECONDS_PER_DAY = 86_400.0

# D4's saturating exponential is the production form. `linear` and `power` are
# here only so CT-4 can run ARCHITECTURE §7 D4's mandated three-form sweep and
# show what unboundedness costs; neither is a legal production setting.
type NoiseForm = Literal["saturating_exponential", "linear", "power"]


@dataclass(frozen=True)
class ReconciliationConfig:
    """C8 configuration.

    Args are all per-pillar triples in `PILLAR_ORDER`
    (`("hawkes", "rmt", "hamilton")`) unless noted.
    """

    # ---- D4: R^(p)(Delta) ---------------------------------------------------
    # [D4] Bounded saturating exponential. Anything else here is a sweep, not
    # a deployment: an unbounded R sends P -> infinity for a structurally
    # absent pillar and makes the FR29 interval infinite.
    noise_form: NoiseForm = "saturating_exponential"

    # [D4] `h_p` initialised to each pillar's native update cadence — Hawkes at
    # bar frequency (FR10, 5-minute bars), RMT and Hamilton daily (FR16, FR23).
    # A signal is appreciably stale after roughly one of its own update cycles.
    h_seconds: Vec3 = (300.0, SECONDS_PER_DAY, SECONDS_PER_DAY)

    # [D4] R_max^(p) = 100 * R_0^(p). At saturation the Kalman gain is ~0, so a
    # fully stale pillar contributes nothing — while P stays finite.
    r_max_ratio: float = 100.0

    # [D4] `noise_saturated[p]` flips at R^(p)(Delta) > 0.95 * R_max, which is
    # also a D2 mask-exclusion trigger.
    saturation_fraction: float = 0.95

    # [OPS] R_0^(p) for Hawkes and RMT is "rolling empirical variance of the
    # sub-score over a trailing window" (D4). Window length is not specified;
    # 60 observations is ~3 months of daily emissions, long enough to be stable
    # and short enough to track a level shift.
    r0_window: int = 60
    # [OPS] Prior R_0 before that window has any content, and the floor applied
    # to every R_0 afterwards. The floor matters: Hamilton's FR23 estimation
    # variance is exactly 0 when the filter is certain of the regime, and R = 0
    # is a gain of 1 — one emission would be treated as ground truth and drive
    # P[p][p] to 0.
    r0_prior: Vec3 = (1.0e-2, 1.0e-2, 1.0e-2)
    r0_floor: float = 1.0e-6

    # ---- FR25: random-walk transition ---------------------------------------
    # [OPS, SPEC OQ9] Process-noise variance per pillar per trading day. The
    # random walk is x_{t+dt} = x_t, P -> P + Q * dt, with dt in trading days,
    # so a 5-minute Hawkes tick costs 1/288 of a day's drift rather than a
    # whole one — the recursion must not depend on how often it is polled.
    # D4 says Q_proc is MLE-fitted alongside h_p and R_max/R_0; until that fit
    # exists this is a documented starting value, swept by FR34.
    q_proc_per_day: Vec3 = (1.0e-3, 1.0e-3, 1.0e-3)

    # ---- Priors -------------------------------------------------------------
    # [OPS] The state a pillar sits at before it has ever been observed. NOT
    # zero by default reasoning: zero is a *meaningful* value in every pillar's
    # units (MTS_t = 0 means no self-excitation, LSD_t = 0 means certainly
    # tranquil), so imputing it would assert calm, which is precisely the
    # failure D2's mask exists to prevent. The mask keeps this value out of the
    # CSRS entirely; it is a placeholder, and callers with a real climatology
    # for a pillar should pass it.
    prior_mean: Vec3 = (0.0, 0.0, 0.0)
    # [OPS] The prior mean is credited with the weight of a single observation
    # (`P[p][p] = R_0^(p)` at t=0) and then inflates toward the D4 ceiling under
    # process noise exactly like any other ageing information. Set True to
    # start at the ceiling instead — i.e. to claim no prior information at all,
    # which is the more conservative reading and produces a flat `P` for a
    # pillar that is never observed.
    prior_variance_at_ceiling: bool = False

    # ---- FR27: forward-fill ablation baseline -------------------------------
    # [OPS] The baseline's constant covariance, in units of R_0. Forward-fill
    # has no uncertainty model at all — FR29 still needs *a* P, so the baseline
    # reports its R_0 and never ages it. That flatness is the point of the
    # comparison, not an oversight.
    forward_fill_variance_ratio: float = 1.0

    # ---- bookkeeping --------------------------------------------------------
    # [OPS] `state_at()` serves states the recursion has already produced. A
    # Hawkes-cadence backtest produces millions of them, so the buffer is
    # bounded and oldest-first: C1 (FR38) is where a state goes to be kept, not
    # this layer's memory.
    history_limit: int = 8192

    def __post_init__(self) -> None:
        if len(self.h_seconds) != len(PILLARS):
            raise ValueError("h_seconds must carry one entry per pillar")
        if any(h <= 0.0 for h in self.h_seconds):
            raise ValueError("every h_p must be positive")
        if self.r_max_ratio <= 1.0:
            raise ValueError("r_max_ratio must exceed 1 (R_max > R_0)")
        if not 0.0 < self.saturation_fraction < 1.0:
            raise ValueError("saturation_fraction must lie in (0, 1)")
        if self.r0_window < 2:
            raise ValueError("r0_window must be at least 2 observations")
        if any(r <= 0.0 for r in self.r0_prior):
            raise ValueError("every r0_prior must be positive")
        if self.r0_floor <= 0.0:
            raise ValueError("r0_floor must be positive")
        if any(q < 0.0 for q in self.q_proc_per_day):
            raise ValueError("process noise cannot be negative")
        if self.forward_fill_variance_ratio <= 0.0:
            raise ValueError("forward_fill_variance_ratio must be positive")
        if self.history_limit < 1:
            raise ValueError("history_limit must be at least 1")
