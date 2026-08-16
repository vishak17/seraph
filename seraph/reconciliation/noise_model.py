"""C8 — Reconciliation Layer (FR26) — D4 observation noise and the D2 mask.

Two mechanisms live here, and they are the whole of Objective 6:

**D4 — `R^(p)(Delta)`.** Observation noise grows with staleness
`Delta = ts - tau`, where `tau` is when the pillar actually computed the value
(S5). The production form is the bounded saturating exponential

    R^(p)(Delta) = R_0 + (R_max - R_0) * (1 - exp(-Delta / h_p))

`linear` and `power` are provided only for ARCHITECTURE §7 D4's mandated
three-form sweep. Their `ceiling()` is infinite, which is exactly the finding
the sweep is meant to produce: with an unbounded R there is nothing to bound
`P` for a pillar that never emits, and FR29's interval diverges.

**D2 — the availability mask.** Which pillars carry genuine information:

| Emission                         | mask  |
|----------------------------------|-------|
| `observed`                       | true  |
| `unavailable`, absence transient | true  |
| `unavailable`, absence structural| false |
| `observed` but R^(p) at ceiling  | false |

Nothing here touches state; every function is pure, so C8's mask and noise
behaviour is testable without constructing a filter.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from typing import Literal

from seraph.reconciliation.config import NoiseForm, ReconciliationConfig
from seraph.shared_types import ISOTimestamp

__all__ = [
    "PillarStatus",
    "R0Tracker",
    "ceiling",
    "epoch_seconds",
    "is_saturated",
    "mask_bit",
    "observation_noise",
    "staleness_seconds",
]

# What the most recent emission said about a pillar. `never` is the state
# before any emission at all has been seen for it — distinct from `structural`
# because it is C8's ignorance, not the pillar's declaration.
type PillarStatus = Literal["never", "observed", "transient", "structural"]


def epoch_seconds(ts: ISOTimestamp) -> float:
    """IST-explicit timestamp -> seconds since epoch.

    `ISOTimestamp`'s pattern guarantees the `+05:30` offset is present, so this
    can never silently interpret a naive timestamp as UTC (AGENTS.md §5).
    """
    return datetime.fromisoformat(ts).timestamp()


def staleness_seconds(ts: ISOTimestamp, tau: ISOTimestamp | None) -> float:
    """`Delta = ts - tau`, or infinity for a pillar never observed.

    Negative deltas are *not* clamped here: `tau > ts` is a contract violation
    the layer must detect and report (CT-4's negative case), which it cannot do
    if this function quietly rounds it up to zero.
    """
    if tau is None:
        return math.inf
    return epoch_seconds(ts) - epoch_seconds(tau)


def observation_noise(
    delta_seconds: float,
    r0: float,
    h_seconds: float,
    r_max_ratio: float,
    form: NoiseForm = "saturating_exponential",
) -> float:
    """D4 `R^(p)(Delta)`. Monotone non-decreasing in `delta_seconds`.

    `R^(p)(0) == r0` for every form — FR26's requirement that a freshly
    computed observation is trusted at exactly its own estimation variance
    (FR23 where the pillar reports one).
    """
    if delta_seconds <= 0.0:
        return r0
    r_max = r_max_ratio * r0
    if form == "saturating_exponential":
        if math.isinf(delta_seconds):
            return r_max
        return r0 + (r_max - r0) * -math.expm1(-delta_seconds / h_seconds)
    if form == "linear":
        return r0 * (1.0 + delta_seconds / h_seconds)
    if form == "power":
        return r0 * (1.0 + delta_seconds / h_seconds) ** 2
    raise ValueError(f"unknown noise form {form!r}")


def ceiling(r0: float, r_max_ratio: float, form: NoiseForm) -> float:
    """`sup_Delta R^(p)(Delta)` — the bound that keeps `P` finite.

    This is D4's decisive property expressed as a number. C8 applies it as a
    covariance ceiling on the predict step: once uncertainty about a pillar
    reaches the noise floor of a fully-stale observation of it, further random-
    walk drift adds nothing that any observation could ever have resolved.
    Equivalently, each latent pillar state carries a stationary prior of
    variance `R_max`. Without it, 200 ticks of structural absence send
    `P[p][p]` to infinity and FR29's interval with it.

    Infinite for the unbounded sweep forms, by construction.
    """
    if form == "saturating_exponential":
        return r_max_ratio * r0
    return math.inf


def is_saturated(r: float, r0: float, cfg: ReconciliationConfig) -> bool:
    """D4 — `R^(p)(Delta) > 0.95 * R_max`, also a D2 mask-exclusion trigger."""
    bound = ceiling(r0, cfg.r_max_ratio, cfg.noise_form)
    if math.isinf(bound):
        # An unbounded form never "saturates"; it just keeps growing. Reporting
        # saturation here would hide that, which is the opposite of what the
        # D4 sweep is for.
        return False
    return r > cfg.saturation_fraction * bound


def mask_bit(status: PillarStatus, saturated: bool) -> bool:
    """D2 — one pillar's mask bit.

    `never` is masked out for the same reason `structural` is: there is no
    observation behind `x_hat[p]`, only the prior, and FR28 must not allocate
    weight to a dimension carrying no information.
    """
    if status in ("never", "structural"):
        return False
    return not saturated


class R0Tracker:
    """Rolling empirical variance of one pillar's sub-score (D4's `R_0^(p)`).

    Hamilton reports `estimation_variance` per FR23 and that is used directly.
    Hawkes and RMT do not, so D4 specifies "rolling empirical variance of the
    sub-score over a trailing window" — this is that window.

    The sample variance is taken over the *values* the pillar emitted, which is
    a proxy for its estimation error, not the error itself. Stated plainly
    rather than dressed up: a pillar whose sub-score genuinely moves a lot is
    assigned a wider `R_0` than one that sits still. The floor and the prior
    stop that proxy from ever reaching 0 or being undefined.
    """

    __slots__ = ("_prior", "_floor", "_values")

    def __init__(self, prior: float, floor: float, window: int) -> None:
        self._prior = prior
        self._floor = floor
        self._values: deque[float] = deque(maxlen=window)

    def observe(self, value: float) -> None:
        self._values.append(value)

    def r0(self, reported: float | None = None) -> float:
        """`R_0` for the next update. `reported` is FR23's variance if given."""
        if reported is not None and math.isfinite(reported):
            return max(reported, self._floor)
        if len(self._values) < 2:
            return max(self._prior, self._floor)
        n = len(self._values)
        mean = sum(self._values) / n
        var = sum((v - mean) ** 2 for v in self._values) / (n - 1)
        return max(var, self._floor)
