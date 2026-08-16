"""Shared types — S5, the pillars -> C8 seam (docs/ARCHITECTURE.md §2).

This is the single most consequential contract in the system: absence is a
first-class *value* here, not an error and not a zero.
"""

from __future__ import annotations

from typing import Annotated, Final, Literal, Protocol

from pydantic import ConfigDict, Field

from seraph.shared_types.common import (
    FrozenModel,
    ISOTimestamp,
    Result,
    Simplex3,
)

# PILLAR ORDER IS GLOBAL AND FIXED. Every Vec3, Mat3 and AvailabilityMask in
# this system uses this order. Violating it silently corrupts the CSRS — no
# error is raised. Kept as a list literal because scripts/check_invariants.py
# matches on exactly this form; consumers should use PILLARS / PILLAR_INDEX.
PILLAR_ORDER = ["hawkes", "rmt", "hamilton"]

type PillarId = Literal["hawkes", "rmt", "hamilton"]

PILLARS: Final[tuple[PillarId, ...]] = ("hawkes", "rmt", "hamilton")
PILLAR_INDEX: Final[dict[PillarId, int]] = {p: i for i, p in enumerate(PILLARS)}


class PillarObservation(FrozenModel):
    """S5 `PillarObservation`.

    `tau` is when the pillar was ACTUALLY COMPUTED — never the query time,
    never a forward-filled time. C8's `R^(p)(ts - tau)` is only correct if it
    is honest.

    No validator enforces `tau <= ts` here on purpose: CT-4 requires C8 to
    *detect* `tau > ts` and return `CONTRACT_VIOLATION`, which it cannot do if
    the shape refuses to carry the offending value.
    """

    pillar: PillarId
    ts: ISOTimestamp
    tau: ISOTimestamp
    value: float
    estimation_variance: float | None = None


class ObservedEmission(FrozenModel):
    kind: Literal["observed"] = "observed"
    obs: PillarObservation


class UnavailableEmission(FrozenModel):
    """Absence as a value.

    `structural` — the pillar cannot exist for this period at all (e.g. Hawkes
    pre-2015, Hamilton before its minimum estimation history). Drives D2 mask
    *exclusion*.

    `transient` — it should exist but this update failed. The pillar stays in
    the mask and D4 age-inflation handles it.
    """

    kind: Literal["unavailable"] = "unavailable"
    pillar: PillarId
    ts: ISOTimestamp
    absence: Literal["structural", "transient"]
    reason: Literal["no_data_coverage", "estimation_failed", "insufficient_history"]


type PillarEmission = Annotated[
    ObservedEmission | UnavailableEmission, Field(discriminator="kind")
]


class PillarCoverage(FrozenModel):
    """D2 — the period over which a pillar can exist at all."""

    # ARCHITECTURE names these `from`/`to`; both are Python keywords, so the
    # wire names live on as aliases and `populate_by_name` keeps both usable.
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    from_ts: ISOTimestamp = Field(alias="from")
    to_ts: ISOTimestamp = Field(alias="to")


class HamiltonDetail(FrozenModel):
    """C7 detail — rides alongside the emission, never inside it."""

    xi: Simplex3  # [tranquil, stressed, crisis]
    p_hat_22: float
    p_hat_33: float
    tau_half_stressed: float
    tau_half_crisis: float


class PillarEngine(Protocol):
    """S5 `PillarEngine`. Implemented by C5, C6 and C7.

    `emit`/`emit_range` return `Ok(UnavailableEmission)` — not `Err` — when a
    pillar has nothing to say. `Err` is reserved for genuine failures of the
    call itself (a missing upstream dependency, a contract violation).
    """

    pillar: PillarId

    async def emit(self, ts: ISOTimestamp) -> Result[PillarEmission]: ...

    async def emit_range(
        self, from_ts: ISOTimestamp, to_ts: ISOTimestamp
    ) -> Result[tuple[PillarEmission, ...]]: ...

    async def coverage(self) -> Result[PillarCoverage]: ...
