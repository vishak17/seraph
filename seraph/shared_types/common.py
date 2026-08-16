"""Shared types — Python realisation of docs/ARCHITECTURE.md §0.

TypeScript in ARCHITECTURE is an interface description language only (§0
"Notation caveat"). The mapping used here:

    TS `interface`/`type`       -> pydantic BaseModel
    TS union on `status`/`kind` -> Literal-discriminated union
    TS `readonly`               -> model_config frozen=True
    TS `Vec3`/`Mat3`            -> fixed-length tuples
    TS camelCase field names    -> snake_case (SPEC §6 entity fields are
                                   snake_case and SPEC is authoritative)

The shapes are binding. The language is not.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, StringConstraints

# --------------------------------------------------------------------------
# Scalars
# --------------------------------------------------------------------------

type ISODate = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")]

# AGENTS.md §5: all timestamps are IST and explicit. The pattern is the
# enforcement point — a naive or UTC timestamp cannot be constructed.
type ISOTimestamp = Annotated[
    str,
    StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+05:30$"),
]

type Symbol = str
type SectorId = str
type RunId = str
type Sha256 = str

type Vec3 = tuple[float, float, float]
type Mat3 = tuple[Vec3, Vec3, Vec3]

SIMPLEX_TOLERANCE = 1e-9


def _validate_simplex(v: Vec3) -> Vec3:
    if any(x < 0.0 for x in v):
        raise ValueError(f"Simplex3 requires all components >= 0, got {v!r}")
    total = sum(v)
    if abs(total - 1.0) > SIMPLEX_TOLERANCE:
        raise ValueError(f"Simplex3 must sum to 1 (+/-1e-9), got {total!r}")
    return v


type Simplex3 = Annotated[Vec3, AfterValidator(_validate_simplex)]

# D2 — which pillars carry genuine information at this timestamp.
# Always in PILLAR_ORDER (see seraph.shared_types.pillars).
type AvailabilityMask = tuple[bool, bool, bool]


class FrozenModel(BaseModel):
    """Base for every `readonly` ARCHITECTURE shape.

    Public on purpose: every component's models subclass this rather than
    re-declaring `ConfigDict(frozen=True, extra="forbid")` locally, which is
    how a `readonly` field quietly becomes mutable in one component only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------
# Warnings
# --------------------------------------------------------------------------

type WarningCode = Literal[
    "PARTIAL_COVERAGE",
    "ESTIMATOR_FALLBACK",
    "STALE_OBSERVATION",
    "LOW_EVENT_COUNT",
    "UNIVERSE_UNDERFILLED",
    "SOURCE_SCHEMA_LEGACY",
    "MASK_DEGRADED",
    "REBALANCE_ADJACENT",
    "COMMON_SUPPORT_LOW",
    "NOISE_SATURATED",
]


class Warning(FrozenModel):  # noqa: A001 — name is the ARCHITECTURE §0 contract name
    """ARCHITECTURE §0 `Warning`. Exported as `SeraphWarning` too."""

    code: WarningCode
    message: str
    context: dict[str, str | float] = {}


# Plain alias, not a `type` statement: this name is used as a constructor
# (`SeraphWarning(...)`) wherever shadowing the builtin `Warning` would be
# confusing, and a TypeAliasType is not callable.
SeraphWarning = Warning


# --------------------------------------------------------------------------
# Errors — discriminated on `kind`
# --------------------------------------------------------------------------


class SourceUnavailable(FrozenModel):
    kind: Literal["SOURCE_UNAVAILABLE"] = "SOURCE_UNAVAILABLE"
    source: str
    http_status: int | None = None
    retryable: Literal[True] = True


class SourceBlocked(FrozenModel):
    kind: Literal["SOURCE_BLOCKED"] = "SOURCE_BLOCKED"
    source: str
    hint: Literal["residential-ip-required"] = "residential-ip-required"
    retryable: Literal[False] = False


class SchemaMismatch(FrozenModel):
    kind: Literal["SCHEMA_MISMATCH"] = "SCHEMA_MISMATCH"
    source: str
    expected: str
    received: str
    retryable: Literal[False] = False


class InsufficientHistory(FrozenModel):
    kind: Literal["INSUFFICIENT_HISTORY"] = "INSUFFICIENT_HISTORY"
    required: int
    available: int
    as_of: ISODate
    retryable: Literal[False] = False


class EstimationDiverged(FrozenModel):
    kind: Literal["ESTIMATION_DIVERGED"] = "ESTIMATION_DIVERGED"
    estimator: str
    iterations: int
    last_objective: float
    retryable: Literal[False] = False


class ConstraintViolated(FrozenModel):
    kind: Literal["CONSTRAINT_VIOLATED"] = "CONSTRAINT_VIOLATED"
    constraint: str
    observed: float
    bound: float
    retryable: Literal[False] = False


class MissingDependency(FrozenModel):
    kind: Literal["MISSING_DEPENDENCY"] = "MISSING_DEPENDENCY"
    entity: str
    as_of: ISOTimestamp
    retryable: Literal[True] = True


class EmptyMask(FrozenModel):
    kind: Literal["EMPTY_MASK"] = "EMPTY_MASK"
    ts: ISOTimestamp
    retryable: Literal[False] = False


class ContractViolation(FrozenModel):
    kind: Literal["CONTRACT_VIOLATION"] = "CONTRACT_VIOLATION"
    field: str
    detail: str
    retryable: Literal[False] = False


type SeraphError = Annotated[
    SourceUnavailable
    | SourceBlocked
    | SchemaMismatch
    | InsufficientHistory
    | EstimationDiverged
    | ConstraintViolated
    | MissingDependency
    | EmptyMask
    | ContractViolation,
    "discriminated on `kind`",
]


# --------------------------------------------------------------------------
# Result<T> = Ok<T> | Err
# --------------------------------------------------------------------------


class Ok[T](FrozenModel):
    status: Literal["ok"] = "ok"
    value: T
    warnings: tuple[Warning, ...] = ()


class Err(FrozenModel):
    status: Literal["error"] = "error"
    error: SeraphError


type Result[T] = Ok[T] | Err


def ok[T](value: T, warnings: tuple[Warning, ...] = ()) -> Ok[T]:
    """Ok<T> constructor. `unavailable` pillar states are Ok, never Err."""
    return Ok[T](value=value, warnings=warnings)


def err(error: SeraphError) -> Err:
    """Err constructor. Never raise across a component boundary — return this."""
    return Err(error=error)
