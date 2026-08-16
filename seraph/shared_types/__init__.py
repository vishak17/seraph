"""Shared types — pydantic mirrors of docs/ARCHITECTURE.md §0/§2.

EVERY component imports its shapes from here. Nothing redefines a shape
locally. Treated as frozen once Phase 1 (L0) lands — ask before editing.

**Partial by design.** Only the slice C7 (Hamilton) and C8 (Reconciliation)
need across their seams is defined so far:

  * `common.py`         — §0 scalars, Simplex3, Warning/WarningCode,
                          SeraphError, Result/Ok/Err
  * `pillars.py`        — S5: PILLAR_ORDER, PillarObservation, PillarEmission,
                          PillarCoverage, HamiltonDetail, PillarEngine
  * `features.py`       — the C4 -> pillars slice of S4: ObservationRow,
                          MacroRow
  * `reconciliation.py` — S6: ReconciledState, ReconciliationMode,
                          ReconciliationLayer

Still to come, in the C1 (L0) session: E1-E14 entity models, `CsrsPoint`,
`FusionWeights`, `UniverseSnapshot`, the store interfaces.
"""

from seraph.shared_types.common import (
    SIMPLEX_TOLERANCE,
    AvailabilityMask,
    ConstraintViolated,
    ContractViolation,
    EmptyMask,
    Err,
    EstimationDiverged,
    FrozenModel,
    InsufficientHistory,
    ISODate,
    ISOTimestamp,
    Mat3,
    MissingDependency,
    Ok,
    Result,
    RunId,
    SchemaMismatch,
    SectorId,
    SeraphError,
    SeraphWarning,
    Sha256,
    Simplex3,
    SourceBlocked,
    SourceUnavailable,
    Symbol,
    Vec3,
    Warning,
    err,
    ok,
)
from seraph.shared_types.features import MacroRow, ObservationRow
from seraph.shared_types.pillars import (
    PILLAR_INDEX,
    PILLAR_ORDER,
    PILLARS,
    HamiltonDetail,
    ObservedEmission,
    PillarCoverage,
    PillarEmission,
    PillarEngine,
    PillarId,
    PillarObservation,
    UnavailableEmission,
)
from seraph.shared_types.reconciliation import (
    PSD_TOLERANCE,
    ReconciledState,
    ReconciliationLayer,
    ReconciliationMode,
)

__all__ = [
    "PILLARS",
    "PILLAR_INDEX",
    "PILLAR_ORDER",
    "PSD_TOLERANCE",
    "SIMPLEX_TOLERANCE",
    "AvailabilityMask",
    "ConstraintViolated",
    "ContractViolation",
    "EmptyMask",
    "Err",
    "EstimationDiverged",
    "FrozenModel",
    "HamiltonDetail",
    "ISODate",
    "ISOTimestamp",
    "InsufficientHistory",
    "MacroRow",
    "Mat3",
    "MissingDependency",
    "ObservationRow",
    "ObservedEmission",
    "Ok",
    "PillarCoverage",
    "PillarEmission",
    "PillarEngine",
    "PillarId",
    "PillarObservation",
    "ReconciledState",
    "ReconciliationLayer",
    "ReconciliationMode",
    "Result",
    "RunId",
    "SchemaMismatch",
    "SectorId",
    "SeraphError",
    "SeraphWarning",
    "Sha256",
    "Simplex3",
    "SourceBlocked",
    "SourceUnavailable",
    "Symbol",
    "UnavailableEmission",
    "Vec3",
    "Warning",
    "err",
    "ok",
]
