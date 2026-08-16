"""C8 — Reconciliation Layer (FR25-FR27).

    layer.py        S6 ReconciliationLayer, both modes (FR25, FR26, FR27)
    kalman.py       the 3-dimensional predict/update recursion
    noise_model.py  D4 R^(p)(Delta), the D4 ceiling, the D2 mask rules
    runner.py       the S5 driver — C8 pulls from C5/C6/C7 (ARCHITECTURE §3)
    pipeline.py     C7 -> C8 wiring; emits S6 states paired with xi for C9
    fitting.py      MLE for D4's h_p, R_max/R_0 and Q_proc (SPEC OQ10)
    output.py       E9 `reconciled_state` rows for the archive (FR38)
    config.py       knobs, each labelled [SPEC] / [D4] / [OPS]

C8 talks to the pillars through S5 (`PillarEmission`, `PillarEngine`) and to C9
through S6 (`ReconciledState`). It imports no component package and reads no
store directly; a caller wires C1/C5/C6/C7 to it.
"""

from seraph.reconciliation.config import ReconciliationConfig
from seraph.reconciliation.fitting import NoiseFit, fit_noise_parameters
from seraph.reconciliation.layer import (
    ForwardFillReconciliationLayer,
    KalmanReconciliationLayer,
)
from seraph.reconciliation.output import e9_row, e9_rows
from seraph.reconciliation.pipeline import (
    PipelineRun,
    ReconciledPoint,
    ReconciliationPipeline,
)
from seraph.reconciliation.runner import ReconciliationRunner, RunReport

__all__ = [
    "ForwardFillReconciliationLayer",
    "KalmanReconciliationLayer",
    "NoiseFit",
    "PipelineRun",
    "ReconciledPoint",
    "ReconciliationConfig",
    "ReconciliationPipeline",
    "ReconciliationRunner",
    "RunReport",
    "e9_row",
    "e9_rows",
    "fit_noise_parameters",
]
