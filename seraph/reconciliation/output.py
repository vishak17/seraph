"""C8 — E9 `reconciled_state` rows for the archive (FR38).

ARCHITECTURE §1 gives C8 exclusive ownership of E9, and FR38 requires every
reconciliation covariance to be archived for retrospective analysis. SPEC §6
E9 is:

    ts · x_hat float[3] · P_t float[3][3] · tau_last_update timestamp[3]
       · mode enum{kalman, forward_fill}

`ReconciledState` already *is* that shape — no second model is defined here,
because two pydantic models for one entity is exactly how the persisted
schema and the runtime shape drift apart. What this module provides is the
explicit column mapping C1's `StoreWriter` needs, including the two fields
ARCHITECTURE added after SPEC §6 was written.

`mask` and `noise_saturated` are archived even though E9 predates them: without
the mask, an archived state cannot be told apart from a full-information one,
and FR35/FR36's ablation rows stop being falsifiable a year later when nobody
remembers which pillars existed. Flagged here rather than silently dropped —
C1's DDL needs two columns SPEC §6 does not list.

Natural key is `(ts, mode)`: the Kalman and forward-fill arms of the FR36
ablation both produce a state at the same timestamp, and they are different
rows, not a conflict.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from seraph.shared_types import ISOTimestamp, ReconciledState

__all__ = ["E9_COLUMNS", "E9_NATURAL_KEY", "e9_row", "e9_rows"]

# SPEC §6 E9, then the two ARCHITECTURE §2 additions.
E9_COLUMNS: tuple[str, ...] = (
    "ts",
    "x_hat",
    "P_t",
    "tau_last_update",
    "mode",
    "mask",
    "noise_saturated",
)

# For C1's `INSERT ... ON CONFLICT DO UPDATE` (writeBatch idempotency, CT-1).
E9_NATURAL_KEY: tuple[str, ...] = ("ts", "mode")


def e9_row(state: ReconciledState) -> dict[str, object]:
    """One archivable E9 row, keyed by SPEC §6 column names.

    Column names are SPEC's (`P_t`, not `p_t`) because the DDL follows SPEC;
    the Python attribute names follow the repo's snake_case convention. This
    function is the one place the two spellings meet.
    """
    return {
        "ts": state.ts,
        "x_hat": list(state.x_hat),
        "P_t": [list(row) for row in state.p_t],
        "tau_last_update": list(state.tau_last_update),
        "mode": state.mode,
        "mask": list(state.mask),
        "noise_saturated": list(state.noise_saturated),
    }


def e9_rows(states: Iterable[ReconciledState]) -> tuple[dict[str, object], ...]:
    """E9 rows for a run, in the order produced."""
    return tuple(e9_row(state) for state in states)


def timestamps(states: Sequence[ReconciledState]) -> tuple[ISOTimestamp, ...]:
    """The run's tick grid — what C10 aligns folds on (O6: identical folds)."""
    return tuple(state.ts for state in states)
