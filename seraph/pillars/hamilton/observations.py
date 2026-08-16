"""C7 — Hamilton Engine (FR20, FR21, FR22) — building y_t and z_t.

C7 owns nothing upstream of this file. `RV_t` (Yang-Zhang), `AR_t` (top-10 PCs)
`BAS_t` (cross-sectional median Abdi-Ranaldo) and `G_t` (FR22) are all C4's,
reached through the S4 interface — never recomputed here, and `AR_t` in
particular is never derived from C6 (that coupling would break FR35).

What this module does own: the transform, the trading-day alignment of a
mixed-frequency z_t (SPEC E4 `[GAP]`), and the standardisation statistics,
which are frozen at fit time so a refit cannot leak future scale into the past.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from seraph.pillars.hamilton.config import HamiltonConfig
from seraph.shared_types import (
    ISODate,
    MacroRow,
    ObservationRow,
    Result,
    SeraphWarning,
)

__all__ = [
    "HamiltonFeatureSource",
    "ObservationPanel",
    "build_panel",
    "design_matrix",
    "panel_fingerprint",
    "standardise",
    "usable_covariates",
    "validate_observations",
]

_LOG_FLOOR = 1e-12


class HamiltonFeatureSource(Protocol):
    """The S4 slice C7 consumes. Implemented by C4; mocked in tests."""

    async def observation_vector(
        self, from_date: ISODate, to_date: ISODate
    ) -> Result[tuple[ObservationRow, ...]]: ...

    async def covariate_vector(
        self, from_date: ISODate, to_date: ISODate
    ) -> Result[tuple[MacroRow, ...]]: ...


@dataclass(frozen=True, eq=False)
class ObservationPanel:
    """Aligned (y_t, z_t) panel on the trading-day grid."""

    dates: tuple[ISODate, ...]
    y: np.ndarray  # (T, 3) transformed, NOT standardised
    z: np.ndarray  # (T, p) covariates, NOT standardised, no intercept column
    covariates: tuple[str, ...]  # names of z's columns, in order
    warnings: tuple[SeraphWarning, ...]
    # The untransformed rows behind `y`, aligned index-for-index. E8 needs the
    # raw RV/AR/BAS levels, and carrying them here means the engine never has
    # to re-fetch and risk two fetches disagreeing.
    rows: tuple[ObservationRow, ...] = ()

    def __len__(self) -> int:
        return len(self.dates)


def validate_observations(rows: tuple[ObservationRow, ...]) -> tuple[str, str] | None:
    """Reject malformed C4 output before it reaches the estimator.

    C7 is downstream of a component that does not exist yet, so every
    assumption the filter makes about y_t is checked here rather than
    discovered as a `nan` regime probability three layers later.

    Returns `(field, detail)` for a `CONTRACT_VIOLATION`, or None if clean.
    """
    seen: set[str] = set()
    for r in rows:
        if r.date in seen:
            return ("ObservationRow.date", f"duplicate observation date {r.date}")
        seen.add(r.date)
        for name, value in (
            ("rv_yang_zhang", r.rv_yang_zhang),
            ("ar_t", r.ar_t),
            ("bas_t", r.bas_t),
        ):
            if not np.isfinite(value):
                return (
                    f"ObservationRow.{name}",
                    f"non-finite {name}={value!r} on {r.date}",
                )
        if r.rv_5min is not None and not np.isfinite(r.rv_5min):
            return ("ObservationRow.rv_5min", f"non-finite rv_5min on {r.date}")
    return None


def panel_fingerprint(panel: ObservationPanel, upto: int | None = None) -> bytes:
    """Content hash of the panel (or a prefix of it).

    Used to decide whether cached parameters and filter state may be reused
    across calls. Comparing content rather than an object identity is what
    makes the cache safe: a source that silently restates history invalidates
    it instead of being trusted.
    """
    n = len(panel) if upto is None else upto
    h = hashlib.blake2b(digest_size=16)
    h.update("\x1f".join(panel.dates[:n]).encode())
    h.update(b"\x00")
    h.update(np.ascontiguousarray(panel.y[:n]).tobytes())
    h.update(b"\x00")
    h.update(np.ascontiguousarray(panel.z[:n]).tobytes())
    h.update(b"\x00")
    h.update("\x1f".join(panel.covariates).encode())
    return h.digest()


def _transform(col: np.ndarray, how: str, name: str) -> tuple[np.ndarray, str | None]:
    """Apply a y-column transform, reporting any fallback it had to make."""
    note: str | None = None
    if how == "identity":
        return col, None
    if how == "log":
        bad = int((col <= 0.0).sum())
        if bad:
            col = np.clip(col, _LOG_FLOOR, None)
            note = f"{name}: {bad} non-positive value(s) floored before log"
            return np.log(col), note
        return np.log(col), None
    if how == "logit":
        bad = int(((col <= 0.0) | (col >= 1.0)).sum())
        clipped = np.clip(col, 1e-9, 1.0 - 1e-9)
        note = (
            f"{name}: {bad} value(s) outside (0,1) clipped before logit"
            if bad
            else None
        )
        return np.log(clipped / (1.0 - clipped)), note
    raise ValueError(f"unknown transform {how!r}")


def build_panel(
    observations: tuple[ObservationRow, ...],
    macro: tuple[MacroRow, ...],
    cfg: HamiltonConfig,
    apply_drop: bool = True,
) -> ObservationPanel:
    """Align, transform and assemble (y_t, z_t).

    The observation grid drives everything: z_t is aligned onto it, never the
    other way round, so a macro release calendar can never add or remove a
    Hamilton update.

    `apply_drop=False` keeps every covariate column (NaNs and all) so the
    caller can make the drop decision per expanding-window fit rather than once
    over the whole loaded span — which is what the engine does, because a drop
    decided on the full span would be a lookahead.
    """
    rows = tuple(sorted(observations, key=lambda r: r.date))
    if not rows:
        return ObservationPanel((), np.empty((0, 3)), np.empty((0, 0)), (), ())

    dates = tuple(r.date for r in rows)
    warnings: list[SeraphWarning] = []

    # ---- y_t (FR20) ---------------------------------------------------------
    rv = np.array([r.rv_yang_zhang for r in rows], dtype=float)
    if cfg.rv_source == "rv5min_where_available":
        spliced = np.array(
            [r.rv_5min if r.rv_5min is not None else r.rv_yang_zhang for r in rows],
            dtype=float,
        )
        n_spliced = sum(1 for r in rows if r.rv_5min is not None)
        rv = spliced
        warnings.append(
            SeraphWarning(
                code="ESTIMATOR_FALLBACK",
                message=(
                    "RV_t splices 5-minute realised volatility over Yang-Zhang "
                    "where available; the estimator changes mid-sample (FR53)"
                ),
                context={"n_rv5min": n_spliced, "n_total": len(rows)},
            )
        )
    ar = np.array([r.ar_t for r in rows], dtype=float)
    bas = np.array([r.bas_t for r in rows], dtype=float)

    cols = []
    for raw, how, name in (
        (rv, cfg.rv_transform, "RV_t"),
        (ar, cfg.ar_transform, "AR_t"),
        (bas, cfg.bas_transform, "BAS_t"),
    ):
        out, note = _transform(raw, how, name)
        if note:
            warnings.append(
                SeraphWarning(
                    code="ESTIMATOR_FALLBACK",
                    message=f"y_t transform fallback — {note}",
                    context={"column": name, "transform": how},
                )
            )
        cols.append(out)
    y = np.column_stack(cols)

    # ---- z_t (FR21, FR22) ---------------------------------------------------
    by_date = {m.date: m for m in macro}
    n_missing_rows = sum(1 for d in dates if d not in by_date)
    raw_z = np.full((len(dates), len(cfg.covariates)), np.nan)
    for t, d in enumerate(dates):
        row = by_date.get(d)
        if row is None:
            continue
        for c, name in enumerate(cfg.covariates):
            val = getattr(row, name, None)
            raw_z[t, c] = np.nan if val is None else float(val)

    if n_missing_rows:
        warnings.append(
            SeraphWarning(
                code="PARTIAL_COVERAGE",
                message="z_t has no macro row on some trading days",
                context={"n_missing_dates": n_missing_rows, "n_total": len(dates)},
            )
        )

    if cfg.covariate_fill == "ffill":
        n_filled = 0
        for c in range(raw_z.shape[1]):
            col = raw_z[:, c]
            last = np.nan
            for t in range(col.shape[0]):
                if np.isnan(col[t]):
                    if not np.isnan(last):
                        col[t] = last
                        n_filled += 1
                else:
                    last = col[t]
        if n_filled:
            warnings.append(
                SeraphWarning(
                    code="PARTIAL_COVERAGE",
                    message=(
                        "z_t forward-filled onto the trading-day grid "
                        "(SPEC E4 leaves mixed-frequency alignment undefined)"
                    ),
                    context={"n_filled_cells": n_filled},
                )
            )

    keep = list(range(raw_z.shape[1]))
    if apply_drop and cfg.drop_incomplete_covariates:
        incomplete = [c for c in keep if np.isnan(raw_z[:, c]).any()]
        if incomplete:
            dropped = [cfg.covariates[c] for c in incomplete]
            keep = [c for c in keep if c not in incomplete]
            warnings.append(
                SeraphWarning(
                    code="PARTIAL_COVERAGE",
                    message=(
                        "covariate dropped from z_t for this window — still null "
                        "after alignment (SPEC OQ5 open for India VIX pre-Nov-2007)"
                    ),
                    context={"dropped": ",".join(dropped)},
                )
            )

    z = raw_z[:, keep] if keep else np.empty((len(dates), 0))
    covariates = tuple(cfg.covariates[c] for c in keep)

    return ObservationPanel(
        dates=dates,
        y=y,
        z=z,
        covariates=covariates,
        warnings=tuple(warnings),
        rows=rows,
    )


def standardise(
    a: np.ndarray, center: np.ndarray | None = None, scale: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score columns, returning the statistics so they can be frozen.

    Statistics computed on a fit window are reused verbatim when filtering
    forward past it — recomputing them per update would let tomorrow's scale
    into today's regime probability.
    """
    if a.size == 0:
        return a, np.zeros(a.shape[1]), np.ones(a.shape[1])
    if center is None:
        center = a.mean(axis=0)
    if scale is None:
        scale = a.std(axis=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
    return (a - center) / scale, center, scale


def usable_covariates(
    z: np.ndarray, names: tuple[str, ...], upto: int
) -> tuple[int, ...]:
    """Column indices of z that are complete on rows [0, upto] inclusive.

    Called per fit window so the covariate set at date t is decided from data
    up to t only. India VIX (null before Nov 2007, SPEC OQ5) is exactly the
    case this exists for: it is absent from z for early windows and enters by
    itself once the series starts.
    """
    if z.size == 0:
        return ()
    window = z[: upto + 1]
    return tuple(c for c in range(z.shape[1]) if not np.isnan(window[:, c]).any())


def design_matrix(z_std: np.ndarray) -> np.ndarray:
    """Prepend the intercept column: z~_t = [1, z_t] (FR21)."""
    return np.column_stack([np.ones(z_std.shape[0]), z_std])
