"""Synthetic C4 feature source for C7 tests.

Deliberately NOT `fixtures/mock_generator.py` — that is T0, owned by D, and it
emits PillarEmission/ReconciledState/CsrsPoint streams. This one goes the other
way: it fakes C7's *inputs* (S4's `ObservationRow` / `MacroRow`) so the Hamilton
engine can be exercised before C4 exists.

The data-generating process is a genuine three-state TVTP chain, so parameter
recovery is a meaningful assertion rather than a smoke test.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from seraph.shared_types import (
    MacroRow,
    ObservationRow,
    ObservedEmission,
    PillarEmission,
    PillarObservation,
    Result,
    ok,
)
from seraph.shared_types.common import Err, Ok, SourceUnavailable, err

# Regime-conditional truth, in the raw (untransformed) space the engine reads.
# Index 0 tranquil, 1 stressed, 2 crisis — ordered by RV, matching
# em.identify_regimes.
TRUE_LOG_RV_MEAN = (-3.0, -2.2, -1.4)
TRUE_LOG_RV_SD = (0.22, 0.26, 0.30)
TRUE_AR_MEAN = (0.35, 0.50, 0.68)
TRUE_AR_SD = (0.04, 0.05, 0.05)
TRUE_LOG_BAS_MEAN = (-6.0, -5.6, -5.1)
TRUE_LOG_BAS_SD = (0.20, 0.22, 0.25)

# Base transition matrix, tilted by the VIX covariate (the TVTP part).
BASE_TRANSITION = np.array(
    [
        [0.96, 0.035, 0.005],
        [0.06, 0.90, 0.04],
        [0.01, 0.14, 0.85],
    ]
)


@dataclass(frozen=True, eq=False)
class SyntheticPanel:
    observations: tuple[ObservationRow, ...]
    macro: tuple[MacroRow, ...]
    states: np.ndarray  # (T,) true regime index


def _trading_days(n: int, start: date) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def make_panel(
    n_days: int = 600,
    seed: int = 11,
    start: date = date(2015, 1, 1),
    vix_missing_first: int = 0,
) -> SyntheticPanel:
    """Simulate a TVTP three-state panel.

    Args:
        n_days: number of trading days.
        vix_missing_first: leave `india_vix` null on this many leading days —
            the SPEC OQ5 shape (India VIX starts Nov 2007), used to exercise
            the covariate-drop path.
    """
    rng = np.random.default_rng(seed)
    days = _trading_days(n_days, start)

    # z_t: a persistent VIX-like series plus slow macro levels.
    vix = 15.0 + np.cumsum(rng.normal(0.0, 0.35, n_days))
    vix = np.clip(vix, 8.0, 60.0)
    repo = 6.0 + 0.5 * np.sin(np.arange(n_days) / 180.0)
    credit = 10.0 + 2.0 * np.cos(np.arange(n_days) / 220.0)
    inr = 70.0 + np.cumsum(rng.normal(0.0, 0.05, n_days))
    brent = 60.0 + np.cumsum(rng.normal(0.0, 0.4, n_days))
    g_t = np.abs(rng.normal(0.0, 0.02, n_days)).cumsum() * np.exp(
        -np.arange(n_days) / 400.0
    )

    # TVTP: high VIX pushes probability mass toward the stressed/crisis columns.
    tilt = (vix - vix.mean()) / max(vix.std(), 1e-9)
    states = np.empty(n_days, dtype=int)
    state = 0
    for t in range(n_days):
        row = BASE_TRANSITION[state].copy()
        row[1] *= float(np.exp(0.5 * tilt[t]))
        row[2] *= float(np.exp(0.8 * tilt[t]))
        row = row / row.sum()
        state = int(rng.choice(3, p=row))
        states[t] = state

    rv = np.exp(
        np.array([TRUE_LOG_RV_MEAN[s] for s in states])
        + np.array([TRUE_LOG_RV_SD[s] for s in states]) * rng.standard_normal(n_days)
    )
    ar = np.clip(
        np.array([TRUE_AR_MEAN[s] for s in states])
        + np.array([TRUE_AR_SD[s] for s in states]) * rng.standard_normal(n_days),
        0.01,
        0.99,
    )
    bas = np.exp(
        np.array([TRUE_LOG_BAS_MEAN[s] for s in states])
        + np.array([TRUE_LOG_BAS_SD[s] for s in states]) * rng.standard_normal(n_days)
    )

    observations = tuple(
        ObservationRow(
            date=d.isoformat(),
            rv_yang_zhang=float(rv[t]),
            rv_5min=None,
            ar_t=float(ar[t]),
            bas_t=float(bas[t]),
        )
        for t, d in enumerate(days)
    )
    macro = tuple(
        MacroRow(
            date=d.isoformat(),
            rbi_repo_rate=float(repo[t]),
            bank_credit_growth_yoy=float(credit[t]),
            india_vix=None if t < vix_missing_first else float(vix[t]),
            vix_available=t >= vix_missing_first,
            inr_twi=float(inr[t]),
            brent_price=float(brent[t]),
            g_t=float(g_t[t]),
            g_t_sector_weighted=float(g_t[t]) * 1.5,
        )
        for t, d in enumerate(days)
    )
    return SyntheticPanel(observations=observations, macro=macro, states=states)


class MockFeatureSource:
    """Implements the `HamiltonFeatureSource` protocol over a SyntheticPanel."""

    def __init__(self, panel: SyntheticPanel, fail: bool = False) -> None:
        self.panel = panel
        self.fail = fail
        self.calls: list[tuple[str, str, str]] = []

    async def observation_vector(
        self, from_date: str, to_date: str
    ) -> Result[tuple[ObservationRow, ...]]:
        self.calls.append(("observation_vector", from_date, to_date))
        if self.fail:
            return err(SourceUnavailable(source="c4-stub"))
        return ok(
            tuple(r for r in self.panel.observations if from_date <= r.date <= to_date)
        )

    async def covariate_vector(
        self, from_date: str, to_date: str
    ) -> Result[tuple[MacroRow, ...]]:
        self.calls.append(("covariate_vector", from_date, to_date))
        if self.fail:
            return err(SourceUnavailable(source="c4-stub"))
        return ok(tuple(r for r in self.panel.macro if from_date <= r.date <= to_date))


def as_ok[T](res: Result[T]) -> Ok[T]:
    """Narrow a Result to Ok, failing the test if it is an Err.

    Consumers of S5 have to narrow the union before touching `.value`; these
    helpers do it once so the assertions below stay readable and the test suite
    type-checks under the same mypy settings as the package.
    """
    assert isinstance(res, Ok), f"expected Ok, got {res!r}"
    return res


def unwrap[T](res: Result[T]) -> T:
    return as_ok(res).value


def observed(res: Result[PillarEmission]) -> PillarObservation:
    """Narrow to an `observed` emission and return its observation."""
    emission = unwrap(res)
    assert isinstance(emission, ObservedEmission), f"expected observed: {emission!r}"
    return emission.obs


__all__ = [
    "Err",
    "MockFeatureSource",
    "SyntheticPanel",
    "as_ok",
    "make_panel",
    "observed",
    "unwrap",
]
