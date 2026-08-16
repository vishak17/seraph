"""C8 — the S5 driver (`C8 -> C5/C6/C7`, ARCHITECTURE §3).

Two kinds of test here. Stub engines pin the runner's *policies* — coverage
priming, engine failure becoming transient absence, a missing pillar simply
never being observed. Then the real `HamiltonEngine` (C7) is driven end to end,
because a protocol both sides satisfy on paper is not the same as a seam that
actually works: C7 emits `tau` at the last NSE close, re-serves it when polled
again, and reports structural absence before `min_history_days` — all three of
which C8 has to handle correctly.

C7 is imported read-only. Nothing in this file changes it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import datetime, timedelta
from typing import Any, Literal

import pytest

from seraph.pillars.hamilton import HamiltonConfig, HamiltonEngine
from seraph.reconciliation import (
    ForwardFillReconciliationLayer,
    KalmanReconciliationLayer,
    ReconciliationRunner,
)
from seraph.shared_types import (
    ISOTimestamp,
    ObservedEmission,
    PillarCoverage,
    PillarEmission,
    PillarId,
    PillarObservation,
    Result,
    SourceUnavailable,
    UnavailableEmission,
    err,
    ok,
)
from tests.pillars.synthetic import MockFeatureSource, make_panel
from tests.reconciliation.helpers import unwrap

START = datetime.fromisoformat("2015-01-01T15:30:00+05:30")


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def day(n: int) -> ISOTimestamp:
    return (START + timedelta(days=n)).isoformat()


class StubEngine:
    """Minimal S5 `PillarEngine`. Not a component — a test double."""

    def __init__(
        self,
        pillar: PillarId,
        *,
        value: float = 1.0,
        coverage: PillarCoverage | None = None,
        fail_emit: bool = False,
        raise_emit: bool = False,
        fail_coverage: bool = False,
        absent: Literal["structural", "transient"] | None = None,
    ) -> None:
        self.pillar: PillarId = pillar
        self.value = value
        self._coverage = coverage
        self.fail_emit = fail_emit
        self.raise_emit = raise_emit
        self.fail_coverage = fail_coverage
        self.absent = absent
        self.emit_calls = 0

    async def emit(self, ts: ISOTimestamp) -> Result[PillarEmission]:
        self.emit_calls += 1
        if self.raise_emit:
            raise RuntimeError("engine blew up")
        if self.fail_emit:
            return err(SourceUnavailable(source=self.pillar, http_status=503))
        if self.absent is not None:
            return ok(
                UnavailableEmission(
                    pillar=self.pillar,
                    ts=ts,
                    absence=self.absent,
                    reason="no_data_coverage",
                )
            )
        return ok(
            ObservedEmission(
                obs=PillarObservation(
                    pillar=self.pillar, ts=ts, tau=ts, value=self.value
                )
            )
        )

    async def emit_range(
        self, from_ts: ISOTimestamp, to_ts: ISOTimestamp
    ) -> Result[tuple[PillarEmission, ...]]:
        return ok(())

    async def coverage(self) -> Result[PillarCoverage]:
        if self.fail_coverage or self._coverage is None:
            return err(SourceUnavailable(source=self.pillar, http_status=503))
        return ok(self._coverage)


# -- policies ----------------------------------------------------------------


def test_runner_polls_every_engine_and_produces_one_state_per_tick() -> None:
    engines = [StubEngine("rmt", value=0.3), StubEngine("hamilton", value=0.7)]
    runner = ReconciliationRunner(engines, KalmanReconciliationLayer())

    report = unwrap(_run(runner.run([day(n) for n in range(5)])))
    assert len(report.states) == 5
    assert all(e.emit_calls == 5 for e in engines)
    assert report.updates == (0, 5, 5)


def test_coverage_priming_reaches_the_mask() -> None:
    """SPEC §4 in one call: Hawkes declares 2015+, so 2014 is masked out."""
    hawkes = StubEngine(
        "hawkes", coverage=PillarCoverage(from_ts=day(10), to_ts=day(400))
    )
    runner = ReconciliationRunner([hawkes], KalmanReconciliationLayer())

    async def scenario() -> Any:
        await runner.prime_coverage()
        return await runner.step(day(0))

    result = _run(scenario())
    assert result.status == "ok"
    assert result.value.mask[0] is False


def test_an_engine_that_cannot_report_coverage_is_not_fatal() -> None:
    runner = ReconciliationRunner(
        [StubEngine("rmt", fail_coverage=True)], KalmanReconciliationLayer()
    )
    warnings = _run(runner.prime_coverage())
    assert any(w.code == "ESTIMATOR_FALLBACK" for w in warnings)
    assert _run(runner.step(day(0))).status == "ok"


@pytest.mark.parametrize("mode", ["error", "raise"])
def test_an_engine_failure_becomes_a_transient_absence(mode: str) -> None:
    """Policy 1: one broken pillar must not take the other two off the air, and
    a failure is never `structural` — it says nothing about whether the pillar
    can exist.

    The pillar is seeded with one good emission first, because "transient
    absence keeps it in the mask" is a statement about a pillar that has
    something to keep; one that has never emitted is masked out by D4's
    ceiling regardless, and rightly so.
    """
    hawkes = StubEngine("hawkes", value=1.0)
    runner = ReconciliationRunner(
        [hawkes, StubEngine("rmt", value=0.3)], KalmanReconciliationLayer()
    )

    async def scenario() -> tuple[Any, Any]:
        seeded = await runner.step(day(0))
        hawkes.fail_emit = mode == "error"
        hawkes.raise_emit = mode == "raise"
        # One Hawkes bar later: too soon for D4's ceiling to have been reached.
        return seeded, await runner.step("2015-01-01T15:35:00+05:30")

    seeded, result = _run(scenario())
    assert seeded.value.mask[0] is True
    assert result.status == "ok", result
    assert result.value.mask[0] is True  # transient keeps it in the mask
    assert result.value.mask[1] is True  # RMT unaffected
    assert any(w.code == "ESTIMATOR_FALLBACK" for w in result.warnings)


def test_repeated_engine_failures_are_counted() -> None:
    runner = ReconciliationRunner(
        [StubEngine("hawkes", fail_emit=True)], KalmanReconciliationLayer()
    )
    report = unwrap(_run(runner.run([day(n) for n in range(4)])))
    assert report.engine_errors == (4, 0, 0)


def test_a_pillar_with_no_engine_is_simply_never_observed() -> None:
    """This is how FR35's 7 ablation subsets are expressed: leave the engine
    out, and D2 masks the pillar rather than the caller zeroing it."""
    runner = ReconciliationRunner(
        [StubEngine("rmt", value=0.3)], KalmanReconciliationLayer()
    )
    report = unwrap(_run(runner.run([day(n) for n in range(3)])))
    last = report.states[-1]
    assert last.mask == (False, True, False)
    assert last.tau_last_update[0] is None and last.tau_last_update[2] is None


def test_two_engines_for_one_pillar_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="two engines"):
        ReconciliationRunner(
            [StubEngine("rmt"), StubEngine("rmt")], KalmanReconciliationLayer()
        )


def test_no_engines_degrades_to_a_pure_predict() -> None:
    runner = ReconciliationRunner([], KalmanReconciliationLayer())
    report = unwrap(_run(runner.run([day(n) for n in range(3)])))
    assert report.states[-1].mask == (False, False, False)


def test_the_runner_drives_the_forward_fill_arm_identically() -> None:
    """FR36 runs both arms over the same stream; the runner must not be part of
    what differs between them."""
    kalman = ReconciliationRunner(
        [StubEngine("rmt", value=0.3)], KalmanReconciliationLayer()
    )
    baseline = ReconciliationRunner(
        [StubEngine("rmt", value=0.3)], ForwardFillReconciliationLayer()
    )
    grid = [day(n) for n in range(4)]
    a = unwrap(_run(kalman.run(grid)))
    b = unwrap(_run(baseline.run(grid)))

    assert [s.ts for s in a.states] == [s.ts for s in b.states]
    assert [s.mask for s in a.states] == [s.mask for s in b.states]
    assert a.states[-1].mode == "kalman"
    assert b.states[-1].mode == "forward_fill"


# -- the real C7 engine ------------------------------------------------------


def _hamilton_engine() -> HamiltonEngine:
    cfg = HamiltonConfig(
        corpus_start="2015-01-01",
        min_history_days=250,
        refit_every_days=120,
        em_max_iter=15,
        logit_max_iter=25,
    )
    return HamiltonEngine(MockFeatureSource(make_panel(n_days=320, seed=11)), cfg)


@pytest.mark.slow
def test_c7_hamilton_drives_the_filter_end_to_end() -> None:
    """The S5 seam against a real pillar, not a stub.

    Asserts the three C7 behaviours C8 has to get right: structural absence
    before `min_history_days` keeps Hamilton out of the mask; the first real
    emission puts it in; and polling on a date C7 has no update for re-serves
    the previous close, which must not count as new information.
    """
    engine = _hamilton_engine()
    layer = KalmanReconciliationLayer()
    runner = ReconciliationRunner([engine], layer)

    async def scenario() -> Any:
        await runner.prime_coverage()
        # Dates chosen either side of C7's 250-day history floor.
        grid = [f"2015-{month:02d}-15T15:30:00+05:30" for month in range(1, 13)] + [
            f"2016-{month:02d}-15T15:30:00+05:30" for month in range(1, 4)
        ]
        return await runner.run(grid)

    report = unwrap(_run(scenario()))
    states = report.states

    # Early on Hamilton cannot exist -> masked out, tau still null.
    assert states[0].mask[2] is False
    assert states[0].tau_last_update[2] is None

    # By the end it is observed, in the mask, and its tau is a real NSE close.
    final = states[-1]
    assert final.mask[2] is True
    assert final.tau_last_update[2] is not None
    assert final.tau_last_update[2].endswith("T15:30:00+05:30")
    # C7 emits daily; the monthly grid means most polls re-serve the last close.
    assert report.updates[2] >= 1
    assert report.engine_errors == (0, 0, 0)


@pytest.mark.slow
def test_polling_c7_twice_on_one_day_does_not_shrink_p_twice() -> None:
    """The FR26 re-delivery rule, against the engine that actually does this."""
    engine = _hamilton_engine()
    layer = KalmanReconciliationLayer()
    runner = ReconciliationRunner([engine], layer)

    async def scenario() -> tuple[Any, Any]:
        ts = "2016-03-15T15:30:00+05:30"
        first = await runner.step(ts)
        second = await runner.step("2016-03-15T15:35:00+05:30")
        return first, second

    first, second = _run(scenario())
    assert first.status == "ok" and second.status == "ok"
    assert layer.update_counts()[2] == 1
    assert layer.redelivery_counts()[2] == 1
    # Second poll only aged the state; it cannot have sharpened it.
    assert second.value.p_t[2][2] >= first.value.p_t[2][2]
