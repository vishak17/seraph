"""C8 — the two rules that decide whether an emission means anything.

**FR26's "new value".** A pillar polled between its own updates re-serves the
same estimate: same `tau`, later `ts`. Folding that in twice is double counting
and its whole effect lands on `P`, where nobody looks until FR29's interval is
already too tight.

**S5 `coverage()` (D2).** Only the pillar can say whether it is quiet or cannot
exist. SPEC §4's coverage matrix is that statement for Hawkes pre-2015.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import datetime, timedelta
from typing import Any

import pytest

from seraph.reconciliation import (
    ForwardFillReconciliationLayer,
    KalmanReconciliationLayer,
    ReconciliationConfig,
)
from seraph.shared_types import (
    ISOTimestamp,
    ObservedEmission,
    PillarCoverage,
    PillarObservation,
)

START = datetime.fromisoformat("2015-01-01T15:30:00+05:30")


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def day(n: int) -> ISOTimestamp:
    return (START + timedelta(days=n)).isoformat()


def obs(
    pillar: str, ts: ISOTimestamp, value: float, tau: ISOTimestamp
) -> ObservedEmission:
    return ObservedEmission(
        obs=PillarObservation(pillar=pillar, ts=ts, tau=tau, value=value)
    )


# -- FR26: re-delivery -------------------------------------------------------


def test_the_same_tau_served_twice_updates_the_filter_once() -> None:
    """The bug this guards: `P` shrinking on re-read rather than on evidence."""

    async def scenario() -> tuple[Any, Any]:
        layer = KalmanReconciliationLayer()
        first = await layer.update((obs("rmt", day(0), 0.5, tau=day(0)),))
        # Same estimate, polled again a day later — C7's emit() does exactly
        # this when queried after its last close.
        again = await layer.update((obs("rmt", day(1), 0.5, tau=day(0)),))
        return first, again

    first, again = _run(scenario())
    assert first.value.p_t[1][1] < 1.0
    # P may only have GROWN across the day: no new information arrived.
    assert again.value.p_t[1][1] > first.value.p_t[1][1]
    assert again.value.tau_last_update[1] == day(0)


def test_redeliveries_are_counted_not_silently_dropped() -> None:
    async def scenario() -> KalmanReconciliationLayer:
        layer = KalmanReconciliationLayer()
        for n in range(5):
            await layer.update((obs("hamilton", day(n), 0.5, tau=day(0)),))
        return layer

    layer = _run(scenario())
    assert layer.update_counts() == (0, 0, 1)
    assert layer.redelivery_counts() == (0, 0, 4)


def test_a_genuinely_new_tau_still_updates() -> None:
    async def scenario() -> tuple[Any, Any]:
        layer = KalmanReconciliationLayer()
        first = await layer.update((obs("rmt", day(0), 0.5, tau=day(0)),))
        second = await layer.update((obs("rmt", day(1), 0.9, tau=day(1)),))
        return first, second

    first, second = _run(scenario())
    assert second.value.tau_last_update[1] == day(1)
    # Moved toward the new observation, and P fell — both only happen on a
    # genuine arrival.
    assert second.value.x_hat[1] > first.value.x_hat[1]
    assert second.value.p_t[1][1] < first.value.p_t[1][1]


def test_the_r0_window_does_not_see_the_same_value_twice() -> None:
    """A re-delivered value entering the rolling variance would make a pillar
    look artificially steady, shrinking `R_0` — the same over-confidence by a
    different route."""

    async def scenario() -> KalmanReconciliationLayer:
        layer = KalmanReconciliationLayer()
        await layer.update((obs("rmt", day(0), 1.0, tau=day(0)),))
        await layer.update((obs("rmt", day(1), 3.0, tau=day(1)),))
        for n in range(2, 30):  # 28 re-deliveries of the same estimate
            await layer.update((obs("rmt", day(n), 3.0, tau=day(1)),))
        return layer

    layer = _run(scenario())
    assert layer.update_counts()[1] == 2
    assert layer._r0(1) == pytest.approx(2.0)  # var([1, 3]), not var of 30 points


def test_forward_fill_applies_the_same_rule() -> None:
    async def scenario() -> ForwardFillReconciliationLayer:
        layer = ForwardFillReconciliationLayer()
        await layer.update((obs("rmt", day(0), 0.5, tau=day(0)),))
        await layer.update((obs("rmt", day(1), 0.5, tau=day(0)),))
        return layer

    layer = _run(scenario())
    assert layer.update_counts() == (0, 1, 0)
    assert layer.redelivery_counts() == (0, 1, 0)


# -- D2: coverage ------------------------------------------------------------


def test_a_pillar_outside_its_coverage_window_is_masked_out() -> None:
    """SPEC §4: Hawkes cannot exist in 2008. The pillar says so once, via
    `coverage()`, and C8 does not need an emission on every tick to know it."""

    async def scenario() -> Any:
        layer = KalmanReconciliationLayer()
        layer.declare_coverage(
            "hawkes", PillarCoverage(from_ts=day(100), to_ts=day(500))
        )
        return await layer.update((obs("rmt", day(0), 0.3, tau=day(0)),))

    result = _run(scenario())
    assert result.value.mask[0] is False  # no Hawkes emission needed to know


def test_coverage_masks_out_even_a_previously_observed_pillar() -> None:
    """Coverage is about the period, not the pillar's history: an observation
    before the window opens does not make the window open earlier."""

    async def scenario() -> tuple[Any, Any]:
        layer = KalmanReconciliationLayer()
        inside = await layer.update((obs("hawkes", day(0), 1.0, tau=day(0)),))
        layer.declare_coverage(
            "hawkes", PillarCoverage(from_ts=day(10), to_ts=day(500))
        )
        after = await layer.predict(day(1))
        return inside, after

    inside, after = _run(scenario())
    assert inside.value.mask[0] is True
    assert after.value.mask[0] is False


def test_an_emission_outside_declared_coverage_is_warned_and_still_applied() -> None:
    """Discarding real data on the strength of metadata would be worse. The
    contradiction is surfaced for C10 instead."""

    async def scenario() -> Any:
        layer = KalmanReconciliationLayer()
        layer.declare_coverage(
            "hawkes", PillarCoverage(from_ts=day(10), to_ts=day(500))
        )
        return await layer.update((obs("hawkes", day(0), 1.0, tau=day(0)),))

    result = _run(scenario())
    assert result.status == "ok"
    assert any(w.code == "PARTIAL_COVERAGE" for w in result.warnings)
    assert result.value.tau_last_update[0] == day(0)  # applied
    assert result.value.mask[0] is False  # but not scoreable


def test_inside_coverage_nothing_changes() -> None:
    async def scenario() -> Any:
        layer = KalmanReconciliationLayer(ReconciliationConfig())
        layer.declare_coverage("rmt", PillarCoverage(from_ts=day(0), to_ts=day(500)))
        return await layer.update((obs("rmt", day(5), 0.3, tau=day(5)),))

    result = _run(scenario())
    assert result.value.mask[1] is True
    assert not any(w.code == "PARTIAL_COVERAGE" for w in result.warnings)
