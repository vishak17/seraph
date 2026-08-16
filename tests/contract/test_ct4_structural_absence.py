"""CT-4 · Pillars <-> C8 — structural absence (docs/ARCHITECTURE.md §5 ②).

**This test is the whole of Objective 6.** It proves the one claim that makes
the reconciliation layer worth building: a pillar that does not exist is
handled as growing-but-bounded *uncertainty*, never as a zero and never as an
error.

Setup, per ARCHITECTURE: 200 consecutive ticks on which `hawkes` returns
`{kind: "unavailable", absence: "structural"}` while RMT and Hamilton observe
normally, followed by Hawkes' first genuine emission.

Asserted:
  * `x_hat[0]` does not collapse toward 0
  * `P[0][0]` grows monotonically, then saturates (D4)
  * `mask[0] is False`, `tau_last_update[0] is None`
  * on the first genuine Hawkes emission `P[0][0]` drops strictly and
    `mask[0]` flips true
Negatives:
  * `tau > ts` -> `CONTRACT_VIOLATION`
  * `absence: "transient"` keeps `mask[0] is True`

Plus ARCHITECTURE §7 D4's mandated sweep of the three candidate `R^(p)(.)`
forms, the FR27 forward-fill baseline on the identical stream, and SPEC §4's
literal case — Hawkes declaring a coverage window that opens in 2015 and the
mask flipping on that date rather than on a message.

The emissions are built inline rather than drawn from T0 (`fixtures/
mock_generator.py`, owner D, not yet written) — the shapes are S5's, so this
test moves onto T0 unchanged the moment it exists.
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
    ReconciliationRunner,
)
from seraph.reconciliation.config import NoiseForm
from seraph.reconciliation.noise_model import ceiling, observation_noise
from seraph.shared_types import (
    ContractViolation,
    ISOTimestamp,
    ObservedEmission,
    PillarCoverage,
    PillarEmission,
    PillarId,
    PillarObservation,
    ReconciledState,
    UnavailableEmission,
    ok,
)

pytestmark = pytest.mark.contract

TICKS = 200
FIRST_CLOSE = datetime.fromisoformat("2014-01-02T15:30:00+05:30")


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


FIRST_BAR = datetime.fromisoformat("2014-01-02T09:15:00+05:30")


def close(n: int) -> ISOTimestamp:
    """NSE close, `n` calendar days after the first tick. IST-explicit."""
    return (FIRST_CLOSE + timedelta(days=n)).isoformat()


def bar(n: int) -> ISOTimestamp:
    """The `n`-th 5-minute bar — Hawkes' native cadence (FR10, D4's `h_p`)."""
    return (FIRST_BAR + timedelta(minutes=5 * n)).isoformat()


def observed(
    pillar: str, ts: ISOTimestamp, value: float, tau: ISOTimestamp | None = None
) -> ObservedEmission:
    return ObservedEmission(
        obs=PillarObservation(pillar=pillar, ts=ts, tau=tau or ts, value=value)
    )


def absent(pillar: str, ts: ISOTimestamp, absence: str) -> UnavailableEmission:
    return UnavailableEmission(
        pillar=pillar,
        ts=ts,
        absence=absence,
        reason="no_data_coverage" if absence == "structural" else "estimation_failed",
    )


def tick(n: int, hawkes_absence: str = "structural") -> tuple[PillarEmission, ...]:
    """One tick: Hawkes absent, RMT and Hamilton observing normally."""
    ts = close(n)
    return (
        absent("hawkes", ts, hawkes_absence),
        observed("rmt", ts, 0.30 + 0.001 * n),
        observed("hamilton", ts, 0.70 - 0.001 * n),
    )


def cfg(**overrides: Any) -> ReconciliationConfig:
    """Hawkes drift raised so the D4 ceiling is reached inside 200 ticks.

    `q_proc` is a SPEC-undefined constant (OQ9) awaiting D4's MLE fit, so
    choosing it per scenario is legitimate; the *shape* of the trajectory this
    test asserts does not depend on its value, only on how many ticks it takes.
    `prior_mean[0] = 0.45` is a plausible MTS_t level and is deliberately
    non-zero — the point of the first assertion is that C8 leaves it alone
    rather than imputing 0.
    """
    return ReconciliationConfig(
        q_proc_per_day=(1.0e-2, 1.0e-3, 1.0e-3),
        prior_mean=(0.45, 0.0, 0.0),
        **overrides,
    )


async def _absence_run(
    layer: KalmanReconciliationLayer | ForwardFillReconciliationLayer,
    hawkes_absence: str = "structural",
) -> list[ReconciledState]:
    states: list[ReconciledState] = []
    for n in range(TICKS):
        result = await layer.update(tick(n, hawkes_absence))
        assert result.status == "ok", result
        states.append(result.value)
    return states


def _trace(state: ReconciledState) -> float:
    return state.p_t[0][0] + state.p_t[1][1] + state.p_t[2][2]


# ---------------------------------------------------------------------------
# The positive case
# ---------------------------------------------------------------------------


def test_structural_absence_is_uncertainty_not_zero() -> None:
    states = _run(_absence_run(KalmanReconciliationLayer(cfg())))

    # x_hat[0] does not collapse toward 0 — it is the prior, untouched.
    assert all(s.x_hat[0] == pytest.approx(0.45) for s in states)
    # mask[0] false and tau_last_update[0] null, on all 200 ticks.
    assert all(s.mask[0] is False for s in states)
    assert all(s.tau_last_update[0] is None for s in states)
    # The other two pillars stay in the mask throughout.
    assert all(s.mask[1] and s.mask[2] for s in states)


def test_hawkes_variance_grows_monotonically_then_saturates() -> None:
    states = _run(_absence_run(KalmanReconciliationLayer(cfg())))
    p00 = [s.p_t[0][0] for s in states]

    assert all(b >= a - 1e-15 for a, b in zip(p00, p00[1:], strict=False))
    assert p00[50] > p00[10] > p00[0]

    # Saturated by the end at D4's bound — not merely "large".
    bound = ceiling(cfg().r0_prior[0], cfg().r_max_ratio, cfg().noise_form)
    assert p00[-1] == pytest.approx(bound, rel=1e-12)
    assert p00[-1] == pytest.approx(p00[-20], rel=1e-12)
    assert states[-1].noise_saturated[0] is True


def test_first_genuine_hawkes_emission_drops_p_and_flips_the_mask() -> None:
    async def scenario() -> tuple[ReconciledState, ReconciledState]:
        layer = KalmanReconciliationLayer(cfg())
        saturated = (await _absence_run(layer))[-1]
        ts = close(TICKS)
        result = await layer.update(
            (
                observed("hawkes", ts, 1.20),
                observed("rmt", ts, 0.50),
                observed("hamilton", ts, 0.50),
            )
        )
        assert result.status == "ok", result
        return saturated, result.value

    saturated, arrived = _run(scenario())

    assert arrived.p_t[0][0] < saturated.p_t[0][0]
    assert arrived.mask[0] is True
    assert arrived.noise_saturated[0] is False
    assert arrived.tau_last_update[0] == close(TICKS)
    # The estimate moves essentially all the way onto the observation: P sat at
    # the ceiling, R^(p)(0) is R_0, so the gain is ~1. This is the "R -> R_max,
    # then a real arrival" mechanism O6 rests on.
    assert arrived.x_hat[0] == pytest.approx(1.20, rel=1e-2)


def test_trace_p_is_non_decreasing_between_updates() -> None:
    """SPEC O6's acceptance criterion in runtime form: non-decreasing between
    arrivals, strictly decreasing at one."""

    async def scenario() -> tuple[list[float], float]:
        layer = KalmanReconciliationLayer(cfg())
        warm = await layer.update(tick(0))
        assert warm.status == "ok"

        traces = [_trace(warm.value)]
        for k in range(1, 6):  # predict-only: no information may arrive
            result = await layer.predict(close(k))
            assert result.status == "ok", result
            traces.append(_trace(result.value))

        arrival = await layer.update(tick(6))
        assert arrival.status == "ok", arrival
        return traces, _trace(arrival.value)

    traces, after_arrival = _run(scenario())
    assert all(b >= a - 1e-15 for a, b in zip(traces, traces[1:], strict=False))
    assert traces[-1] > traces[0]
    assert after_arrival < traces[-1]


def test_state_at_serves_a_state_the_recursion_produced() -> None:
    async def scenario() -> tuple[Any, Any]:
        layer = KalmanReconciliationLayer(cfg())
        states = await _absence_run(layer)
        return await layer.state_at(states[5].ts), await layer.state_at(close(10_000))

    served, missing = _run(scenario())
    assert served.status == "ok"
    assert served.value.ts == close(5)
    # A state that was never produced is an error, not a rewound guess.
    assert missing.status == "error"
    assert missing.error.kind == "CONTRACT_VIOLATION"


# ---------------------------------------------------------------------------
# The negatives
# ---------------------------------------------------------------------------


def test_tau_after_ts_is_a_contract_violation() -> None:
    layer = KalmanReconciliationLayer(cfg())
    result = _run(
        layer.update((observed("rmt", close(0), 0.3, tau=close(1)),))  # computed later
    )
    assert result.status == "error"
    assert isinstance(result.error, ContractViolation)
    assert result.error.field == "PillarObservation.tau"


def test_transient_absence_keeps_the_pillar_in_the_mask() -> None:
    """D2's whole distinction: transient absence inflates noise, it does not
    remove the pillar.

    Run at Hawkes' own 5-minute cadence, because that is what D4's `h_p` is
    measured in: one missed bar is `Delta = h_p`, which inflates `R` to ~63% of
    the ceiling — stale, still informative, still masked in.
    """

    async def scenario() -> tuple[Any, Any]:
        layer = KalmanReconciliationLayer(cfg())
        seeded = await layer.update((observed("hawkes", bar(0), 0.9),))
        assert seeded.status == "ok", seeded
        return seeded, await layer.update((absent("hawkes", bar(1), "transient"),))

    seeded, result = _run(scenario())
    assert seeded.value.mask[0] is True
    assert result.status == "ok", result
    assert result.value.mask[0] is True
    assert result.value.noise_saturated[0] is False
    assert result.value.tau_last_update[0] == bar(0)  # still the last real one


def test_a_never_ending_transient_absence_eventually_saturates_out() -> None:
    """The transient/structural line is about *why*, not about forever.

    A transient gap that runs long enough hits the D4 ceiling and D2 drops the
    pillar anyway, with NOISE_SATURATED saying so. At the 0.95 threshold that
    happens at `Delta ~ 3 * h_p` — three missed bars for Hawkes. Aggressive,
    and deliberately so: a Hawkes score is a bar-frequency object and a
    15-minute-old one has nothing to say about the current bar.
    """

    async def scenario() -> list[Any]:
        layer = KalmanReconciliationLayer(cfg())
        await layer.update((observed("hawkes", bar(0), 0.9),))
        results = []
        for n in range(1, 6):
            result = await layer.update((absent("hawkes", bar(n), "transient"),))
            assert result.status == "ok", result
            results.append(result)
        return results

    results = _run(scenario())
    assert results[0].value.mask[0] is True  # one bar stale: still in
    last = results[-1]
    assert last.value.noise_saturated[0] is True
    assert last.value.mask[0] is False
    assert any(w.code == "NOISE_SATURATED" for w in last.warnings)


def test_emissions_may_not_run_backwards() -> None:
    async def scenario() -> Any:
        layer = KalmanReconciliationLayer(cfg())
        assert (await layer.update(tick(5))).status == "ok"
        return await layer.update(tick(4))

    result = _run(scenario())
    assert result.status == "error"
    assert isinstance(result.error, ContractViolation)
    assert result.error.field == "PillarEmission.ts"


# ---------------------------------------------------------------------------
# ARCHITECTURE §7 D4: "still sweep three forms in CT-4"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["saturating_exponential", "linear", "power"])
def test_every_candidate_noise_form_is_monotone_in_staleness(form: NoiseForm) -> None:
    """Requirements 1 and 2 of D4 — monotone in `Delta` (FR26), `R(0) == R_0`."""
    values = [
        observation_noise(d, r0=0.01, h_seconds=86_400.0, r_max_ratio=100.0, form=form)
        for d in (0.0, 3_600.0, 86_400.0, 5 * 86_400.0, 100 * 86_400.0)
    ]
    assert all(b >= a for a, b in zip(values, values[1:], strict=False))
    assert values[0] == pytest.approx(0.01)


def test_only_the_saturating_form_is_bounded() -> None:
    """Requirement 3 — and the argument that decided D4.

    An unbounded `R` gives nothing to bound `P` with, so a structurally absent
    pillar's variance diverges and FR29's confidence interval with it. Stated
    here as a number rather than a claim.
    """
    far = 10_000 * 86_400.0  # ~27 years, i.e. longer than the whole corpus
    args = {"r0": 0.01, "h_seconds": 86_400.0, "r_max_ratio": 100.0}

    assert observation_noise(far, form="saturating_exponential", **args) == (
        pytest.approx(1.0)
    )
    assert ceiling(0.01, 100.0, "saturating_exponential") == pytest.approx(1.0)

    # Both unbounded forms blow straight past R_max = 1.0 and keep going.
    assert observation_noise(far, form="linear", **args) == pytest.approx(100.01)
    assert observation_noise(far, form="power", **args) > 1e6
    assert ceiling(0.01, 100.0, "linear") == float("inf")
    assert ceiling(0.01, 100.0, "power") == float("inf")


@pytest.mark.parametrize("form", ["linear", "power"])
def test_unbounded_forms_leave_p_unbounded(form: NoiseForm) -> None:
    """The sweep's system-level consequence: with no ceiling, 200 ticks of
    structural absence end with a variance that is still climbing."""
    states = _run(_absence_run(KalmanReconciliationLayer(cfg(noise_form=form))))
    p00 = [s.p_t[0][0] for s in states]
    assert p00[-1] > p00[-2] > p00[-20]  # never settles
    # Already twice the bounded form's ceiling at 200 ticks, and still linear
    # in elapsed time: over the real 2005-2015 Hawkes gap it is ~2,500x.
    assert p00[-1] > 2 * ceiling(0.01, 100.0, "saturating_exponential")
    # ...and D2 never masks it out, because nothing ever "saturates".
    assert states[-1].noise_saturated[0] is False


# ---------------------------------------------------------------------------
# FR27 — the forward-fill baseline, on the same stream
# ---------------------------------------------------------------------------


def test_forward_fill_baseline_has_no_answer_for_a_missing_pillar() -> None:
    """Why O6 exists, as a test rather than a paragraph.

    Forward-fill carries the last value at a flat variance: after 200 ticks of
    absence it is exactly as confident as it was on tick 1. The D2 mask is the
    only thing standing between that and the CSRS, which is why both modes
    share it.
    """
    states = _run(_absence_run(ForwardFillReconciliationLayer(cfg())))
    first, last = states[0], states[-1]

    assert last.mode == "forward_fill"
    assert last.p_t[0][0] == pytest.approx(first.p_t[0][0])
    assert last.noise_saturated == (False, False, False)
    assert last.mask[0] is False  # structural absence still masks out
    assert last.x_hat[1] == pytest.approx(0.30 + 0.001 * (TICKS - 1))


# ---------------------------------------------------------------------------
# SPEC §4 — the coverage matrix, driven through the S5 runner
# ---------------------------------------------------------------------------


class _SilentUntil:
    """A pillar that cannot exist before `opens` and observes after it.

    Stands in for C5, which does not exist yet. What matters is the S5
    behaviour it reproduces: `coverage()` declares the window (SPEC §4's
    "~2015/16 onwards"), and before that window the engine has nothing at all
    to say — not a value, not even an `unavailable` message on every tick.
    """

    pillar: PillarId = "hawkes"

    def __init__(self, opens: ISOTimestamp, closes: ISOTimestamp) -> None:
        self.opens = opens
        self.closes = closes

    async def emit(self, ts: ISOTimestamp) -> Any:
        if ts < self.opens:
            return ok(
                UnavailableEmission(
                    pillar="hawkes",
                    ts=ts,
                    absence="structural",
                    reason="no_data_coverage",
                )
            )
        return ok(observed("hawkes", ts, 1.10))

    async def emit_range(self, from_ts: ISOTimestamp, to_ts: ISOTimestamp) -> Any:
        return ok(())

    async def coverage(self) -> Any:
        return ok(PillarCoverage(from_ts=self.opens, to_ts=self.closes))


def test_the_mask_flips_on_the_coverage_boundary_not_on_a_message() -> None:
    """SPEC §4's coverage matrix, end to end through `ReconciliationRunner`.

    The 2008 and 2013 epochs are scored with `mask[0] == False` because Hawkes
    *declared* it cannot exist there — the D2 exclusion does not depend on the
    pillar remembering to say so on every one of ~2,500 ticks. This is the
    difference between Table B being a defensible ablation row and being an
    artefact of message plumbing.
    """
    opens = "2015-01-01T15:30:00+05:30"
    engine = _SilentUntil(opens, "2026-12-31T15:30:00+05:30")
    layer = KalmanReconciliationLayer(cfg())
    runner = ReconciliationRunner([engine], layer)

    grid = [
        "2014-12-30T15:30:00+05:30",
        "2014-12-31T15:30:00+05:30",
        opens,
        "2015-01-02T15:30:00+05:30",
    ]

    async def scenario() -> Any:
        await runner.prime_coverage()
        return await runner.run(grid)

    report = _run(scenario())
    assert report.status == "ok", report
    states = report.value.states

    before, after = states[:2], states[2:]
    assert all(s.mask[0] is False for s in before)
    assert all(s.tau_last_update[0] is None for s in before)
    assert all(s.x_hat[0] == pytest.approx(0.45) for s in before)  # prior, not 0

    assert all(s.mask[0] is True for s in after)
    assert after[0].p_t[0][0] < before[-1].p_t[0][0]  # first arrival sharpens it
    assert report.value.updates[0] == 2


def test_a_pillar_inside_coverage_that_goes_quiet_is_not_structural() -> None:
    """The converse, and the reason coverage and absence are separate ideas: a
    silent pillar *inside* its window is transiently absent, so D4 ages it out
    rather than D2 excluding it outright."""
    engine = _SilentUntil("2014-01-01T15:30:00+05:30", "2026-12-31T15:30:00+05:30")
    layer = KalmanReconciliationLayer(cfg())
    runner = ReconciliationRunner([engine], layer)

    async def scenario() -> Any:
        await runner.prime_coverage()
        return await runner.step(bar(0))

    result = _run(scenario())
    assert result.status == "ok", result
    assert result.value.mask[0] is True  # observed, inside coverage
