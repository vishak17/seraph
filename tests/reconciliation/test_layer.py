"""C8 — the S6 seam: `ReconciliationLayer` and `ReconciledState`.

CT-4 covers the absence mechanism end to end. What is left here is the seam's
smaller print — batch/incremental equivalence, FR23's estimation variance
actually reaching `R_0`, the typed rejections, and the S6 shape's own PSD
assertion.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from seraph.reconciliation import (
    ForwardFillReconciliationLayer,
    KalmanReconciliationLayer,
    ReconciliationConfig,
)
from seraph.shared_types import (
    ContractViolation,
    ISOTimestamp,
    Mat3,
    ObservedEmission,
    PillarObservation,
    ReconciledState,
    UnavailableEmission,
)

START = datetime.fromisoformat("2020-01-01T09:15:00+05:30")


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def at(minutes: int) -> ISOTimestamp:
    return (START + timedelta(minutes=minutes)).isoformat()


def obs(
    pillar: str,
    ts: ISOTimestamp,
    value: float,
    tau: ISOTimestamp | None = None,
    variance: float | None = None,
) -> ObservedEmission:
    return ObservedEmission(
        obs=PillarObservation(
            pillar=pillar,
            ts=ts,
            tau=tau or ts,
            value=value,
            estimation_variance=variance,
        )
    )


# -- FR25/FR26 bookkeeping ---------------------------------------------------


def test_batched_and_incremental_delivery_agree_exactly() -> None:
    """C10 runs "byte-identical folds" (O6 acceptance). A batch replayed from
    the store must give the state a live one-at-a-time stream gave."""
    stream = [
        obs("rmt", at(0), 0.30),
        obs("hamilton", at(0), 0.70, variance=0.02),
        obs("hawkes", at(5), 1.10),
        obs("rmt", at(10), 0.35),
    ]

    async def scenario() -> tuple[ReconciledState, ReconciledState]:
        batched = await KalmanReconciliationLayer().update(tuple(stream))
        incremental = KalmanReconciliationLayer()
        last = None
        for emission in stream:
            last = await incremental.update((emission,))
        assert batched.status == "ok" and last is not None and last.status == "ok"
        return batched.value, last.value

    batched, incremental = _run(scenario())
    assert batched.model_dump() == incremental.model_dump()


def test_out_of_order_emissions_inside_one_batch_are_sorted_not_rejected() -> None:
    """Ordering *within* a call is C8's problem to solve; ordering *across*
    calls is a contract violation, since the state has already moved."""

    async def scenario() -> Any:
        return await KalmanReconciliationLayer().update(
            (obs("rmt", at(10), 0.35), obs("rmt", at(0), 0.30))
        )

    result = _run(scenario())
    assert result.status == "ok", result
    assert result.value.ts == at(10)


def test_reported_estimation_variance_reaches_the_gain() -> None:
    """FR23 -> D4 `R_0`: a confident pillar moves the state further than an
    uncertain one, from the same prior and the same observation."""

    async def move(variance: float) -> float:
        layer = KalmanReconciliationLayer(ReconciliationConfig(prior_mean=(0, 0, 0)))
        result = await layer.update((obs("hamilton", at(0), 1.0, variance=variance),))
        assert result.status == "ok", result
        return result.value.x_hat[2]

    confident = _run(move(1e-6))
    unsure = _run(move(1.0))
    assert confident > unsure
    assert confident == pytest.approx(1.0, rel=1e-3)


def test_a_stale_emission_is_warned_about_and_trusted_less() -> None:
    async def scenario() -> tuple[Any, Any]:
        fresh = KalmanReconciliationLayer()
        stale = KalmanReconciliationLayer()
        # RMT's h_p is one day; tau a week back is deep into age inflation.
        week_ago = (START - timedelta(days=7)).isoformat()
        return (
            await fresh.update((obs("rmt", at(0), 1.0),)),
            await stale.update((obs("rmt", at(0), 1.0, tau=week_ago),)),
        )

    fresh, stale = _run(scenario())
    assert fresh.status == "ok" and stale.status == "ok"
    assert stale.value.x_hat[1] < fresh.value.x_hat[1]
    assert any(w.code == "STALE_OBSERVATION" for w in stale.warnings)
    assert not any(w.code == "STALE_OBSERVATION" for w in fresh.warnings)


def test_history_is_bounded_and_evicts_oldest_first() -> None:
    async def scenario() -> tuple[Any, Any]:
        layer = KalmanReconciliationLayer(ReconciliationConfig(history_limit=5))
        for n in range(8):
            assert (await layer.update((obs("rmt", at(n), 0.3),))).status == "ok"
        return await layer.state_at(at(0)), await layer.state_at(at(7))

    evicted, kept = _run(scenario())
    assert evicted.status == "error"
    assert kept.status == "ok"


# -- typed rejections ---------------------------------------------------------


def test_empty_update_is_rejected_rather_than_silently_a_predict() -> None:
    result = _run(KalmanReconciliationLayer().update(()))
    assert result.status == "error"
    assert isinstance(result.error, ContractViolation)
    assert result.error.field == "update.emissions"


def test_non_finite_value_is_rejected() -> None:
    result = _run(
        KalmanReconciliationLayer().update((obs("rmt", at(0), float("nan")),))
    )
    assert result.status == "error"
    assert isinstance(result.error, ContractViolation)
    assert result.error.field == "PillarObservation.value"


def test_negative_estimation_variance_is_rejected() -> None:
    result = _run(
        KalmanReconciliationLayer().update((obs("rmt", at(0), 0.3, variance=-1.0),))
    )
    assert result.status == "error"
    assert isinstance(result.error, ContractViolation)
    assert result.error.field == "PillarObservation.estimation_variance"


def test_predicting_backwards_is_rejected() -> None:
    async def scenario() -> Any:
        layer = KalmanReconciliationLayer()
        await layer.update((obs("rmt", at(10), 0.3),))
        return await layer.predict(at(5))

    result = _run(scenario())
    assert result.status == "error"
    assert isinstance(result.error, ContractViolation)
    assert result.error.field == "predict.to_ts"


def test_a_rejected_emission_leaves_the_state_untouched() -> None:
    """An `Err` must not half-apply a batch — C10 replays these."""

    async def scenario() -> tuple[Any, Any, Any]:
        layer = KalmanReconciliationLayer()
        good = await layer.update((obs("rmt", at(0), 0.3),))
        bad = await layer.update(
            (obs("hamilton", at(5), 0.7), obs("rmt", at(5), 0.4, tau=at(9)))
        )
        after = await layer.predict(at(5))
        return good, bad, after

    good, bad, after = _run(scenario())
    assert bad.status == "error"
    # The valid Hamilton emission in the same batch was not applied either.
    assert after.value.x_hat[2] == good.value.x_hat[2]
    assert after.value.tau_last_update[2] is None


# -- FR27 baseline -------------------------------------------------------------


def test_forward_fill_carries_the_last_value_and_never_ages_it() -> None:
    async def scenario() -> tuple[Any, Any]:
        layer = ForwardFillReconciliationLayer()
        first = await layer.update((obs("rmt", at(0), 0.3),))
        later = await layer.predict(at(10_000))  # a week later
        return first, later

    first, later = _run(scenario())
    assert later.value.x_hat[1] == first.value.x_hat[1] == 0.3
    assert later.value.p_t == first.value.p_t
    assert later.value.mode == "forward_fill"


def test_both_modes_agree_on_the_mask() -> None:
    """The FR36 ablation isolates the recursion; if the two arms disagreed on
    D2 it would be measuring the mask as well."""
    stream = (
        UnavailableEmission(
            pillar="hawkes",
            ts=at(0),
            absence="structural",
            reason="no_data_coverage",
        ),
        obs("rmt", at(0), 0.3),
        obs("hamilton", at(0), 0.7),
    )

    async def scenario() -> tuple[Any, Any]:
        return (
            await KalmanReconciliationLayer().update(stream),
            await ForwardFillReconciliationLayer().update(stream),
        )

    kalman, forward_fill = _run(scenario())
    assert kalman.value.mask == forward_fill.value.mask == (False, True, True)


# -- the S6 shape itself -------------------------------------------------------


def _state(p: Mat3) -> ReconciledState:
    return ReconciledState(
        ts=at(0),
        x_hat=(0.0, 0.0, 0.0),
        p_t=p,
        tau_last_update=(None, None, None),
        mask=(True, True, True),
        noise_saturated=(False, False, False),
        mode="kalman",
    )


def test_reconciled_state_rejects_an_asymmetric_covariance() -> None:
    with pytest.raises(ValidationError, match="not symmetric"):
        _state(((1.0, 0.5, 0.0), (0.4, 1.0, 0.0), (0.0, 0.0, 1.0)))


def test_reconciled_state_rejects_a_negative_variance() -> None:
    with pytest.raises(ValidationError, match="positive semi-definite"):
        _state(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))


def test_reconciled_state_rejects_an_indefinite_covariance() -> None:
    """The realistic failure: diagonals fine, correlation above 1."""
    with pytest.raises(ValidationError, match="positive semi-definite"):
        _state(((1.0, 1.5, 0.0), (1.5, 1.0, 0.0), (0.0, 0.0, 1.0)))


def test_reconciled_state_accepts_a_singular_covariance() -> None:
    """PSD, not PD: a pillar updated at R = 0 legitimately has zero variance."""
    state = _state(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    assert state.p_t[0][0] == 0.0


def test_reconciled_state_is_frozen() -> None:
    state = _state(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    with pytest.raises(ValidationError):
        state.mask = (False, False, False)  # type: ignore[misc]
