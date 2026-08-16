"""C8 — D4 noise and the D2 mask rules, in isolation.

These are pure functions, so the D2 decision table can be asserted row by row
rather than inferred from filter behaviour. Getting a row wrong here is silent:
a structurally-absent pillar left in the mask still produces a number, just a
diluted one.
"""

from __future__ import annotations

import math

import pytest

from seraph.reconciliation.config import SECONDS_PER_DAY, ReconciliationConfig
from seraph.reconciliation.noise_model import (
    R0Tracker,
    ceiling,
    epoch_seconds,
    is_saturated,
    mask_bit,
    observation_noise,
    staleness_seconds,
)

R0 = 0.01
H = SECONDS_PER_DAY
RATIO = 100.0


# -- D4: R^(p)(Delta) --------------------------------------------------------


def test_r_at_zero_staleness_is_the_estimation_variance() -> None:
    """D4 requirement 2 — `R^(p)(0)` is FR23's reported variance, untouched."""
    assert observation_noise(0.0, R0, H, RATIO) == R0


def test_saturating_exponential_matches_the_written_form() -> None:
    delta = 3.0 * H
    expected = R0 + (RATIO * R0 - R0) * (1.0 - math.exp(-3.0))
    assert observation_noise(delta, R0, H, RATIO) == pytest.approx(expected)


def test_r_approaches_r_max_from_below_and_never_exceeds_it() -> None:
    r_max = RATIO * R0
    for k in (1, 5, 20, 100, 10_000):
        r = observation_noise(k * H, R0, H, RATIO)
        assert r < r_max or r == pytest.approx(r_max)
    assert observation_noise(math.inf, R0, H, RATIO) == pytest.approx(r_max)


def test_one_native_cadence_of_staleness_costs_about_63_percent() -> None:
    """Why `h_p` is initialised to the pillar's own update cadence (D4): one
    cycle late is most of the way to worthless, which is the intended reading."""
    r = observation_noise(H, R0, H, RATIO)
    assert (r - R0) / (RATIO * R0 - R0) == pytest.approx(1 - math.exp(-1.0))


def test_negative_staleness_cannot_shrink_r_below_r0() -> None:
    """`tau > ts` is rejected upstream; if it ever leaked through, the noise
    model must not reward it with sub-floor variance."""
    assert observation_noise(-100.0, R0, H, RATIO) == R0


def test_saturation_flag_fires_at_the_documented_fraction() -> None:
    cfg = ReconciliationConfig()
    r_max = RATIO * R0
    assert is_saturated(0.96 * r_max, R0, cfg) is True
    assert is_saturated(0.94 * r_max, R0, cfg) is False


def test_unbounded_forms_never_report_saturation() -> None:
    """Deliberate: a form with no ceiling has nothing to saturate against, and
    reporting otherwise would hide exactly what the D4 sweep is measuring."""
    cfg = ReconciliationConfig(noise_form="linear")
    assert is_saturated(1e12, R0, cfg) is False
    assert ceiling(R0, RATIO, "linear") == math.inf


def test_unknown_noise_form_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown noise form"):
        observation_noise(1.0, R0, H, RATIO, form="quadratic")  # type: ignore[arg-type]


# -- staleness ---------------------------------------------------------------


def test_staleness_is_infinite_for_a_never_observed_pillar() -> None:
    assert staleness_seconds("2020-01-01T09:15:00+05:30", None) == math.inf


def test_staleness_is_ist_aware() -> None:
    """AGENTS.md §5 — a timestamp is IST-explicit or it does not parse."""
    a = "2020-01-01T09:15:00+05:30"
    b = "2020-01-01T15:30:00+05:30"
    assert staleness_seconds(b, a) == pytest.approx(6.25 * 3600.0)
    assert epoch_seconds(b) - epoch_seconds(a) == pytest.approx(6.25 * 3600.0)


def test_staleness_is_negative_when_tau_is_in_the_future() -> None:
    """Not clamped — the layer needs to see this to report CONTRACT_VIOLATION."""
    assert (
        staleness_seconds("2020-01-01T09:15:00+05:30", "2020-01-02T09:15:00+05:30") < 0
    )


# -- D2: the mask decision table ---------------------------------------------


@pytest.mark.parametrize(
    ("status", "saturated", "expected"),
    [
        ("observed", False, True),  # normal
        ("transient", False, True),  # should exist; D4 age-inflation handles it
        ("structural", False, False),  # cannot exist for this period
        ("observed", True, False),  # present but contributing nothing
        ("transient", True, False),
        ("structural", True, False),
        ("never", False, False),  # no observation behind x_hat[p] at all
    ],
)
def test_d2_mask_table(status: str, saturated: bool, expected: bool) -> None:
    assert mask_bit(status, saturated) is expected  # type: ignore[arg-type]


# -- R_0 -----------------------------------------------------------------------


def test_reported_estimation_variance_wins() -> None:
    """FR23: Hamilton reports its own variance; D4 says use it as `R_0`."""
    tracker = R0Tracker(prior=1.0, floor=1e-6, window=10)
    for v in (0.1, 0.2, 0.3):
        tracker.observe(v)
    assert tracker.r0(reported=0.042) == 0.042


def test_reported_variance_is_still_floored() -> None:
    """A certain Hamilton filter reports variance 0; R = 0 is a gain of 1, i.e.
    one emission treated as ground truth."""
    tracker = R0Tracker(prior=1.0, floor=1e-6, window=10)
    assert tracker.r0(reported=0.0) == 1e-6


def test_rolling_variance_is_used_once_the_window_has_content() -> None:
    tracker = R0Tracker(prior=0.5, floor=1e-12, window=4)
    assert tracker.r0() == 0.5  # prior, nothing observed
    tracker.observe(1.0)
    assert tracker.r0() == 0.5  # one point has no sample variance
    tracker.observe(3.0)
    assert tracker.r0() == pytest.approx(2.0)  # var([1, 3]) with ddof=1


def test_the_window_actually_rolls() -> None:
    tracker = R0Tracker(prior=0.5, floor=1e-12, window=3)
    for v in (100.0, 0.0, 1.0, 1.0, 1.0):
        tracker.observe(v)
    assert tracker.r0() == pytest.approx(0.0, abs=1e-12)  # the outlier aged out
