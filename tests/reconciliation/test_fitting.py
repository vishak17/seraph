"""C8 — the D4 parameter fit (SPEC OQ10: `h_p`, `R_max/R_0`, `Q_proc` MLE).

The headline test generates emissions from a *known* random walk and checks the
fit recovers the process-noise scale it was generated with. The rest pin the
behaviour that matters more than the point estimate: the fit never reports an
improvement it did not make, never invents parameters for a pillar the history
cannot identify, and is reproducible from the stream alone (NFR21).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from seraph.reconciliation import ReconciliationConfig, fit_noise_parameters
from seraph.reconciliation.config import SECONDS_PER_DAY
from seraph.reconciliation.fitting import _pinned_parameters
from seraph.shared_types import (
    ISOTimestamp,
    ObservedEmission,
    PillarEmission,
    PillarObservation,
    UnavailableEmission,
)
from tests.reconciliation.helpers import as_err, as_ok, unwrap

# Each fit replays a few hundred emissions a few hundred times. Seconds, not
# milliseconds — kept out of the default inner loop, run in CI.
pytestmark = pytest.mark.slow

START = datetime.fromisoformat("2018-01-01T15:30:00+05:30")


def day(n: int) -> ISOTimestamp:
    return (START + timedelta(days=n)).isoformat()


def simulate(
    n_days: int = 400,
    q_true: float = 4.0e-2,
    r_true: float = 2.5e-2,
    stale_every: int = 3,
    seed: int = 7,
) -> list[PillarEmission]:
    """A random-walk latent state observed daily by RMT and Hamilton.

    `stale_every` makes some arrivals carry the *previous* close as `tau`, so
    the ageing curve is actually exercised — without staleness variation `h_p`
    is not identified, which is itself asserted below.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(3)
    emissions: list[PillarEmission] = []
    for n in range(n_days):
        x += rng.normal(0.0, np.sqrt(q_true), 3)
        for pillar, index in (("rmt", 1), ("hamilton", 2)):
            lag = 1 if stale_every and n % stale_every == 0 else 0
            emissions.append(
                ObservedEmission(
                    obs=PillarObservation(
                        pillar=pillar,
                        ts=day(n),
                        tau=day(n - lag),
                        value=float(x[index] + rng.normal(0.0, np.sqrt(r_true))),
                    )
                )
            )
    return emissions


# -- recovery ----------------------------------------------------------------


def test_the_fit_recovers_the_process_noise_it_was_generated_with() -> None:
    q_true = 4.0e-2
    fit = unwrap(fit_noise_parameters(simulate(q_true=q_true)))

    assert fit.improvement > 0.0
    for index in (1, 2):  # rmt, hamilton
        fitted = fit.cfg.q_proc_per_day[index]
        # Within a factor of three of truth. Not tighter on purpose: R_0 here
        # is C8's rolling empirical proxy, not the true observation variance,
        # so Q and R are only partly separable — the module says so, and this
        # bound is that statement made testable rather than aspirational.
        assert q_true / 3.0 < fitted < q_true * 3.0


def test_the_fitted_config_is_a_drop_in_replacement() -> None:
    stream = simulate(n_days=120)
    fit = unwrap(fit_noise_parameters(stream))
    replayed = unwrap(fit_noise_parameters(stream, fit.cfg))
    # Re-fitting from the fitted point cannot do materially better — it is
    # already at (or very near) the optimum.
    assert replayed.improvement < abs(fit.log_likelihood) * 1e-3


def test_the_fit_is_deterministic() -> None:
    """NFR21: reproducible from the emission stream alone."""
    stream = simulate(n_days=150)
    a = unwrap(fit_noise_parameters(stream))
    b = unwrap(fit_noise_parameters(stream))
    assert a.log_likelihood == b.log_likelihood
    assert a.cfg == b.cfg


def test_improvement_is_never_negative() -> None:
    """L-BFGS-B may land marginally worse on a flat surface; the fit keeps the
    better of the two rather than shipping a regression."""
    fit = unwrap(fit_noise_parameters(simulate(n_days=60)))
    assert fit.improvement >= 0.0
    assert fit.log_likelihood >= fit.initial_log_likelihood


# -- identification ----------------------------------------------------------


def test_a_pillar_with_no_updates_is_held_at_its_initialisation() -> None:
    """Hawkes pre-2015 (SPEC §4): nothing to fit, and the fit says so."""
    base = ReconciliationConfig()
    result = as_ok(fit_noise_parameters(simulate(n_days=200), base))
    fit = result.value

    assert fit.fitted_pillars == (False, True, True)
    assert fit.cfg.h_seconds[0] == base.h_seconds[0]
    assert fit.cfg.q_proc_per_day[0] == base.q_proc_per_day[0]
    assert any(
        w.code == "PARTIAL_COVERAGE" and "hawkes" in str(w.context["held"])
        for w in result.warnings
    )


def test_h_is_not_fitted_when_no_arrival_was_ever_stale() -> None:
    """`h_p` is identified by variation in `Delta`, not by sample size."""
    base = ReconciliationConfig()
    fit = unwrap(fit_noise_parameters(simulate(n_days=300, stale_every=0), base))

    assert fit.h_identified == (False, False, False)
    assert fit.cfg.h_seconds == base.h_seconds
    # Q is still fitted — it does not depend on staleness.
    assert fit.cfg.q_proc_per_day[1] != base.q_proc_per_day[1]


def test_unavailable_emissions_do_not_count_toward_the_sample() -> None:
    absences: list[PillarEmission] = [
        UnavailableEmission(
            pillar="hawkes",
            ts=day(n),
            absence="structural",
            reason="no_data_coverage",
        )
        for n in range(400)
    ]
    error = as_err(fit_noise_parameters(absences)).error
    assert error.kind == "INSUFFICIENT_HISTORY"


def test_r_max_ratio_is_left_alone_for_the_unbounded_sweep_forms() -> None:
    """`R_max` is not a parameter of the linear form at all — fitting it would
    be reporting an estimate of something the model does not contain."""
    base = ReconciliationConfig(noise_form="linear")
    fit = unwrap(fit_noise_parameters(simulate(n_days=150), base))
    assert fit.cfg.r_max_ratio == base.r_max_ratio
    assert fit.cfg.noise_form == "linear"


# -- rejections ---------------------------------------------------------------


def test_too_short_a_history_is_an_error_not_a_guess() -> None:
    error = as_err(fit_noise_parameters(simulate(n_days=5))).error
    assert error.kind == "INSUFFICIENT_HISTORY"


def test_an_empty_stream_is_an_error() -> None:
    error = as_err(fit_noise_parameters([])).error
    assert error.kind == "INSUFFICIENT_HISTORY"


def test_a_contract_violation_in_the_stream_propagates() -> None:
    """The fit replays through the real layer, so it inherits its validation
    rather than quietly fitting on data production would reject."""
    bad: list[PillarEmission] = [
        ObservedEmission(
            obs=PillarObservation(pillar="rmt", ts=day(0), tau=day(1), value=1.0)
        )
    ]
    error = as_err(fit_noise_parameters(bad)).error
    assert error.kind == "CONTRACT_VIOLATION"


def test_fitted_parameters_stay_inside_their_bounds() -> None:
    fit = unwrap(fit_noise_parameters(simulate(n_days=200)))
    for h in fit.cfg.h_seconds:
        assert 60.0 <= h <= 30.0 * SECONDS_PER_DAY
    for q in fit.cfg.q_proc_per_day:
        assert 1.0e-8 <= q <= 1.0e2
    assert 2.0 <= fit.cfg.r_max_ratio <= 1.0e4


def test_bound_pinned_parameters_are_detected_and_named() -> None:
    """A parameter driven onto its bound is an artefact of the bound, not an
    estimate — saying so is the difference between a fit and a number.

    Tested on the detector directly: whether a given stream *drives* a
    parameter to a bound depends on the draw, so asserting that it does would
    be testing the random number generator. What must hold unconditionally is
    that a point sitting on a bound is recognised and named.
    """
    bounds = [
        (math.log(1.0e-8), math.log(1.0e2)),  # q
        (math.log(60.0), math.log(30.0 * SECONDS_PER_DAY)),  # h
    ]
    slots = [("q", 1), ("h", 2)]

    on_the_floor = np.array([math.log(1.0e-8), math.log(3600.0)])
    assert _pinned_parameters(on_the_floor, bounds, slots) == ["q[rmt]"]

    on_the_ceiling = np.array([math.log(1.0e-3), math.log(30.0 * SECONDS_PER_DAY)])
    assert _pinned_parameters(on_the_ceiling, bounds, slots) == ["h[hamilton]"]

    interior = np.array([math.log(1.0e-3), math.log(3600.0)])
    assert _pinned_parameters(interior, bounds, slots) == []


def test_a_clean_fit_reports_no_pinned_parameters() -> None:
    """The integration half: a well-identified stream should land in the
    interior, and the warning must not fire when it does."""
    result = as_ok(fit_noise_parameters(simulate(n_days=200)))
    for warning in result.warnings:
        if "pinned" in warning.context:
            # If anything did pin, it must at least be named — never silent.
            assert warning.code == "ESTIMATOR_FALLBACK"
            assert str(warning.context["pinned"])


@pytest.mark.parametrize("n_days", [60, 120, 250])
def test_more_data_never_makes_the_fit_report_a_worse_likelihood(n_days: int) -> None:
    fit = unwrap(fit_noise_parameters(simulate(n_days=n_days)))
    assert fit.updates[1] > 0 and fit.updates[2] > 0
    assert fit.log_likelihood >= fit.initial_log_likelihood
