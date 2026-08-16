"""C4 — `G_t`, the tariff-shock covariate (FR22, Objective 8).

    G_t = sum_k |delta_r_k| * exp(-(t - tau_k) / eta)      over tau_k <= t

One dated event table (SPEC E5), one exponential decay. `G_t` feeds C7's `z_t`,
which is how a tariff announcement is allowed to move the TVTP transition
probabilities without being smuggled into `y_t` as if it were a market
observable.

`eta` is a SPEC-undefined constant (SPEC §8 item 9, roadmap §4: "pick
reasonable starting values and let the ablation runner tell you if they
matter"). The default here is 60 trading days — about a quarter, long enough
that a tariff announcement is still visible in `z_t` when its second-round
effects land, short enough that events a year apart do not superpose into one
permanent level shift. Documented as a starting value, swept by FR34, never
presented as spec.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from seraph.shared_types import ISODate

__all__ = ["DEFAULT_ETA_DAYS", "TariffEvent", "tariff_covariate"]

# [OPS] SPEC §8 item 9 leaves eta undefined. See the module docstring.
DEFAULT_ETA_DAYS = 60.0


@dataclass(frozen=True)
class TariffEvent:
    """One row of SPEC E5 `tariff_events`, reduced to what FR22 reads.

    `severity_score` and `sectors_affected` are E5 fields the full C4 carries;
    FR22's formula uses only the announcement date and the tariff-rate change,
    so only those two are required here.
    """

    tau_k: ISODate
    delta_r_k: float
    event_id: str = ""
    source_ref: str = ""


def tariff_covariate(
    dates: tuple[ISODate, ...],
    events: tuple[TariffEvent, ...],
    eta_days: float = DEFAULT_ETA_DAYS,
) -> tuple[float, ...]:
    """`G_t` on the supplied trading-day grid.

    Decay is measured in *calendar* days, matching how the events themselves
    are dated; a trading-day clock would make `G_t` depend on the exchange
    calendar for no modelling reason. Zero before the first event — which is
    the correct value, not a missing one, and is why this returns floats rather
    than `None`s that C7 would then drop as an incomplete covariate.
    """
    if eta_days <= 0.0:
        raise ValueError("eta_days must be positive")

    parsed = sorted(
        ((date.fromisoformat(e.tau_k), abs(e.delta_r_k)) for e in events),
        key=lambda pair: pair[0],
    )
    out: list[float] = []
    for iso in dates:
        day = date.fromisoformat(iso)
        total = 0.0
        for tau_k, magnitude in parsed:
            if tau_k > day:
                break
            total += magnitude * math.exp(-(day - tau_k).days / eta_days)
        out.append(total)
    return tuple(out)
