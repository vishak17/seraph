"""C4 — the S4 slice C7 consumes: `observationVector` / `covariateVector`.

This is the seam ARCHITECTURE §2 draws as `C4 --S4--> C7`. C7 declares the
protocol (`HamiltonFeatureSource`) and never reaches past it; this module is
the first real implementation of it, replacing the test double C7 was built
against.

    y_t = (RV_t, AR_t, BAS_t)      FR20  -> ObservationRow
    z_t = macro covariates + G_t   FR21, FR22 -> MacroRow

**Minimal, and specific about how.** The three `y_t` components are computed
from daily OHLC with the estimators FR20/FR6 name (Yang-Zhang, top-10 PC
absorption, Abdi-Ranaldo), which is the whole of what C7 needs. What is *not*
here, and is owner A/B's C4 session: `rv_5min` from E1 intraday bars, FR50 jump
extraction, the `S_t` quintile grid, Corwin-Schultz/Roll fallbacks, and FR53's
estimator-agreement check. `rv_5min` is therefore always `None`, which is the
honest value before intraday coverage and which C7's default config already
expects.

Rows are emitted only where every rolling window behind them is complete, so
the first `max(window)` trading days of a panel produce no observations at all.
C7 reads that as insufficient history and reports structural absence, which is
correct: those dates genuinely have no `y_t`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from seraph.features.cross_sectional import absorption_ratio, cross_sectional_spread
from seraph.features.daily_panel import DailyPanel
from seraph.features.tariff_covariate import (
    DEFAULT_ETA_DAYS,
    TariffEvent,
    tariff_covariate,
)
from seraph.features.volatility import yang_zhang_variance
from seraph.shared_types import (
    ContractViolation,
    ISODate,
    MacroRow,
    ObservationRow,
    Result,
    SeraphWarning,
    err,
    ok,
)

__all__ = ["FeatureConfig", "PanelFeatureSource"]


@dataclass(frozen=True)
class FeatureConfig:
    """Rolling windows for the `y_t` estimators.

    All three are `[OPS]` choices — FR20/FR6 name the estimators, not their
    window lengths. One month for the two volatility/spread windows and one
    quarter for the covariance window is the conventional split: `AR_t` is a
    cross-sectional covariance over ~500 names and needs more rows than a
    univariate variance does before it is anything but noise.
    """

    rv_window: int = 21
    ar_window: int = 63
    bas_window: int = 21
    ar_components: int = 10  # FR20: "top-10 principal components"
    min_symbols: int = 12
    eta_days: float = DEFAULT_ETA_DAYS  # FR22

    def warmup(self) -> int:
        return max(self.rv_window, self.ar_window, self.bas_window)

    def __post_init__(self) -> None:
        if min(self.rv_window, self.ar_window, self.bas_window) < 2:
            raise ValueError("every rolling window needs at least 2 days")
        if self.ar_components < 1:
            raise ValueError("ar_components must be positive")


@dataclass
class PanelFeatureSource:
    """`HamiltonFeatureSource` over a `DailyPanel` (S4, FR20/FR21/FR22).

    Args:
        panel: the daily OHLC slice of E3. In production C1's
            `StoreReader.dailyBars()` fills it.
        macro: E4 rows, one per date, as ingested by C2. `G_t` is *not* read
            from these — C4 owns it (FR22) and overwrites whatever arrives.
        events: E5 tariff events driving `G_t`.
        cfg: rolling windows.

    Derived series are computed once, on first request, over the whole panel
    and then sliced. Recomputing per query would make a backtest quadratic, and
    windowing on a sub-range would silently change the values for the same date
    depending on how much history the caller happened to ask for — the kind of
    non-determinism that shows up months later as an unreproducible AUC.
    """

    panel: DailyPanel
    macro: tuple[MacroRow, ...] = ()
    events: tuple[TariffEvent, ...] = ()
    cfg: FeatureConfig = field(default_factory=FeatureConfig)

    _rows: tuple[ObservationRow, ...] | None = field(default=None, init=False)
    _macro: tuple[MacroRow, ...] | None = field(default=None, init=False)

    # -- S4 -------------------------------------------------------------------

    async def observation_vector(
        self, from_date: ISODate, to_date: ISODate
    ) -> Result[tuple[ObservationRow, ...]]:
        built = self._observations()
        if isinstance(built, str):
            return err(ContractViolation(field="C4:DailyPanel", detail=built))
        rows = tuple(r for r in built if from_date <= r.date <= to_date)
        warnings: list[SeraphWarning] = []
        if not rows:
            warnings.append(
                SeraphWarning(
                    code="PARTIAL_COVERAGE",
                    message=(
                        "no observation rows in range — the panel's first "
                        f"{self.cfg.warmup()} trading days have no complete "
                        "estimator window behind them"
                    ),
                    context={"from": from_date, "to": to_date},
                )
            )
        return ok(rows, warnings=tuple(warnings))

    async def covariate_vector(
        self, from_date: ISODate, to_date: ISODate
    ) -> Result[tuple[MacroRow, ...]]:
        rows = tuple(r for r in self._covariates() if from_date <= r.date <= to_date)
        return ok(rows)

    # -- derivation -----------------------------------------------------------

    def _observations(self) -> tuple[ObservationRow, ...] | str:
        if self._rows is not None:
            return self._rows

        violation = self.panel.validate()
        if violation is not None:
            return violation

        log_open, log_high, log_low, log_close = self.panel.market_ohlc()
        rv = yang_zhang_variance(
            log_open, log_high, log_low, log_close, self.cfg.rv_window
        )
        ar = absorption_ratio(
            self.panel.log_returns(),
            self.cfg.ar_window,
            self.cfg.ar_components,
            self.cfg.min_symbols,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            bas = cross_sectional_spread(
                np.log(self.panel.high),
                np.log(self.panel.low),
                np.log(self.panel.close),
                self.cfg.bas_window,
            )

        rows = tuple(
            ObservationRow(
                date=self.panel.dates[i],
                rv_yang_zhang=float(rv[i]),
                rv_5min=None,  # E1 intraday is C2/C4 work; never faked here
                ar_t=float(ar[i]),
                bas_t=float(bas[i]),
            )
            for i in range(len(self.panel))
            if np.isfinite(rv[i]) and np.isfinite(ar[i]) and np.isfinite(bas[i])
        )
        self._rows = rows
        return rows

    def _covariates(self) -> tuple[MacroRow, ...]:
        """E4 rows with `G_t` written in — FR22 says C4 owns that column."""
        if self._macro is not None:
            return self._macro

        supplied = {row.date: row for row in self.macro}
        dates = self.panel.dates
        g_t = tariff_covariate(dates, self.events, self.cfg.eta_days)

        rows = tuple(
            (supplied.get(d) or MacroRow(date=d)).model_copy(update={"g_t": g})
            for d, g in zip(dates, g_t, strict=True)
        )
        self._macro = rows
        return rows
