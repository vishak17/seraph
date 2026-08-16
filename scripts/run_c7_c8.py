#!/usr/bin/env python3
"""Run the C4 -> C7 -> C8 chain and write E9 rows. Idempotent.

AGENTS.md §2 picks "cron + idempotent scripts" over a workflow engine, so this
is what a scheduled reconciliation run looks like: one process, deterministic
given its inputs, safe to re-run after a failure. Re-running it overwrites its
own output with byte-identical content — it never appends, and it holds no
state between runs.

Until C1 and C2 exist there is nothing to read real bars from, so the default
input is the generated market in `fixtures/synthetic_market.py`. That is
flagged in the output rather than hidden: a run on generated data is labelled
`synthetic` in its manifest, and no result from it is reportable under FR35.

    python scripts/run_c7_c8.py --days 560 --out .runs/c7c8
    python scripts/run_c7_c8.py --mode forward_fill      # FR36's baseline arm
    python scripts/run_c7_c8.py --fit-noise-frac 0.7     # D4 MLE on the train fold

Output (`--out`):
    manifest.json   what was run, with the config that produced it
    e8.jsonl        E8 `hamilton_output` rows — C7's archive (FR21, FR23, FR24)
    e9.jsonl        E9 `reconciled_state` rows — C1's `writeBatch` input (FR38)
    xi.jsonl        the regime probabilities C9 needs alongside them (FR28)

E8 and E9 are both written because FR38 archives *both* halves — the regime
probabilities and half-lives C7 produced, and the reconciliation covariance C8
derived from them. E9 alone cannot reconstruct why `x_hat[2]` moved.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path

from fixtures.synthetic_market import generate_market
from seraph.features import FeatureConfig, PanelFeatureSource
from seraph.pillars.hamilton import HamiltonConfig, HamiltonEngine
from seraph.reconciliation import (
    PipelineRun,
    ReconciliationConfig,
    ReconciliationPipeline,
)
from seraph.shared_types import Err

FROM_TS = "1900-01-01T15:30:00+05:30"
TO_TS = "2099-12-31T15:30:00+05:30"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=560)
    parser.add_argument("--symbols", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--mode", choices=("kalman", "forward_fill"), default="kalman")
    parser.add_argument(
        "--fit-noise-frac",
        type=float,
        default=None,
        help=(
            "fraction of the corpus to MLE-fit D4's noise parameters on "
            "(training fold only — fitting through the evaluation window is "
            "leakage C10 cannot see)"
        ),
    )
    parser.add_argument("--out", type=Path, default=Path(".runs/c7c8"))
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    market = generate_market(n_days=args.days, n_symbols=args.symbols, seed=args.seed)
    source = PanelFeatureSource(
        panel=market.panel,
        macro=market.macro,
        events=market.events,
        cfg=FeatureConfig(),
    )
    hamilton_cfg = HamiltonConfig(
        corpus_start=market.panel.dates[0],
        min_history_days=250,
        refit_every_days=250,
    )
    # Built explicitly rather than through `from_hamilton_source` so the E8
    # archive below is a typed call on C7 rather than a hasattr probe on an
    # S5-typed engine. The wiring is identical: one engine, serving both the
    # emissions C8 reconciles and the xi C9 weights with.
    engine = HamiltonEngine(source, hamilton_cfg)
    pipeline = ReconciliationPipeline(
        [engine],
        detail_source=engine,
        cfg=ReconciliationConfig(),
        mode=args.mode,
    )

    fit_through = None
    if args.fit_noise_frac is not None:
        if not 0.0 < args.fit_noise_frac < 1.0:
            print("--fit-noise-frac must lie strictly between 0 and 1")
            return 2
        cut = market.panel.dates[int(args.fit_noise_frac * (args.days - 1))]
        fit_through = f"{cut}T15:30:00+05:30"

    result = await pipeline.run(FROM_TS, TO_TS, fit_noise_through=fit_through)
    if isinstance(result, Err):
        print(f"run failed: {result.error.kind} — {result.error!r}")
        if result.error.kind == "INSUFFICIENT_HISTORY" and fit_through is not None:
            print(
                "  the training fold ends before the pillars start emitting: C4 "
                "needs ~63 days of warm-up and C7 another 250 before its first "
                "estimate, so raise --days or --fit-noise-frac"
            )
        return 1
    run = result.value

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    e8 = await engine.outputs(FROM_TS, TO_TS)
    if isinstance(e8, Err):
        print(f"E8 archive unavailable: {e8.error.kind}")
        return 1
    _write_jsonl(out / "e8.jsonl", [row.model_dump() for row in e8.value])

    _write_jsonl(out / "e9.jsonl", run.e9())
    _write_jsonl(
        out / "xi.jsonl",
        [
            {"ts": p.ts, "xi": list(p.xi) if p.xi else None, "mask": list(p.state.mask)}
            for p in run.points
        ],
    )

    scoreable = sum(1 for p in run.points if p.scoreable)
    manifest = {
        "data": "synthetic",  # never reportable under FR35 — see the docstring
        "seed": args.seed,
        "days": args.days,
        "symbols": args.symbols,
        "mode": run.mode,
        "points": len(run.points),
        "scoreable_points": scoreable,
        "e8_rows": len(e8.value),
        "updates_by_pillar": list(run.report.updates),
        "redeliveries_by_pillar": list(run.report.redeliveries),
        "engine_errors_by_pillar": list(run.report.engine_errors),
        "reconciliation_config": asdict(run.cfg),
        "noise_fit": _fit_summary(run),
        "warnings": sorted({w.code for w in run.warnings}),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"[{run.mode}] {len(run.points)} states, {scoreable} scoreable -> {out}")
    print(
        f"  updates={run.report.updates} redeliveries={run.report.redeliveries} "
        f"engine_errors={run.report.engine_errors}"
    )
    if run.noise_fit is not None:
        fit = run.noise_fit
        print(
            f"  D4 MLE: loglik {fit.initial_log_likelihood:.2f} -> "
            f"{fit.log_likelihood:.2f} (+{fit.improvement:.2f}), "
            f"fitted={fit.fitted_pillars}"
        )
    for code in sorted({w.code for w in run.warnings}):
        print(f"  warning: {code}")
    return 0


def _fit_summary(run: PipelineRun) -> dict[str, object] | None:
    fit = run.noise_fit
    if fit is None:
        return None
    return {
        "log_likelihood": fit.log_likelihood,
        "initial_log_likelihood": fit.initial_log_likelihood,
        "improvement": fit.improvement,
        "updates": list(fit.updates),
        "fitted_pillars": list(fit.fitted_pillars),
        "h_identified": list(fit.h_identified),
        "converged": fit.converged,
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Whole-file write: re-running replaces, never appends (idempotency)."""
    lines = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    path.write_text(lines, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
