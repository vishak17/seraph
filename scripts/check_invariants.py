#!/usr/bin/env python3
"""
check_invariants.py — mechanical subset of AGENTS.md §5.

Not everything in §5 is checkable by a script (some genuinely need review),
but a handful are grep-able, and a failing check is worth more than a rule
sitting deep in a context window an agent may or may not still be attending
to. Run this alongside pytest, in CI, and as a pre-commit hook.

Exit code 0 = all mechanical checks pass (or nothing to check yet).
Exit code 1 = at least one violation found.

This is a starting point, not a complete implementation — extend it as real
components land. Each check degrades gracefully to "nothing found yet" if the
relevant directory doesn't exist, since this is meant to run from day one of
the repo, not just once the codebase is complete.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SERAPH_SRC = REPO_ROOT / "seraph"

EXPECTED_PILLAR_ORDER = '["hawkes", "rmt", "hamilton"]'
IST_OFFSET_RE = re.compile(r"\+05:30")
NAIVE_DATETIME_RE = re.compile(r"datetime\.utcnow\(\)|datetime\.now\(\)(?!\s*\(tz)")


def _iter_py_files(*subdirs: str) -> list[Path]:
    files: list[Path] = []
    for sub in subdirs:
        d = SERAPH_SRC / sub
        if d.exists():
            files.extend(sorted(d.rglob("*.py")))
    return files


def check_pillar_order() -> list[str]:
    """PILLAR_ORDER must be defined identically everywhere it's redefined."""
    violations = []
    hits = []
    for f in SERAPH_SRC.rglob("*.py") if SERAPH_SRC.exists() else []:
        text = f.read_text(errors="ignore")
        for m in re.finditer(r"PILLAR_ORDER\s*=\s*(\[[^\]]*\])", text):
            hits.append((f, m.group(1)))
    for f, literal in hits:
        normalized = re.sub(r"\s+", " ", literal.replace("'", '"')).strip()
        expected = re.sub(r"\s+", " ", EXPECTED_PILLAR_ORDER).strip()
        if normalized != expected:
            violations.append(
                f"{f.relative_to(REPO_ROOT)}: PILLAR_ORDER = {literal!r} "
                f"does not match the fixed order {EXPECTED_PILLAR_ORDER}"
            )
    return violations


def check_fusion_purity() -> list[str]:
    """C9 (fusion/) must never import C10 (validation/) or touch the DB layer."""
    violations = []
    for f in _iter_py_files("fusion"):
        text = f.read_text(errors="ignore")
        if re.search(r"^\s*(from|import)\s+.*\bvalidation\b", text, re.MULTILINE):
            violations.append(
                f"{f.relative_to(REPO_ROOT)}: imports from validation/ (C10) "
                f"— breaks C9's purity requirement (CT-6)"
            )
        if re.search(r"\b(asyncpg|psycopg)\b", text):
            violations.append(
                f"{f.relative_to(REPO_ROOT)}: references a DB driver directly "
                f"— C9 must have no DB calls"
            )
    return violations


def check_ar_t_placement() -> list[str]:
    """AR_t (cross-sectional) must live in features/, never in pillars/rmt/."""
    violations = []
    for f in _iter_py_files("pillars"):
        if "rmt" not in f.relative_to(SERAPH_SRC).parts:
            continue
        text = f.read_text(errors="ignore")
        if re.search(r"\bAR_t\b", text) and "def cross_sectional" not in text:
            violations.append(
                f"{f.relative_to(REPO_ROOT)}: references AR_t inside the RMT "
                f"engine — AR_t belongs in features/cross_sectional.py (C4)"
            )
    return violations


def check_bounded_observation_noise() -> list[str]:
    """AGENTS.md §5: `R^(p)(Delta)` must be D4's bounded saturating form.

    The unbounded `linear`/`power` variants exist only for CT-4's mandated
    three-form sweep. Shipping one as C8's default sends `P -> infinity` for a
    structurally absent pillar and makes the FR29 interval infinite — a defect
    that produces plausible numbers everywhere except the epochs that matter.
    """
    violations = []
    for f in _iter_py_files("reconciliation"):
        text = f.read_text(errors="ignore")
        for m in re.finditer(r"noise_form:\s*NoiseForm\s*=\s*\"(\w+)\"", text):
            if m.group(1) != "saturating_exponential":
                violations.append(
                    f"{f.relative_to(REPO_ROOT)}: default noise_form is "
                    f"{m.group(1)!r} — D4 requires the bounded saturating "
                    f"exponential in production"
                )
    return violations


def check_naive_timestamps() -> list[str]:
    """Flag likely-naive/UTC-implicit datetime usage in store/ingestion code."""
    violations = []
    for f in _iter_py_files("store", "ingestion"):
        text = f.read_text(errors="ignore")
        for m in NAIVE_DATETIME_RE.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            violations.append(
                f"{f.relative_to(REPO_ROOT)}:{line_no}: naive/UTC-implicit "
                f"datetime call — timestamps must be explicit IST (+05:30)"
            )
    return violations


def check_frozen_shared_types() -> list[str]:
    """Best-effort: pydantic models in shared_types/ should set frozen=True.

    This can't know which ones ARCHITECTURE marks `readonly` without parsing
    ARCHITECTURE.md, so it reports a count for a human to eyeball rather than
    hard-failing on every non-frozen model.
    """
    notes: list[str] = []
    shared_types = SERAPH_SRC / "shared_types"
    if not shared_types.exists():
        return notes
    total, frozen = 0, 0
    for f in shared_types.rglob("*.py"):
        text = f.read_text(errors="ignore")
        for _ in re.finditer(r"class\s+\w+\(BaseModel\)", text):
            total += 1
        if "frozen=True" in text or "frozen = True" in text:
            frozen += text.count("frozen=True") + text.count("frozen = True")
    if total:
        notes.append(
            f"[info] shared_types/: {total} BaseModel class(es) found, "
            f"{frozen} frozen=True usage(s). Cross-check against everything "
            f"ARCHITECTURE.md marks `readonly` — this check can't do that part."
        )
    return notes


CHECKS = [
    ("PILLAR_ORDER consistency", check_pillar_order),
    ("C9 fusion purity (no validation/ import, no DB driver)", check_fusion_purity),
    ("AR_t placement (C4, not C6)", check_ar_t_placement),
    ("C8 observation noise is bounded (D4)", check_bounded_observation_noise),
    ("Explicit IST timestamps", check_naive_timestamps),
    ("shared_types frozen= audit (info only)", check_frozen_shared_types),
]


def main() -> int:
    if not SERAPH_SRC.exists():
        print(
            "seraph/ not found yet — nothing to check. "
            "(Run this again once Phase 1 lands.)"
        )
        return 0

    had_violation = False
    for name, fn in CHECKS:
        results = fn()
        hard_violations = [r for r in results if not r.startswith("[info]")]
        info_notes = [r for r in results if r.startswith("[info]")]
        status = "FAIL" if hard_violations else "ok"
        print(f"[{status}] {name}")
        for v in hard_violations:
            print(f"    - {v}")
            had_violation = True
        for n in info_notes:
            print(f"    {n}")

    return 1 if had_violation else 0


if __name__ == "__main__":
    sys.exit(main())
