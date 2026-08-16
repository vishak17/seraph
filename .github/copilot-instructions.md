See `AGENTS.md` at the repo root — this project's full rules, stack, and
invariants live there, and Copilot reads it natively. This file is a short
Copilot-specific backstop.

Quick invariants: `PILLAR_ORDER` is fixed
(`["hawkes", "rmt", "hamilton"]`); `docs/SPEC.md` / `docs/ARCHITECTURE.md`
win over existing code on any conflict; `docs/archive/SPEC-DELTA-01.md` is
superseded — never use it as a source unless the task is explicitly about
report §5.1.2 or viva prep.
