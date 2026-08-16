# SERAPH AI-context kit — what this is and how to install it

Drop the contents of this kit into your `seraph/` repo root (it's laid out to
overlay directly — `AGENTS.md`, `.cursor/`, etc. all land where each tool
expects them). Delete whichever tool-specific files your team doesn't
actually use; keeping unused ones does no harm, but there's no point
maintaining them.

## The tiering model

- **Tier 0 — `AGENTS.md`.** Always loaded, every session, every tool. This is
  your old `SERAPH-TECH-STACK.md`, restructured slightly. It's the single
  source of truth — every other file in this kit is a thin pointer back to it,
  not a fork of it. Edit rules here, nowhere else.
- **Tier 1 — `docs/`.** `SPEC.md`, `ARCHITECTURE.md`, `DATA-ACQUISITION.md`,
  `SERAPH-BUILD-ROADMAP.md`. Loaded on demand — reference these by path in a
  prompt ("read docs/ARCHITECTURE.md §7 before touching C8"), don't expect a
  tool to have them in context automatically.
- **Tier 2 — `docs/archive/`.** `SPEC-DELTA-01.md`. Deliberately excluded from
  every tool's indexing (see below) because it's superseded but reads like
  current spec — a RAG-based search can't tell "historical, contains a
  documented bug" from "current" the way a person reading the banner can. Kept
  only for report §5.1.2 and the viva.

## Per-tool wiring, and what each ignore file actually does

| Tool | Reads `AGENTS.md` natively? | Extra file this kit adds | What it's for |
|---|---|---|---|
| Cursor | Yes | `.cursor/rules/*.mdc` | `seraph-core.mdc` (always on) restates the 5 easiest-to-violate rules so they survive even in a thin context window; `pillars.mdc` auto-attaches only when you're inside `pillars/`, `reconciliation/`, or `fusion/`. `.cursorindexingignore` keeps `docs/archive/` out of semantic search *without* blocking it if you deliberately `@`-mention it. |
| Claude Code | **No** — reads `CLAUDE.md` instead, as of Aug 2026 | `CLAUDE.md` (first line: `@AGENTS.md`), `.claude/settings.json` | The settings file `deny`s the `Read` tool on `docs/archive/**` (best-effort — also covers `Grep`/`Glob`, but *not* a raw `cat` run through `Bash`, so it's a nudge plus a backstop, not a hard wall) and `ask`s before any edit to `shared_types/`. |
| Windsurf / Devin Desktop | Increasingly yes (Cascade reads AGENTS.md dynamically) | `.devinignore`, `.windsurfrules` | Windsurf was rebranded Devin Desktop in June 2026; current builds prefer `.devinignore`, older ones look for `.codeiumignore`/`.windsurfrules`. This kit ships both so it works either way. `.windsurfrules` also carries the critical invariants as plain text, in case a teammate's build doesn't pick up AGENTS.md at all. |
| GitHub Copilot | Yes | `.github/copilot-instructions.md` | Thin backstop pointing back to `AGENTS.md`. |

**None of these ignore/deny mechanisms are airtight.** They stop automatic
retrieval and casual reads, not a determined `cat docs/archive/SPEC-DELTA-01.md`
in a terminal. That's fine here — the goal is "don't let the model
accidentally resurface a documented bug while answering an unrelated
question," not "hide a secret." If you ever put something genuinely sensitive
in this repo (credentials, private data-vendor terms), don't rely on these
files for that — use real access controls.

## Making rules actually stick, not just visible

A markdown rules file is context, not enforcement — a long enough session can
still drift past it. Where a rule in `AGENTS.md` §5 is mechanically checkable,
`scripts/check_invariants.py` checks it instead of hoping it's still being
attended to 40 turns in:

```bash
python scripts/check_invariants.py
```

It currently checks: `PILLAR_ORDER` consistency, C9 purity (no `validation/`
import, no DB driver in `fusion/`), `AR_t` placement, and naive/UTC-implicit
timestamps. It's intentionally a starting point — extend it as real components
land, and consider wiring it into a pre-commit hook or your CI once there's a
CI. It's safe to run today: with no `seraph/` tree yet, it just says so and
exits 0.

## What this kit deliberately doesn't do

It doesn't touch your actual `seraph/` source tree, your `pyproject.toml`, or
`tests/contract/` — those come from `SERAPH-BUILD-ROADMAP.md`'s phase plan,
not from this kit. This is purely the context/rules layer.
