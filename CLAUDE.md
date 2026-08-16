@AGENTS.md

# Claude Code notes (SERAPH)

Everything binding lives in `AGENTS.md` above (imported into every session).
This file only adds mechanics specific to Claude Code.

- **Session scope:** one component per session — see AGENTS.md §8's ownership
  table and `docs/SERAPH-BUILD-ROADMAP.md`'s phase list. Don't let one session's
  diff span two owners' directories.
- **Before reporting a component done:** actually run
  `pytest tests/contract/test_ctN_*.py -v` for its contract test and show the
  result. A plausible-looking diff that was never run doesn't count as done.
- **`docs/archive/SPEC-DELTA-01.md` is denied by `.claude/settings.json`.** It's
  superseded historical material (see AGENTS.md §0). If a task is specifically
  about report §5.1.2 or viva prep, say that explicitly and I'll grant it —
  otherwise don't go looking for it. Note the deny rule blocks the `Read` tool
  (and best-effort covers `Grep`/`Glob`) but not a raw `cat` inside `Bash` —
  don't route around it that way either.
- **`seraph/shared_types/` is ask-gated**, not denied — confirm with the human
  before editing it, since every component imports from there.
- **Auto memory** is on by default and is local to your machine, not shared
  with teammates. Run `/memory` occasionally to see what Claude has taught
  itself on this project and prune anything that drifts from AGENTS.md.
