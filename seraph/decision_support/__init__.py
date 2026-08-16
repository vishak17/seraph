"""C11 — Decision Support (FR43-FR49).

Owns E11 and E12: the seven role-specific artefacts — portfolio overlay,
template explanation, market-level flow diagnostic (downgraded per SPEC v2),
historical playbook, shock simulator, policy-path simulator, regulatory action
log. Simulators call C9's `score()` directly; they never fork scoring logic
(CT-7).

Phase 8 (L7) — not implemented yet.
"""
