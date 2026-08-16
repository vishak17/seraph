"""C9 — Fusion Engine (FR28, FR29, FR31).

Owns E10: the rolling 5-year empirical-CDF standardiser, exact 8-subset
Shapley decomposition, and the D2 mask renormalisation.

`score()` is a PURE function: no DB calls, no hidden state, and it never
imports C10 (weights flow one-directionally C10 -> FusionWeights -> C9, per
D1). Identical inputs must give byte-identical output (CT-6).

Phase 7 (L6) — buildable against T0 mocks. Not implemented yet.
"""
