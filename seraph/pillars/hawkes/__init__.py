"""C5 — Hawkes Engine (FR10-FR15).

Owns E6: kernel parameterisation, the spectral-radius regulariser enforcing
n(S_t) < 1 on every window, Phi(S_t), and the contagion network. Models
negative price-jump events, not order-flow events (SPEC v2 observation-channel
change). Emissions before the intraday-coverage start date are `unavailable`
with `structural` absence — Ok, never Err.

Phase 5 (L4), built last, highest technical risk (R2) — not implemented yet.
"""
