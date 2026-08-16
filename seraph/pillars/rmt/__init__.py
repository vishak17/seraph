"""C6 — RMT Engine (FR16-FR19).

Owns E7: the rolling eigendecomposition (T=1000, N=500, Q=2), Marchenko-Pastur
bounds, the 1,000-replication bootstrap, and the D3 common-support rotation
speed. The Absorption Ratio is computed in C4 (features/cross_sectional.py)
and must never be computed here — that would make Hamilton depend on RMT and
break FR35.

Phase 5 (L4), built first of the three pillars — not implemented yet.
"""
