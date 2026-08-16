"""T0 — mock data generator.

Emits synthetic PillarEmission / ReconciledState / CsrsPoint streams
conforming to docs/ARCHITECTURE.md §2, including deliberately-`unavailable`
periods of both `structural` and `transient` absence. This is what lets
C8/C9/C10/C11/C12 be built and tested before any real pillar exists.

Phase 1, built alongside C1 — `mock_generator.py` not implemented yet.
"""
