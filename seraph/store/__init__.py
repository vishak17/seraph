"""C1 — Store & Schema (FR8, FR38).

Owns all DDL, hypertable partitioning, migrations, and the only connection
pool. Entities E1-E14 exist nowhere else. Interfaces S1 (StoreWriter) and
S2 (StoreReader) per docs/ARCHITECTURE.md §2.

Phase 1 (L0) — not implemented yet.
"""
