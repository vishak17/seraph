"""C2 — Ingestion Layer (FR1-FR4, FR7, FR51).

Owns all outbound network I/O, Kite credentials, retry/backoff, per-source
watermarks, and both bhavcopy schema readers (legacy + UDiFF). Lands raw
external series unmodified except for schema normalisation.

Phase 2 (L1) — not implemented yet.
"""
