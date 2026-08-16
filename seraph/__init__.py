"""SERAPH — systemic-risk early-warning framework for the Indian NSE.

Twelve components (C1-C12) across nine layers (L0-L8); see docs/ARCHITECTURE.md
§1 for the component map and docs/SERAPH-BUILD-ROADMAP.md §2 for build order.
Every component imports its shapes from `seraph.shared_types` — nothing
redefines a shape locally.
"""

__version__ = "0.0.0"
