"""The three pillar engines (L4).

`PILLAR_ORDER = ["hawkes", "rmt", "hamilton"]` is global and fixed — every
Vec3, Mat3 and AvailabilityMask uses it. Build order is the reverse of that
ordering: RMT (C6) first, Hamilton (C7) second, Hawkes (C5) last.
"""
