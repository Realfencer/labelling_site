"""The ONE p95 latency limit (milliseconds, CPU) every deep post-process and every collector gate keeps.

Imported by find_best_model.py (deep post-process, in the site and in the collector sandbox) and by
backend/collector/config.py. Change it here only.
"""

P95_LIMIT_MS = 150.0

# p95 (ms) of the reference winner (Alice36/bev_1_929) on an idle box. Timing it alongside a model tells how
# much slower than idle the box is right now; a model's p95 is then scaled by that ratio ("idle-equivalent").
REF_IDLE_P95_MS = 89.0
