# Predictions for the regeneration, written before running

Environment: CUDA 13.1, driver 610.74, WSL2, buffers padded by 64 KiB in every mode.
The noise and polling checks already showed v0 at ~246.5 GB/s, not the notebook's 232.9.

1. Baseline: v0 ~246.5 GB/s, v1 (G = 102,400) ~247.1, v1 (G = 768) ~243.9, v2 ~244.6,
   each within 0.05% of the polling-check runs.
2. M4: rep 0 of v0, the first configuration after a cold start, may still be rejected
   for an SM clock dip.
3. Offset sweep: if the old 6% penalty came from buffer placement, it has moved rather
   than vanished, and at least one shift lands near 233 GB/s. If every shift gives ~247,
   the penalty is absent in this environment.
4. Swizzle sweep: v0 has no penalty to remove at S = 1, so every row lands near 247 GB/s.
5. E4: at base clock, v2 is unchanged; v0 and v1 slow down by more than the 0.10-0.16%
   they lost at 2400 MHz.
6. Profiler: 4 sectors per request for v0 and v1, 16 for v2. Instruction counts match
   A7 (v0: 16 per warp; v1: 64 or 75) within a few instructions per warp, after
   recounting from the new SASS.
