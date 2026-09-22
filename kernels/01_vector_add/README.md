# 01 · Vector add

`c[i] = a[i] + b[i]` · fp32 · n = 2²⁶ (256 MiB per array) · RTX 4060 Laptop GPU

**Full investigation: [EXPERIMENTS.md](EXPERIMENTS.md).** The model I built before measuring, four measurement experiments and eight on the kernel, what each could and couldn't distinguish, and a derivation for every number.

> [TODO: the numbers below predate the current harness. Regenerate with the commands in
> [Reproduce](#reproduce), update them here and in EXPERIMENTS.md, then delete this note.]

## Summary

Grid-stride vector add beat the naive kernel by 6% (247.4 vs 232.9 GB/s). Instruction overhead, memory-level parallelism, coalescing and wave quantization were each ruled out. The gap turned out to belong to neither kernel: with the naive kernel untouched, offsetting two of its three buffers by 128 and 256 bytes takes it from 233 to 246 GB/s. Naive pays a buffer-placement penalty, and grid-stride sidesteps it through the order in which it touches memory. Every penalty-free configuration lands at ~247 GB/s, 94.1% of theoretical peak.

For most of the investigation I asked why grid-stride was fast. The answer was that naive was slow.

## Open question

Offsetting `b` and `c` by 128 and 256 bytes removes the penalty. Offsetting them by 256 and 512 bytes brings it back; 1 and 2 KiB removes it again. The penalty depends on particular relative offsets, not on how far apart the buffers are. Which pair of streams collides, and at what level of the memory system? The 256/512 result is a single run so far; replicating it comes first ([E8](EXPERIMENTS.md#e8-is-it-the-buffers-placement), [open questions](EXPERIMENTS.md#open-questions)).

## Results

Turbo mode (memory clock 8201 MHz, theoretical peak 262.4 GB/s), SM clock ~2730 MHz, median of accepted reps.

| Configuration | ms | GB/s | % of peak |
|---|---|---|---|
| v0 naive | 3.4573 | 232.9 | 88.8% |
| v1 grid-stride, 102,400 blocks | 3.2549 | 247.4 | 94.3% |
| v2 `float4` | 3.3942 | 237.3 | 90.4% |
| v0, `b` and `c` shifted by 128 and 256 B | 3.2723 | 246.1 | 93.8% |
| v0, blocks swizzled into 8 clusters | 3.2599 | 247.0 | 94.1% |

The last two rows change nothing in v0's code path: one moves its buffers, the other reorders its blocks.

## Ruled out

| Explanation | Ruled out by |
|---|---|
| Instruction and scheduling overhead | SM clock cut 43%: v0 unchanged, gap shrank ([E4](EXPERIMENTS.md#e4-does-the-deficit-live-in-the-sm-accidental)) |
| Memory-level parallelism | SASS: both kernels keep 8 bytes in flight per thread ([E5](EXPERIMENTS.md#e5-is-it-memory-level-parallelism)) |
| Coalescing | Exact sector count, zero waste ([E6](EXPERIMENTS.md#e6-is-it-coalescing)) |
| Extra DRAM traffic | Counters: reads exact to 0.04% ([E1](EXPERIMENTS.md#e1-is-traffic-really-3n)) |
| Wave quantization | Tail loss ≤ 0.1% at the grids used ([Part 0](EXPERIMENTS.md#wave-quantization-does-the-grid-feed-the-sms-to-the-end)) |

## Reproduce

From the repository root. Every mode prints the environment header first, and every configuration is verified before it is timed.

```bash
make SM=sm_89
R=kernels/01_vector_add/results && mkdir -p $R

./bin/01_vector_add                  2>&1 | tee $R/baseline.txt        # E2, E4 at boost clock
./bin/01_vector_add --grid-sweep     2>&1 | tee $R/grid_sweep.txt      # E3
./bin/01_vector_add --swizzle-sweep  2>&1 | tee $R/swizzle_sweep.txt   # E7
for i in 1 2 3; do                                                     # E8, open question 1
  ./bin/01_vector_add --offset-sweep 2>&1 | tee $R/offset_sweep_$i.txt
done
./bin/01_vector_add --open           2>&1 | tee $R/open.txt            # open questions 2-4

# E4 at base clock: ncu pins 1545 MHz for the whole process; the harness times as usual
ncu --kernel-name vecadd_naive --launch-count 1 ./bin/01_vector_add 2>&1 | tee $R/clock_base.txt

# E1, E5, E6: one launch per configuration (v0, v1, v2, v0 shifted, swizzle S=8, v0 at the largest shift)
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,smsp__inst_executed.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
lts__t_sector_hit_rate.pct \
    --kernel-name regex:vecadd --launch-count 6 ./bin/01_vector_add --profile 2>&1 | tee $R/ncu.txt

cuobjdump -sass ./bin/01_vector_add > $R/sass.txt                     # E5
compute-sanitizer ./bin/01_vector_add --profile 2>&1 | tee $R/sanitizer.txt
```

## Raw results

| File | Experiment |
|---|---|
| `results/baseline.txt` | E2; E4 at boost clock |
| `results/clock_base.txt` | E4 at base clock |
| `results/grid_sweep.txt` | E3 |
| `results/ncu.txt` | E1, E5 (instruction counts), E6 |
| `results/sass.txt` | E5 |
| `results/swizzle_sweep.txt` | E7 |
| `results/offset_sweep_1.txt` to `_3.txt` | E8, open question 1 |
| `results/open.txt` | open questions 2–4 |
| `results/sanitizer.txt` | memory-safety check for every configuration |
