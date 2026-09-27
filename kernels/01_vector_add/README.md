# 01 · Vector add

`c[i] = a[i] + b[i]` · fp32 · n = 2²⁶ (256 MiB per array) · RTX 4060 Laptop GPU

**Full investigation: [EXPERIMENTS.md](EXPERIMENTS.md).** The model I built before measuring, four measurement experiments and eight on the kernel, what each could and couldn't distinguish, and a derivation for every number.

> Regenerated under the current harness and environment. The original numbers, and how they changed, are in [EXPERIMENTS.md, Part III](EXPERIMENTS.md#part-iii-regeneration).

## Summary

In the original runs, grid-stride vector add beat the naive kernel by 6% (247.4 vs 232.9 GB/s). Instruction overhead, memory-level parallelism, coalescing and wave quantization were each ruled out. The gap belonged to neither kernel: with the naive kernel untouched, shifting two of its three buffers by 128 and 256 bytes took it from 233 to 246 GB/s, and shifting them by 256 and 512 bytes brought the penalty back.

Regenerated with the current harness (CUDA 13.1, WSL2), the penalty reproduces exactly where it was found, 234.4 GB/s at the 256/512-byte shift in three of three fresh processes. But the default placement no longer lands on it, so naive and grid-stride now tie at ~247 GB/s, 94% of theoretical peak. Within one process, several configurations start in the slow state and switch to full speed a few seconds later, with nothing in the program changed.

For most of the investigation I asked why grid-stride was fast. The answer was that naive was slow, and whether it is slow depends on where its buffers land.

## Open question

With nothing in the program changed, several buffer placements start in the slow state and switch to full speed after one to six reps, and never switch back. Only the 256/512-byte shift stays slow in every rep. What changes underneath a running program: where Windows places the buffers in physical memory, an adaptive memory-controller policy, or a power state left over from idle? A native-Linux run would separate the first from the others ([R8](EXPERIMENTS.md#r8-offset-sweep-three-fresh-processes), [open questions](EXPERIMENTS.md#open-questions)).

## Results

Turbo mode (memory clock 8201 MHz, theoretical peak 262.4 GB/s), SM clock ~2730 MHz, median of accepted reps, noise σ ≈ 0.01% ([methodology](../../docs/methodology.md#deviation-flag)).

| Configuration | ms | GB/s | % of peak |
| --- | --- | --- | --- |
| v0 naive | 3.2663 | 246.5 | 93.9% |
| v1 grid-stride, 102,400 blocks | 3.2591 | 247.1 | 94.2% |
| v1 grid-stride, 131,072 blocks (best) | 3.2517 | 247.7 | 94.4% |
| v2 `float4` | 3.2928 | 244.6 | 93.2% |
| v0, `b` and `c` shifted by 256 and 512 B | 3.4348–3.4357 | 234.4–234.5 | 89.3% |
| swizzle, 4 clusters, same 256/512 B shift | 3.2692 | 246.3 | 93.9% |

The last two rows share one placement: the naive kernel pays the penalty there in every run, and reordering its blocks removes it. In the original runs, v0 ran at 232.9 GB/s and grid-stride won by 6%; [Part III](EXPERIMENTS.md#part-iii-regeneration) compares every number.

## Ruled out

| Explanation | Ruled out by |
| --- | --- |
| Instruction and scheduling overhead | Costs at most ~1.4% even with the SM clock cut 43%, far short of the ~5% penalty ([E4](EXPERIMENTS.md#e4-does-the-deficit-live-in-the-sm-accidental), [R4](EXPERIMENTS.md#r4-e4-in-one-session)) |
| Memory-level parallelism | SASS: both kernels keep 8 bytes in flight per thread; instruction counts match the listing exactly ([E5](EXPERIMENTS.md#e5-is-it-memory-level-parallelism), [R5](EXPERIMENTS.md#r5-profiler-counts)) |
| Coalescing | Exact sector count, 4 sectors per request ([E6](EXPERIMENTS.md#e6-is-it-coalescing)) |
| Extra DRAM traffic | Counters: reads within 0.06% ([E1](EXPERIMENTS.md#e1-is-traffic-really-3n)) |
| Wave quantization | Tail loss ≤ 0.1% at the grids used ([Part 0](EXPERIMENTS.md#wave-quantization-does-the-grid-feed-the-sms-to-the-end)) |

## Reproduce

From the repository root, on AC power in turbo mode, with GPU-using Windows apps closed. Every mode prints the environment header first, and every configuration is verified before it is timed.

```
make SM=sm_89
R=kernels/01_vector_add/results && mkdir -p $R

compute-sanitizer --tool memcheck ./bin/01_vector_add --profile 2>&1 | tee $R/sanitizer.txt   # memory safety

./bin/01_vector_add                  2>&1 | tee $R/baseline.txt        # E2, E4 at boost clock; start from an idle GPU
./bin/01_vector_add --grid-sweep     2>&1 | tee $R/grid_sweep.txt      # E3
./bin/01_vector_add --swizzle-sweep  2>&1 | tee $R/swizzle_sweep.txt   # E7
for i in 1 2 3; do                                                     # E8, open question 1
  ./bin/01_vector_add --offset-sweep 2>&1 | tee $R/offset_sweep_$i.txt
done
./bin/01_vector_add --open           2>&1 | tee $R/open.txt            # open questions 2-4

# E4 at base clock. Under WSL2 Nsight Compute does not pin the clock, so lock it from an
# administrator PowerShell on Windows first:   nvidia-smi -lgc 1545,1545
./bin/01_vector_add 2>&1 | tee $R/clock_base.txt
# ...then unlock:                               nvidia-smi -rgc

# E1, E5, E6: one launch per configuration
M=dram__bytes_read.sum,dram__bytes_write.sum,smsp__inst_executed.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,lts__t_sector_hit_rate.pct
ncu --metrics $M --kernel-name regex:vecadd --launch-count 6 ./bin/01_vector_add --profile 2>&1 | tee $R/ncu.txt
grep -c "n/a" $R/ncu.txt                                               # must print 0

cuobjdump -sass ./bin/01_vector_add > $R/sass.txt                     # E5
```

The checks behind the harness itself (noise floor, polling, clock locking) are described with their commands in [methodology](../../docs/methodology.md).

## Raw results

| File | Experiment |
| --- | --- |
| `results/predictions.md` | predictions for the regeneration, committed before running |
| `results/sanitizer.txt` | memory-safety check for every configuration |
| `results/noise.txt` | noise floor: 30 reps per configuration |
| `results/poll5_*.txt`, `results/poll0_*.txt` | polling check |
| `results/lgc_test.txt` | clock lock at 2400 MHz from the Windows host |
| `results/baseline.txt` | E2 and R2; E4 at boost clock; R3 cold start |
| `results/clock_base.txt` | E4 and R4 at 1545 MHz, locked from the Windows host |
| `results/clock_base_ncu.txt` | E4 under Nsight Compute, which no longer pins the clock |
| `results/grid_sweep.txt` | E3 and R6 |
| `results/ncu.txt` | E1, E5, E6 and R5 |
| `results/sass.txt` | E5 |
| `results/swizzle_sweep.txt` | E7 and R7 |
| `results/offset_sweep_1.txt` to `_3.txt` | E8 and R8, open question 1 |
| `results/open.txt` | open questions 2–4 |
