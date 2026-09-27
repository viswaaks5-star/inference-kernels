# 01 · Vector add: experiments

The full investigation behind the [summary](README.md): the model I built before measuring, every experiment with its hypothesis and prediction, what each one could and could not distinguish, and what went wrong. Format: [docs/experiment-format.md](../../docs/experiment-format.md) .

> [TODO: every number below was produced by an earlier version of the harness, before the fixes
> in this commit (clock sampling inside the timed region, convergence-based warm-up, true median,
> `__restrict__` on v0). Regenerate with the commands in [README.md](README.md#reproduce), update
> every number, then delete this note.]

## Reading guide

- **Two minutes:** [Summary](#summary) and the [hypothesis ledger](#hypothesis-ledger).
- **Fifteen minutes:** add [Part 0](#part-0-the-model-before-measuring), [E4](#e4-does-the-deficit-live-in-the-sm-accidental), [E5](#e5-is-it-memory-level-parallelism), [E7](#e7-is-it-the-address-to-block-mapping) and [E8](#e8-is-it-the-buffers-placement).
- **Everything:** the rest, and the [derivations](#appendix-derivations) behind every number.

## Contents

- [Summary](#summary)
- [Part 0: the model, before measuring](#part-0-the-model-before-measuring)
- [Part I: making the measurement trustworthy](#part-i-making-the-measurement-trustworthy) (M1–M4)
- [Part II: explaining the gap](#part-ii-explaining-the-gap) (E1–E8)
- [Conclusion](#conclusion)
- [Hypothesis ledger](#hypothesis-ledger)
- [Threats to validity](#threats-to-validity)
- [What transfers to inference](#what-transfers-to-inference)
- [Open questions](#open-questions)
- [Errors caught along the way](#errors-caught-along-the-way)
- [Timeline](#timeline)
- [Appendix: derivations](#appendix-derivations)
- [How this was done](#how-this-was-done)

## Summary

I wrote three versions of vector add (naive, grid-stride, `float4`) and predicted all three would land near the bandwidth ceiling. Grid-stride beat naive by 6% (247.4 vs 232.9 GB/s), and I set out to explain why.

Before explaining anything, I made the measurement trustworthy. Theoretical peak turned out not to be a constant: the laptop's turbo mode runs memory 200 MHz above the maximum NVML itself reports, so the harness reads the clock live. The SM clock needs half a second to settle, so warm-up waits for it. And the byte count every GB/s figure rests on was checked against DRAM counters.

Then I tested the standard explanations one at a time. Instruction overhead died when a profiler accidentally cut the SM clock by 43% and naive didn't slow down at all. Memory-level parallelism died when the SASS showed both kernels keep exactly 8 bytes in flight per thread. Coalescing died on an exact sector count. The arithmetic never supported wave quantization.

What survived was stranger. Remapping which block handles which chunk, with no loop and nothing else changed, reproduces grid-stride's full gain. And with the naive kernel untouched, shifting two of its three buffers by 128 and 256 bytes takes it from 233 to 246 GB/s. The gap was never a property of either kernel: naive pays a buffer-placement penalty, and grid-stride sidesteps it through the order in which it touches memory. Every penalty-free configuration lands at ~247 GB/s, 94.1% of theoretical peak.

For most of the investigation I asked why grid-stride was fast. The answer was that naive was slow.

## Part 0: the model, before measuring

### Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU: AD107, sm_89 |
| Compute | 24 SMs × 128 FP32 lanes = 3,072 lanes; SM clock ~2730 MHz under load |
| Memory | 8 GB GDDR6, 128-bit bus; 8201 MHz in turbo mode (NVML's maximum: 8001) |
| Theoretical bandwidth | 2 × 16 B × 8201 MHz = 262.4 GB/s ([A1](#a1-theoretical-bandwidth)) |
| L2 | 32 MiB |
| Occupancy | 1,536 threads/SM → 6 blocks of 256 → 144 resident blocks on the GPU |

Problem: `c[i] = a[i] + b[i]`, fp32, n = 2²⁶. That is 256 MiB per array and 805.3 MB moved per launch, 24× the L2.

The size is deliberate. My first version used n = 2²⁰, where the three arrays total 12 MiB and fit in the 32 MiB L2. Back-to-back launches would then have measured L2 bandwidth and reported it as DRAM bandwidth.

| Variant | Code | Grid × block | Per thread |
|---|---|---|---|
| v0 naive | one thread per element | 262,144 × 256 | 1 element |
| v1 grid-stride | loop, stride = grid × block | 768 × 256 initially; 102,400 × 256 after E3 | 341 elements; later 2–3 |
| v2 `float4` | 128-bit loads and stores | 65,536 × 256 | 4 elements |

All pointers are `__restrict__`. Every configuration is verified against a host reference before it is timed.

### Roofline: what limits it?

One addition per element; three 4-byte accesses per element (read `a[i]`, read `b[i]`, write `c[i]`). Arithmetic intensity is 1/12 ≈ 0.083 FLOP/B.

The ridge point, where compute and bandwidth limits cross, is the FP32 peak over the memory peak: 16.8 TFLOP/s ÷ 262.4 GB/s ≈ 64 FLOP/B ([A2](#a2-roofline)). Vector add sits about 770× to the left of it. At full bandwidth the SMs' arithmetic units are more than 99% idle.

Three things follow before any measurement. The metric is GB/s, not FLOP/s. The target is DRAM bandwidth. And no change to the arithmetic can matter; only how memory is accessed can.

### Little's Law: is enough data in flight?

A memory system delivering bandwidth B through latency L must have B × L bytes outstanding at every instant. At ~250 GB/s and an assumed ~500 ns loaded DRAM latency, that is ~125 KB ([A3](#a3-littles-law)). The latency is an assumption; I have not measured it.

The GPU holds 36,864 threads at once, so each needs only ~3.4 bytes in flight:

| | bytes in flight per thread | total | vs 125 KB |
|---|---|---|---|
| v0 | 8 (two 4-byte loads) | 295 KB | 2.4× |
| v1 | 8 (established by SASS in [E5](#e5-is-it-memory-level-parallelism)) | 295 KB | 2.4× |
| v2 | 32 (two 16-byte loads) | 1.18 MB | 9.4× |

Every variant clears the requirement. None should be latency-bound. If one underperforms, the cause should be found somewhere other than a shortage of parallelism.

### Wave quantization: does the grid feed the SMs to the end?

144 blocks run at once, so a grid of G blocks runs in W = G/144 waves. During the last, partial wave, part of the GPU idles. With equal-length waves, the lost fraction is at most (1 − frac(W)) / ⌈W⌉ ([A4](#a4-wave-quantization)).

| Grid | Waves | Upper bound on tail loss |
|---|---|---|
| 768 (v1 initial: 32 × SM count) | 5.33 | 11.1% |
| 65,536 (v2) | 455.1 | 0.2% |
| 102,400 (v1 after E3) | 711.1 | 0.1% |
| 262,144 (v0) | 1,820.4 | 0.03% |

At 768 blocks the tail is material, which is why I moved v1 off the 32 × SM-count heuristic. At every other size it is negligible.

### Predictions, written before the first run

1. All three variants land close to sustainable bandwidth. None is short of memory parallelism or of grid-level work.
2. v1 gains from lower scheduling and instruction overhead, by decoupling grid size from problem size.
3. v2 gains from wider loads cutting the number of memory instructions.

## Part I: making the measurement trustworthy

Every result in this notebook is bandwidth = bytes ÷ time, and every "% of peak" divides that by a theoretical peak. Each of the three can be wrong. This part covers time and peak; [E1](#e1-is-traffic-really-3n) covers bytes.

The facts established here belong to the machine, not to vector add, so [methodology](../../docs/methodology.md#machine-characterization) states them for every kernel. This part records how they were found.

### M1. Is peak bandwidth a constant?

**Hypothesis.** Peak bandwidth can be computed once, from the spec-sheet memory clock.

**Prediction.** If so, NVML reports the memory clock at 8001 MHz (16 Gbps) throughout a run.

**Setup.** The harness reads `NVML_CLOCK_MEM` during every rep and computes peak from that reading instead of a constant.

**Result.** NVML reported 8201 MHz, above the 8001 MHz that `clocks.max.memory` gives as the maximum.

**Verdict.** Falsified. The denominator must be read live. The difference is not cosmetic:

| Assumed memory clock | Peak | v1 at 247.4 GB/s |
|---|---|---|
| 8001 MHz | 256.0 GB/s | 96.6% |
| 8201 MHz | 262.4 GB/s | 94.3% |

A 2.3-point spread in the headline number, larger than some of the effects this notebook measures.

**Next.** Does the clock move during a run, and why is it above the maximum? (M2, M3)

### M2. Does the memory clock move under load?

**Hypothesis.** The memory clock changes with load or temperature, so a per-rep reading is necessary.

**Prediction.** If so, `nvidia-smi` shows different memory clocks at idle and under sustained load.

**Setup.** `nvidia-smi` polled about every 0.6 s, first idle, then under sustained vector add, then while cooling down.

**Result.**

| State | Memory | SM | Power | Temperature | Clock-event reasons |
|---|---|---|---|---|---|
| idle | 8201 MHz | 1890 MHz | 10.5 W | 39 °C | `0x1` GpuIdle |
| sustained load | 8201 MHz | 2715 MHz | 56.3 W | 60 °C | `0x0` none |
| cooling down | 8201 MHz | 1890 MHz | 10.9 W | 56 °C | `0x1` GpuIdle |

[ATTACH: ../../docs/evidence/clock_idle_vs_load.txt]

**Verdict.** Falsified on this machine: the memory clock never moved. The SM clock is what moves. Under load NVML reports no throttle reasons at all, so the measurement conditions are clean: 56 W and 60 °C are far from this GPU's limits.

The per-rep reading stays anyway. On battery, under thermal stress or on another GPU, the memory clock could move, and a harness that assumes otherwise fails silently.

An idle memory clock of 8201 MHz is unusual: most GPUs drop memory to a few hundred MHz when idle. A display attached to the GPU is the likely reason; I haven't confirmed it.

### M3. Why 8201 MHz when the maximum is 8001?

**Hypothesis.** The laptop vendor's "turbo" power mode applies a runtime memory overclock that the driver honours but NVML's maximum doesn't include.

**Prediction.** In the balanced power mode the memory clock reads 8001 MHz. In turbo it reads 8201. The maximum reads 8001 in both.

**Setup.** `nvidia-smi --query-gpu=clocks.current.memory,clocks.max.memory --format=csv`, once in each mode.

**Result.**

| Power mode | current | max |
|---|---|---|
| balanced | 8001 MHz | 8001 MHz |
| turbo | 8201 MHz | 8001 MHz |

[ATTACH: ../../docs/evidence/power_mode_toggle.txt]

**Verdict.** Supported. The +200 MHz is exactly a round offset, the signature of a configuration knob rather than measurement error. NVML's maximum comes from the VBIOS P-state table, which is static; the vendor's runtime offset never enters it. So NVML reports a current clock above its own maximum, and neither number is wrong.

The data had pointed here before the toggle confirmed it. Against the 8001 MHz peak, the best vector add would reach 96.6% of theoretical. A stream that both reads and writes GDDR6 loses bandwidth to refresh and to read/write bus turnaround, so 96.6% was already suspicious, and the smaller denominator was the likelier culprit.

**Consequences.** `clocks.max.memory` is not a usable ceiling on this machine. The power mode must be recorded with every result, since it moves the denominator by 2.5%. All numbers here are in turbo mode.

### M4. Why is the first rep always rejected?

**Hypothesis.** The SM clock is still ramping from idle when rep 0 starts. The harness rejects any rep whose SM clock moves by more than 2%, so rep 0 fails.

**Prediction.** Lengthening the warm-up clears the rejection past some threshold, and that threshold is the ramp time.

**Setup.** Warm-up of 100, 200, 300, 400 and 500 ms before rep 0. Warm-up ran in batches of 25 launches, about 85 ms each, so the effective warm-up is the first multiple of ~85 ms above the setting.

**Result.** Rep 0 was rejected at every setting up to 400 ms (~425 ms effective) and accepted at 500 ms (~510 ms effective).

[ATTACH: ../../docs/evidence/warmup_sweep.txt]

**Verdict.** Supported. From idle, the SM clock needs roughly half a second under load to settle (1890 → ~2730 MHz).

**What went wrong.** The next harness version used a fixed 480 ms warm-up, which is about 510 ms after batching: right at the edge of the threshold I had just measured. Rep 0 kept being rejected in most runs. A constant tuned to sit on a measured threshold is fragile by construction. It also encodes one machine's ramp as a number that is wrong on any other machine, or on this one when the GPU starts warm.

**Next.** The current harness launches until the SM clock holds within 1% for 200 ms, and reports how long that took. [TODO: record whether rep 0 is still rejected after regeneration, and the reason the harness prints if so.]

## Part II: explaining the gap

### E1. Is traffic really 3N?

**Hypothesis.** Each launch moves exactly 3N bytes: two reads and one write, nothing else.

**Prediction.** `dram__bytes_read.sum` = 536.87 MB and `dram__bytes_write.sum` = 268.44 MB. The alternative worth ruling out is write-allocate: if L2 fetched each line from DRAM before overwriting it, reads would approach 3N = 805 MB.

**Setup.** Nsight Compute, one profiled launch of v0 (`--profile` mode).

**Result.**

| | predicted | measured | Δ |
|---|---|---|---|
| DRAM read | 536.87 MB | 537.10 MB | +0.04% |
| DRAM write | 268.44 MB | 254.90 MB | −5.0% |

[ATTACH: results/ncu.txt]

**Verdict.** Supported, for the steady state the harness measures.

- **Reads are exact.** There is no write-allocate traffic, since a coalesced warp writes whole 32-byte sectors, and L2 has nothing to fetch before writing them. There is also no reuse: 512 MB of reads stream through a 32 MiB cache.
- **Writes are 13.5 MB short.** L2 is write-back: a dirty line reaches DRAM only when it is evicted. When the kernel ends, its last writes are still in L2. 13.5 MB is 40% of the cache; the rest held the read stream. In the harness's 200 back-to-back launches, each launch evicts the previous one's leftovers, so per-launch writes are N in steady state.

**An independent bound.** The ceiling constrains the byte model without any profiler. If true traffic exceeded 3N by a fraction x, true bandwidth would be 247.4 × (1 + x) GB/s, which cannot exceed 262.4. So x ≤ 6%, and write-allocate (x = 33%) is ruled out by arithmetic alone ([A10](#a10-the-ceiling-bounds-the-byte-model)).

**Can't distinguish.** One profiled launch, with caches flushed by the profiler, is the cold-boundary case. Steady-state writes are inferred, not measured.

**Next.** Profile a launch that directly follows another launch of the same kernel, with `--cache-control none`. Writes should come in near N.

### E2. Baseline

**Setup.** All three variants, turbo mode, SM clock ~2730 MHz, v1 at 768 blocks (`./bin/01_vector_add`, which also runs the tuned 102,400 used from E3 on).

**Result.**

| Variant | ms | GB/s | % of 262.4 |
|---|---|---|---|
| v0 naive | 3.4573 | 232.9 | 88.8% |
| v1 grid-stride, G = 768 | 3.2828 | 245.3 | 93.5% |
| v2 `float4` | 3.3942 | 237.3 | 90.4% |

[ATTACH: results/baseline.txt]

**Against the predictions.**

1. Held: all three sit between 88.8% and 93.5% of theoretical peak.
2. v1 is fastest, as predicted, though the explanation remained to be tested.
3. v2 gained only 1.9% over v0, far less than its 4× wider loads suggested.

**The question this raised,** and the one I pursued through E3–E6, was: why is v1 5.3% faster than v0? It turned out to be the wrong question.

### E3. Is it grid-level parallelism and overhead?

**Hypothesis.** v0 launches one thread per element: 262,144 blocks, each paying launch, index arithmetic and retirement for a single element. More grid-level parallelism helps until there are enough outstanding requests to feed the memory system. Past that point, the extra blocks only add scheduling and instruction overhead.

**Prediction.** Sweeping v1's grid size shows a peak. Too few blocks starve the memory system; too many add overhead.

**Setup.** v1 at 768, 1,536, 3,072, 6,144, 12,288, 24,576, 49,152, 65,536, 102,400, 131,072, 196,608 and 262,144 blocks (`--grid-sweep`). At 262,144 the stride equals n, so every thread runs exactly one iteration: v0's work plus the loop's setup, a control built into the sweep.

**Result.** Peak of 247.4 GB/s (94.3%) at 102,400 blocks, declining above ~200,000.

[ATTACH: results/grid_sweep.txt, as a table or plot]

**Verdict.** It looked supported. It was not a clean test.

**Can't distinguish.** For a fixed n, grid size is not one variable. Changing it moves three things together:

1. how many blocks are launched and retired;
2. elements per thread, which changes which compiled code path runs ([E5](#e5-is-it-memory-level-parallelism));
3. which addresses are in flight at the same time ([E7](#e7-is-it-the-address-to-block-mapping)).

Any of the three could produce this curve.

**What went wrong.** I read a confounded curve as confirmation of overhead. I also attributed the optimum at 102,400 to wave quantization, believing it gave a whole number of waves. It gives 711.1, and the tail bound there is 0.1%, far too small to explain a 1% difference.

**Next.** A test that changes the cost of instructions without changing anything else. It arrived by accident (E4).

### E4. Does the deficit live in the SM? (accidental)

**Hypothesis under test.** E3's reading: v0 loses to instruction and scheduling work.

**Prediction.** Instructions execute on the SM clock. If v0's deficit is instruction work, it scales with 1/f_SM. Cutting the SM clock from 2730 to 1545 MHz makes each instruction take 1.77× longer, so the absolute deficit should grow from 0.20 ms to about 0.36 ms.

**Setup.** Not planned. Nsight Compute's default `--clock-control base` pinned the SM clock at 1545 MHz for the entire profiled process, while the harness went on timing every kernel normally. The memory clock stayed at 8201 MHz. v1 at 102,400 blocks at both clocks.

**Result.**

| | SM 2730 MHz | SM 1545 MHz | change |
|---|---|---|---|
| v0 | 3.4573 ms · 232.9 GB/s | 3.4505 ms · 233.4 GB/s | +0.2% |
| v1 | 3.2549 ms · 247.4 GB/s | 3.2814 ms · 245.4 GB/s | −0.8% |
| v2 | 3.3940 ms · 237.3 GB/s | 3.3949 ms · 237.2 GB/s | −0.04% |
| v0 − v1 | 0.2024 ms | 0.1691 ms | shrank |

[ATTACH: results/baseline.txt, results/clock_base.txt]

[TODO: rerun both clocks in one session with identical configurations and replace this table.]

**Verdict.** Falsified. A 43% cut in the SM clock left v0 unchanged, and the deficit shrank instead of growing to 0.36 ms. Whatever costs v0 its 6% does not slow down when the SM clock does.

This is also the roofline confirmed by experiment. Until here, "memory-bound" was an inference from arithmetic intensity. Varying the compute clock by nearly half and seeing v0 and v2 not respond at all is direct evidence.

**What went wrong.** Nothing in the experiment; the lesson is about instrumentation. I noticed only because every result line printed the SM clock beside it, and the strongest test in this investigation was one I never designed. An earlier analysis of this data also compared v1 at 768 blocks (boost clock) with v1 at 102,400 (base clock): two variables at once, the same confound as E3. The table above compares like with like.

**Loose end.** v1 is not fully clock-insensitive: −0.8%. See [open question 5](#open-questions).

**Next.** If the cost isn't instructions, is it latency hiding? That is a memory-side explanation that could still differ between the kernels (E5).

### E5. Is it memory-level parallelism?

**Hypothesis.** v1's loop lets each thread keep loads from several iterations in flight, so it hides DRAM latency better than v0.

**Prediction.** GB/s tracks bytes in flight per thread, so the order is v0 < v1 < v2.

**Setup.** (a) Compare against v2, whose in-flight bytes are fixed by its instructions. (b) Read the SASS of all three kernels (`cuobjdump -sass`) and work out which code path v1 actually executes at 102,400 blocks.

**Result (a).** The order is v0 < v2 < v1. v2 has four times v0's bytes in flight and gains 1.9%. v1 gains 6.2%. If memory-level parallelism drove the gap, v2 would lead.

**Result (b).** The listing shows what each warp can execute. The scoreboard bits in each instruction's control word ([A6](#a6-reading-scoreboard-barriers-from-the-control-word)) show where it must stop and wait.

v0 issues both loads, then blocks on their shared barrier:

```
LDG.E.CONSTANT R4, [R4.64]        /* 0x000ea8... */   sets barrier 2
LDG.E.CONSTANT R3, [R2.64]        /* 0x000ea2... */   sets barrier 2
IMAD.WIDE R6, R6, R7, c[0x0][0x170]
FADD R9, R4, R3                   /* 0x004fca... */   waits on barrier 2
STG.E [R6.64], R9
```

Two 4-byte loads in flight: 8 bytes per thread.

v2 does the same with `LDG.E.128`: two 16-byte loads before its first `FADD` waits, so 32 bytes per thread. The `.CONSTANT` suffix on every load is `const __restrict__` visible in the machine code: the compiler proved the data read-only and routed it through the non-coherent path.

v1 is where the listing misleads. nvcc compiled the loop into two copies:

```
0x00e0  LOP3.LUT R7, RZ, R2, RZ, 0x33, !PT     R7 = ~(stride + i)
0x00f0  IADD3 R7, R7, c[0x0][0x178], R0        R7 = n - i - 1
        (0x00a0-0x01e0, 21 instructions in all, ends with R2 = (n - i - 1) / stride)
0x0200  ISETP.GE.U32.AND P1, PT, R2, 0x3, PT   P1 = (trip_count >= 4)
0x0210  LOP3.LUT P0, R4, R4, 0x3, RZ, 0xc0     R4 = trip_count & 3
0x0220  @!P0 BRA 0x330                         skip the prologue if no remainder
0x0280  scalar prologue: 2 LDG, FADD waits on both, STG, loop
0x0340  @!P1 EXIT                              leave unless trip_count >= 4
0x0350  unrolled body: 8 LDG in flight across barriers 2-5, then 4 FADD, 4 STG
```

The scalar prologue runs `trip_count mod 4` iterations, and the unrolled-by-4 body runs the rest, all or nothing. The trip count is `(n − i − 1)/stride + 1` ([A5](#a5-trip-count-from-the-sass)). At 102,400 blocks it is 3 for 14,680,064 threads and 2 for the other 11,534,336. `P1` is false for every thread in the grid.

The unrolled body with its eight loads never executes. Every thread runs the scalar prologue, whose `FADD` blocks until both loads return: 8 bytes in flight, identical to v0.

The block from `0x00a0` to `0x01e0` computes `n − i − 1` and divides it by the stride. The division exists because the stride is a runtime value and GPUs have no integer-divide instruction; the compiler synthesizes one from a floating-point reciprocal (`MUFU.RCP`), a refinement step and two conditional corrections. Every thread pays for those 21 instructions before issuing its first load.

[ATTACH: results/sass.txt, regenerated from this commit]

**Verdict.** Falsified twice. v0 and v1 keep identical bytes in flight, and the variant with the most is not the fastest.

**What went wrong.** I recognised that SASS is a static listing and that execution depends on runtime trip counts. But I estimated that part of the unrolled body would run, for up to 24 bytes per thread. Tracing the guard shows the body is all-or-nothing, and at this grid size it is never entered.

**Prediction still open.** From the SASS and the trip counts, v0 executes 16 instructions per warp, 33.55M in total. v1 executes 64 or 75, 57.48M in total ([A7](#a7-instruction-counts)). The faster kernel should execute 71% more instructions. [TODO: measure `smsp__inst_executed.sum` for both and record the result here, whether or not it matches.]

**Next.** The two kernels execute different instructions and have the same memory parallelism. Do they present the memory system with the same requests (E6)?

### E6. Is it coalescing?

**Hypothesis.** v0's loads are partly uncoalesced, fetching sectors they don't use.

**Prediction.** Fully coalesced, v0's loads fetch exactly 2 × 2²⁶ × 4 B ÷ 32 B = 16,777,216 sectors ([A8](#a8-sectors-per-request)). Any waste shows up as more.

**Setup.** `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`, one launch of v0. The metric name decodes as: L1/texture unit, sectors touched, through the load/store pipe, by global-memory loads, summed over all units.

**Result.** 16,777,216, exactly.

[ATTACH: results/ncu.txt]

**Verdict.** Falsified. No sector is wasted.

**What went wrong.** This was the only metric that survived its command. The other three names had typos (a single underscore where the grammar needs a double one, and "issus" for "issue"), and Nsight Compute returns `n/a` for an unknown metric instead of raising an error. The instrument failed silently, and it did so while I was trying to check another instrument.

[TODO: record sectors per request from `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`, now in the profiling command. Expected: 4 for v0 and v1, 16 for v2.]

**Next.** Instruction work, memory parallelism, coalescing and traffic are all ruled out as differences. What is left?

### E7. Is it the address-to-block mapping?

**Hypothesis.** v1's loop doesn't matter. What matters is which addresses are in flight at the same time. v0's resident blocks all work on one contiguous window. v1's blocks sit at different loop iterations, so they touch several windows far apart.

**Prediction.** A naive kernel that only permutes which block handles which chunk of the array, with no loop and one element per thread, reproduces v1's gain.

**Setup.** A swizzled v0 at 262,144 blocks (`--swizzle-sweep`):

```cuda
constexpr int S = 1 << LOG2S;
const int g = blockIdx.x & (S - 1);                  // which cluster
const int r = blockIdx.x >> LOG2S;                   // position within it
const int i = (g * (gridDim.x >> LOG2S) + r) * blockDim.x + threadIdx.x;
if (i < n) c[i] = a[i] + b[i];
```

The design holds every earlier suspect fixed:

- elements per thread: 1;
- bytes in flight per thread: 8;
- coalescing: every warp still reads 128 contiguous bytes;
- traffic: 3N;
- block count: 262,144;
- instructions: v0's plus a few integer operations;
- loop: none.

S = 1 is exactly v0. The only thing that changes is which addresses are concurrent. The ~144 resident blocks form min(S, 144) clusters of contiguous blocks, spaced G/S blocks apart ([A9](#a9-the-swizzle)).

**Result.**

| LOG2S | clusters | blocks per cluster | cluster spacing | ms | GB/s |
|---|---|---|---|---|---|
| 0 | 1 | 144 | n/a | 3.4564 | 233.0 |
| 1 | 2 | 72 | 128 MiB | 3.4579 | 232.9 |
| 2 | 4 | 36 | 64 MiB | 3.2614 | 246.9 |
| 3 | 8 | 18 | 32 MiB | 3.2599 | 247.0 |
| 4 | 16 | 9 | 16 MiB | 3.2632 | 246.8 |
| 5 | 32 | 4.5 | 8 MiB | 3.2560 | 247.3 |
| 6 | 64 | 2.25 | 4 MiB | 3.2616 | 246.9 |
| 10 | 144 | 1 | 256 KiB | 3.2633 | 246.8 |
| 16 | 144 | 1 | 4 KiB | 3.3197 | 242.6 |

[ATTACH: results/swizzle_sweep.txt]

**Verdict.** Supported. Permuting blocks alone reproduces v1's full gain. The effect is a step between two and four clusters, flat across a 16× range of cluster counts. Holding 144 single-block clusters fixed and packing them 4 KiB apart instead of 256 KiB brings part of the penalty back.

**Consistent with E3.** At 768 blocks, v1's trip count is ~341, so the unrolled body runs, and each thread keeps four iterations' loads in flight at once. Those are four concurrent address clusters, 768 KiB apart, and that configuration already reached 245.3 GB/s before any tuning.

**Can't distinguish.** Every thread still reads `a[i]` and `b[i]` and writes `c[i]` at the same `i`, so the distance between the three buffers is identical in every row of this table. The swizzle changes the schedule, meaning which addresses are concurrent, but not the placement. Either could be the real variable.

**Next.** Hold the schedule fixed at v0's and move the buffers instead (E8).

### E8. Is it the buffers' placement?

**Hypothesis.** The three 256 MiB buffers sit at relative offsets that make `a[i]`, `b[i]` and `c[i]` collide somewhere in the memory system. Grid-stride and the swizzle cannot remove the collision: for every `i`, `&b[i] − &a[i] = d_b − d_a`, a constant no access order can change. What they change is which addresses are concurrent, and that can make the collision cheap.

**Prediction.** With v0's mapping unchanged, shifting `b` and `c` relative to `a` recovers ~247 GB/s.

**Setup.** `b` is read from `b0 + OFF` and `c` written at `c0 + 2·OFF` (in floats), inside allocations 64 KiB longer than the arrays (`--offset-sweep`). OFF is a multiple of 32 floats, so every warp still reads 128 aligned bytes, and coalescing is untouched. `a` never moves. Kernel: v0.

**Result.**

| OFF (floats) | `b` shift | `c` shift | ms | GB/s |
|---|---|---|---|---|
| 0 | 0 | 0 | 3.4570 | 233.0 |
| 32 | 128 B | 256 B | 3.2723 | 246.1 |
| 64 | 256 B | 512 B | 3.4352 | 234.4 |
| 256 | 1 KiB | 2 KiB | 3.2615 | 246.9 |
| 1024 | 4 KiB | 8 KiB | 3.2697 | 246.3 |
| 8192 | 32 KiB | 64 KiB | 3.2696 | 246.3 |

In the OFF = 32 run one accepted rep sat 1.04% from the median; the other reps and runs stayed within 0.5%.

[ATTACH: results/offset_sweep_1.txt to _3.txt]

**Verdict.** Supported. Placement alone accounts for the gap: v0, with its kernel and schedule untouched, reaches 246–247 GB/s.

**The result is not monotonic.** 128/256 B removes the penalty, 256/512 B brings it back, and 1/2 KiB removes it again. So the penalty depends on particular relative offsets, not on how far apart the buffers are.

I separate what the data establishes from what it suggests:

- **Established:** the penalty depends on the buffers' relative placement, and certain offsets trigger it.
- **Consistent with:** an address-to-DRAM mapping under which those offsets land the three streams on conflicting resources. Modern GPUs hash physical address bits across memory channels and banks, and a hash produces exactly this kind of irregular pattern.
- **Plausible but unconfirmed:** the conflict is between DRAM rows. Each bank keeps one row open. Reaching a different row in the same bank costs a precharge and an activate, tens of nanoseconds during which that bank delivers nothing. If `a[i]`, `b[i]` and `c[i]` land in the same banks on different rows, the three streams keep closing each other's rows. Shifting `b` and `c` moves them to other banks. The swizzle leaves them colliding but spreads each moment's requests over more rows and banks, which gives the memory controller room to batch row hits. This GPU exposes no bank- or row-level counters, so I can't observe it.

It is not classical partition camping, which is load imbalance across memory partitions. Each stream's resident window here (~144 KiB per array) almost certainly spans every channel evenly.

**Replication.** Within the run, every accepted rep sat within 0.5% of the 234.4 median, so the OFF = 64 result is not noise inside one process. It has not yet been reproduced in fresh processes, where the driver may place the buffers at different physical addresses. The offsets themselves survive translation from virtual to physical addresses; the base addresses may not.

**What went wrong.** [TODO: if the first run of this sweep used 8 KB of padding, record here that at OFF = 8192 `c` was shifted 64 KiB, 56 KiB past its allocation. The allocator's 2 MiB rounding kept it from faulting, but that row came from an out-of-bounds configuration and has been rerun with 64 KiB padding. `compute-sanitizer` would have caught it.]

## Conclusion

1. **v0's deficit is a buffer-placement penalty, not a kernel inefficiency.** With the kernel and its schedule untouched, moving two buffers by a few hundred bytes recovers it ([E8](#e8-is-it-the-buffers-placement)).
2. **Grid-stride won by accident.** It cannot change the distance between buffers; it changes which addresses are in flight together, which makes the collision cheap. A block swizzle does the same with no loop at all ([E7](#e7-is-it-the-address-to-block-mapping)).
3. **Ruled out with evidence:** instruction and scheduling overhead ([E4](#e4-does-the-deficit-live-in-the-sm-accidental)), memory-level parallelism ([E5](#e5-is-it-memory-level-parallelism)), coalescing ([E6](#e6-is-it-coalescing)), traffic volume ([E1](#e1-is-traffic-really-3n)) and wave quantization ([Part 0](#wave-quantization-does-the-grid-feed-the-sms-to-the-end)).
4. **The practical ceiling is ~247 GB/s, 94.1% of 262.4.** Three unrelated routes converge on it: tuned grid-stride (247.4), a swizzle with four or more clusters (246.8–247.3) and buffer offsets (246.1–246.9). Convergence from different directions is the evidence. The remaining 6% goes to refresh, bus turnaround and controller overhead, and no configuration I've tried recovers any of it.
5. **Mechanism:** consistent with collisions in the address-to-DRAM mapping, most plausibly row-buffer conflicts among the three streams. Not confirmed below the partition level.

The first prediction in Part 0 held: every variant can reach the ceiling. The other two, about why v1 and v2 would gain, were wrong in instructive ways.

**Related work.** Ruetsch and Micikevicius, *Optimizing Matrix Transpose in CUDA* (NVIDIA, 2009), describe partition camping in older GPUs and fix it with diagonal block reordering. That is a block swizzle: the same family of fix, for a related but different effect.

## Hypothesis ledger

| | Hypothesis | Origin | Tested by | Verdict |
|---|---|---|---|---|
| H1 | Peak bandwidth is a constant from the spec sheet | own | M1 | falsified: 8201 vs 8001 MHz |
| H2 | The memory clock moves under load | own | M2 | falsified on this machine; check kept |
| H3 | 8201 MHz is a vendor turbo offset | own | M3 | supported |
| H4 | Rep 0 fails because the SM clock is still ramping | joint | M4 | supported: ~0.5 s ramp |
| H5 | Traffic exceeds 3N (write-allocate) | discussion | E1 | falsified: reads exact; ceiling bounds extra traffic at 6% |
| H6 | v0 loses to instruction and scheduling overhead | own | E3, E4 | falsified by E4 |
| H7 | Wave quantization explains the 102,400 optimum | own | arithmetic | falsified: tail ≤ 0.1% |
| H8 | v1 keeps more bytes in flight per thread | discussion | E5 | falsified: 8 B each, and v2 isn't fastest |
| H9 | v0's loads are partly uncoalesced | discussion | E6 | falsified: exact sector count |
| H10 | Which addresses are concurrent matters, not the loop | discussion | E7 | supported |
| H11 | Buffer placement causes a collision; concurrency changes its cost | discussion | E8 | supported; replication pending |

"Own" means I proposed it; "discussion" means it came out of working through the problem with Claude (see [How this was done](#how-this-was-done)); "joint" means both.

## Threats to validity

1. **The OFF = 64 anomaly is one process's run.** It is stable within that process, but untested across fresh allocations.
2. **Physical placement is invisible.** Offsets under a page survive translation from virtual to physical addresses; where the driver puts each 256 MiB buffer does not, and may differ between runs.
3. **Clocks are gated, not locked.** 3. **Clocks are gated, not locked.** A lock is accepted only from the Windows host, not from inside WSL2 ([methodology](../../docs/methodology.md#limitations)). Results are taken unlocked, so the harness rejects reps where clocks move instead of preventing movement.
4. **The harness may perturb what it measures.** It polls NVML every 5 ms during timed reps. Checked: medians with `--poll-ms 5` and `--poll-ms 0` agree within 0.03% in every configuration, inside the run-to-run range ([methodology](../../docs/methodology.md#how-a-rep-is-timed)). Excursions shorter than 5 ms are invisible.
5. **The ceiling is empirical.** ~247 GB/s is the best of the configurations I tried for a 2-read/1-write stream. A different read/write mix may reach more.
6. **The profiler measures a different run.** One launch, caches flushed, clocks pinned. Steady-state behaviour is inferred from it, not observed.
7. **The control-bit layout is reverse-engineered,** not documented by NVIDIA ([A6](#a6-reading-scoreboard-barriers-from-the-control-word)). Its decode was self-consistent across every instruction in these kernels, which is evidence, not proof.
8. **The DRAM latency in Part 0 is assumed.** At 1.2 µs instead of 500 ns, v0's Little's Law margin would be gone (0.98×) while v2's would still be 3.9×. E8 limits the concern in practice: once placement is fixed, v0 reaches the same ceiling as everything else, so 8 bytes per thread is enough whatever the latency is.
9. **One machine.** Nothing here is claimed to hold on another GPU.

## What transfers to inference

**The ceiling that matters is measured, not specified.** This card's spec sheet implies 256 GB/s; the live clock gives 262.4; kernels actually reach ~247. A tokens-per-second model built on the spec sheet or on NVML is off by up to 6% before any kernel runs.

**Batch-1 decode is this problem at scale.** Generating a token streams every weight through memory once. For 2.5 GB of fp16 weights, about 1.2B parameters, that caps decode near 247 ÷ 2.5 ≈ 99 tokens/s on this card, before KV-cache reads.

**Any kernel that streams several large buffers in lockstep has this exposure.** A fused gate-and-up projection reads two weight matrices together. Decode attention reads K and V caches together. A fused residual-add and RMSNorm reads two activation buffers and writes one. In each, how the buffers are placed relative to each other is an allocator decision, invisible in the kernel's code, and E8 shows it can be worth 6%.

**Paged KV caches change which addresses are concurrent by design.** Whether that helps the way the swizzle did, or hurts, is an empirical question for kernel 07.

**The measurement discipline carries over unchanged.** Power mode, clock state and warm-up move an end-to-end tokens/s number exactly as they move a GB/s number.

## Open questions

| # | Question | Test |
|---|---|---|
| 1 | Does the 256/512 B penalty reproduce? | `--offset-sweep` in three fresh processes |
| 2 | Which pair collides: the two reads, or a read and the write? | `--open`: shift only `b`, leaving `a` and `c` aligned; then only `c`, leaving `a` and `b` aligned |
| 3 | Does the swizzle cure the penalty at any placement? | `--open`: S = 4 with OFF = 64 |
| 4 | Is v2's modest gain the same effect? Its resident window is 576 KiB per array, 4× v0's. | `--open`: v2 with OFF = 32. If it reaches ~247, vectorization contributed almost nothing here |
| 5 | Why does v1 lose 0.8% at base clock when v0 doesn't? Candidate: the 21-instruction division block runs before each thread's first load, so its latency scales with the SM clock. | Same-session rerun of E4; then the swizzle, which has no division, at base clock should lose nothing |
| 6 | What is the loaded DRAM latency Part 0 assumes? | A pointer-chase microbenchmark |
| 7 | Does placement matter inside real inference kernels? | Measure in 03 (GEMV) and 07 (decode attention) |

For question 2, my prediction from the SASS is that `a`–`b` matters most: v0 issues those two loads back to back, while the store waits a full DRAM round trip. The counterpoint is that other warps' stores overlap in time with reads at nearby `i`, so I hold that prediction loosely.

## Errors caught along the way

- **The framing.** I asked why v1 was fast. Every hypothesis in E3–E6 examined what v1 did differently; the answer was in what the environment did to v0.
- **A confounded sweep read as confirmation** (E3). Grid size moved three variables at once.
- **Wave quantization cited for an optimum** where the tail bound is 0.1% (E3).
- **A static listing read as dynamic behaviour** (E5), until I traced the guard.
- **A clock comparison across two configurations** (E4): the same confound as E3, caught before publication.
- **Two definitions of tail loss mixed** in an early draft of Part 0. The notebook uses one formula throughout.
- **An overstated inference from the offset data** in an early draft: I claimed a simple bit-slice mapping couldn't produce it. With unknown base addresses one can. Only the non-monotonicity is robust.
- **A warm-up constant set on a measured threshold** (M4).
- **Silent tool failures:** typo'd metric names returning `n/a` (E6), a throttle query failure reported as "no throttle", and log messages naming causes nobody had checked. The current harness prints what it measured, not what it assumes.
- **Documentation written from the plan rather than the code.** An earlier methodology described a convergence-based warm-up the code didn't yet have, and v0 had lost `__restrict__` after its SASS was taken.

## Timeline

The experiments are numbered in the order the argument needs; this is the order they happened.

| Step | What happened | Section |
|---|---|---|
| 1 | Wrote v0, v1, v2; the three-question model; predictions | Part 0 |
| 2 | Baseline, v1 at 768 blocks | E2 |
| 3 | Grid sweep; concluded overhead | E3 |
| 4 | Made peak bandwidth live; found 8201 MHz | M1 |
| 5 | Polled clocks at idle and under load | M2 |
| 6 | Found rep 0 always rejected; swept warm-up | M4 |
| 7 | Toggled the power mode | M3 |
| 8 | First profiling run: DRAM traffic, and the SM clock pinned by accident | E1, E4 |
| 9 | Memory-level-parallelism hypothesis; learned to read SASS | E5 |
| 10 | Sector count, the one metric that survived a typo'd command | E6 |
| 11 | Swizzle sweep | E7 |
| 12 | Buffer-offset sweep | E8 |

## Appendix: derivations

### A1. Theoretical bandwidth

GDDR6 transfers two bits per pin per reported clock cycle (NVML's 8001 MHz ↔ 16 Gbps per pin).

```
peak = 2 × (bus bits / 8) × memory clock
     = 2 × 16 B × 8201 MHz = 262.4 GB/s        (turbo)
     = 2 × 16 B × 8001 MHz = 256.0 GB/s        (balanced)
```

### A2. Roofline

```
FP32 peak     = 24 SMs × 128 lanes × 2 FLOP/FMA × 2.73 GHz = 16.8 TFLOP/s
ridge point   = 16.8 TFLOP/s ÷ 262.4 GB/s ≈ 64 FLOP/B
intensity     = 1 FLOP ÷ 12 B ≈ 0.083 FLOP/B, about 770× below the ridge
```

### A3. Little's Law

```
bytes in flight   = bandwidth × latency = 250 GB/s × 500 ns ≈ 125 KB
resident threads  = 24 SMs × 6 blocks × 256 threads = 36,864
needed per thread = 125,000 B ÷ 36,864 ≈ 3.4 B
v0: 8 B × 36,864  = 295 KB   (2.4×)
v2: 32 B × 36,864 = 1.18 MB  (9.4×)
```

### A4. Wave quantization

With W = G ÷ 144 waves of equal length, a fraction 1 − frac(W) of the GPU's block slots sits empty during the final wave, which lasts 1/⌈W⌉ of the run:

```
tail loss ≤ (1 − frac(W)) ÷ ⌈W⌉
G = 768:      W = 5.33,    0.667 ÷ 6    = 11.1%
G = 102,400:  W = 711.1,   0.889 ÷ 712  = 0.1%
G = 262,144:  W = 1820.4,  0.556 ÷ 1821 = 0.03%
```

It is an upper bound: fewer resident blocks can still pull substantial bandwidth, so real tail loss is smaller.

### A5. Trip count from the SASS

Using `~x = −x − 1`:

```
R7 = ~(stride + i) + n + stride = −stride − i − 1 + n + stride = n − i − 1
trip_count = (n − i − 1) / stride + 1          (the number of i, i+s, i+2s, … below n)
```

At G = 102,400: stride = 102,400 × 256 = 26,214,400 and n = 67,108,864.

```
i <  n − 2·stride = 14,680,064  →  trip 3    (14,680,064 threads)
i ≥ 14,680,064                  →  trip 2    (11,534,336 threads)
check: 14,680,064 × 3 + 11,534,336 × 2 = 67,108,864 = n
```

The unrolled body needs `trip_count ≥ 4`, so no thread reaches it. At G = 768 the stride is 196,608 and trip counts are 341 or 342, so there the body runs.

### A6. Reading scoreboard barriers from the control word

Since Volta, each 128-bit instruction carries scheduling metadata computed by the compiler. The layout below is from reverse-engineering work (Jia et al., *Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking*, 2018); NVIDIA does not document it. The 23 control bits are the top bits of the second word the disassembler prints:

| Field | Bits | Meaning |
|---|---|---|
| stall | [3:0] | cycles before the next instruction issues |
| yield | [4] | hint to switch warps |
| write barrier | [7:5] | barrier set when the result lands (7 = none) |
| read barrier | [10:8] | barrier for operand reads (7 = none) |
| wait mask | [16:11] | barriers this instruction waits on |

v0's first load, `0x000ea8…`: dropping bit 40 from the top 24 bits gives `0b11101010100`: stall 4, yield, **sets barrier 2**, waits on nothing.

v0's add, `0x004fca…`: `0b00000000010011111100101`: stall 5, sets no barrier, **wait mask `0b000100`, barrier 2**.

The shortcut: a second word starting `000` waits on nothing; anything else stalls. In v1's unrolled body the four adds read `004`, `008`, `010` and `020`: waits on barriers 2, 3, 4 and 5, one per unrolled iteration. So eight loads are in flight when that body runs.

### A7. Instruction counts

Counted from the listing, including predicated-off instructions, which still issue:

```
v0: 16 per warp × 2,097,152 warps                        = 33,554,432
v1: 35 setup (incl. the 21-instruction division block) + 5 prologue entry + 2 exit = 42 fixed
    + 11 per scalar iteration
    trip 3: 75 per warp × 458,752 warps                  = 34,406,400
    trip 2: 64 per warp × 360,448 warps                  = 23,068,672
    total                                                = 57,475,072  (+71%)
```

Trip counts are uniform within each warp, since the boundary at 14,680,064 is a multiple of 32, so there is no divergence. If the measurement differs by a few instructions per warp, uniform-datapath instructions such as `ULDC` being counted differently is the likely reason.

### A8. Sectors per request

```
sectors  = 2 arrays × 2^26 elements × 4 B ÷ 32 B = 16,777,216
requests = 2 loads × 2,097,152 warps             = 4,194,304
sectors per request = 4: 32 threads × 4 B = 128 B = 4 sectors, the minimum
```

v2 moves 32 × 16 B = 512 B per request, so it should show 16.

### A9. The swizzle

For a block index b below S, `g = b` and `r = 0`, so the block lands at `b × (G/S)`. Resident blocks therefore form min(S, 144) clusters spaced G/S blocks apart. Each block covers 256 × 4 B = 1 KiB of each array, so the spacing in bytes is G/S KiB.

The mapping is a bijection whenever S divides G, which holds for every S used here since G = 2¹⁸: each of S clusters receives exactly G/S consecutive blocks. All nine instantiations were checked exhaustively.

### A10. The ceiling bounds the byte model

If true traffic were (1 + x) × 3N, true bandwidth would be 247.4 × (1 + x) GB/s, and it cannot exceed peak:

```
247.4 × (1 + x) ≤ 262.4   →   x ≤ 6.1%
write-allocate would mean x = 33% (4N instead of 3N): 329.9 GB/s, impossible
```

## How this was done

I worked through this investigation with Claude, Anthropic's AI model, as a sparring partner. The measurement work in Part I started from my own observations: the dynamic denominator, the 8201 MHz reading, the turbo hypothesis and the warm-up threshold. Several later hypotheses and experiment designs came out of that back-and-forth, including the memory-level-parallelism test, the control-bit and trip-count analysis of the SASS, the block swizzle and the buffer-offset control; the [hypothesis ledger](#hypothesis-ledger) marks which. Every experiment ran on my machine, and every number comes from my logs.
