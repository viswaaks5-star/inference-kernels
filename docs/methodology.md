# Methodology

Every number in this repository comes from one harness ([`common/benchmark.cuh`](../common/benchmark.cuh)) under the rules below. The principle: the harness checks its own assumptions instead of trusting them, and prints the evidence next to every result. Where a check doesn't exist yet, it is listed under [limitations](#limitations).

## What is timed

The kernel launch only. Allocation, host↔device copies and verification sit outside every timed region. In an inference engine the weights are resident on the device before the first token; timing their upload would measure something that happens once.

## Correctness before timing

Every configuration is launched once on a zeroed output buffer and checked element by element against a host reference, with relative tolerance, before it is timed. A configuration that fails is never timed. Floats hold every integer exactly only up to 2²⁴, so exact equality would reject a correct kernel at n = 2²⁶.

## How a rep is timed

1. 25 warm-up launches, then a device synchronize, so the timed region starts on an idle stream.
2. An NVML clock sample, the start event, 200 back-to-back launches, the stop event.
3. While the GPU works through the queued launches, the host samples NVML every 5 ms until the stop event completes, then samples once more. A rep of ~680 ms gets ~130 samples, so a clock change anywhere inside the timed region is seen.
4. Time per launch is the event-measured time divided by 200.

CUDA events, not host clocks: a launch returns to the host as soon as the work is queued, so a host clock measures enqueue time, not execution.

Seven reps are taken. The reported time is the median of the accepted reps; with an even number accepted, the mean of the two middle values.

`--poll-ms 0` samples only at the start and end of each rep. It exists to check that sampling doesn't perturb the measurement. Checked: three runs of each setting, alternated in fresh processes ([results/](../kernels/01_vector_add/results/), `poll5_*.txt` and `poll0_*.txt`). The medians agree within the run-to-run range of each setting:

| Configuration | poll 0 (mean of 3 medians) | poll 5 | difference | run-to-run range |
|---|---|---|---|---|
| v0 naive | 3.2670 ms | 3.2672 ms | 0.005% | 0.03–0.04% |
| v1 grid-stride, G = 768 | 3.3024 ms | 3.3032 ms | 0.024% | 0.07–0.09% |
| v1 grid-stride, G = 102,400 | 3.2588 ms | 3.2586 ms | 0.004% | 0.01–0.02% |
| v2 `float4` | 3.2926 ms | 3.2928 ms | 0.006% | 0.01% |

Sampling every 5 ms does not measurably perturb the result. It does change what gets rejected: `--poll-ms 0` accepted all 84 reps and `--poll-ms 5` rejected 2 (both v0, the first configuration after startup), because sampling only at a rep's ends cannot see a clock dip in its middle. The default stays at 5 ms.

## Clock state

### The denominator is read live

Theoretical DRAM bandwidth is

```
peak = 2 × (bus width / 8) × memory clock
```

and on this machine the memory clock is not the spec-sheet value ([machine characterization](#machine-characterization)). The harness computes peak from the memory clock NVML reports during the timed region. If accepted reps saw different memory clocks, it uses the highest, which gives the least flattering "% of peak".

### Warm-up runs until the clock converges

The GPU idles at 1890 MHz SM clock and boosts to ~2730 MHz under load, which takes roughly half a second ([SM clock ramp](#sm-clock-ramp)). Measuring before the ramp finishes measures a slower GPU than the one being reported.

Before the first rep of every configuration, the harness launches the kernel in batches of four and samples the SM clock every 20 ms. It declares the clock steady when the last 10 samples (200 ms) agree within 1%, and never before 250 ms have passed. It gives up after 5 s and says so in the output. A fixed warm-up would encode one machine's ramp as a constant; convergence adapts to a cold GPU, a warm one, and a different power mode.

Every result line reports how long the ramp took and the clock it settled at.

### Rejection

A rep is rejected if any of these hold across its samples:

- **The memory clock changed at all.** It moves in discrete P-states, so any change is a state change, and the denominator depends on it.
- **The SM clock moved by more than 2%.**
- **NVML reported a hardware slowdown, hardware or software thermal slowdown, power-brake slowdown, or software power cap.** `GpuIdle`, `ApplicationsClocksSetting` (Nsight Compute's clock lock) and `DisplayClockSetting` are reported but don't reject.

Each rejected rep is printed with its clock ranges and decoded reasons. If the reasons query is unsupported, the log says "unknown" rather than "none".

### Deviation flag

Accepted reps more than 0.5% from the median are listed under the result, and every result reports the spread between the fastest and slowest accepted rep. The 0.5% threshold is a guess, not a measured noise floor. Accepted reps more than 0.2% from the median are listed under the result, and every result reports the spread between the fastest and slowest accepted rep. The threshold is measured: 30 reps of each baseline configuration, with the flag temporarily at 0 so every rep is printed ([results/noise.txt](../kernels/01_vector_add/results/noise.txt)). σ is estimated robustly, as 1.4826 × the median absolute deviation ([tools/sigma.py](../tools/sigma.py)).

| Configuration | σ | 3σ | drift, first half vs second half |
|---|---|---|---|
| v0 naive | 0.011% | 0.034% | +0.008% |
| v1 grid-stride, G = 768 | 0.070% | 0.209% | +0.012% |
| v1 grid-stride, G = 102,400 | 0.005% | 0.014% | −0.003% |
| v2 `float4` | 0.014% | 0.041% | −0.006% |

The flag sits at 3σ of the noisiest configuration, so normal noise never triggers it. Drift is below σ everywhere, so the spread is noise, not heating. G = 768 is the outlier, consistent with its 5.33 waves: with few, long-running blocks, when the last partial wave finishes varies from rep to rep (not confirmed). This is noise within one process; between processes the buffers' physical placement changes, which this measurement doesn't cover.
The calibrated flag earned its place on first use. In the offset sweeps it exposed reps that sit in one of two discrete states, ~3.27 and ~3.44 ms, within a single configuration ([01, R8](../kernels/01_vector_add/EXPERIMENTS.md#r8-offset-sweep-three-fresh-processes)). When reps split like that, the median reports whichever state held for at least four of seven reps, and only the flagged-rep list shows the other.

## Byte counts are measured, not assumed

GB/s is bytes moved divided by time, and "bytes moved" is a model. Before a kernel's bandwidth is reported, its byte model is checked against DRAM counters:

```bash
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ...
```

For vector add, the model (3N: read `a` and `b`, write `c`) matched reads to within 0.06% in every profiled configuration. Writes came in 5% low in a single profiled launch because dirty lines are still resident in the 32 MiB L2 when the kernel ends. Across back-to-back launches, each launch's leftovers are written back during the next, so steady-state traffic is 3N. Details: [01, E1](../kernels/01_vector_add/EXPERIMENTS.md#e1-is-traffic-really-3n).

## Two ceilings

Bandwidth results are reported against both:

- **Theoretical peak**, from the live memory clock: 262.4 GB/s in turbo mode.
- **Empirical ceiling**, the highest bandwidth an access pattern reaches on this machine. For a 2-read/1-write stream it is ~247 GB/s (94.1%), reached by three unrelated configurations that converge on the same value ([01, conclusion](../kernels/01_vector_add/EXPERIMENTS.md#conclusion)).

The gap between the two is refresh, bus turnaround and memory-controller overhead. No configuration I have tried recovers any of it.

## Every log carries its own context

Each run begins with a header, printed by the program itself:

```
GIT_SHA: <commit the binary was built from>
device   : NVIDIA GeForce RTX 4060 Laptop GPU (sm_89, 24 SMs, 1536 threads/SM)
memory   : 128-bit bus, L2 32.0 MiB
software : CUDA runtime X.Y, driver NNN.NN (supports CUDA X.Y)
os       : <distribution>, kernel <release>
clocks   : memory 8201 MHz at startup (NVML max 8001 MHz), SM max NNNN MHz
resident : blocks/SM v0 6, v1 6, v2 6 at 256 threads/block (144, 144, 144 on the GPU)
```

The clocks line is read at startup, before any load, when the GPU may still be idle: it has read 7001 MHz in turbo mode. The memory clock in each result line, sampled during the timed reps, is the one to trust; 8201 against an NVML maximum of 8001 means turbo. All output goes to stdout. Save runs with `2>&1 | tee` so errors land in the same file.

## Profiling rules

**Counts come from the profiler, time from the harness, and the two are never mixed.** Nsight Compute replays each kernel several times, flushes caches between replays and serializes execution. Its durations describe a different run from the harness's.

**`--profile` launches each configuration once**, in a fixed order, so `ncu --launch-count 6` profiles each exactly once. Timing loops would otherwise make ncu replay thousands of launches. The last configuration uses the largest buffer shift, so the same mode serves the memory-safety check below.

**Nsight Compute can pin the SM clock, but not here.** By default (`--clock-control base`) it locks the SM clock for the whole profiled process, including kernels it isn't profiling, while the harness keeps timing normally. In the original setup that pinned 1545 MHz and produced the strongest single experiment in 01 ([E4](../kernels/01_vector_add/EXPERIMENTS.md#e4-does-the-deficit-live-in-the-sm-accidental)). Under WSL2 it does not: every rep under Nsight Compute ran at 2715–2730 MHz ([01, R4](../kernels/01_vector_add/EXPERIMENTS.md#r4-e4-in-one-session)). Clock-sensitivity experiments now lock the clock from the Windows host ([limitations](#limitations)), and the SM clock column in every result line shows which clock actually applied.

**Metric names fail silently.** A misspelled metric returns `n/a` with no error. The separator between unit and quantity is a double underscore: `dram__bytes.sum`, not `dram_bytes.sum`.

**Memory safety is checked once per kernel** with `compute-sanitizer ./bin/<kernel> --profile`. Buffer-shift experiments write near the ends of their allocations, which is exactly what it catches. For 01,memcheck reports 0 errors across all six profile launches, including the largest shift ('c' + 64 Kib, which ends exactly at the end of its padding ). Under WSL2 the tool needs the Windows GPU debugger interface enabled (registry value `HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger\EnableInterface` = 1). Without it, the tool fails to attach but still prints an error summary, which looks like a result and isn't 

**Evidence is committed as text.** Harness output, `ncu` console output and SASS dumps live in each kernel's `results/` directory. Binary `.ncu-rep` files are not committed: they are megabytes each and stay in git history permanently. Export the part a claim depends on.

## Machine characterization

Facts about the test machine that every kernel inherits. How each was established is recorded in [01, Part I](../kernels/01_vector_add/EXPERIMENTS.md#part-i-making-the-measurement-trustworthy).

### Memory clock is constant under load

NVML polled every 0.5 s through idle (1.2–10.7 W, 42 °C), a baseline run (51–60 W, up to 53 °C) and cool-down read 8200–8201 MHz throughout, and exactly 8201 MHz under load ([evidence/clock_idle_vs_load.txt](evidence/clock_idle_vs_load.txt)). The exception is the start of a run after the GPU has been idle: the first rep can begin at 7001 MHz and step up to 8201 inside it. The per-rep check rejected two such reps during the regeneration ([01, R3](../kernels/01_vector_add/EXPERIMENTS.md#r3-rep-0-after-a-cold-start)), so it stays.

### Turbo mode sets the memory clock above NVML's maximum

```bash
nvidia-smi --query-gpu=clocks.current.memory,clocks.max.memory --format=csv
```

| Power mode | current | max |
|---|---|---|
| balanced | 8001 MHz | 8001 MHz |
| turbo | 8201 MHz | 8001 MHz |

The +200 MHz is a vendor runtime offset that the driver honours but the VBIOS P-state table, where NVML's maximum comes from, does not include. Two consequences: `clocks.max.memory` is not a usable ceiling on this machine, and the power mode must be recorded with every result, since it moves the denominator by 2.5%.

The data had flagged this before the mode toggle confirmed it. Against the 8001 MHz peak (256.0 GB/s), the best vector add would sit at 96.6% of theoretical, high enough for a read-and-write stream on GDDR6 to make the smaller denominator suspect.

*Raw output not preserved.*

### SM clock ramp

From idle, the SM clock needs roughly 0.5–1 s of sustained load to reach steady state (1890 → ~2730 MHz), and it climbs through intermediate values: 2415 MHz appears in the idle-to-load log above. Fixed warm-ups of 200–400 ms left rep 0 rejected in every run of the earlier harness, which is why warm-up is now convergence-based. Convergence has its own failure: after idle it can settle on an intermediate clock (2040–2115 MHz seen), hold it long enough to pass the 1%-for-200-ms test, and rep 0 is then rejected as the clock keeps climbing ([01, R3](../kernels/01_vector_add/EXPERIMENTS.md#r3-rep-0-after-a-cold-start)). The rejection gate catches it; no reported median includes such a rep.

### Clock-event reasons

Under sustained load NVML reports no reasons (`0x0`). At idle it reports `GpuIdle` (`0x1`). Under Nsight Compute in WSL2 no clock-setting reason appears, because Nsight Compute doesn't set the clock here ([01, R4](../kernels/01_vector_add/EXPERIMENTS.md#r4-e4-in-one-session)). Under a lock set from the Windows host with `nvidia-smi -lgc`, the harness reports `GpuIdle` on every result line rather than `ApplicationsClocksSetting`, so a lock is visible only in the SM clock column. `GpuIdle` also appears inconsistently in unlocked runs; it is reported, never used to reject.

## Limitations

- **Laptop GPU.** - **Clocks are gated, not locked.** From inside WSL2, `sudo nvidia-smi -lgc 2400,2400` is refused (`Unknown Error`, exit code 255). From an administrator PowerShell on the Windows host, the same lock is accepted and holds: every rep of the baseline ran at exactly 2400 MHz ([results/lgc_test.txt](../kernels/01_vector_add/results/lgc_test.txt)). Results are still taken unlocked, at the natural boost clock (~2715–2730 MHz), with the gating described above, to match the conditions under which the noise floor and the polling check were measured. At 2400 MHz, v0 and v1 run 0.10–0.16% slower and v2 is unchanged, so locked and unlocked numbers are not interchangeable. The lock remains available for experiments that vary the SM clock deliberately.
- **DRAM latency is assumed**, ~500 ns, wherever Little's Law is applied. It is not measured; a pointer-chase microbenchmark would measure it.
- **No bank- or row-level DRAM counters** are exposed on this GPU, so effects below the memory-partition level are inferred, not observed.
- **NVML sampling runs on the host.** A clock excursion shorter than the 5 ms sampling interval can be missed.
- **WSL2.** The Windows driver (WDDM) owns the GPU. Windows' video memory manager decides where buffers physically live, and clock locking, debugger access for compute-sanitizer and profiler clock control all go through the Windows host. Effects that depend on physical placement may differ on native Linux.
- **One machine.** Nothing here is claimed to transfer to another GPU.

## Reporting checklist

Every published number states: GPU, CUDA version, driver version, OS, power mode, memory clock, SM clock, problem size, dtype, what was timed, reps accepted out of reps taken, and the ceiling or baseline it is compared against. The log header covers all of these except power mode, which it implies through the memory clock.
