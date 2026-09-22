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

`--poll-ms 0` samples only at the start and end of each rep. It exists to check that sampling doesn't perturb the measurement. [TODO: run the baseline with `--poll-ms 5` and `--poll-ms 0`, and record here that the medians agree within noise.]

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

Accepted reps more than 0.5% from the median are listed under the result, and every result reports the spread between the fastest and slowest accepted rep. The 0.5% threshold is a guess, not a measured noise floor. [TODO: measure run-to-run variation (≥30 reps, clocks steady), set the threshold at 3σ, and record σ here.]

## Byte counts are measured, not assumed

GB/s is bytes moved divided by time, and "bytes moved" is a model. Before a kernel's bandwidth is reported, its byte model is checked against DRAM counters:

```bash
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ...
```

For vector add, the model (3N: read `a` and `b`, write `c`) matched reads to 0.04%. Writes came in 5% low in a single profiled launch because dirty lines are still resident in the 32 MiB L2 when the kernel ends. Across back-to-back launches, each launch's leftovers are written back during the next, so steady-state traffic is 3N. Details: [01, E1](../kernels/01_vector_add/EXPERIMENTS.md#e1-is-traffic-really-3n).

## Two ceilings

Bandwidth results are reported against both:

- **Theoretical peak**, from the live memory clock: 262.4 GB/s in turbo mode.
- **Empirical ceiling**, the highest bandwidth an access pattern reaches on this machine. For a 2-read/1-write stream it is ~247 GB/s (94.1%), reached by three unrelated configurations that converge on the same value ([01, conclusion](../kernels/01_vector_add/EXPERIMENTS.md#conclusion)).

The gap between the two is refresh, bus turnaround and memory-controller overhead. No configuration I have tried recovers any of it.

## Every log carries its own context

Each run begins with a header, printed by the program itself:

```
device   : NVIDIA GeForce RTX 4060 Laptop GPU (sm_89, 24 SMs, 1536 threads/SM)
memory   : 128-bit bus, L2 32.0 MiB
software : CUDA runtime X.Y, driver NNN.NN (supports CUDA X.Y)
os       : <distribution>, kernel <release>
clocks   : memory 8201 MHz at startup (NVML max 8001 MHz), SM max NNNN MHz
resident : blocks/SM v0 6, v1 6, v2 6 at 256 threads/block (144, 144, 144 on the GPU)
```

The memory line records the power mode indirectly: 8201 against an NVML maximum of 8001 means turbo. All output goes to stdout. Save runs with `2>&1 | tee` so errors land in the same file.

## Profiling rules

**Counts come from the profiler, time from the harness, and the two are never mixed.** Nsight Compute replays each kernel several times, flushes caches between replays and serializes execution. Its durations describe a different run from the harness's.

**`--profile` launches each configuration once**, in a fixed order, so `ncu --launch-count 6` profiles each exactly once. Timing loops would otherwise make ncu replay thousands of launches. The last configuration uses the largest buffer shift, so the same mode serves the memory-safety check below.

**Nsight Compute pins the SM clock.** By default (`--clock-control base`) it locks the SM clock, to 1545 MHz on this machine, for the whole profiled process, including kernels it isn't profiling. The harness keeps timing normally, so any profiling session doubles as a clock-sensitivity test. That side effect produced the strongest single experiment in 01 ([E4](../kernels/01_vector_add/EXPERIMENTS.md#e4-does-the-deficit-live-in-the-sm-accidental)).

**Metric names fail silently.** A misspelled metric returns `n/a` with no error. The separator between unit and quantity is a double underscore: `dram__bytes.sum`, not `dram_bytes.sum`.

**Memory safety is checked once per kernel** with `compute-sanitizer ./bin/<kernel> --profile`. Buffer-shift experiments write near the ends of their allocations, which is exactly what it catches. [TODO: run and record a clean result.]

**Evidence is committed as text.** Harness output, `ncu` console output and SASS dumps live in each kernel's `results/` directory. Binary `.ncu-rep` files are not committed: they are megabytes each and stay in git history permanently. Export the part a claim depends on.

## Machine characterization

Facts about the test machine that every kernel inherits. How each was established is recorded in [01, Part I](../kernels/01_vector_add/EXPERIMENTS.md#part-i-making-the-measurement-trustworthy).

### Memory clock is constant under load

NVML polled every ~0.6 s through idle (10.5 W, 39 °C) and sustained load (56 W, 60 °C) read 8201 MHz in every sample. The per-rep reading is defensive on this machine, since nothing has moved the memory clock yet. It stays, because on battery, under thermal stress or on another GPU something would.

[ATTACH: evidence/clock_idle_vs_load.txt]

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

[ATTACH: evidence/power_mode_toggle.txt]

### SM clock ramp

From idle, the SM clock needs roughly 0.5 s of sustained load to reach steady state (1890 → ~2730 MHz). Fixed warm-ups of 200–400 ms left rep 0 rejected in every run; ~500 ms cleared it. That measurement is why warm-up is now convergence-based rather than timed.

[ATTACH: evidence/warmup_sweep.txt]

### Clock-event reasons

Under sustained vector-add load NVML reports no reasons (`0x0`). At idle it reports `GpuIdle` (`0x1`). [TODO: run the baseline under Nsight Compute and record the decoded reason the harness now prints. Expected: `ApplicationsClocksSetting`, the profiler's own clock lock.]

## Limitations

- **Laptop GPU.** [TODO: confirm whether `sudo nvidia-smi -lgc` is refused on this machine. If it works, lock clocks and say so here; if not, say that clocks are gated and reported instead of locked.]
- **One machine.** Nothing here is claimed to transfer to another GPU.
- **DRAM latency is assumed**, ~500 ns, wherever Little's Law is applied. It is not measured; a pointer-chase microbenchmark would measure it.
- **No bank- or row-level DRAM counters** are exposed on this GPU, so effects below the memory-partition level are inferred, not observed.
- **NVML sampling runs on the host.** A clock excursion shorter than the 5 ms sampling interval can be missed.

## Reporting checklist

Every published number states: GPU, CUDA version, driver version, OS, power mode, memory clock, SM clock, problem size, dtype, what was timed, reps accepted out of reps taken, and the ceiling or baseline it is compared against. The log header covers all of these except power mode, which it implies through the memory clock.
