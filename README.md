# cuda-kernels

CUDA kernels for LLM inference, written from scratch and measured on a laptop GPU. For each one I try to explain why the number is what it is, and to write down exactly where the explanation stops.

Each kernel directory is a lab notebook: a short README with the finding, and an `EXPERIMENTS.md` with the model I wrote before measuring, each hypothesis and what it predicted, what the hardware said, and what went wrong. Falsified hypotheses stay in.

## Latest finding: [01 vector add](kernels/01_vector_add/)

Grid-stride vector add beat the naive kernel by 6% (247.4 vs 232.9 GB/s). Instruction overhead, memory-level parallelism, coalescing and wave quantization were each ruled out. The gap turned out to belong to neither kernel. With the naive kernel untouched, offsetting two of its three buffers by 128 and 256 bytes takes it from 233 to 246 GB/s: it was paying a buffer-placement penalty, and grid-stride sidestepped it through the order in which it touches memory. Every penalty-free configuration lands at ~247 GB/s, 94.1% of theoretical peak. [Full investigation →](kernels/01_vector_add/EXPERIMENTS.md)

## Why start with bandwidth

At batch size 1, generating each token streams every weight through the memory system once. Decode is a bandwidth problem before it is anything else. On this card, 2.5 GB of fp16 weights (about 1.2B parameters) caps batch-1 decode near 247 ÷ 2.5 ≈ 100 tokens/s, before KV-cache reads. Knowing the real ceiling, and what quietly costs 6% of it, comes before writing a GEMV or an attention kernel.

## Kernels

| # | Kernel | Bound by | Result | Status |
|---|---|---|---|---|
| 01 | [vector add](kernels/01_vector_add/) | DRAM bandwidth | ceiling ~247 GB/s (94.1%); naive's 6% deficit traced to buffer placement | done |
| 02 | reduction | DRAM bandwidth | | next |
| 03 | GEMV (decode) | DRAM bandwidth | | |
| 04 | RMSNorm | DRAM bandwidth | | |
| 05 | softmax (online) | DRAM bandwidth | | |
| 06 | RoPE | DRAM bandwidth | | |
| 07 | decode attention | DRAM bandwidth | | |
| 08 | GEMM (prefill) | tensor-core throughput | | |
| 09 | prefill attention | tensor-core throughput | | |
| 10 | weight-only int4 GEMV | DRAM bandwidth | | |

The order follows the decode path. On an 8 GB laptop GPU, batch-1 decode is the realistic workload, and every kernel through 07 is judged against the ceiling measured in 01. A GEMV row is a dot product, so reduction comes first. Compute-bound prefill kernels come after.

The inference engine these kernels feed into will live in its own repository.

## Hardware

NVIDIA GeForce RTX 4060 Laptop GPU: AD107, sm_89, 24 SMs, 128-bit GDDR6, 32 MiB L2 · CUDA [TODO] · driver [TODO] · [TODO: OS]

Every results file begins with a header the program prints itself (CUDA and driver versions, OS, memory clock against NVML's maximum, resident blocks), so the line above can be copied from any of them.

Numbers are taken in the laptop vendor's turbo power mode, which raises the memory clock from 8001 to 8201 MHz, above the maximum NVML itself reports. That offset moves every "% of peak" figure by 2.5%, so the harness reads the live clock instead of trusting a spec sheet. Details in [methodology](docs/methodology.md#machine-characterization).

## Method

Timing uses CUDA events, never host clocks. Warm-up runs until the SM clock converges rather than for a fixed time. Inside every timed rep the harness samples NVML every 5 ms; a rep is rejected if the memory clock moves at all, the SM clock moves more than 2%, or NVML reports a slowdown, and each rejection is printed with its reason. Theoretical peak is computed from the memory clock seen during the run. A kernel's byte model is checked against DRAM hardware counters before any bandwidth figure built on it is reported, and profiler counts are never mixed with harness times.

Full rules: [docs/methodology.md](docs/methodology.md). Notebook format: [docs/experiment-format.md](docs/experiment-format.md).

## Build and run

Requires the CUDA toolkit (11.8+ for sm_89) and NVML (the header ships with the toolkit, the library with the driver).

```bash
git clone git@github.com:[TODO]/cuda-kernels.git
cd cuda-kernels
make                          # builds every kernel for the local GPU
./bin/01_vector_add           # baseline; see the kernel's README for every mode
```

CI compiles for sm_75, sm_86 and sm_89. GitHub's runners have no GPU, so it proves the code builds; every number comes from the machine above.

## License

MIT
