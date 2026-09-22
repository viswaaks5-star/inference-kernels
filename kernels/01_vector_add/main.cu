// 01 · vector add. Summary in README.md, full investigation in EXPERIMENTS.md.
//
//   01_vector_add                    E2   baseline: v0, v1 at G=768 and G=102400, v2
//   01_vector_add --grid-sweep       E3   v1 across grid sizes
//   01_vector_add --swizzle-sweep    E7   naive with a block swizzle, 1 to 65536 clusters
//   01_vector_add --offset-sweep     E8   naive with b and c shifted relative to a
//   01_vector_add --open                  open questions 2-4
//   01_vector_add --profile               one launch per configuration, for Nsight Compute
//
//   --poll-ms N    NVML sampling interval inside a rep (default 5; 0 = start and end only)
//
// Save a run with its environment header:
//   ./bin/01_vector_add --offset-sweep 2>&1 | tee kernels/01_vector_add/results/offset_sweep.txt

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "benchmark.cuh"
#include "cuda_utils.cuh"

namespace {

constexpr int kBlock = 256;
constexpr int kN = 1 << 26;                            // 2^26 floats: 256 MiB per array
constexpr size_t kBytes = static_cast<size_t>(kN) * sizeof(float);
constexpr size_t kTraffic = 3 * kBytes;                // read a, read b, write c (validated in E1)
constexpr int kMaxShift = 16384;                       // floats (64 KiB): the largest shift any buffer takes
constexpr size_t kAllocBytes = kBytes + static_cast<size_t>(kMaxShift) * sizeof(float);
constexpr int kGridStrideBlocks = 102400;              // v1's tuned grid (E3)

constexpr int kGridSweep[] = {768, 1536, 3072, 6144, 12288, 24576, 49152,
                              65536, 102400, 131072, 196608, 262144};
constexpr int kSwizzleSweep[] = {0, 1, 2, 3, 4, 5, 6, 10, 16};   // log2 of the cluster count
constexpr int kOffsetSweep[] = {0, 32, 64, 256, 1024, 8192};      // b's shift in floats; c gets twice this

static_assert(kN % 4 == 0, "v2 needs n divisible by 4");
static_assert((kN / kBlock) % (1 << 16) == 0, "the swizzle needs the grid divisible by every S");

// ----------------------------------------------------------------------- kernels

// v0: one thread per element.
__global__ void vecadd_naive(const float* __restrict__ a, const float* __restrict__ b,
                             float* __restrict__ c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// v1: grid-stride loop; the grid is sized to the machine, not to n.
__global__ void vecadd_grid_stride(const float* __restrict__ a, const float* __restrict__ b,
                                   float* __restrict__ c, int n) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        c[i] = a[i] + b[i];
}

// v2: 128-bit loads and stores. Needs 16-byte-aligned pointers and n divisible by 4.
__global__ void vecadd_vec4(const float4* __restrict__ a, const float4* __restrict__ b,
                            float4* __restrict__ c, int n4) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        const float4 x = a[i];
        const float4 y = b[i];
        c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
}

// E7: v0 with the block-to-chunk mapping permuted. The ~144 resident blocks form
// min(S, 144) clusters of contiguous blocks, spaced gridDim.x / S blocks apart.
// LOG2S = 0 is exactly v0. A bijection because gridDim.x is divisible by S.
template <int LOG2S>
__global__ void vecadd_swizzle(const float* __restrict__ a, const float* __restrict__ b,
                               float* __restrict__ c, int n) {
    constexpr int S = 1 << LOG2S;
    const int per = gridDim.x >> LOG2S;       // blocks per cluster
    const int g = blockIdx.x & (S - 1);       // which cluster
    const int r = blockIdx.x >> LOG2S;        // position within it
    const int i = (g * per + r) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

void launch_swizzle(int log2s, int grid, const float* a, const float* b, float* c, int n) {
    switch (log2s) {
        case 0:  vecadd_swizzle<0><<<grid, kBlock>>>(a, b, c, n);  return;
        case 1:  vecadd_swizzle<1><<<grid, kBlock>>>(a, b, c, n);  return;
        case 2:  vecadd_swizzle<2><<<grid, kBlock>>>(a, b, c, n);  return;
        case 3:  vecadd_swizzle<3><<<grid, kBlock>>>(a, b, c, n);  return;
        case 4:  vecadd_swizzle<4><<<grid, kBlock>>>(a, b, c, n);  return;
        case 5:  vecadd_swizzle<5><<<grid, kBlock>>>(a, b, c, n);  return;
        case 6:  vecadd_swizzle<6><<<grid, kBlock>>>(a, b, c, n);  return;
        case 10: vecadd_swizzle<10><<<grid, kBlock>>>(a, b, c, n); return;
        case 16: vecadd_swizzle<16><<<grid, kBlock>>>(a, b, c, n); return;
        default:
            std::fprintf(stderr, "no vecadd_swizzle instantiation for LOG2S = %d\n", log2s);
            std::exit(EXIT_FAILURE);
    }
}

// ---------------------------------------------------------------- configurations

enum class Variant { Naive, GridStride, Vec4, Swizzle };

struct Config {
    std::string label;
    Variant variant = Variant::Naive;
    int grid = 0;        // blocks
    int log2s = 0;       // swizzle only
    int shift_b = 0;     // floats from the start of b's allocation
    int shift_c = 0;     // floats from the start of c's allocation
};

std::string shift_label(int sb, int sc) {
    if (sb == 0 && sc == 0) return "";
    char buf[48];
    std::snprintf(buf, sizeof(buf), " b+%dB c+%dB", sb * 4, sc * 4);
    return buf;
}

Config v0(int sb = 0, int sc = 0) {
    return {"v0_naive" + shift_label(sb, sc), Variant::Naive, kN / kBlock, 0, sb, sc};
}
Config v1(int grid) {
    return {"v1_grid_stride G=" + std::to_string(grid), Variant::GridStride, grid, 0, 0, 0};
}
Config v2(int sb = 0, int sc = 0) {
    return {"v2_vec4" + shift_label(sb, sc), Variant::Vec4, kN / 4 / kBlock, 0, sb, sc};
}
Config swizzle(int log2s, int sb = 0, int sc = 0) {
    return {"swizzle S=2^" + std::to_string(log2s) + shift_label(sb, sc),
            Variant::Swizzle, kN / kBlock, log2s, sb, sc};
}

struct Context {
    gpu::DeviceInfo dev{};
    nvmlDevice_t nvdev{};
    gpu::BenchConfig bench{};
    std::vector<float> h_a, h_b, h_c;
    float* a0 = nullptr;          // allocations, each kMaxShift floats longer than the array
    float* b0 = nullptr;
    float* c0 = nullptr;
    int b_uploaded_at = -1;       // shift at which b's data currently sits
};

struct Ptrs {
    float* a;
    float* b;
    float* c;
};

// Point b and c at their shifted positions, uploading b if it moved. Shifts are
// multiples of 32 floats (128 B): coalescing and float4 alignment stay unchanged.
Ptrs place(Context& ctx, int sb, int sc) {
    for (int s : {sb, sc}) {
        if (s < 0 || s > kMaxShift || s % 32 != 0) {
            std::fprintf(stderr, "shift %d must be a multiple of 32 floats in [0, %d]\n", s, kMaxShift);
            std::exit(EXIT_FAILURE);
        }
    }
    if (ctx.b_uploaded_at != sb) {
        CUDA_CHECK(cudaMemcpy(ctx.b0 + sb, ctx.h_b.data(), kBytes, cudaMemcpyHostToDevice));
        ctx.b_uploaded_at = sb;
    }
    return {ctx.a0, ctx.b0 + sb, ctx.c0 + sc};
}

// Returns a callable that performs one launch of the configured kernel.
auto make_launch(const Config& cfg, const Ptrs& p) {
    const Variant variant = cfg.variant;
    const int grid = cfg.grid;
    const int log2s = cfg.log2s;
    return [=] {
        switch (variant) {
            case Variant::Naive:
                vecadd_naive<<<grid, kBlock>>>(p.a, p.b, p.c, kN);
                break;
            case Variant::GridStride:
                vecadd_grid_stride<<<grid, kBlock>>>(p.a, p.b, p.c, kN);
                break;
            case Variant::Vec4:
                vecadd_vec4<<<grid, kBlock>>>(reinterpret_cast<const float4*>(p.a),
                                              reinterpret_cast<const float4*>(p.b),
                                              reinterpret_cast<float4*>(p.c), kN / 4);
                break;
            case Variant::Swizzle:
                launch_swizzle(log2s, grid, p.a, p.b, p.c, kN);
                break;
        }
    };
}

// Relative tolerance: floats hold every integer only up to 2^24, and b[i] = 2i
// already rounds above that, so exact equality would fail on a correct kernel.
bool verify(const std::vector<float>& c) {
    for (int i = 0; i < kN; ++i) {
        const float expected = 3.0f * static_cast<float>(i);
        const float tol = 1e-5f * std::fabs(expected) + 1e-5f;
        if (std::fabs(c[i] - expected) > tol) {
            std::fprintf(stderr, "verification failed: c[%d] = %f, expected %f\n", i, c[i], expected);
            return false;
        }
    }
    return true;
}

void run(Context& ctx, const Config& cfg) {
    const Ptrs p = place(ctx, cfg.shift_b, cfg.shift_c);
    auto launch = make_launch(cfg, p);

    // Correctness before timing: clear c, launch once, check every element.
    CUDA_CHECK(cudaMemset(p.c, 0, kBytes));
    launch();
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(ctx.h_c.data(), p.c, kBytes, cudaMemcpyDeviceToHost));
    if (!verify(ctx.h_c)) {
        std::fprintf(stderr, "%s: wrong result, not timing it\n", cfg.label.c_str());
        std::exit(EXIT_FAILURE);
    }

    const gpu::Result r = gpu::benchmark(ctx.nvdev, launch, ctx.bench);
    gpu::print_result(cfg.label.c_str(), r, kTraffic, ctx.dev.mem_bus_bits);
}

void print_occupancy(const gpu::DeviceInfo& dev) {
    int n0 = 0, n1 = 0, n2 = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n0, vecadd_naive, kBlock, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n1, vecadd_grid_stride, kBlock, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n2, vecadd_vec4, kBlock, 0));
    std::printf("resident : blocks/SM v0 %d, v1 %d, v2 %d at %d threads/block (%d, %d, %d on the GPU)\n",
                n0, n1, n2, kBlock, n0 * dev.sm_count, n1 * dev.sm_count, n2 * dev.sm_count);
}

// One launch per configuration, in this order, so ncu --launch-count 6 profiles each once.
// The last one uses the largest shift, which is what compute-sanitizer most needs to see.
void profile(Context& ctx) {
    const std::vector<Config> cfgs = {v0(), v1(kGridStrideBlocks), v2(), v0(32, 64), swizzle(3),
                                      v0(kMaxShift / 2, kMaxShift)};
    for (size_t k = 0; k < cfgs.size(); ++k) {
        const Ptrs p = place(ctx, cfgs[k].shift_b, cfgs[k].shift_c);
        std::printf("profile launch %zu: %s\n", k + 1, cfgs[k].label.c_str());
        make_launch(cfgs[k], p)();
        CUDA_CHECK_LAST();
    }
}

void usage() {
    std::fprintf(stderr,
                 "usage: 01_vector_add [--baseline | --grid-sweep | --swizzle-sweep | "
                 "--offset-sweep | --open | --profile] [--poll-ms N]\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string mode = "--baseline";
    gpu::BenchConfig bench;
    for (int k = 1; k < argc; ++k) {
        const std::string arg = argv[k];
        if (arg == "--poll-ms" && k + 1 < argc) {
            bench.poll_ms = std::atoi(argv[++k]);
        } else if (arg == "--baseline" || arg == "--grid-sweep" || arg == "--swizzle-sweep" ||
                   arg == "--offset-sweep" || arg == "--open" || arg == "--profile") {
            mode = arg;
        } else {
            usage();
            return EXIT_FAILURE;
        }
    }

    const gpu::Nvml nvml;
    Context ctx;
    ctx.dev = gpu::query_device();
    ctx.nvdev = nvml.dev;
    ctx.bench = bench;

    gpu::print_device(ctx.dev);
    gpu::print_environment(ctx.nvdev);
    print_occupancy(ctx.dev);
    std::printf("problem  : n = %d floats, %.1f MiB per array, %.1f MB moved per launch, "
                "buffers padded by %zu KiB for shifts\n",
                kN, kBytes / (1024.0 * 1024.0), kTraffic / 1.0e6,
                static_cast<size_t>(kMaxShift) * sizeof(float) / 1024);
    if (bench.poll_ms > 0) std::printf("mode     : %s, clocks sampled every %d ms inside each rep\n", mode.c_str(), bench.poll_ms);
    else std::printf("mode     : %s, clocks sampled at start and end of each rep only\n", mode.c_str());

    ctx.h_a.resize(kN);
    ctx.h_b.resize(kN);
    ctx.h_c.assign(kN, 0.0f);
    for (int i = 0; i < kN; ++i) {
        ctx.h_a[i] = static_cast<float>(i);
        ctx.h_b[i] = 2.0f * static_cast<float>(i);
    }

    CUDA_CHECK(cudaMalloc(&ctx.a0, kAllocBytes));
    CUDA_CHECK(cudaMalloc(&ctx.b0, kAllocBytes));
    CUDA_CHECK(cudaMalloc(&ctx.c0, kAllocBytes));
    CUDA_CHECK(cudaMemcpy(ctx.a0, ctx.h_a.data(), kBytes, cudaMemcpyHostToDevice));

    if (mode == "--profile") {
        profile(ctx);
    } else {
        std::vector<Config> cfgs;
        if (mode == "--baseline") {
            cfgs = {v0(), v1(768), v1(kGridStrideBlocks), v2()};   // 768 = 32 x SMs, E2's original grid
        } else if (mode == "--grid-sweep") {
            for (int g : kGridSweep) cfgs.push_back(v1(g));
        } else if (mode == "--swizzle-sweep") {
            for (int l : kSwizzleSweep) cfgs.push_back(swizzle(l));
        } else if (mode == "--offset-sweep") {
            for (int s : kOffsetSweep) cfgs.push_back(v0(s, 2 * s));
        } else if (mode == "--open") {
            cfgs = {v0(),                               // control
                    v0(32, 0), v0(0, 32),               // Q2: which pair collides
                    v0(64, 128),                        // the anomaly, same process as the control
                    swizzle(2), swizzle(2, 64, 128),    // Q3: does the swizzle cure it at any placement
                    v2(), v2(32, 64)};                  // Q4: is v2 penalised too
        }
        gpu::print_header();
        for (const Config& cfg : cfgs) run(ctx, cfg);
        std::printf("\nall configurations verified before timing\n");
    }

    CUDA_CHECK(cudaFree(ctx.a0));
    CUDA_CHECK(cudaFree(ctx.b0));
    CUDA_CHECK(cudaFree(ctx.c0));
    return EXIT_SUCCESS;
}
