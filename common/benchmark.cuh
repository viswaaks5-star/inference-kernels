#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_utils.cuh"

namespace gpu {

// Clock-event reasons that invalidate a rep. GpuIdle, ApplicationsClocksSetting
// (set by Nsight Compute's clock lock) and DisplayClockSetting are reported, not rejected.
constexpr unsigned long long kRejectReasons =
      nvmlClocksThrottleReasonHwSlowdown
    | nvmlClocksThrottleReasonSwPowerCap
    | nvmlClocksThrottleReasonSwThermalSlowdown
    | nvmlClocksThrottleReasonHwThermalSlowdown
    | nvmlClocksThrottleReasonHwPowerBrakeSlowdown;

struct BenchConfig {
    int reps = 7;                   // result = median over accepted reps
    int iters = 200;                // back-to-back launches per rep
    int rep_warmup = 25;            // launches before each rep; clocks are already steady
    int poll_ms = 5;                // NVML sampling interval inside a rep; <= 0 samples start and end only
    double sm_spread_tol = 0.02;    // reject a rep whose SM clock moved by more than 2%
    double deviation_flag = 0.002;  // flag accepted reps more than 0.5% from the median (not yet calibrated)
    // Warm-up ramp: launch until the SM clock stops moving.
    double ramp_sample_ms = 20.0;   // one SM clock sample per 20 ms of launches
    int ramp_window = 10;           // the last 10 samples (200 ms)...
    double ramp_tol = 0.01;         // ...must agree to within 1%
    double ramp_min_ms = 250.0;     // never declare convergence earlier than this
    double ramp_timeout_ms = 5000.0;
};

struct Ramp {
    double ms{};
    unsigned sm_mhz{};
    bool converged{};
};

struct Rep {
    int index{};
    float ms{};                     // per launch
    unsigned mem_lo{}, mem_hi{};
    unsigned sm_lo{}, sm_hi{};
    unsigned long long reasons{};
    bool reasons_known{true};
    int samples{};
    bool accepted{};
};

struct Result {
    float median_ms{}, min_ms{}, max_ms{};
    int accepted{}, total{};
    unsigned mem_lo{}, mem_hi{}, sm_lo{}, sm_hi{};
    unsigned long long reasons{};
    bool reasons_known{true};
    double samples_per_rep{};
    Ramp ramp{};
    std::vector<std::string> notes;   // rejected and flagged reps, printed under the result
};

namespace detail {
inline double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace detail

// Launch until the SM clock has stopped moving, instead of for a fixed time:
// the ramp differs between machines, power modes, and a cold or warm GPU.
template <typename LaunchFn>
Ramp ramp_to_steady_state(nvmlDevice_t nvdev, LaunchFn& launch, const BenchConfig& cfg) {
    const auto t0 = std::chrono::steady_clock::now();
    double last_sample_ms = -1.0e30;
    std::vector<unsigned> sm;
    Ramp r{};
    for (;;) {
        for (int i = 0; i < 4; ++i) launch();
        CUDA_CHECK_LAST();
        const double t = detail::ms_since(t0);
        if (t - last_sample_ms >= cfg.ramp_sample_ms) {
            last_sample_ms = t;
            sm.push_back(sample_clock(nvdev).sm_mhz);
            if (t >= cfg.ramp_min_ms && static_cast<int>(sm.size()) >= cfg.ramp_window) {
                const auto [lo, hi] = std::minmax_element(sm.end() - cfg.ramp_window, sm.end());
                if (*hi - *lo <= cfg.ramp_tol * *hi) {
                    r.converged = true;
                    break;
                }
            }
        }
        if (t > cfg.ramp_timeout_ms) break;
    }
    r.ms = detail::ms_since(t0);
    r.sm_mhz = sm.empty() ? 0 : sm.back();
    return r;
}

// One rep: `iters` back-to-back launches between two CUDA events, with the clocks
// sampled from NVML throughout the timed region, not only around it.
template <typename LaunchFn>
Rep time_rep(nvmlDevice_t nvdev, LaunchFn& launch, const BenchConfig& cfg, int index) {
    for (int i = 0; i < cfg.rep_warmup; ++i) launch();
    CUDA_CHECK_LAST();  // drain, so the sampling window below covers only the timed region

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<Sample> samples;
    samples.push_back(sample_clock(nvdev));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < cfg.iters; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));

    if (cfg.poll_ms > 0) {
        // The launches above were only queued. Sample while the GPU works through them.
        for (;;) {
            const cudaError_t q = cudaEventQuery(stop);
            if (q == cudaSuccess) break;
            if (q != cudaErrorNotReady) CUDA_CHECK(q);
            samples.push_back(sample_clock(nvdev));
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg.poll_ms));
        }
        // cudaErrorNotReady is a status, not a failure; clear it in case the runtime kept it.
        const cudaError_t residual = cudaGetLastError();
        if (residual != cudaSuccess && residual != cudaErrorNotReady) CUDA_CHECK(residual);
    } else {
        CUDA_CHECK(cudaEventSynchronize(stop));
    }
    samples.push_back(sample_clock(nvdev));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    Rep r{};
    r.index = index;
    r.ms = ms / static_cast<float>(cfg.iters);
    r.samples = static_cast<int>(samples.size());
    r.mem_lo = r.sm_lo = ~0u;
    for (const Sample& s : samples) {
        r.mem_lo = std::min(r.mem_lo, s.mem_mhz);
        r.mem_hi = std::max(r.mem_hi, s.mem_mhz);
        r.sm_lo = std::min(r.sm_lo, s.sm_mhz);
        r.sm_hi = std::max(r.sm_hi, s.sm_mhz);
        r.reasons |= s.reasons;
        r.reasons_known = r.reasons_known && s.reasons_known;
    }
    // The memory clock moves in discrete P-states, so any change at all is a state change.
    const bool mem_steady = r.mem_lo == r.mem_hi;
    const bool sm_steady = (r.sm_hi - r.sm_lo) <= cfg.sm_spread_tol * r.sm_hi;
    const bool no_bad_reason = (r.reasons & kRejectReasons) == 0;
    r.accepted = mem_steady && sm_steady && no_bad_reason;
    return r;
}

template <typename LaunchFn>
Result benchmark(nvmlDevice_t nvdev, LaunchFn&& launch, const BenchConfig& cfg) {
    Result res{};
    res.ramp = ramp_to_steady_state(nvdev, launch, cfg);

    std::vector<Rep> good;
    long total_samples = 0;
    char buf[192];
    for (int i = 0; i < cfg.reps; ++i) {
        const Rep r = time_rep(nvdev, launch, cfg, i);
        if (r.accepted) {
            good.push_back(r);
            total_samples += r.samples;
        } else {
            std::snprintf(buf, sizeof(buf),
                          "rep %d rejected: mem %u-%u MHz, SM %u-%u MHz (%.1f%%), reasons: %s",
                          r.index, r.mem_lo, r.mem_hi, r.sm_lo, r.sm_hi,
                          100.0 * (r.sm_hi - r.sm_lo) / r.sm_hi,
                          clock_reasons_str(r.reasons, r.reasons_known).c_str());
            res.notes.emplace_back(buf);
        }
    }
    res.total = cfg.reps;
    res.accepted = static_cast<int>(good.size());
    if (good.empty()) {
        for (const std::string& n : res.notes) std::printf("    %s\n", n.c_str());
        std::fprintf(stderr, "all %d reps rejected; no result\n", cfg.reps);
        std::exit(EXIT_FAILURE);
    }

    std::sort(good.begin(), good.end(), [](const Rep& x, const Rep& y) { return x.ms < y.ms; });
    const size_t k = good.size();
    // True median: with an even count (typically 6 of 7 accepted), average the two middle values.
    res.median_ms = (k % 2 == 1) ? good[k / 2].ms : 0.5f * (good[k / 2 - 1].ms + good[k / 2].ms);
    res.min_ms = good.front().ms;
    res.max_ms = good.back().ms;
    res.samples_per_rep = static_cast<double>(total_samples) / static_cast<double>(k);

    res.mem_lo = res.sm_lo = ~0u;
    for (const Rep& g : good) {
        res.mem_lo = std::min(res.mem_lo, g.mem_lo);
        res.mem_hi = std::max(res.mem_hi, g.mem_hi);
        res.sm_lo = std::min(res.sm_lo, g.sm_lo);
        res.sm_hi = std::max(res.sm_hi, g.sm_hi);
        res.reasons |= g.reasons;
        res.reasons_known = res.reasons_known && g.reasons_known;
        const double dev = (g.ms - res.median_ms) / res.median_ms;
        if (std::fabs(dev) > cfg.deviation_flag) {
            std::snprintf(buf, sizeof(buf),
                          "rep %d: %.4f ms, %+.2f%% from the median (flag threshold %.1f%%)",
                          g.index, g.ms, 100.0 * dev, 100.0 * cfg.deviation_flag);
            res.notes.emplace_back(buf);
        }
    }
    return res;
}

inline double bandwidth_gbs(size_t bytes, float ms) {
    return static_cast<double>(bytes) / (static_cast<double>(ms) * 1.0e6);
}

inline void print_header() {
    std::printf("\n%-30s %9s %8s %7s %9s %11s %5s %7s\n",
                "kernel", "ms", "GB/s", "%peak", "mem MHz", "SM MHz", "reps", "spread");
    std::printf("%s\n", std::string(92, '-').c_str());
}

// %peak uses the highest memory clock seen in any accepted rep: the conservative denominator.
inline void print_result(const char* name, const Result& r, size_t bytes, int bus_bits) {
    const double gbs = bandwidth_gbs(bytes, r.median_ms);
    const double peak = peak_bandwidth_gbs(r.mem_hi, bus_bits);
    char mem[24], sm[24], reps[16];
    if (r.mem_lo == r.mem_hi) std::snprintf(mem, sizeof(mem), "%u", r.mem_hi);
    else std::snprintf(mem, sizeof(mem), "%u-%u", r.mem_lo, r.mem_hi);
    std::snprintf(sm, sizeof(sm), "%u-%u", r.sm_lo, r.sm_hi);
    std::snprintf(reps, sizeof(reps), "%d/%d", r.accepted, r.total);
    std::printf("%-30s %9.4f %8.1f %6.1f%% %9s %11s %5s %6.2f%%\n",
                name, r.median_ms, gbs, 100.0 * gbs / peak, mem, sm, reps,
                100.0 * (r.max_ms - r.min_ms) / r.median_ms);
    std::printf("%-30s ramp %.0f ms to SM %u MHz%s, %.0f clock samples/rep, reasons: %s\n",
                "", r.ramp.ms, r.ramp.sm_mhz, r.ramp.converged ? "" : " (NOT converged)",
                r.samples_per_rep, clock_reasons_str(r.reasons, r.reasons_known).c_str());
    for (const std::string& note : r.notes) std::printf("%-30s %s\n", "", note.c_str());
}

}  // namespace gpu
