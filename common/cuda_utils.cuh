#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include <sys/utsname.h>

#include <cuda_runtime.h>
#include <nvml.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        const cudaError_t err_ = (call);                                          \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error at %s:%d\n  call:  %s\n  error: %s (%d)\n", \
                         __FILE__, __LINE__, #call, cudaGetErrorString(err_),     \
                         static_cast<int>(err_));                                 \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

// Launches are asynchronous. cudaGetLastError catches a bad launch configuration;
// the synchronize catches faults raised while the kernel ran. Never use inside a timed loop.
#define CUDA_CHECK_LAST()                                                         \
    do {                                                                          \
        CUDA_CHECK(cudaGetLastError());                                           \
        CUDA_CHECK(cudaDeviceSynchronize());                                      \
    } while (0)

#define NVML_CHECK(call)                                                          \
    do {                                                                          \
        const nvmlReturn_t r_ = (call);                                           \
        if (r_ != NVML_SUCCESS) {                                                 \
            std::fprintf(stderr, "NVML error at %s:%d\n  call:  %s\n  error: %s\n", \
                         __FILE__, __LINE__, #call, nvmlErrorString(r_));         \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

namespace gpu {

struct DeviceInfo {
    char name[256];
    int sm_count;
    int major, minor;
    int mem_bus_bits;
    int l2_bytes;
    int max_threads_per_sm;
};

inline DeviceInfo query_device(int device = 0) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    DeviceInfo d{};
    std::snprintf(d.name, sizeof(d.name), "%s", prop.name);
    d.sm_count = prop.multiProcessorCount;
    d.major = prop.major;
    d.minor = prop.minor;
    d.mem_bus_bits = prop.memoryBusWidth;
    d.l2_bytes = prop.l2CacheSize;
    d.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    return d;
}

// Theoretical DRAM bandwidth in GB/s. NVML reports the GDDR6 memory clock at half
// the per-pin data rate (8001 MHz <-> 16 Gbps), hence the factor of 2.
inline double peak_bandwidth_gbs(unsigned mem_mhz, int bus_bits) {
    return 2.0 * (bus_bits / 8.0) * mem_mhz / 1.0e3;
}

inline void print_device(const DeviceInfo& d) {
    std::printf("device   : %s (sm_%d%d, %d SMs, %d threads/SM)\n",
                d.name, d.major, d.minor, d.sm_count, d.max_threads_per_sm);
    std::printf("memory   : %d-bit bus, L2 %.1f MiB\n",
                d.mem_bus_bits, d.l2_bytes / (1024.0 * 1024.0));
}

struct Sample {
    unsigned mem_mhz{};
    unsigned sm_mhz{};
    unsigned long long reasons{};   // NVML clock-event ("throttle") reason bits
    bool reasons_known{};
};

inline Sample sample_clock(nvmlDevice_t d) {
    Sample s{};
    NVML_CHECK(nvmlDeviceGetClockInfo(d, NVML_CLOCK_MEM, &s.mem_mhz));
    NVML_CHECK(nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &s.sm_mhz));
    // Not every platform supports this query. Record "unknown" rather than
    // reporting "no reasons" when the query itself failed.
    s.reasons_known = nvmlDeviceGetCurrentClocksEventReasons(d, &s.reasons) == NVML_SUCCESS;
    if (!s.reasons_known) s.reasons = 0;
    return s;
}

// Names for NVML clock-event reason bits, so logs say which reason fired.
inline std::string clock_reasons_str(unsigned long long r, bool known = true) {
    if (!known) return "unknown";
    if (r == 0) return "none";
    static const struct { unsigned long long bit; const char* name; } kNames[] = {
        {0x001ULL, "GpuIdle"},
        {0x002ULL, "ApplicationsClocksSetting"},
        {0x004ULL, "SwPowerCap"},
        {0x008ULL, "HwSlowdown"},
        {0x010ULL, "SyncBoost"},
        {0x020ULL, "SwThermalSlowdown"},
        {0x040ULL, "HwThermalSlowdown"},
        {0x080ULL, "HwPowerBrakeSlowdown"},
        {0x100ULL, "DisplayClockSetting"},
    };
    std::string out;
    unsigned long long named = 0;
    for (const auto& k : kNames) {
        named |= k.bit;
        if (r & k.bit) {
            if (!out.empty()) out += '|';
            out += k.name;
        }
    }
    if (r & ~named) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "0x%llx", r & ~named);
        if (!out.empty()) out += '|';
        out += buf;
    }
    return out;
}

inline nvmlDevice_t nvml_device_for_cuda(int cuda_device = 0) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cuda_device));

    unsigned count = 0;
    NVML_CHECK(nvmlDeviceGetCount_v2(&count));
    for (unsigned i = 0; i < count; ++i) {
        nvmlDevice_t d{};
        NVML_CHECK(nvmlDeviceGetHandleByIndex_v2(i, &d));
        nvmlPciInfo_t pci{};
        NVML_CHECK(nvmlDeviceGetPciInfo_v3(d, &pci));
        if (static_cast<int>(pci.bus) == prop.pciBusID &&
            static_cast<int>(pci.device) == prop.pciDeviceID &&
            static_cast<int>(pci.domain) == prop.pciDomainID) {
            return d;
        }
    }
    std::fprintf(stderr, "no NVML device matches CUDA device %d\n", cuda_device);
    std::exit(EXIT_FAILURE);
}

inline std::string os_description() {
    std::ifstream f("/etc/os-release");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("PRETTY_NAME=", 0) == 0) {
            std::string v = line.substr(12);
            if (v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
            return v;
        }
    }
    return "unknown OS";
}

// Everything a reported number depends on, printed at the top of every log,
// so any saved result file carries its own reporting context.
inline void print_environment(nvmlDevice_t d) {
    int runtime = 0, driver_api = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaDriverGetVersion(&driver_api));

    char driver[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = {};
    NVML_CHECK(nvmlSystemGetDriverVersion(driver, static_cast<unsigned>(sizeof(driver))));

    utsname u{};
    if (uname(&u) != 0) std::snprintf(u.release, sizeof(u.release), "unknown");

    unsigned mem_now = 0, mem_max = 0, sm_max = 0;
    NVML_CHECK(nvmlDeviceGetClockInfo(d, NVML_CLOCK_MEM, &mem_now));
    NVML_CHECK(nvmlDeviceGetMaxClockInfo(d, NVML_CLOCK_MEM, &mem_max));
    NVML_CHECK(nvmlDeviceGetMaxClockInfo(d, NVML_CLOCK_SM, &sm_max));

    std::printf("software : CUDA runtime %d.%d, driver %s (supports CUDA %d.%d)\n",
                runtime / 1000, (runtime % 1000) / 10, driver,
                driver_api / 1000, (driver_api % 1000) / 10);
    std::printf("os       : %s, kernel %s\n", os_description().c_str(), u.release);
    std::printf("clocks   : memory %u MHz at startup (NVML max %u MHz), SM max %u MHz\n",
                mem_now, mem_max, sm_max);
}

// Owns NVML initialisation for the lifetime of the program.
struct Nvml {
    nvmlDevice_t dev{};

    explicit Nvml(int cuda_device = 0) {
        NVML_CHECK(nvmlInit_v2());
        dev = nvml_device_for_cuda(cuda_device);
    }
    ~Nvml() { nvmlShutdown(); }

    Nvml(const Nvml&) = delete;
    Nvml& operator=(const Nvml&) = delete;
};

}  // namespace gpu
