NVCC      ?= nvcc
CUDA_HOME ?= /usr/local/cuda
# `native` detects the local GPU (CUDA >= 11.5, GPU present at build time).
# CI has no GPU, so it passes an explicit target: make SM=sm_89
SM        ?= native
NVCCFLAGS := -O3 -std=c++17 -arch=$(SM) -lineinfo -Icommon -DGIT_SHA=\"$(shell git rev-parse --short HEAD)\"
# NVML. The toolkit's stub library lets the link succeed on machines with no driver
# (CI); at run time the driver's libnvidia-ml.so.1 is loaded instead.
LDLIBS    := -L$(CUDA_HOME)/lib64/stubs -lnvidia-ml
BIN       := bin

SOURCES := $(wildcard kernels/*/main.cu)
TARGETS := $(patsubst kernels/%/main.cu,$(BIN)/%,$(SOURCES))

.PHONY: all run clean
all: $(TARGETS)

$(BIN)/%: kernels/%/main.cu $(wildcard common/*.cuh) | $(BIN)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

$(BIN):
	mkdir -p $(BIN)

run: all
	@for t in $(TARGETS); do echo "=== $$t"; ./$$t; echo; done

clean:
	rm -rf $(BIN)
