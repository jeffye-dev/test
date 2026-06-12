# Flash Attention CUDA Lab

This directory contains a minimal CUDA flash-attention-style example in `src/flash_attention_min.cu`.

## Prerequisites

- CUDA compiler: `/usr/local/cuda-12.9/bin/nvcc`
- CMake: `cmake >= 3.22`
- Nsight Compute CLI: `/usr/local/cuda-12.9/bin/ncu`
- Nsight Systems CLI: `/usr/local/cuda-12.9/bin/nsys`
- Target GPU architecture used here: `sm_100` for NVIDIA B200

## Install ncu-ui nsys-ui on local machine
https://developer.nvidia.com/nsight-compute
https://developer.nvidia.com/nsight-systems

## Build with nvcc

```bash
cd /root/develop/linke/test/cuda_lab/fa
mkdir -p build
/usr/local/cuda-12.9/bin/nvcc \
  -std=c++17 \
  -O3 \
  -arch=sm_100 \
  src/flash_attention_min.cu \
  -o build/flash_attention_min
```

Run the binary:

```bash
./build/flash_attention_min
```

Expected output should report a small max absolute error, for example:

```text
max abs err vs reference = 5.960e-07  (should be < 1e-3)
```

## Build with CMake

```bash
cd /root/develop/linke/test/cuda_lab/fa
cmake -S . -B build-cmake \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake
```

Run the CMake-built binary:

```bash
./build-cmake/flash_attention_min
```

The same numerical check should pass with max absolute error below `1e-3`.

## Profile with Nsight Compute

Nsight Compute (`ncu`) is used for deep single-kernel analysis. It replays the selected kernel multiple times and reports SM utilization, memory throughput, launch shape, occupancy, cache behavior, and optimization hints.

Run this example:

```bash
cd /root/develop/linke/test/cuda_lab/fa
mkdir -p profiles
/usr/local/cuda-12.9/bin/ncu \
  --force-overwrite \
  --target-processes all \
  --kernel-name flash_attn_min \
  --set basic \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section Occupancy \
  --export profiles/flash_attention_min_ncu \
  ./build/flash_attention_min
```

Open or inspect the generated report:

```bash
/usr/local/cuda-12.9/bin/ncu-ui profiles/flash_attention_min_ncu.ncu-rep
/usr/local/cuda-12.9/bin/ncu --import profiles/flash_attention_min_ncu.ncu-rep --page details
```

Observed output on B200 with CUDA 12.9:

```text
Report: profiles/flash_attention_min_ncu.ncu-rep
Kernel: flash_attn_min(const float *, const float *, const float *, float *, int)
Launch: grid=(256,1,1), block=(64,1,1), device CC=10.0
Duration: 63.90 us
Compute (SM) Throughput: 7.69%
Memory Throughput: 17.58%
DRAM Throughput: 0.05%
L1/TEX Cache Throughput: 19.37%
L2 Cache Throughput: 0.71%
Registers Per Thread: 40
Static Shared Memory Per Block: 33.28 KB
Theoretical Occupancy: 18.75%
Achieved Occupancy: 5.45%
Waves Per SM: 0.29
```

How to interpret this example:

- `Duration` is the measured GPU kernel time under NCU replay. It is higher than the NSYS timeline value because NCU instrumentation/replay adds overhead and measures a profiled pass.
- `Compute (SM) Throughput` is low, so the kernel does not keep the SM arithmetic pipelines busy.
- `DRAM Throughput` is near zero, while `L1/TEX Cache Throughput` is higher, because this tiny example reuses a small K/V tile in shared memory and does not stream enough data to stress HBM.
- `Waves Per SM = 0.29` means the whole launch is too small for a 148-SM B200. There are only 256 blocks, so the GPU cannot sustain many full waves of work.
- `Theoretical Occupancy = 18.75%` is limited mostly by shared memory: each block uses about `33.28 KB` static shared memory for `Ks`, `Vs`, `q`, `scores`, and `reduce`.
- `Achieved Occupancy = 5.45%` is even lower than theoretical occupancy because the workload is short and imbalanced; many threads wait at `__syncthreads()` or execute serial loops.
- NCU's optimization hints are expected for this teaching kernel: use more blocks/work, reduce shared memory per block, reduce synchronization, and replace serial reductions with warp/block reductions for a production implementation.

## Profile with Nsight Systems

Nsight Systems (`nsys`) is used for application-level timeline analysis. It shows CUDA API calls, kernel launches, memory copies, CPU runtime calls, and cross-thread timing. Use it before NCU when you need to understand where total application time goes.

Run this example:

```bash
cd /root/develop/linke/test/cuda_lab/fa
mkdir -p profiles
/usr/local/cuda-12.9/bin/nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  --output=profiles/flash_attention_min_nsys \
  ./build/flash_attention_min
```

Open or inspect the generated report:

```bash
/usr/local/cuda-12.9/bin/nsys-ui profiles/flash_attention_min_nsys.nsys-rep
/usr/local/cuda-12.9/bin/nsys stats profiles/flash_attention_min_nsys.nsys-rep
sqlite3 profiles/flash_attention_min_nsys.sqlite '.tables'
```

Observed output on B200 with CUDA 12.9:

```text
Generated:
  profiles/flash_attention_min_nsys.nsys-rep
  profiles/flash_attention_min_nsys.sqlite

CUDA API summary:
  cudaMalloc: 4 calls, total 412.884 ms
  cudaLaunchKernel: 1 call, total 716.265 us API time
  cudaMemcpy: 4 calls, total 288.164 us API time
  cudaDeviceSynchronize: 1 call, total 32.213 us API time
  cudaFree: 4 calls, total 316.118 us API time

CUDA GPU kernel summary:
  flash_attn_min(...): 1 instance, 32.320 us GPU time

CUDA GPU memory summary:
  Host-to-Device: 3 copies, total 0.197 MB, total 16.256 us
  Device-to-Host: 1 copy, total 0.066 MB, total 5.504 us
```

How to interpret this example:

- NSYS shows end-to-end application behavior. Here `cudaMalloc` dominates CPU-side CUDA API time because the program allocates device memory on the critical path.
- The actual GPU kernel is very short: one `flash_attn_min` launch takes about `32.320 us` in the timeline.
- H2D copies transfer Q/K/V, each about `N * D * sizeof(float) = 256 * 64 * 4 = 65536` bytes, so three H2D copies total about `0.197 MB`.
- D2H copies transfer O once, about `0.066 MB`.
- The warning about missing NVTX data is expected because this example does not call NVTX range APIs.
- For larger experiments, add NVTX ranges around allocation, H2D, kernel, D2H, and CPU reference code to make the NSYS timeline easier to read.

## NCU vs NSYS

- Use `nsys` first to answer: which phase dominates total runtime, how many kernels launch, whether memory copies overlap with compute, and whether CPU-side CUDA APIs are expensive.
- Use `ncu` next to answer: why a specific kernel is slow, whether it is compute-bound or memory-bound, what occupancy is achieved, how much shared memory/register pressure exists, and which source-level optimization hints apply.
- In this tiny teaching example, `nsys` shows the application is dominated by setup/API overhead, while `ncu` shows the single CUDA kernel is too small and low-occupancy for a B200.

## Generated Files

```text
build/flash_attention_min
build-cmake/flash_attention_min
profiles/flash_attention_min_ncu.ncu-rep
profiles/flash_attention_min_nsys.nsys-rep
profiles/flash_attention_min_nsys.sqlite
```

## Notes

- The current source emits a harmless CUDA warning for an unused shared variable `reduce`.
- If running on a different GPU architecture, change `sm_100` in the nvcc command and `CUDA_ARCHITECTURES` in `CMakeLists.txt` accordingly.
- `ncu` may take longer than normal execution because it replays kernels to collect metrics.
- `nsys --stats=true` can generate both `.nsys-rep` and `.sqlite`; the `.sqlite` file is useful for scripted analysis.
