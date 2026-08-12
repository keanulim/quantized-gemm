# quantized-gemm

CUDA INT8 general matrix multiply (GEMM) with per-tensor quantization, shared-memory tiling, CPU/GPU validation, and a cuBLAS baseline.

**Version:** v1 (complete)

## v1 scope

v1 implements a complete INT8 GEMM pipeline on GPU:

1. **CPU scale finding** — symmetric per-tensor scaling (`scale = max(|x|) / 127`)
2. **GPU quantize** — FP32 → INT8 elementwise kernels
3. **GPU tiled GEMM** — 16×16×16 shared-memory tiles, INT32 accumulation
4. **GPU dequantize** — INT32 → FP32 with `scale_a * scale_b`
5. **CPU FP32 reference** — end-to-end correctness baseline
6. **cuBLAS INT8 baseline** — `cublasGemmEx` with INT32 output for perf comparison
7. **Modal runners** — test and benchmark on cloud GPU (no local `nvcc` required)

Not in v1: vectorized loads, CUTLASS, FP8, GPU scale reduction.

## Pipeline

```
FP32 A, B (host)
    │  CPU: find scale_a, scale_b
    ▼
H2D copy
    ▼
quantize_a / quantize_b  →  INT8 A_q, B_q
    ▼
gemm (tiled)             →  INT32 C_q
    ▼
dequantize               →  FP32 C
    ▼
D2H copy
```

Math:

```
C ≈ scale_a × scale_b × (A_q @ B_q)
```

Matrices are **row-major**:

- `A[M×K]`, `B[K×N]`, `C[M×N]`
- `A[row,col] = A[row * K + col]`

## Tiled GEMM (custom kernel)

- **Block:** 16×16 threads
- **Grid:** one block per 16×16 output tile of `C`
- **Shared memory:** `As[16×16]`, `Bs[16×16]` per k-tile
- **Outer loop:** `kStart` steps along K in chunks of 16
- **Inner loop:** each thread `(ty, tx)` accumulates  
  `acc += As[ty][k] * Bs[k][tx]`
- **Output:** INT32 accumulator per element (never INT8)

## cuBLAS baseline

Benchmark uses `cublasGemmEx` with:

- Inputs: `CUDA_R_8I`
- Output: `CUDA_R_32I`
- Compute: `CUBLAS_COMPUTE_32I`

Row-major GEMM is mapped via `CUBLAS_OP_T` on both operands. A 128×128 correctness check verifies custom and cuBLAS INT32 outputs match exactly before timing.

## Project layout

```
src/
  reference/          CPU FP32 GEMM + test helpers
    gemm.h / gemm.cpp
    main.cpp
  kernels/
    gemm.h            Public launch + benchmark API
    gemm.cu           Quantize, GEMM, dequantize, tests, benchmark
modal/
  test_gemm.py        Correctness tests on Modal T4
  bench_gemm.py       Kernel benchmark on Modal T4
docs/
  concepts.md         Quantization primer
Makefile              Local build (requires nvcc) + Modal shortcuts
```

## Setup

### Local (Linux machine with CUDA)

Requires `nvcc`, NVIDIA GPU, and cuBLAS (included with CUDA toolkit).

```bash
make run        # correctness tests
make run-bench  # custom vs cuBLAS benchmark
```

### Modal (Mac or machine without local CUDA)

```bash
pip install -r requirements-modal.txt
modal setup

make modal-test   # correctness
make modal-bench  # benchmark
```

## Correctness tests

Compares full GPU INT8 pipeline output against CPU FP32 `gemm_fp32`.

Cases:

- 3×3 fixed matrices
- 2×4×3 rectangular GEMM
- 17×17 non-multiple-of-16 size (tile edge coverage)

```bash
make modal-test
```

## Benchmark

Times **only the GEMM kernel** (custom tiled vs cuBLAS) using `cudaEvent`. Excludes quantize, dequantize, and host/PCIe work.

Output CSV columns:

```
M,K,N,custom_ms,custom_gflops,cublas_ms,cublas_gflops,cublas_speedup
```

`cublas_speedup` = `custom_ms / cublas_ms` (>1 means cuBLAS is faster).

Default square sizes: 256, 512, 1024, 2048, 4096.

```bash
make modal-bench
```

Example v1 result (Modal T4):

```
M,K,N,custom_ms,custom_gflops,cublas_ms,cublas_gflops,cublas_speedup
256,256,256,0.106,315,0.028,1196,3.8
512,512,512,0.738,364,0.063,4243,11.7
1024,1024,1024,2.954,727,0.106,20227,27.8
2048,2048,2048,22.476,764,0.725,23693,31.0
4096,4096,4096,188.78,728,6.015,22850,31.4
```

Custom kernel matches cuBLAS exactly (128×128 INT32 check). cuBLAS is ~31× faster at 4096³ on T4 — expected for a v1 educational tiled kernel vs a heavily optimized library.

## API

```cpp
// Full INT8 pipeline: host FP32 in → host FP32 out
void gemm_launch(const float* A, const float* B, float* C, int M, int K, int N);

// Kernel-only benchmarks (device buffers allocated internally)
void benchmark_gemm_kernel(int M, int K, int N, int warmup_iters, int bench_iters);
void benchmark_cublas_gemm_kernel(int M, int K, int N, int warmup_iters, int bench_iters);
```

## Resume one-liner

Built a CUDA INT8 tiled GEMM with per-tensor quantization and shared-memory blocking; validated against CPU FP32 and cuBLAS INT32 baselines; benchmarked on cloud GPU.

## Roadmap (post-v1)

- [ ] Vectorized global loads (int4 / char4)
- [ ] CUTLASS INT8 kernel
- [ ] Naive vs tiled benchmark plot in README
- [ ] GPU scale reduction
- [ ] FP8 (E4M3 / E5M2)

## References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS API](https://docs.nvidia.com/cuda/cublas/index.html)
- [Modal GPU docs](https://modal.com/docs/guide/gpu)
