# quantized-gemm

CUDA INT8 general matrix multiply (GEMM) with per-tensor quantization, shared-memory tiling, and CPU/GPU validation.

**Version:** v1

## v1 scope

v1 implements a complete INT8 GEMM pipeline on GPU:

1. **CPU scale finding** — symmetric per-tensor scaling (`scale = max(|x|) / 127`)
2. **GPU quantize** — FP32 → INT8 elementwise kernels
3. **GPU tiled GEMM** — 16×16×16 shared-memory tiles, INT32 accumulation
4. **GPU dequantize** — INT32 → FP32 with `scale_a * scale_b`
5. **CPU FP32 reference** — correctness baseline in `src/reference/`
6. **Modal runners** — test and benchmark on cloud GPU (no local `nvcc` required)

Not in v1: vectorized loads, cuBLAS/CUTLASS baselines, FP8, GPU scale reduction.

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

## Tiled GEMM (v1 kernel)

- **Block:** 16×16 threads
- **Grid:** one block per 16×16 output tile of `C`
- **Shared memory:** `As[16×16]`, `Bs[16×16]` per k-tile
- **Outer loop:** `kStart` steps along K in chunks of 16
- **Inner loop:** each thread `(ty, tx)` accumulates  
  `acc += As[ty][k] * Bs[k][tx]`
- **Output:** INT32 accumulator per element (never INT8)

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

Requires `nvcc` and an NVIDIA GPU.

```bash
make run        # correctness tests
make run-bench  # kernel benchmark
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

Pass threshold: relative error vs FP32 reference (typically < 1–10% depending on size).

```bash
make modal-test
```

## Benchmark

Times **only the tiled `gemm` kernel** using `cudaEvent` (warmup + averaged iterations). Excludes quantize, dequantize, and host/PCIe work.

Output CSV columns:

```
M,K,N,avg_ms,gflops
```

Default square sizes: 256, 512, 1024, 2048, 4096.

GFLOPS formula:

```
GFLOPS = 2 * M * N * K / (seconds * 1e9)
```

```bash
make modal-bench
```

Example v1 result (Modal T4):

```
M,K,N,avg_ms,gflops
256,256,256,0.088,381
512,512,512,0.607,442
1024,1024,1024,2.976,722
2048,2048,2048,22.463,765
4096,4096,4096,188.961,727
```

Use benchmarks for before/after comparisons on the same GPU, not as absolute peak claims.

## API

```cpp
// Full INT8 pipeline: host FP32 in → host FP32 out
void gemm_launch(const float* A, const float* B, float* C, int M, int K, int N);

// Kernel-only benchmark (device buffers allocated internally)
void benchmark_gemm_kernel(int M, int K, int N, int warmup_iters, int bench_iters);
```

## Roadmap (post-v1)

- [ ] Vectorized global loads (int4 / char4)
- [ ] cuBLAS INT8 baseline
- [ ] Naive vs tiled benchmark comparison
- [ ] GPU scale reduction
- [ ] FP8 (E4M3 / E5M2)
- [ ] Register blocking / double buffering

## References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Modal GPU docs](https://modal.com/docs/guide/gpu)
