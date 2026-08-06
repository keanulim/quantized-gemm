# Quantized GEMM (INT8 / FP8)

General matrix multiply `C = A × B` using low-precision weights/activations.

## What this project is

**GEMM** = General Matrix Multiply. The core operation behind linear layers in neural nets.

**Quantization** = store/compute with fewer bits (INT8 = 8-bit integers, FP8 = 8-bit floats) instead of FP32/FP16. You trade a bit of accuracy for speed and memory.

**Pipeline (simplified):**
1. Quantize FP32 matrices → INT8/FP8 + scale factors
2. Multiply in low precision (often with INT32 accumulation)
3. Dequantize back to FP32 for output

## Learning path (recommended order)

1. **Reference FP32 GEMM** — naive triple loop on CPU; verify correctness.
2. **Quantization basics** — per-tensor vs per-channel scaling; symmetric vs asymmetric.
3. **Reference INT8 GEMM** — quantize A/B, int32 dot products, dequantize C.
4. **Correctness tests** — compare against FP32 baseline; measure max/mean error.
5. **FP8 variant** — E4M3 / E5M2 formats; same structure as INT8.
6. **Optimized kernels** — SIMD (AVX2/NEON), blocking/tiling, optional GPU later.
7. **Benchmarks** — GFLOPS, memory bandwidth, accuracy vs speed tradeoffs.

## Target formats

| Format | Typical use |
|--------|-------------|
| INT8   | Weights & activations; int32 accumulators |
| FP8 E4M3 | Activations (more precision) |
| FP8 E5M2 | Weights (wider range) |

## Success criteria

- [ ] Reference FP32 GEMM passes unit tests
- [ ] INT8 GEMM within acceptable error vs FP32
- [ ] FP8 GEMM within acceptable error vs FP32
- [ ] Documented quantization scheme and error bounds
- [ ] Benchmark comparing naive vs optimized (optional)
