#pragma once

void gemm_launch(const float* A, const float* B, float* C, int M, int K, int N);

// Times only the tiled INT8 GEMM kernel (excludes quantize/dequantize and host work).
void benchmark_gemm_kernel(int M, int K, int N, int warmup_iters, int bench_iters);
