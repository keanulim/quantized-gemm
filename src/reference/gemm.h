#pragma once

// Row-major GEMM reference: C[M×N] = A[M×K] × B[K×N]
//
// Layout (0-based):
//   A[row, col] = A[row * K + col]
//   B[row, col] = B[row * N + col]
//   C[row, col] = C[row * N + col]

void gemm_fp32(const float* A, const float* B, float* C, int M, int N, int K);

void print_matrix(const float* mat, int rows, int cols, const char* name);

bool matrices_allclose(const float* expected, const float* actual, int size,
                       float rtol = 1e-5f, float atol = 1e-8f);
