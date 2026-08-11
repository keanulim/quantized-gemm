#include "gemm.h"

#include <cmath>
#include <iostream>

void gemm_fp32(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

void print_matrix(const float* mat, int rows, int cols, const char* name) {
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << mat[r * cols + c];
            if (c + 1 < cols) {
                std::cout << ' ';
            }
        }
        std::cout << '\n';
    }
}

bool matrices_allclose(const float* expected, const float* actual, int size,
                       float rtol, float atol) {
    for (int i = 0; i < size; ++i) {
        const float diff = std::fabs(expected[i] - actual[i]);
        const float threshold = atol + rtol * std::fabs(expected[i]);
        if (diff > threshold) {
            std::cout << "Mismatch at index " << i << ": expected " << expected[i]
                      << ", got " << actual[i] << '\n';
            return false;
        }
    }
    return true;
}
