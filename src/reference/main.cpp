#include "gemm.h"

#include <iostream>
#include <vector>

namespace {

bool run_square_test() {
    // A[3×3] × B[3×3] = C[3×3]
    const int M = 3, N = 3, K = 3;
    const std::vector<float> A = {1, 3, 4, 4, 3, 1, 2, 2, 4};
    const std::vector<float> B = {4, 3, 2, 4, 5, 2, 2, 6, 8};
    const std::vector<float> expected = {24, 42, 40, 30, 33, 22, 24, 40, 40};

    std::vector<float> C(M * N, 0.0f);
    gemm_fp32(A.data(), B.data(), C.data(), M, N, K);

    print_matrix(C.data(), M, N, "C (square test)");
    return matrices_allclose(expected.data(), C.data(), M * N);
}

bool run_rectangular_test() {
    // A[2×4] × B[4×3] = C[2×3]
    const int M = 2, N = 3, K = 4;
    const std::vector<float> A = {
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    const std::vector<float> B = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        1, 1, 1,
    };
    const std::vector<float> expected = {
        5, 6, 7,
        13, 14, 15,
    };

    std::vector<float> C(M * N, 0.0f);
    gemm_fp32(A.data(), B.data(), C.data(), M, N, K);

    print_matrix(C.data(), M, N, "C (rectangular test)");
    return matrices_allclose(expected.data(), C.data(), M * N);
}

}  // namespace

int main() {
    const bool square_ok = run_square_test();
    const bool rect_ok = run_rectangular_test();

    if (square_ok && rect_ok) {
        std::cout << "All reference GEMM tests passed.\n";
        return 0;
    }

    std::cout << "Reference GEMM tests failed.\n";
    return 1;
}
