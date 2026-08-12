#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "../reference/gemm.h"
#include "gemm.h"

__global__ void quantize_a(const float *A, int8_t* d_q_A,
int M, int K, float scale_a){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < M * K){
        d_q_A[i] = (int8_t)roundf(A[i]/scale_a);
    }
        //quantize, turns into INT8
        //each thread does one
        //q_A[] = round(x/scale_A); //confine to [-128,127]
        
}

__global__ void quantize_b(const float *B, int8_t* d_q_B,
int K, int N, float scale_b){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < K * N){
        d_q_B[i] = (int8_t)roundf(B[i]/scale_b);
    }
            //quantize, turns into INT8
            //each thread does one
           
            //q_B[] = round(x/scale_B); 
}

__global__ void gemm(const int8_t *A, const int8_t *B, int32_t* C_int,
    int M, int N, int K){
        //input: int8 matrices
       //matmul
       //INT32_C[M x N] = q_A dot q_B; (INT32 accumulation) 
        const int TILE = 16;
       //each thread owns one C output
       int tx = threadIdx.x;
       int ty = threadIdx.y;
       int bRow = blockIdx.y;
       int bCol = blockIdx.x;
       int row = bRow * TILE + ty;
       int col = bCol * TILE + tx;
        //load A and B tile 
        __shared__ int8_t tileA[TILE * TILE];
        __shared__ int8_t tileB[TILE * TILE];

        int32_t acc = 0;
        for(int kStart = 0; kStart < K; kStart += TILE){
            tileA[ty * TILE + tx] = (row < M && kStart + tx < K) ? A[row * K + (kStart + tx)] : 0;
            tileB[ty * TILE + tx] = (kStart + ty < K && col < N) ? B[(kStart + ty) * N + col] : 0;
            __syncthreads();

            for (int k = 0; k < min(TILE, K - kStart); ++k) {  
                acc += tileA[ty * TILE + k] * tileB[k * TILE + tx];
            }
            __syncthreads();
        }
        if (row < M && col < N) C_int[row * N + col] = acc;

      
        
}

__global__ void dequantize(const int32_t *C_int,
    int M, int N, float *C, float scale_a, float scale_b){
       //dequantize with scale back to FP32
       //FP32_C[M x N] = scale_a x scale_b x INT32_C[]

       int i = blockDim.x * blockIdx.x + threadIdx.x;
       if(i < M * N){
        C[i] = (float)C_int[i] * scale_a * scale_b;
       }
}

void gemm_launch(const float *A, const float *B, float *C, int M, int K, int N){
    //input FP32 matrices
    //16 x 16 threads per block
    dim3 dimBlock(16,16);
    dim3 dimBlockQ(256);
    // 16 x 16 / M x N blocks
    dim3 dimGridA((M * K + 255)/ 256);
    dim3 dimGridB((K * N + 255)/ 256);
    dim3 dimGridGemm((N + 15) / 16, (M + 15) / 16);
    dim3 dimGridDQ((M * N + 255) / 256);
    //allocate 2 FP32 matrices, 2 INT8 matrices, 1 fp32 matrix
    //find max on cpu
    float max_a = -INFINITY;
    float max_b = -INFINITY;

    if(M * K != 0){
        for(int i = 0; i < M * K; i++){
            if(fabsf(A[i]) > max_a) max_a = fabsf(A[i]);
        }
    }
    if(K * N != 0){
        for(int i = 0; i < K * N; i++){
            if(fabsf(B[i]) > max_b) max_b = fabsf(B[i]);
        }   
    }
    
    float scale_a = (max_a == 0.f) ? 1.f: max_a / 127.f;
    float scale_b = (max_b == 0.f) ? 1.f: max_b / 127.f;

    int sizeA = M*K * sizeof(float);
    int sizeB = K*N * sizeof(float);
    int sizeC = M*N * sizeof(float);
    int sizeC_int = M*N * sizeof(int32_t);
    int sizeq_A = M*K * sizeof(int8_t);
    int sizeq_B = K*N * sizeof(int8_t);
    

    float *d_A, *d_B, *d_C;
    int8_t *d_q_A, *d_q_B;
    int32_t *d_C_int;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_q_A, sizeq_A);
    cudaMalloc(&d_q_B, sizeq_B);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    quantize_a<<<dimGridA, dimBlockQ>>>(d_A, d_q_A, M, K, scale_a);
    quantize_b<<<dimGridB, dimBlockQ>>>(d_B, d_q_B, K, N, scale_b);

    cudaMalloc(&d_C_int, sizeC_int);
    cudaMalloc(&d_C, sizeC);

    gemm<<<dimGridGemm, dimBlock>>>(d_q_A, d_q_B, d_C_int, M, N, K);
    dequantize<<<dimGridDQ, dimBlockQ>>>(d_C_int, M, N, d_C, scale_a, scale_b);

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_q_A);
    cudaFree(d_q_B);
    cudaFree(d_C_int);
}

#ifdef RUN_INT8_GEMM_TEST
static float max_abs_diff(const float* a, const float* b, int n) {
    float max_diff = 0.f;
    for (int i = 0; i < n; ++i) {
        max_diff = fmaxf(max_diff, fabsf(a[i] - b[i]));
    }
    return max_diff;
}

static float max_abs_val(const float* a, int n) {
    float max_val = 0.f;
    for (int i = 0; i < n; ++i) {
        max_val = fmaxf(max_val, fabsf(a[i]));
    }
    return max_val;
}

static bool run_test_case(const char* name, const float* A, const float* B,
                          int M, int K, int N, float max_rel_error) {
    std::vector<float> C_gpu(M * N, 0.f);
    std::vector<float> C_ref(M * N, 0.f);

    gemm_launch(A, B, C_gpu.data(), M, K, N);
    gemm_fp32(A, B, C_ref.data(), M, N, K);

    const float abs_diff = max_abs_diff(C_gpu.data(), C_ref.data(), M * N);
    const float ref_scale = fmaxf(max_abs_val(C_ref.data(), M * N), 1e-6f);
    const float rel_error = abs_diff / ref_scale;

    std::cout << "\n[" << name << "] M=" << M << " K=" << K << " N=" << N << "\n";
    print_matrix(C_ref.data(), M, N, "CPU FP32 reference");
    print_matrix(C_gpu.data(), M, N, "GPU INT8 pipeline");
    std::cout << "max abs diff: " << abs_diff << "\n";
    std::cout << "relative error: " << rel_error << "\n";

    if (rel_error <= max_rel_error) {
        std::cout << "PASS\n";
        return true;
    }

    std::cout << "FAIL (relative error > " << max_rel_error << ")\n";
    return false;
}

static int run_int8_gemm_tests() {
    bool ok = true;

    {
        const int M = 3, K = 3, N = 3;
        const float A[] = {1, 3, 4, 4, 3, 1, 2, 2, 4};
        const float B[] = {4, 3, 2, 4, 5, 2, 2, 6, 8};
        ok &= run_test_case("square-3x3", A, B, M, K, N, 0.05f);
    }

    {
        const int M = 2, K = 4, N = 3;
        const float A[] = {1, 2, 3, 4, 5, 6, 7, 8};
        const float B[] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
        ok &= run_test_case("rect-2x4x3", A, B, M, K, N, 0.05f);
    }

    {
        const int M = 17, K = 17, N = 17;
        std::vector<float> A(M * K);
        std::vector<float> B(K * N);
        for (int i = 0; i < M * K; ++i) {
            A[i] = static_cast<float>((i % 7) - 3);
        }
        for (int i = 0; i < K * N; ++i) {
            B[i] = static_cast<float>((i % 5) - 2);
        }
        ok &= run_test_case("tile-edge-17", A.data(), B.data(), M, K, N, 0.10f);
    }

    std::cout << (ok ? "\nAll tests passed.\n" : "\nSome tests failed.\n");
    return ok ? 0 : 1;
}
#endif

static float gflops(int M, int N, int K, float ms) {
    if (ms <= 0.f) {
        return 0.f;
    }
    const double ops = 2.0 * static_cast<double>(M) * N * K;
    return static_cast<float>(ops / (static_cast<double>(ms) * 1e6));
}

void benchmark_gemm_kernel(int M, int K, int N, int warmup_iters, int bench_iters) {
    const int sizeq_A = M * K * static_cast<int>(sizeof(int8_t));
    const int sizeq_B = K * N * static_cast<int>(sizeof(int8_t));
    const int sizeC_int = M * N * static_cast<int>(sizeof(int32_t));

    int8_t *d_q_A = nullptr;
    int8_t *d_q_B = nullptr;
    int32_t *d_C_int = nullptr;

    cudaMalloc(&d_q_A, sizeq_A);
    cudaMalloc(&d_q_B, sizeq_B);
    cudaMalloc(&d_C_int, sizeC_int);
    cudaMemset(d_q_A, 1, sizeq_A);
    cudaMemset(d_q_B, 1, sizeq_B);
    cudaMemset(d_C_int, 0, sizeC_int);

    const dim3 dimBlock(16, 16);
    const dim3 dimGridGemm((N + 15) / 16, (M + 15) / 16);

    for (int i = 0; i < warmup_iters; ++i) {
        gemm<<<dimGridGemm, dimBlock>>>(d_q_A, d_q_B, d_C_int, M, N, K);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; ++i) {
        gemm<<<dimGridGemm, dimBlock>>>(d_q_A, d_q_B, d_C_int, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.f;
    cudaEventElapsedTime(&total_ms, start, stop);
    const float avg_ms = total_ms / static_cast<float>(bench_iters);

    std::cout << M << "," << K << "," << N << "," << avg_ms << ","
              << gflops(M, N, K, avg_ms) << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_q_A);
    cudaFree(d_q_B);
    cudaFree(d_C_int);
}

static int run_int8_gemm_benchmarks() {
    const int sizes[] = {256, 512, 1024, 2048, 4096};
    const int warmup_iters = 5;
    const int bench_iters = 100;

    std::cout << "INT8 tiled GEMM kernel benchmark\n";
    std::cout << "warmup_iters=" << warmup_iters << " bench_iters=" << bench_iters << "\n";
    std::cout << "M,K,N,avg_ms,gflops\n";

    for (int n : sizes) {
        benchmark_gemm_kernel(n, n, n, warmup_iters, bench_iters);
    }

    return 0;
}

#ifdef RUN_INT8_GEMM_BENCH
int main() {
    return run_int8_gemm_benchmarks();
}
#elif defined(RUN_INT8_GEMM_TEST)
int main() {
    return run_int8_gemm_tests();
}
#else
int main() {
    return 0;
}
#endif