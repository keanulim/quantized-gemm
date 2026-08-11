#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

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
}

__global__ void dequantize(const int32_t *C_int,
    int M, int N, float *C, float scale_a, float scale_b){
       //dequantize with scale back to FP32
       //FP32_C[M x N] = scale_a x scale_b x INT32_C[]
}

void gemm_launch(const float *A, const float *B, float *C, int M, int K, int N){
    //input FP32 matrices
    //16 x 16 threads per block
    dim3 dimBlock(256);
    // 16 x 16 / M x N blocks
    dim3 dimGridA((M * K + 255)/ 256);
    dim3 dimGridB((K * N + 255)/ 256);
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
    int sizeq_A = M*K * sizeof(int8_t);
    int sizeq_B = K*N * sizeof(int8_t);

    float *d_A, *d_B;
    int8_t *d_q_A, *d_q_B;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_q_A, sizeq_A);
    cudaMalloc(&d_q_B, sizeq_B);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    quantize_a<<<dimGridA, dimBlock>>>(d_A, d_q_A, M, K, scale_a);
    quantize_b<<<dimGridB, dimBlock>>>(d_B, d_q_B, K, N, scale_b);

    cudaDeviceSynchronize();

    std::vector<int8_t> h_q_A(M * K);
    cudaMemcpy(h_q_A.data(), d_q_A, sizeq_A, cudaMemcpyDeviceToHost);

    int mismatches = 0;
    for (int i = 0; i < M * K; ++i) {
        const int8_t expected = static_cast<int8_t>(roundf(A[i] / scale_a));
        if (h_q_A[i] != expected) {
            ++mismatches;
            std::cout << "q_A mismatch at " << i << ": gpu=" << static_cast<int>(h_q_A[i])
                      << " expected=" << static_cast<int>(expected)
                      << " (A[i]=" << A[i] << ", scale_a=" << scale_a << ")\n";
        }
    }

    std::cout << "q_A verification: " << (M * K - mismatches) << "/" << (M * K)
              << " elements match\n";

    //call gemm
    //call dequantize

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_q_A);
    cudaFree(d_q_B);
}