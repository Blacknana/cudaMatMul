#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <stdio.h>
#include <iostream>
#include "Common.cuh"
#include "MatMulOnCPU.h"
using namespace std;

const int Row = 1024;
const int Col = 1024;

int main()
{
    float* A = (float*)malloc(sizeof(float) * Row * Col);
    float* B = (float*)malloc(sizeof(float) * Row * Col);
    float* C = (float*)malloc(sizeof(float) * Row * Col);
    float* C_ref = (float*)malloc(sizeof(float) * Row * Col);
    mySetMatValue(A, Row, Col);
    mySetMatValue(B, Row, Col);
    myMatMulOnCPU(A, B, C_ref, Col);


    myCudaDetermineGPU();
    cublasHandle_t handle = 0;
    float alpha = 1, beta = 0;
    
    //malloc device memory
    float* d_dataA, * d_dataB, * d_dataC;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK(cudaMalloc((void**)&d_dataA, sizeof(float) * Row * Col));
    CHECK(cudaMalloc((void**)&d_dataB, sizeof(float) * Row * Col));
    CHECK(cudaMalloc((void**)&d_dataC, sizeof(float) * Row * Col));
    CHECK_CUBLAS(cublasSetMatrix(Row, Col, sizeof(float), A, Row, d_dataA, Row));
    CHECK_CUBLAS(cublasSetMatrix(Row, Col, sizeof(float), B, Row, d_dataB, Row));

    //init timing
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    //warm up
    for (int i = 0; i < 10; i++)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Row, Col, Col,
            &alpha, d_dataA, Row, d_dataB, Row, &beta, d_dataC, Row));

    CHECK(cudaEventRecord(start, 0));
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Row, Col, Col,
        &alpha, d_dataA, Row, d_dataB, Row, &beta, d_dataC, Row));
    CHECK(cudaEventRecord(stop, 0));

    //check result
    CHECK_CUBLAS(cublasGetMatrix(Row, Col, sizeof(float), d_dataC, Row, C, Row));
    if (myMatCmp(C, C_ref, Row * Col))
    {
        printf("Error: Wrong result!\n");
        exit(-1);
    }

    float elapsedTime;
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Total time: %fms\n", elapsedTime);

    //free resource
    free(A);
    free(B);
    free(C);
    CHECK(cudaFree(d_dataA));
    CHECK(cudaFree(d_dataB));
    CHECK(cudaFree(d_dataC));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}