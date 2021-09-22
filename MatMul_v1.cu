#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <random>
#include "Common.cuh"
#include "MatMulOnCPU.h"
using namespace std;

const int Row = 1024;
const int Col = 1024;

__global__ void myMatMulOnGPU(float* M, float* N, float* P, int width)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    float sum = 0;
    for (int k = 0; k < width; k++)
        sum += M[i * width + k] * N[k * width + j];
    P[i * width + j] = sum;
}

int main()
{
    float* A = (float*)malloc(sizeof(float) * Row * Col);
    float* B = (float*)malloc(sizeof(float) * Row * Col);
    float* C = (float*)malloc(sizeof(float) * Row * Col);
    float* C_ref = (float*)malloc(sizeof(float) * Row * Col);
    mySetMatValue(A, Row, Col);
    mySetMatValue(B, Row, Col);
    myMatMulOnCPU(A, B, C_ref, Col);


    //malloc device memory
    float* d_dataA, * d_dataB, * d_dataC;
    myCudaDetermineGPU();
    CHECK(cudaMalloc((void**)&d_dataA, sizeof(float) * Row * Col));
    CHECK(cudaMalloc((void**)&d_dataB, sizeof(float) * Row * Col));
    CHECK(cudaMalloc((void**)&d_dataC, sizeof(float) * Row * Col));
    CHECK(cudaMemcpy(d_dataA, A, sizeof(float) * Row * Col, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dataB, B, sizeof(float) * Row * Col, cudaMemcpyHostToDevice));

    //init timing
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    //init block and grid dim
    dim3 threadPerBlock(16, 16);
    dim3 blockNumber((Col + threadPerBlock.x - 1) / threadPerBlock.x, (Row + threadPerBlock.y - 1) / threadPerBlock.y);
    printf("Block(%d,%d)   Grid(%d,%d).\n", threadPerBlock.x, threadPerBlock.y, blockNumber.x, blockNumber.y);

    //warm up
    for (int i = 0; i < 10; i++)
        myMatMulOnGPU << <blockNumber, threadPerBlock >> > (d_dataA, d_dataB, d_dataC, Col);
    CHECK(cudaGetLastError());

    CHECK(cudaEventRecord(start, 0));
    myMatMulOnGPU << <blockNumber, threadPerBlock >> > (d_dataA, d_dataB, d_dataC, Col);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    //check result
    CHECK(cudaMemcpy(C, d_dataC, sizeof(float) * Row * Col, cudaMemcpyDeviceToHost));
    if (myMatCmp(C, C_ref, Row * Col))
    {
        printf("Error: Wrong result!\n");
        exit(-1);
    }

    float elapsedTime;
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Total time: %fms", elapsedTime);

    //free resource
    free(A);
    free(B);
    free(C);
    CHECK(cudaFree(d_dataA));
    CHECK(cudaFree(d_dataB));
    CHECK(cudaFree(d_dataC));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return 0;
}