#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "Common.cuh"
#include "MatMulOnCPU.h"
using namespace std;

const int Row = 1024;
const int Col = 1024;
const int Block_x = 32;
const int Block_y = 16;

__global__ void myMatMulOnGPU(float* M, float* N, float* P, int width)
{
    int blockRowIdx = blockIdx.x;
    int blockColIdx = blockIdx.y;
    const int blockRowStride = Block_x * 2;
    const int blockColStride = Block_y * 2;
    const int banksize = Block_y * 2;
    int row = threadIdx.x;
    int col = threadIdx.y;

    // loadstore-compute pipeline
    int load_idx = 0;
    int write_idx = 1;
    __shared__ float Msub[2][banksize * blockRowStride];
    __shared__ float Nsub[2][blockColStride * banksize];

    float sum[4] = {0, 0, 0, 0};

    for (int i = 0; i < (width / banksize); i++) {
        // each thread get 4 element(128bit) from M continuously
        int M_loc = (i * banksize + col * 2 + row * 4 / blockRowStride) * width + (blockRowIdx * blockRowStride + row * 4 % blockRowStride);
        int Msub_loc = (col * 2) * blockRowStride + row * 4;
        for (int j = 0; j < 4; j++)
            Msub[write_idx][Msub_loc + j] = M[M_loc + j];

        // each thread get 2 element(64bit) from N continuously
        int N_loc = (blockColIdx * blockColStride + col * 2 + row * 2 / banksize) * width + (i * banksize + row * 2 % banksize);
        int Nsub_loc = (col * 2) * banksize + row * 2;
        for (int j = 0; j < 2; j++)
            Nsub[write_idx][Nsub_loc + j] = N[N_loc + j];

        // make sure that the sub-matrices are loaded
        __syncthreads();

        // switch
        load_idx ^= 1;
        write_idx ^= 1;

        // each thread compute 4 element
        // i.e. (x, y), (x + Block_x, y), (x, y + Block_y), (x + Block_x, y + Block_y)
        for (int j = 0; j < banksize; j++)
        {
            sum[0] += Msub[load_idx][j * blockRowStride + row] * Nsub[load_idx][col * banksize + j];
            sum[1] += Msub[load_idx][j * blockRowStride + row + Block_x] * Nsub[load_idx][col * banksize + j];
            sum[2] += Msub[load_idx][j * blockRowStride + row] * Nsub[load_idx][(col + Block_y) * banksize + j];
            sum[3] += Msub[load_idx][j * blockRowStride + row + Block_x] * Nsub[load_idx][(col + Block_y) * banksize + j];
        }
    }

    P[(blockColIdx * blockColStride + col) * width + (blockRowIdx * blockRowStride + row)] = sum[0];
    P[(blockColIdx * blockColStride + col) * width + (blockRowIdx * blockRowStride + row + Block_x)] = sum[1];
    P[(blockColIdx * blockColStride + col + Block_y) * width + (blockRowIdx * blockRowStride + row)] = sum[2];
    P[(blockColIdx * blockColStride + col + Block_y) * width + (blockRowIdx * blockRowStride + row + Block_x)] = sum[3];
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
    dim3 threadPerBlock(Block_x, Block_y);
    dim3 blockNumber((Col + threadPerBlock.x - 1) / (2 * threadPerBlock.x), (Row + threadPerBlock.y - 1) / (2 * threadPerBlock.y));
    printf("Block(%d,%d)   Grid(%d,%d).\n", threadPerBlock.x, threadPerBlock.y, blockNumber.x, blockNumber.y);

    //warm up
    for (int i = 0; i < 10; i++)
        myMatMulOnGPU << <blockNumber, threadPerBlock >> > (d_dataA, d_dataB, d_dataC, Col);
    CHECK(cudaGetLastError());

    CHECK(cudaEventRecord(start, 0));
    myMatMulOnGPU << <blockNumber, threadPerBlock >> > (d_dataA, d_dataB, d_dataC, Col);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop, 0));

    //check result
    CHECK(cudaMemcpy(C, d_dataC, sizeof(float) * Row * Col, cudaMemcpyDeviceToHost));
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

    return 0;
}