#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "Common.cuh"
#include "MatMulOnCPU.h"
using namespace std;

const int Row = 1024;
const int Col = 1024;
const int Blocksize = 32;

__global__ void myMatMulOnGPU(float* M, float* N, float* P, int width)
{
    int blockRow = blockIdx.x;
    int blockCol = blockIdx.y;
    const int blockstride = Blocksize * 2;
    int row = threadIdx.x;
    int col = threadIdx.y;

    float sum[4] = {0, 0, 0, 0};

    for (int i = 0; i < (width / blockstride); i++) {

        __shared__ float Msub[blockstride * blockstride];
        __shared__ float Nsub[blockstride * blockstride];

        // each thread get 4 element(128bit) from M and N continuously
        int M_loc = (i * blockstride + col * 2 + row * 4 / blockstride) * width + (blockRow * blockstride + row * 4 % blockstride);
        int N_loc = (blockCol * blockstride + col * 2 + row * 4 / blockstride) * width + (i * blockstride + row * 4 % blockstride);
        int MNsub_loc = (col * 2) * blockstride + row * 4;

        for (int j = 0; j < 4; j++)
            Msub[MNsub_loc + j] = M[M_loc + j];
        for (int j = 0; j < 4; j++)
            Nsub[MNsub_loc + j] = N[N_loc + j];

        // make sure that the sub-matrices are loaded
        __syncthreads();

        // each thread compute 4 element
        // i.e. (x, y), (x + Blocksize, y), (x, y + Blocksize), (x + Blocksize, y + Blocksize)
        for (int j = 0; j < blockstride; j++)
        {
            sum[0] += Msub[j * blockstride + row] * Nsub[col * blockstride + j];
            sum[1] += Msub[j * blockstride + row + Blocksize] * Nsub[col * blockstride + j];
            sum[2] += Msub[j * blockstride + row] * Nsub[(col + Blocksize) * blockstride + j];
            sum[3] += Msub[j * blockstride + row + Blocksize] * Nsub[(col + Blocksize) * blockstride + j];
        }
        
        // make sure that preceding computation is done
        // before the next iteration
        __syncthreads();
    }

    P[(blockCol * blockstride + col) * width + (blockRow * blockstride + row)] = sum[0];
    P[(blockCol * blockstride + col) * width + (blockRow * blockstride + row + Blocksize)] = sum[1];
    P[(blockCol * blockstride + col + Blocksize) * width + (blockRow * blockstride + row)] = sum[2];
    P[(blockCol * blockstride + col + Blocksize) * width + (blockRow * blockstride + row + Blocksize)] = sum[3];
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
    dim3 threadPerBlock(Blocksize, Blocksize);
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