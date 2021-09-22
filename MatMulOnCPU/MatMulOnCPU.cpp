#include "MatMulOnCPU.h"
#include <iostream>
#include <random>
using namespace std;

void mySetMatValue(float* A, int row, int col)
{
    random_device rd;  //Will be used to obtain a seed for the random number engine
    mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    uniform_real_distribution<> dis(1.0, 2.0);
    srand((int)time(NULL));
    for (int i = 0; i < row * col; i++) {
        A[i] = dis(gen);
    }
}

void myMatMulOnCPU(float* M, float* N, float* P, int width)
{
    float sum;
    for (int i = 0; i < width; i++)
        for (int j = 0; j < width; j++)
        {
            sum = 0;
            for (int k = 0; k < width; k++)
                sum += M[i * width + k] * N[k * width + j];
            P[i * width + j] = sum;
        }
}

int myMatCmp(float* M, float* N, int size)
{
    for (int i = 0; i < size; i++)
        if (fabs(M[i] - N[i]) >= 1e-3)
            return 1;
    return 0;
}