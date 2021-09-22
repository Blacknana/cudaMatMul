#ifndef _MAT_MUL_CPU_H
#define _MAT_MUL_CPU_H

void mySetMatValue(float* A, int row, int col);
void myMatMulOnCPU(float* M, float* N, float* P, int width);
int myMatCmp(float* M, float* N, int size);

#endif
