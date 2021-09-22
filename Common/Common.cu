#include "Common.cuh"
#include "cuda_runtime.h"
#include <iostream>

void myCudaDetermineGPU() {
    int numDevices;
    CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices > 1) {
        int maxMultiprocessors = 0, maxDevice = 0;
        for (int device = 0; device < numDevices; device++) {
            cudaDeviceProp props;
            CHECK(cudaGetDeviceProperties(&props, device));
            if (maxMultiprocessors < props.multiProcessorCount) {
                maxMultiprocessors = props.multiProcessorCount;
                maxDevice = device;
            }
        }
        CHECK(cudaSetDevice(maxDevice));
    }
}