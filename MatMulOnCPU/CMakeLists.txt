add_library(MatMulOnCPU MatMulOnCPU.cpp)

set_target_properties(MatMulOnCPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON) 