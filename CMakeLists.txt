cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(MAT_MUL LANGUAGES CXX CUDA)

add_subdirectory(MatMulOnCPU)
add_subdirectory(Common)

add_executable(MatMul_v1 MatMul_v1.cu)

set_target_properties(MatMul_v1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(MatMul_v1 PUBLIC MatMulOnCPU)
target_link_libraries(MatMul_v1 PUBLIC Common)
target_include_directories(MatMul_v1 PUBLIC 
                           "${PROJECT_SOURCE_DIR}/MatMulOnCPU" 
                           "${PROJECT_SOURCE_DIR}/Common"
                           ) 