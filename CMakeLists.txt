cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(MAT_MUL LANGUAGES CXX CUDA)

include_directories ("${PROJECT_SOURCE_DIR}/MatMulOnCPU"
                     "${PROJECT_SOURCE_DIR}/Common")
add_subdirectory(MatMulOnCPU)
add_subdirectory(Common)

add_executable(MatMul_v1 MatMul_v1.cu)
set_target_properties(MatMul_v1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(MatMul_v1 PUBLIC MatMulOnCPU Common)

find_package(cuBLAS)
add_executable(MatMul_v2 MatMul_v2.cu)
set_target_properties(MatMul_v2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(MatMul_v2 PUBLIC MatMulOnCPU Common ${CUBLAS_LIBRARIES})

add_executable(MatMul_v3 MatMul_v3.cu)
set_target_properties(MatMul_v3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(MatMul_v3 PUBLIC MatMulOnCPU Common)

add_executable(MatMul_v4 MatMul_v4.cu)
set_target_properties(MatMul_v4 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(MatMul_v4 PUBLIC MatMulOnCPU Common)