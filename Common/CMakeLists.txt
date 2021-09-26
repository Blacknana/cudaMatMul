add_library(Common Common.cu)

set_target_properties(Common 
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      CUDA_ARCHITECTURES "70;70") 