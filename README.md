# cudaMatMul
Practice cuda by matrix multiplication. 

Author: Sunqianqi

- version 1: use single loop(1.10ms)
- version 2: use cuBLAS(0.77ms)
- version 3: utilize shared memory based on v1(0.98ms)
- version 4: utilize prefetch based on v3(0.59ms)

##### Device Info:

| Device                                        | NVIDIA A100-PCIE-40GB                                  |
| --------------------------------------------- | ------------------------------------------------------ |
| CUDA Driver Version / Runtime Version         | 11.4 / 11.2                                            |
| CUDA Capability Major/Minor version number    | 8.0                                                    |
| Total amount of global memory                 | 39.59 GBytes (42505273344 bytes)                       |
| GPU Clock rate                                | 1410 MHz (1.41 GHz)                                    |
| Memory Clock rate                             | 1215 Mhz                                               |
| Memory Bus Width                              | 5120-bit                                               |
| L2 Cache Size                                 | 41943040 bytes                                         |
| Max Texture Dimension Size (x,y,z)            | 1D=(131072), 2D=(131072,65536), 3D=(16384,16384,16384) |
| Max Layered Texture Size (dim) x layers       | 1D=(32768) x 2048, 2D=(32768,32768) x 2048             |
| Total amount of constant memory               | 65536 bytes                                            |
| Total amount of shared memory per block       | 49152 bytes                                            |
| Total number of registers available per block | 65536                                                  |
| Warp size                                     | 32                                                     |
| Maximum number of threads per multiprocessor  | 2048                                                   |
| Maximum number of threads per block           | 1024                                                   |
| Maximum sizes of each dimension of a block    | 1024 x 1024 x 64                                       |
| Maximum sizes of each dimension of a grid     | 2147483647 x 65535 x 65535                             |
| Maximum memory pitch                          | 2147483647 bytes                                       |
