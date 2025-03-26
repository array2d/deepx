#ifndef DEEPX_TENSORFUNC_MATMUL_QWANG_CUH
#define DEEPX_TENSORFUNC_MATMUL_QWANG_CUH

#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/matmul.hpp"

namespace deepx::tensorfunc
{

    #define BLOCK_SIZE 32

    template <typename T>
    __global__ void matmul_kernel(T *C, const T *A, const T *B,
                                     int M, int N, int K);  
 


    template <typename T>
    void launch_matmul(T *d_C, const T *d_A, const T *d_B,
                       int M, int N, int K);

    extern template void launch_matmul<double>(double *d_C, const double *d_A, const double *d_B,
                                        int M, int N, int K);
    extern template void launch_matmul<float>(float *d_C, const float *d_A, const float *d_B,
                                        int M, int N, int K);
    extern template void launch_matmul<half>(half *d_C, const half *d_A, const half *d_B,
                                        int M, int N, int K);
    extern template void launch_matmul<nv_bfloat16>(nv_bfloat16 *d_C, const nv_bfloat16 *d_A, const nv_bfloat16 *d_B,
                                        int M, int N, int K);
    extern template void launch_matmul<int64_t>(int64_t *d_C, const int64_t *d_A, const int64_t *d_B,
                                        int M, int N, int K);
    extern template void launch_matmul<int32_t>(int32_t *d_C, const int32_t *d_A, const int32_t *d_B,
                                        int M, int N, int K);
    extern template void launch_matmul<int16_t>(int16_t *d_C, const int16_t *d_A, const int16_t *d_B,
                                        int M, int N, int K);
    extern template void launch_matmul<int8_t>(int8_t *d_C, const int8_t *d_A, const int8_t *d_B,
                                        int M, int N, int K);
}
#endif