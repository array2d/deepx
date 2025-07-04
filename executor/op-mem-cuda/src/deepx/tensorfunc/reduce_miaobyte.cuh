#ifndef DEEPX_TENSORFUNC_REDUCE_MIAOBYTE_CUH
#define DEEPX_TENSORFUNC_REDUCE_MIAOBYTE_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace deepx::tensorfunc
{
    // sum
    template <int DIM, typename T>
    __global__ void  sum_kernel(const T *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                     const int *reduced_dims, const bool keepdims,
                                     T *result_data, const int *result_strides, const int result_dim);
  
    template <typename T>
    void launch_sum(const T *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                          const int *reduced_dims, const bool keepdims,
                          T *result_data, const int *result_strides, const int result_dim);

  
    //prod

    template <int DIM, typename T>
    __global__ void  prod_kernel(const T *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                      const int *reduced_dims, const bool keepdims,
                                      T *result_data, const int *result_strides, const int result_dim);

    template <typename T>
    void launch_prod(const T *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                           const int *reduced_dims, const bool keepdims,
                           T *result_data, const int *result_strides, const int result_dim);
   
    //max
    template <int DIM, typename T>
    __global__ void reducemax_kernel(const T *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                      const int *reduced_dims, const bool keepdims,
                                      T *result_data, const int *result_strides, const int result_dim);
                                      
    template <typename T>
    void launch_reducemax(const T *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                          const int *reduced_dims, const bool keepdims,
                          T *result_data, const int *result_strides, const int result_dim);

   
    //min
    template <int DIM, typename T>
    __global__ void reducemin_kernel(const T *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                                      const int *reduced_dims, const bool keepdims,
                                      T *result_data, const int *result_strides, const int result_dim);

    template <typename T>
    void launch_reducemin(const T *tensor_data, const int *tensor_strides, const int tensor_dim, const int tensor_len,
                          const int *reduced_dims, const bool keepdims,
                          T *result_data, const int *result_strides, const int result_dim);
 
}

#endif //DEEPX_TENSORFUNC_REDUCE_MIAOBYTE_CUH
