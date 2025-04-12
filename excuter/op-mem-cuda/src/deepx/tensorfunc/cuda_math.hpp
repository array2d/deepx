#ifndef DEEPX_TENSORFUNC_CUDA_MATH_HPP
#define DEEPX_TENSORFUNC_CUDA_MATH_HPP

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

namespace deepx::tensorfunc
{

    // max
    template <typename T>
    __device__ void deepx_max(const T *a, const T *b, T *out);

    template <>
    __device__ void deepx_max<double>(const double *a, const double *b, double *out)
    {
        *out = fmax(*a, *b);
    }
    
    template <>
    __device__ void deepx_max<float>(const float *a, const float *b, float *out)
    {
        *out = fmaxf(*a, *b);
    }

    template <>
    __device__ void deepx_max<half>(const half *a, const half *b, half *out)
    {
        *out = __hmax(*a, *b);
    }

    template <>
    __device__ void deepx_max<nv_bfloat16>(const nv_bfloat16 *a, const nv_bfloat16 *b, nv_bfloat16 *out)
    {
        *out = __hmax(*a, *b);
    }
    template <>
    __device__ void deepx_max<int64_t>(const int64_t *a, const int64_t *b, int64_t *out)
    {
        *out = *a>*b?*a:*b;
    }
    template <>
    __device__ void deepx_max<int32_t>(const int32_t *a, const int32_t *b, int32_t *out)
    {
        *out = *a>*b?*a:*b;
    }
    template <>
    __device__ void deepx_max<int16_t>(const int16_t *a, const int16_t *b, int16_t *out)
    {
        *out = *a>*b?*a:*b;
    }
    template <>
    __device__ void deepx_max<int8_t>(const int8_t *a, const int8_t *b, int8_t *out)
    {
        *out = *a>*b?*a:*b;
    }
    
 
    // min
    template <typename T>
    __device__ void deepx_min(const T *a, const T *b, T *out);
 

    template <>
    __device__ void deepx_min<double>(const double *a, const double *b, double *out)
    {
        *out = fmin(*a, *b);
    }

    template <>
    __device__ void deepx_min<float>(const float *a, const float *b, float *out)
    {
        *out = fminf(*a, *b);
    }


    template <>
    __device__ void deepx_min<half>(const half *a, const half *b, half *out)
    {
        *out = __hmin(*a, *b);
    }

    template <>
    __device__ void deepx_min<nv_bfloat16>(const nv_bfloat16 *a,const nv_bfloat16 *b, nv_bfloat16 *out)
    {
        *out = __hmin(*a, *b);
    }

    template <>
    __device__ void deepx_min<int64_t>(const int64_t *a, const int64_t *b, int64_t *out)
    {
        *out = *a<*b?*a:*b;
    }

    template <>
    __device__ void deepx_min<int32_t>(const int32_t *a, const int32_t *b, int32_t *out)
    {
        *out = *a<*b?*a:*b;
    }

    template <>
    __device__ void deepx_min<int16_t>(const int16_t *a, const int16_t *b, int16_t *out)
    {
        *out = *a<*b?*a:*b;
    }

    template <>
    __device__ void deepx_min<int8_t>(const int8_t *a, const int8_t *b, int8_t *out)
    {
        *out =  *a<*b?*a:*b;
    }   

 

}

#endif
