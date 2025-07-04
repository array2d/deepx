#ifndef DEEPX_TENSORFUNC_CUDA_MATH_CUH
#define DEEPX_TENSORFUNC_CUDA_MATH_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

namespace deepx::tensorfunc
{

    // sqrt
    template <typename T>
    __device__ __forceinline__ void deepx_sqrt(const T *a, T *out);

    template <>
    __device__ __forceinline__ void deepx_sqrt<double>(const double *a, double *out)
    {
        *out = sqrt(*a);
    }

    template <>
    __device__ __forceinline__ void deepx_sqrt<float>(const float *a, float *out)
    {
        *out = sqrtf(*a);
    }

    template <>
    __device__ __forceinline__ void deepx_sqrt<half>(const half *a, half *out)
    {
        *out = hsqrt(*a);
    }

    template <>
    __device__ __forceinline__ void deepx_sqrt<nv_bfloat16>(const nv_bfloat16 *a, nv_bfloat16 *out)
    {
        *out = hsqrt(*a);
    }

    // pow
    template <typename T>
    __device__ __forceinline__ void deepx_pow(const T *a, const T *b, T *out);

    template <>
    __device__ __forceinline__ void deepx_pow<double>(const double *a, const double *b, double *out)
    {
        *out = pow(*a, *b);
    }

    template <>
    __device__ __forceinline__ void deepx_pow<float>(const float *a, const float *b, float *out)
    {
        *out = powf(*a, *b);
    }

    // log
    template <typename T>
    __device__ __forceinline__ void deepx_log(const T *a, T *out);

    template <>
    __device__ __forceinline__ void deepx_log<double>(const double *a, double *out)
    {
        *out = log(*a);
    }

    template <>
    __device__ __forceinline__ void deepx_log<float>(const float *a, float *out)
    {
        *out = logf(*a);
    }

    template <>
    __device__ __forceinline__ void deepx_log<half>(const half *a, half *out)
    {
        *out = hlog(*a);
    }

    template <>
    __device__ __forceinline__ void deepx_log<nv_bfloat16>(const nv_bfloat16 *a, nv_bfloat16 *out)
    {
        *out = hlog(*a);
    }

    // exp
    template <typename T>
    __device__ __forceinline__ void deepx_exp(const T *a, T *out);

    template <>
    __device__ __forceinline__ void deepx_exp<double>(const double *a, double *out)
    {
        *out = exp(*a);
    }

    template <>
    __device__ __forceinline__ void deepx_exp<float>(const float *a, float *out)
    {
        *out = expf(*a);
    }

    template <>
    __device__ __forceinline__ void deepx_exp<half>(const half *a, half *out)
    {
        *out = hexp(*a);
    }

    template <>
    __device__ __forceinline__ void deepx_exp<nv_bfloat16>(const nv_bfloat16 *a, nv_bfloat16 *out)
    {
        *out = hexp(*a);
    }

    // max
    template <typename T>
    __device__ __forceinline__ void deepx_max(const T *a, const T *b, T *out);

    template <>
    __device__ __forceinline__ void deepx_max<double>(const double *a, const double *b, double *out)
    {
        *out = fmax(*a, *b);
    }

    template <>
    __device__ __forceinline__ void deepx_max<float>(const float *a, const float *b, float *out)
    {
        *out = fmaxf(*a, *b);
    }

    template <>
    __device__ __forceinline__ void deepx_max<half>(const half *a, const half *b, half *out)
    {
        *out = __hmax(*a, *b);
    }

    template <>
    __device__ __forceinline__ void deepx_max<nv_bfloat16>(const nv_bfloat16 *a, const nv_bfloat16 *b, nv_bfloat16 *out)
    {
        *out = __hmax(*a, *b);
    }
    template <>
    __device__ __forceinline__ void deepx_max<int64_t>(const int64_t *a, const int64_t *b, int64_t *out)
    {
        *out = *a > *b ? *a : *b;
    }
    template <>
    __device__ __forceinline__ void deepx_max<int32_t>(const int32_t *a, const int32_t *b, int32_t *out)
    {
        *out = *a > *b ? *a : *b;
    }
    template <>
    __device__ __forceinline__ void deepx_max<int16_t>(const int16_t *a, const int16_t *b, int16_t *out)
    {
        *out = *a > *b ? *a : *b;
    }
    template <>
    __device__ __forceinline__ void deepx_max<int8_t>(const int8_t *a, const int8_t *b, int8_t *out)
    {
        *out = *a > *b ? *a : *b;
    }

    // min
    template <typename T>
    __device__ __forceinline__ void deepx_min(const T *a, const T *b, T *out);

    template <>
    __device__ __forceinline__ void deepx_min<double>(const double *a, const double *b, double *out)
    {
        *out = fmin(*a, *b);
    }

    template <>
    __device__ __forceinline__ void deepx_min<float>(const float *a, const float *b, float *out)
    {
        *out = fminf(*a, *b);
    }

    template <>
    __device__ __forceinline__ void deepx_min<half>(const half *a, const half *b, half *out)
    {
        *out = __hmin(*a, *b);
    }

    template <>
    __device__ __forceinline__ void deepx_min<nv_bfloat16>(const nv_bfloat16 *a, const nv_bfloat16 *b, nv_bfloat16 *out)
    {
        *out = __hmin(*a, *b);
    }

    template <>
    __device__ __forceinline__ void deepx_min<int64_t>(const int64_t *a, const int64_t *b, int64_t *out)
    {
        *out = *a < *b ? *a : *b;
    }

    template <>
    __device__ __forceinline__ void deepx_min<int32_t>(const int32_t *a, const int32_t *b, int32_t *out)
    {
        *out = *a < *b ? *a : *b;
    }

    template <>
    __device__ __forceinline__ void deepx_min<int16_t>(const int16_t *a, const int16_t *b, int16_t *out)
    {
        *out = *a < *b ? *a : *b;
    }

    template <>
    __device__ __forceinline__ void deepx_min<int8_t>(const int8_t *a, const int8_t *b, int8_t *out)
    {
        *out = *a < *b ? *a : *b;
    }

    //todtype
    template <typename T, typename Dtype>
    __device__ __forceinline__ Dtype deepx_todtype(const T &a)
    {
        return static_cast<Dtype>(a);
    }
    //float<->half
    template <>
    __device__ __forceinline__ half deepx_todtype<float, half>(const float &src)
    {
        return __float2half(src);
    }

    template <>
    __device__ __forceinline__ float deepx_todtype<half, float>(const half &src)
    {
        return __half2float(src);
    }
    //float<->bfloat16
    template <>
    __device__ __forceinline__ nv_bfloat16 deepx_todtype<float, nv_bfloat16>(const float &src)
    {
        return __float2bfloat16(src);
    }
    template <>
    __device__ __forceinline__ float deepx_todtype<nv_bfloat16, float>(const nv_bfloat16 &src)
    {
        return __bfloat162float(src);
    }
 
}

#endif // DEEPX_TENSORFUNC_CUDA_MATH_CUH