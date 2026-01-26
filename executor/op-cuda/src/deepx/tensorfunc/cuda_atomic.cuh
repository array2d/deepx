#ifndef DEEPX_TENSORFUNC_CUDA_ATOMIC_CUH
#define DEEPX_TENSORFUNC_CUDA_ATOMIC_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
namespace deepx::tensorfunc
{
     // atomicAdd
    template <typename T>
    __device__  __forceinline__ void deepx_atomicAdd(T *a, T b);

    template <>
    __device__ __forceinline__ void deepx_atomicAdd<double>(double *a, double b)
    {
        atomicAdd(a, b);
    }

    template <>
    __device__ __forceinline__ void deepx_atomicAdd<float>(float *a, float b)
    {
        atomicAdd(a, b);
    }

    template <>
    __device__  __forceinline__ void deepx_atomicAdd<half>(half *a, half b)
    {
        atomicAdd(a, b);
    }

    template <>
    __device__  __forceinline__ void deepx_atomicAdd<nv_bfloat16>(nv_bfloat16 *a, nv_bfloat16 b)
    {
        atomicAdd(a, b);
    }

    template <>
    __device__  __forceinline__ void deepx_atomicAdd<int64_t>(int64_t *a, int64_t b)
    {
        int64_t old = *a;
        int64_t assumed;
        do
        {
            assumed = old;
            old = atomicCAS((unsigned long long *)a, (unsigned long long)assumed, (unsigned long long)(assumed + b));
        } while (assumed != old);
        *a = old + b;
    }

    template <>
    __device__  __forceinline__ void deepx_atomicAdd<int32_t>(int32_t *a, int32_t b)
    {
        atomicAdd(a, b);
    }

    template <>
    __device__  __forceinline__ void deepx_atomicAdd<int16_t>(int16_t *a, int16_t b)
    {
        unsigned int *address_as_uint = (unsigned int *)((char *)a - ((size_t)a & 2));
        unsigned int old = *address_as_uint;
        unsigned int assumed;

        do
        {
            assumed = old;
            unsigned int new_val;
            if ((size_t)a & 2)
            {
                new_val = (old & 0x0000FFFF) | (((unsigned short)(((old >> 16) & 0xFFFF) + b)) << 16);
            }
            else
            {
                new_val = (old & 0xFFFF0000) | ((unsigned short)((old & 0xFFFF) + b));
            }
            old = atomicCAS(address_as_uint, assumed, new_val);
        } while (assumed != old);
    }

    template <>
    __device__  __forceinline__ void deepx_atomicAdd<int8_t>(int8_t *a, int8_t b)
    {
        unsigned int *address_as_uint = (unsigned int *)((char *)a - ((size_t)a & 3));
        unsigned int old = *address_as_uint;
        unsigned int assumed;
        unsigned int byte_offset = ((size_t)a & 3) * 8;
        unsigned int mask = 0xFF << byte_offset;

        do
        {
            assumed = old;
            unsigned char byte_val = (old >> byte_offset) & 0xFF;
            byte_val += b;
            unsigned int new_val = (old & ~mask) | (byte_val << byte_offset);
            old = atomicCAS(address_as_uint, assumed, new_val);
        } while (assumed != old);
    }


    // atomicMul
       // atomicMul
    template <typename T>
    __device__  __forceinline__ void deepx_atomicMul(T *a, T b);

    template <>
    __device__  __forceinline__ void deepx_atomicMul<double>(double *a, double b)
    {
        double old = *a;
        double assumed;
        do
        {
            assumed = old;
            old = __longlong_as_double(atomicCAS((unsigned long long int*)a, 
                                               __double_as_longlong(assumed),
                                               __double_as_longlong(assumed * b)));
        } while (assumed != old);
    }

    template <>
    __device__  __forceinline__ void deepx_atomicMul<float>(float *a, float b)
    {
        float old = *a;
        float assumed;
        do
        {
            assumed = old;
            old = __int_as_float(atomicCAS((int*)a, 
                                          __float_as_int(assumed),
                                          __float_as_int(assumed * b)));
        } while (assumed != old);
    }
    
    template <>
    __device__  __forceinline__ void deepx_atomicMul<half>(half *a, half b)
    {
        unsigned int *address_as_uint = (unsigned int *)((char *)a - ((size_t)a & 2));
        unsigned int old = *address_as_uint;
        unsigned int assumed;

        do
        {
            assumed = old;
            half assumed_half;
            if ((size_t)a & 2)
            {
                assumed_half = __ushort_as_half((unsigned short)(old >> 16));
                half new_half = __hmul(assumed_half, b);
                unsigned int new_val = (old & 0x0000FFFF) | ((unsigned int)__half_as_ushort(new_half) << 16);
                old = atomicCAS(address_as_uint, assumed, new_val);
            }
            else
            {
                assumed_half = __ushort_as_half((unsigned short)(old & 0xFFFF));
                half new_half = __hmul(assumed_half, b);
                unsigned int new_val = (old & 0xFFFF0000) | __half_as_ushort(new_half);
                old = atomicCAS(address_as_uint, assumed, new_val);
            }
        } while (assumed != old);
    }

    template <>
    __device__  __forceinline__ void deepx_atomicMul<nv_bfloat16>(nv_bfloat16 *a, nv_bfloat16 b)
    {
        unsigned int *address_as_uint = (unsigned int *)((char *)a - ((size_t)a & 2));
        unsigned int old = *address_as_uint;
        unsigned int assumed;

        do
        {
            assumed = old;
            nv_bfloat16 assumed_bf16;
            if ((size_t)a & 2)
            {
                assumed_bf16 = __ushort_as_bfloat16((unsigned short)(old >> 16));
                nv_bfloat16 new_bf16 = __hmul(assumed_bf16, b);
                unsigned int new_val = (old & 0x0000FFFF) | ((unsigned int)__bfloat16_as_ushort(new_bf16) << 16);
                old = atomicCAS(address_as_uint, assumed, new_val);
            }
            else
            {
                assumed_bf16 = __ushort_as_bfloat16((unsigned short)(old & 0xFFFF));
                nv_bfloat16 new_bf16 = __hmul(assumed_bf16, b);
                unsigned int new_val = (old & 0xFFFF0000) | __bfloat16_as_ushort(new_bf16);
                old = atomicCAS(address_as_uint, assumed, new_val);
            }
        } while (assumed != old);
    }

    template <>
    __device__  __forceinline__ void deepx_atomicMul<int64_t>(int64_t *a, int64_t b)
    {
        int64_t old = *a;
        int64_t assumed;
        do
        {
            assumed = old;
            old = atomicCAS((unsigned long long *)a, 
                          (unsigned long long)assumed, 
                          (unsigned long long)(assumed * b));
        } while (assumed != old);
    }

    template <>
    __device__  __forceinline__ void deepx_atomicMul<int32_t>(int32_t *a, int32_t b)
    {
        int32_t old = *a;
        int32_t assumed;
        do
        {
            assumed = old;
            old = atomicCAS((int32_t *)a, assumed, assumed * b);
        } while (assumed != old);
    }

    template <>
    __device__  __forceinline__ void deepx_atomicMul<int16_t>(int16_t *a, int16_t b)
    {
        unsigned int *address_as_uint = (unsigned int *)((char *)a - ((size_t)a & 2));
        unsigned int old = *address_as_uint;
        unsigned int assumed;

        do
        {
            assumed = old;
            unsigned int new_val;
            if ((size_t)a & 2)
            {
                int16_t assumed_short = (int16_t)(old >> 16);
                new_val = (old & 0x0000FFFF) | (((unsigned short)(assumed_short * b)) << 16);
            }
            else
            {
                int16_t assumed_short = (int16_t)(old & 0xFFFF);
                new_val = (old & 0xFFFF0000) | ((unsigned short)(assumed_short * b));
            }
            old = atomicCAS(address_as_uint, assumed, new_val);
        } while (assumed != old);
    }

    template <>
    __device__  __forceinline__ void deepx_atomicMul<int8_t>(int8_t *a, int8_t b)
    {
        unsigned int *address_as_uint = (unsigned int *)((char *)a - ((size_t)a & 3));
        unsigned int old = *address_as_uint;
        unsigned int assumed;
        unsigned int byte_offset = ((size_t)a & 3) * 8;
        unsigned int mask = 0xFF << byte_offset;

        do
        {
            assumed = old;
            int8_t byte_val = (old >> byte_offset) & 0xFF;
            byte_val *= b;
            unsigned int new_val = (old & ~mask) | ((byte_val & 0xFF) << byte_offset);
            old = atomicCAS(address_as_uint, assumed, new_val);
        } while (assumed != old);
    }
}

#endif
