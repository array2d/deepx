#include <metal_stdlib>
using namespace metal;

// miaobyte elementwise add kernels (specialized per dtype)

kernel void add_f32(device const float* A [[buffer(0)]],
                    device const float* B [[buffer(1)]],
                    device float*       C [[buffer(2)]],
                    constant uint&      n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = A[gid] + B[gid]; }
}

kernel void add_f16(device const half* A [[buffer(0)]],
                    device const half* B [[buffer(1)]],
                    device half*       C [[buffer(2)]],
                    constant uint&     n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = A[gid] + B[gid]; }
}

kernel void add_i8(device const char* A [[buffer(0)]],
                   device const char* B [[buffer(1)]],
                   device char*       C [[buffer(2)]],
                   constant uint&     n [[buffer(3)]],
                   uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = (char)(A[gid] + B[gid]); }
}

kernel void add_i16(device const short* A [[buffer(0)]],
                    device const short* B [[buffer(1)]],
                    device short*       C [[buffer(2)]],
                    constant uint&      n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = (short)(A[gid] + B[gid]); }
}

kernel void add_i32(device const int* A [[buffer(0)]],
                    device const int* B [[buffer(1)]],
                    device int*       C [[buffer(2)]],
                    constant uint&    n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = A[gid] + B[gid]; }
}

kernel void add_i64(device const long* A [[buffer(0)]],
                    device const long* B [[buffer(1)]],
                    device long*       C [[buffer(2)]],
                    constant uint&     n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = A[gid] + B[gid]; }
}
