#include <metal_stdlib>
using namespace metal;

// ═══════════════════════════════════════════════════════════
// miaobyte elementwise kernels (specialized per dtype)
// ops: add / sub / mul / div / relu
// ═══════════════════════════════════════════════════════════

// ── ADD ──

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

// ── SUB ──

kernel void sub_f32(device const float* A [[buffer(0)]],
                    device const float* B [[buffer(1)]],
                    device float*       C [[buffer(2)]],
                    constant uint&      n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = A[gid] - B[gid]; }
}

kernel void sub_f16(device const half* A [[buffer(0)]],
                    device const half* B [[buffer(1)]],
                    device half*       C [[buffer(2)]],
                    constant uint&     n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = A[gid] - B[gid]; }
}

kernel void sub_i8(device const char* A [[buffer(0)]],
                   device const char* B [[buffer(1)]],
                   device char*       C [[buffer(2)]],
                   constant uint&     n [[buffer(3)]],
                   uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = (char)(A[gid] - B[gid]); }
}

kernel void sub_i16(device const short* A [[buffer(0)]],
                    device const short* B [[buffer(1)]],
                    device short*       C [[buffer(2)]],
                    constant uint&      n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = (short)(A[gid] - B[gid]); }
}

kernel void sub_i32(device const int* A [[buffer(0)]],
                    device const int* B [[buffer(1)]],
                    device int*       C [[buffer(2)]],
                    constant uint&    n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = A[gid] - B[gid]; }
}

kernel void sub_i64(device const long* A [[buffer(0)]],
                    device const long* B [[buffer(1)]],
                    device long*       C [[buffer(2)]],
                    constant uint&     n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = A[gid] - B[gid]; }
}

// ── MUL ──

kernel void mul_f32(device const float* A [[buffer(0)]],
                    device const float* B [[buffer(1)]],
                    device float*       C [[buffer(2)]],
                    constant uint&      n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = A[gid] * B[gid]; }
}

kernel void mul_f16(device const half* A [[buffer(0)]],
                    device const half* B [[buffer(1)]],
                    device half*       C [[buffer(2)]],
                    constant uint&     n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = A[gid] * B[gid]; }
}

kernel void mul_i8(device const char* A [[buffer(0)]],
                   device const char* B [[buffer(1)]],
                   device char*       C [[buffer(2)]],
                   constant uint&     n [[buffer(3)]],
                   uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = (char)(A[gid] * B[gid]); }
}

kernel void mul_i16(device const short* A [[buffer(0)]],
                    device const short* B [[buffer(1)]],
                    device short*       C [[buffer(2)]],
                    constant uint&      n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = (short)(A[gid] * B[gid]); }
}

kernel void mul_i32(device const int* A [[buffer(0)]],
                    device const int* B [[buffer(1)]],
                    device int*       C [[buffer(2)]],
                    constant uint&    n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = A[gid] * B[gid]; }
}

kernel void mul_i64(device const long* A [[buffer(0)]],
                    device const long* B [[buffer(1)]],
                    device long*       C [[buffer(2)]],
                    constant uint&     n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = A[gid] * B[gid]; }
}

// ── DIV (浮点 only, 整数除零无保护) ──

kernel void div_f32(device const float* A [[buffer(0)]],
                    device const float* B [[buffer(1)]],
                    device float*       C [[buffer(2)]],
                    constant uint&      n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = A[gid] / B[gid]; }
}

kernel void div_f16(device const half* A [[buffer(0)]],
                    device const half* B [[buffer(1)]],
                    device half*       C [[buffer(2)]],
                    constant uint&     n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = A[gid] / B[gid]; }
}

// ── RELU ──

kernel void relu_f32(device const float* X [[buffer(0)]],
                     device float*       Y [[buffer(1)]],
                     constant uint&      n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = max(X[gid], 0.0f); }
}

kernel void relu_f16(device const half* X [[buffer(0)]],
                     device half*       Y [[buffer(1)]],
                     constant uint&     n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = max(X[gid], half(0.0)); }
}

kernel void relu_i8(device const char* X [[buffer(0)]],
                    device char*       Y [[buffer(1)]],
                    constant uint&     n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = max(X[gid], char(0)); }
}

kernel void relu_i16(device const short* X [[buffer(0)]],
                     device short*       Y [[buffer(1)]],
                     constant uint&      n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = max(X[gid], short(0)); }
}

kernel void relu_i32(device const int* X [[buffer(0)]],
                     device int*       Y [[buffer(1)]],
                     constant uint&    n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = max(X[gid], 0); }
}

kernel void relu_i64(device const long* X [[buffer(0)]],
                     device long*       Y [[buffer(1)]],
                     constant uint&     n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = max(X[gid], long(0)); }
}
