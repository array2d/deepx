#include <metal_stdlib>
using namespace metal;

// ═══════════════════════════════════════════════════════════
// miaobyte elementwise kernels (specialized per dtype)
// ops: add / sub / mul / div / max / min (binary)
//      relu / neg / abs / sqrt / exp / log / sin / cos / tan (unary)
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

// ── DIV ──

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

// ── MAX ──

kernel void max_f32(device const float* A [[buffer(0)]],
                    device const float* B [[buffer(1)]],
                    device float*       C [[buffer(2)]],
                    constant uint&      n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = max(A[gid], B[gid]); }
}

kernel void max_f16(device const half* A [[buffer(0)]],
                    device const half* B [[buffer(1)]],
                    device half*       C [[buffer(2)]],
                    constant uint&     n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = max(A[gid], B[gid]); }
}

kernel void max_i8(device const char* A [[buffer(0)]],
                   device const char* B [[buffer(1)]],
                   device char*       C [[buffer(2)]],
                   constant uint&     n [[buffer(3)]],
                   uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = max(A[gid], B[gid]); }
}

kernel void max_i16(device const short* A [[buffer(0)]],
                    device const short* B [[buffer(1)]],
                    device short*       C [[buffer(2)]],
                    constant uint&      n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = max(A[gid], B[gid]); }
}

kernel void max_i32(device const int* A [[buffer(0)]],
                    device const int* B [[buffer(1)]],
                    device int*       C [[buffer(2)]],
                    constant uint&    n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = max(A[gid], B[gid]); }
}

kernel void max_i64(device const long* A [[buffer(0)]],
                    device const long* B [[buffer(1)]],
                    device long*       C [[buffer(2)]],
                    constant uint&     n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = max(A[gid], B[gid]); }
}

// ── MIN ──

kernel void min_f32(device const float* A [[buffer(0)]],
                    device const float* B [[buffer(1)]],
                    device float*       C [[buffer(2)]],
                    constant uint&      n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = min(A[gid], B[gid]); }
}

kernel void min_f16(device const half* A [[buffer(0)]],
                    device const half* B [[buffer(1)]],
                    device half*       C [[buffer(2)]],
                    constant uint&     n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = min(A[gid], B[gid]); }
}

kernel void min_i8(device const char* A [[buffer(0)]],
                   device const char* B [[buffer(1)]],
                   device char*       C [[buffer(2)]],
                   constant uint&     n [[buffer(3)]],
                   uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = min(A[gid], B[gid]); }
}

kernel void min_i16(device const short* A [[buffer(0)]],
                    device const short* B [[buffer(1)]],
                    device short*       C [[buffer(2)]],
                    constant uint&      n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = min(A[gid], B[gid]); }
}

kernel void min_i32(device const int* A [[buffer(0)]],
                    device const int* B [[buffer(1)]],
                    device int*       C [[buffer(2)]],
                    constant uint&    n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = min(A[gid], B[gid]); }
}

kernel void min_i64(device const long* A [[buffer(0)]],
                    device const long* B [[buffer(1)]],
                    device long*       C [[buffer(2)]],
                    constant uint&     n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = min(A[gid], B[gid]); }
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

// ── NEG ──

kernel void neg_f32(device const float* X [[buffer(0)]],
                    device float*       Y [[buffer(1)]],
                    constant uint&      n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = -X[gid]; }
}

kernel void neg_f16(device const half* X [[buffer(0)]],
                    device half*       Y [[buffer(1)]],
                    constant uint&     n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = -X[gid]; }
}

kernel void neg_i8(device const char* X [[buffer(0)]],
                   device char*       Y [[buffer(1)]],
                   constant uint&     n [[buffer(2)]],
                   uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = (char)(-X[gid]); }
}

kernel void neg_i16(device const short* X [[buffer(0)]],
                    device short*       Y [[buffer(1)]],
                    constant uint&      n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = (short)(-X[gid]); }
}

kernel void neg_i32(device const int* X [[buffer(0)]],
                    device int*       Y [[buffer(1)]],
                    constant uint&    n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = -X[gid]; }
}

kernel void neg_i64(device const long* X [[buffer(0)]],
                    device long*       Y [[buffer(1)]],
                    constant uint&     n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = -X[gid]; }
}

// ── ABS ──

kernel void abs_f32(device const float* X [[buffer(0)]],
                    device float*       Y [[buffer(1)]],
                    constant uint&      n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = abs(X[gid]); }
}

kernel void abs_f16(device const half* X [[buffer(0)]],
                    device half*       Y [[buffer(1)]],
                    constant uint&     n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = abs(X[gid]); }
}

kernel void abs_i8(device const char* X [[buffer(0)]],
                   device char*       Y [[buffer(1)]],
                   constant uint&     n [[buffer(2)]],
                   uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = abs(X[gid]); }
}

kernel void abs_i16(device const short* X [[buffer(0)]],
                    device short*       Y [[buffer(1)]],
                    constant uint&      n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = abs(X[gid]); }
}

kernel void abs_i32(device const int* X [[buffer(0)]],
                    device int*       Y [[buffer(1)]],
                    constant uint&    n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = abs(X[gid]); }
}

kernel void abs_i64(device const long* X [[buffer(0)]],
                    device long*       Y [[buffer(1)]],
                    constant uint&     n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = abs(X[gid]); }
}

// ── SQRT (浮点 only) ──

kernel void sqrt_f32(device const float* X [[buffer(0)]],
                     device float*       Y [[buffer(1)]],
                     constant uint&      n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = sqrt(X[gid]); }
}

kernel void sqrt_f16(device const half* X [[buffer(0)]],
                     device half*       Y [[buffer(1)]],
                     constant uint&     n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = sqrt(X[gid]); }
}

// ── EXP (浮点 only) ──

kernel void exp_f32(device const float* X [[buffer(0)]],
                    device float*       Y [[buffer(1)]],
                    constant uint&      n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = exp(X[gid]); }
}

kernel void exp_f16(device const half* X [[buffer(0)]],
                    device half*       Y [[buffer(1)]],
                    constant uint&     n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = exp(X[gid]); }
}

// ── LOG (浮点 only) ──

kernel void log_f32(device const float* X [[buffer(0)]],
                    device float*       Y [[buffer(1)]],
                    constant uint&      n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = log(X[gid]); }
}

kernel void log_f16(device const half* X [[buffer(0)]],
                    device half*       Y [[buffer(1)]],
                    constant uint&     n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = log(X[gid]); }
}

// ── SIN (浮点 only) ──

kernel void sin_f32(device const float* X [[buffer(0)]],
                    device float*       Y [[buffer(1)]],
                    constant uint&      n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = sin(X[gid]); }
}

kernel void sin_f16(device const half* X [[buffer(0)]],
                    device half*       Y [[buffer(1)]],
                    constant uint&     n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = sin(X[gid]); }
}

// ── COS (浮点 only) ──

kernel void cos_f32(device const float* X [[buffer(0)]],
                    device float*       Y [[buffer(1)]],
                    constant uint&      n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = cos(X[gid]); }
}

kernel void cos_f16(device const half* X [[buffer(0)]],
                    device half*       Y [[buffer(1)]],
                    constant uint&     n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = cos(X[gid]); }
}

// ── TAN (浮点 only) ──

kernel void tan_f32(device const float* X [[buffer(0)]],
                    device float*       Y [[buffer(1)]],
                    constant uint&      n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = tan(X[gid]); }
}

kernel void tan_f16(device const half* X [[buffer(0)]],
                    device half*       Y [[buffer(1)]],
                    constant uint&     n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = tan(X[gid]); }
}
