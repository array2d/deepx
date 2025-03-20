#include "deepx/tensorfunc/cuda.hpp"

#include <cuda_fp64.h>
#include <cuda_fp32.h>
#include "deepx/tensor.hpp"

#include "deepx/tensorfunc/matmul.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/cuda.hpp"

namespace deepx::tensorfunc {

  #define BLOCK_SIZE 32

  __global__ void fp64MatmulKernel(double *C, const double *A, const double *B, 
                                   int M, int N, int K) {
      // 定义共享内存块，用于缓存A和B的矩阵块
      __shared__ double tileA[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ double tileB[BLOCK_SIZE][BLOCK_SIZE];
  
      // 计算当前线程处理的全局矩阵位置
      int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
      int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  
      double sum = 0.0;
  
      // 分块循环处理整个K维度
      for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
          // 计算当前块的起始位置
          int tiledK = t * BLOCK_SIZE;
  
          // 加载A的块到共享内存（行优先）
          int loadA_col = tiledK + threadIdx.x;
          if (row < M && loadA_col < K) {
              tileA[threadIdx.y][threadIdx.x] = A[row * K + loadA_col];
          } else {
              tileA[threadIdx.y][threadIdx.x] = 0.0;  // 填充0处理边界
          }
  
          // 加载B的块到共享内存（列优先等效处理）
          int loadB_row = tiledK + threadIdx.y;
          if (col < N && loadB_row < K) {
              tileB[threadIdx.y][threadIdx.x] = B[loadB_row * N + col];
          } else {
              tileB[threadIdx.y][threadIdx.x] = 0.0;  // 填充0处理边界
          }
  
          __syncthreads();  // 确保块加载完成
  
          // 计算当前块的矩阵乘法贡献
          for (int k = 0; k < BLOCK_SIZE; ++k) {
              sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
          }
  
          __syncthreads();  // 确保计算完成再加载下一块
      }
  
      // 只将有效范围内的结果写入全局内存
      if (row < M && col < N) {
          C[row * N + col] = sum;
      }
  }
  
  // 主机函数调用内核
  void fp64Matmul(double *d_C, const double *d_A, const double *d_B, 
                  int M, int N, int K) {
      dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
      dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                   (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
      
      fp64MatmulKernel<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, M, N, K);
  }
  

}

} // namespace tensorfunc
} // namespace deepx
