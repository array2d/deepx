#ifndef DEEPX_TENSORFUNC_MATMUL_CBLAS_HPP
#define DEEPX_TENSORFUNC_MATMUL_CBLAS_HPP

#include <cblas.h> // 如果使用 OpenBLAS
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/matmul.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{
  template <>
  struct matmulDispatcher<cblas, float>
  {
    static void matmul(const Tensor<float> &a, const Tensor<float> &b, Tensor<float> &c)
    {
      if (!check_matmul_shape(a.shape, b.shape))
      {
        throw std::invalid_argument("a.shape could matmul with b.shape");
      }
      // 计算batch size (将除最后两维外的所有维度展平)
      int64_t batch_size = 1;
      for (int i = 0; i < a.shape.dim - 2; ++i)
      {
        batch_size *= a.shape[i];
      }

      // 获取矩阵维度
      int64_t m = a.shape[-2]; // 倒数第二维
      int64_t k = a.shape[-1]; // 最后一维
      int64_t n = b.shape[-1]; // B的最后一维

      // 设置每个矩阵的步长
      int64_t lda = k;
      int64_t ldb = n;
      int64_t ldc = n;

      // 计算每个batch的指针偏移
      std::vector<const float *> a_array(batch_size);
      std::vector<const float *> b_array(batch_size);
      std::vector<float *> c_array(batch_size);

      for (int64_t i = 0; i < batch_size; ++i)
      {
        a_array[i] = a.data + i * m * k;
        b_array[i] = b.data + i * k * n;
        c_array[i] = c.data + i * m * n;
      }

      for (int64_t i = 0; i < batch_size; ++i)
      {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k,
                    1.0f,
                    a_array[i], lda,
                    b_array[i], ldb,
                    0.0f,
                    c_array[i], ldc);
      }
    }
  };

  template <>
  struct matmulDispatcher<cblas, double>
  {
    static void matmul(const Tensor<double> &a, const Tensor<double> &b, Tensor<double> &c)
    {
      if (!check_matmul_shape(a.shape, b.shape))
      {
        throw std::invalid_argument("a.shape could matmul with b.shape");
      }
      // 计算batch size (将除最后两维外的所有维度展平)
      int64_t batch_size = 1;
      for (int i = 0; i < a.shape.dim - 2; ++i)
      {
        batch_size *= a.shape[i];
      }

      // 获取矩阵维度
      int64_t m = a.shape[-2]; // 倒数第二维
      int64_t k = a.shape[-1]; // 最后一维
      int64_t n = b.shape[-1]; // B的最后一维

      // 设置每个矩阵的步长
      int64_t lda = k;
      int64_t ldb = n;
      int64_t ldc = n;

      // 计算每个batch的指针偏移
      std::vector<const double *> a_array(batch_size);
      std::vector<const double *> b_array(batch_size);
      std::vector<double *> c_array(batch_size);

      for (int64_t i = 0; i < batch_size; ++i)
      {
        a_array[i] = a.data + i * m * k;
        b_array[i] = b.data + i * k * n;
        c_array[i] = c.data + i * m * n;
      }

      for (int64_t i = 0; i < batch_size; ++i)
      {
        // C = α * op(A) * op(B) + β * C
        cblas_dgemm(CblasRowMajor, // 存储顺序
                    CblasNoTrans,  // op(A) = A
                    CblasNoTrans,  // op(B) = B
                    m, n, k,       // A[m×k], B[k×n], C[m×n]
                    1.0,           // α = 1.0
                    a_array[i],    // A矩阵指针
                    lda,           // A的leading dimension（行主序时为列数k）
                    b_array[i],    // B矩阵指针
                    ldb,           // B的leading dimension（行主序时为列数n）
                    0.0,           // β = 0.0
                    c_array[i],    // C矩阵指针
                    ldc);          // C的leading dimension（行主序时为列数n）
      }
    }
  };

  template <typename T>
  struct matmuladdDispatcher<cblas, T>
  {
    static void matmuladd(const Tensor<T> &a, const Tensor<T> &b, const T &alpha, const T &beta, Tensor<T> &c)
    {
      if (!check_shape(a.shape, b.shape))
      {
        throw std::invalid_argument("a.shape could matmul with b.shape");
      }
      c.shape.rangeParallel(c.shape.dim - 2, [&](const std::vector<int> &indices)
                            {
                        int aIdx=a.shape.linearat(indices);
                        int bIdx=b.shape.linearat(indices);
                        int cIdx=c.shape.linearat(indices);
                        int m=a.shape[-2];
                        int k=a.shape[-1];
                        int n=b.shape[-1];
                        for(int i=0;i<m;i++){
                            for(int j=0;j<n;j++){
                                T sum=0;
                                for(int l=0;l<k;l++){
                                    sum+=a.data[aIdx+i*k+l]*b.data[bIdx+l*n+j];
                                }
                                c.data[cIdx+i*n+j]=alpha*sum+beta*c.data[cIdx+i*n+j];
                            }
                        } });
    }
  };

  template <>
  struct matmuladdDispatcher<cblas, float>
  {
    static void matmuladd(const Tensor<float> &a, const Tensor<float> &b, const float &alpha, const float &beta, Tensor<float> &c)
    {
      if (!check_matmul_shape(a.shape, b.shape))
      {
        throw std::invalid_argument("a.shape could matmul with b.shape");
      }
      // 计算batch size (将除最后两维外的所有维度展平)
      // 计算batch size (将除最后两维外的所有维度展平)
      int64_t batch_size = 1;
      for (int i = 0; i < a.shape.dim - 2; ++i)
      {
        batch_size *= a.shape[i];
      }

      // 获取矩阵维度
      int64_t m = a.shape[-2]; // 倒数第二维
      int64_t k = a.shape[-1]; // 最后一维
      int64_t n = b.shape[-1]; // B的最后一维

      // 设置每个矩阵的步长
      int64_t lda = k;
      int64_t ldb = n;
      int64_t ldc = n;

      // 计算每个batch的指针偏移
      std::vector<const float *> a_array(batch_size);
      std::vector<const float *> b_array(batch_size);
      std::vector<float *> c_array(batch_size);

      for (int64_t i = 0; i < batch_size; ++i)
      {
        a_array[i] = a.data + i * m * k;
        b_array[i] = b.data + i * k * n;
        c_array[i] = c.data + i * m * n;
      }

      for (int64_t i = 0; i < batch_size; ++i)
      {
        // C = α * op(A) * op(B) + β * C
        cblas_sgemm(CblasRowMajor, // 存储顺序
                    CblasNoTrans,  // op(A) = A
                    CblasNoTrans,  // op(B) = B
                    m, n, k,       // A[m×k], B[k×n], C[m×n]
                    alpha,         // α = 1.0
                    a_array[i],    // A矩阵指针
                    lda,           // A的leading dimension（行主序时为列数k）
                    b_array[i],    // B矩阵指针
                    ldb,           // B的leading dimension（行主序时为列数n）
                    beta,          // β = 0.0
                    c_array[i],    // C矩阵指针
                    ldc);          // C的leading dimension（行主序时为列数n）
      }
    }
  };

  template <>
  struct matmuladdDispatcher<cblas, double>
  {
    static void matmuladd(const Tensor<double> &a, const Tensor<double> &b, const double &alpha, const double &beta, Tensor<double> &c)
    {
      if (!check_matmul_shape(a.shape, b.shape))
      {
        throw std::invalid_argument("a.shape could matmul with b.shape");
      }
      // 计算batch size (将除最后两维外的所有维度展平)
      // 计算batch size (将除最后两维外的所有维度展平)
      int64_t batch_size = 1;
      for (int i = 0; i < a.shape.dim - 2; ++i)
      {
        batch_size *= a.shape[i];
      }

      // 获取矩阵维度
      int64_t m = a.shape[-2]; // 倒数第二维
      int64_t k = a.shape[-1]; // 最后一维
      int64_t n = b.shape[-1]; // B的最后一维

      // 设置每个矩阵的步长
      int64_t lda = k;
      int64_t ldb = n;
      int64_t ldc = n;

      // 计算每个batch的指针偏移
      std::vector<const double *> a_array(batch_size);
      std::vector<const double *> b_array(batch_size);
      std::vector<double *> c_array(batch_size);

      for (int64_t i = 0; i < batch_size; ++i)
      {
        a_array[i] = a.data + i * m * k;
        b_array[i] = b.data + i * k * n;
        c_array[i] = c.data + i * m * n;
      }

      for (int64_t i = 0; i < batch_size; ++i)
      {
        // C = α * op(A) * op(B) + β * C
        cblas_dgemm(CblasRowMajor, // 存储顺序
                    CblasNoTrans,  // op(A) = A
                    CblasNoTrans,  // op(B) = B
                    m, n, k,       // A[m×k], B[k×n], C[m×n]
                    alpha,         // α = 1.0
                    a_array[i],    // A矩阵指针
                    lda,           // A的leading dimension（行主序时为列数k）
                    b_array[i],    // B矩阵指针
                    ldb,           // B的leading dimension（行主序时为列数n）
                    beta,          // β = 0.0
                    c_array[i],    // C矩阵指针
                    ldc);          // C的leading dimension（行主序时为列数n）
      }
    }
  };
}
#endif // DEEPX_TENSORFUNC_MATMUL_CBLAS_HPP