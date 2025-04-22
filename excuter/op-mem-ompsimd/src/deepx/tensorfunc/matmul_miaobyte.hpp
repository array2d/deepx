#ifndef DEEPX_TENSORFUNC_MATMUL_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_MATMUL_MIAOBYTE_HPP

#include "deepx/tensorfunc/matmul.hpp"

namespace deepx::tensorfunc
{
    template <typename T>
    struct matmulDispatcher<miaobyte,T>
    {
        static void matmul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            if (!check_matmul_shape(A.shape, B.shape))
            {
                throw std::invalid_argument("A.shape could matmul with B.shape");
            }
            //TODO
            //这里如果对二维矩阵运算，则omp并行不起来，因为C.shape.dim - 2刚好=0
            C.shape.rangeParallel(C.shape.dim - 2, [&](const std::vector<int> &indices)
                                  {
                        int aIdx=A.shape.linearat(indices);
                        int bIdx=B.shape.linearat(indices);
                        int cIdx=C.shape.linearat(indices);
                        int m=A.shape[-2];
                        int k=A.shape[-1];
                        int n=B.shape[-1];
                        for(int i=0;i<m;i++){
                            for(int j=0;j<n;j++){
                                T sum=0;
                                for(int l=0;l<k;l++){
                                    sum+=A.data[aIdx+i*k+l]*B.data[bIdx+l*n+j];
                                }
                                C.data[cIdx+i*n+j]=sum;
                            }
                        } });
        }
    };

}

#endif
