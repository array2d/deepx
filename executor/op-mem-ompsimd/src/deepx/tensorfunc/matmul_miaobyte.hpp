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
            //这里需要进一步优化
            C.shape.rangeParallel(C.shape.dim(), [&A,&B,&C](const int idx,const std::vector<int> &indices,ThreadLocalVectors &tlv) {
                
                // int m=A.shape[-2];
                int k=A.shape[-1];
                // int n=B.shape[-1];
     
                std::copy(indices.begin(),indices.end()-2,tlv.get(0).begin());
                tlv.get(0)[indices.size()-2]=A.shape[-2];
                tlv.get(0)[indices.size()-1]=indices[-1];
                int aIdx=A.shape.linearat(tlv.get(0));
                std::copy(indices.begin(),indices.end()-2,tlv.get(1).begin());
                tlv.get(1)[indices.size()-2]=0;
                tlv.get(1)[indices.size()-1]=indices[-2];
                int bIdx=B.shape.linearat(tlv.get(1));
                int bstride=k;
                
                T sum=0;
                for(int l=0;l<k;l++){
                    sum+=A.data[aIdx+l]+B.data[bIdx+l*bstride];
                }
                C.data[idx]=sum;
            },{A.shape.dim(),B.shape.dim()});
        }
    };

}

#endif
