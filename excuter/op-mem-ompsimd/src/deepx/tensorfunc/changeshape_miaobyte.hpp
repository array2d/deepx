#ifndef DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_HPP
#define DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_HPP

#include <stdexcept>
#include <vector>

#include "deepx/tensor.hpp"
#include "deepx/shape_changeshape.hpp"
#include "deepx/tensorfunc/changeshape.hpp"
#include "deepx/tensorfunc/authors.hpp"
namespace deepx::tensorfunc
{
    // reshape
    template <typename T>
    struct reshapeDispatcher<miaobyte, T>
    {
        static void reshape(const Tensor<T> &tensor, const std::vector<int> &shape, Tensor<T> &output)
        { // 参数改为单个tensor引用

            int new_prod = 1;
            for (int dim : shape)
            {
                new_prod *= dim;
            }

            if (tensor.shape.size != new_prod)
            {
                throw std::invalid_argument("Shape size mismatch");
            }
            Shape newshape(shape);
            if (tensor.data == output.data)
            {
                output.shape.shape = newshape.shape;
                output.shape.strides = newshape.strides;
            }
            else
            {
                output.shape.shape = newshape.shape;
                output.shape.strides = newshape.strides;
                output.copyer(tensor.data, output.data, tensor.shape.size);
            }
        }
    };
    // transpose
    template <typename T>
    struct transposeDispatcher<miaobyte, T>
    {
        static void transpose(const Tensor<T> &tensor, const std::vector<int> &dim_order, Tensor<T> &output)
        {

            if (dim_order.size() != tensor.shape.dim)
            {
                throw std::invalid_argument("dimOrder size does not match the number of dimensions in the TensorCPU.");
            }
            if (output.shape.size != tensor.shape.size)
            {
                throw std::runtime_error("transpose error!shape");
            }
            output.shape.rangeParallel(dim_order.size(), [&tensor, &output, &dim_order](int idx_linear, const std::vector<int> &indices, ThreadLocalVectors &tlv)
                                       {
                                        
                            for (size_t i = 0; i < dim_order.size(); ++i) {
                                tlv.get(0)[dim_order[i]] = indices[i];
                            }
                            output.data[idx_linear]= tensor.data[tensor.shape.linearat(tlv.get(0))]; }, {tensor.shape.dim});
        }
    };
    // concat
    template <typename T>
    struct concatDispatcher<miaobyte, T>
    {
        static void concat(const vector<Tensor<T> *> tensors, const int axis, Tensor<T> &result)
        {
            // checkshape
            if (!checkShapeConcat(tensors, axis, result))
            {
                throw TensorShapeError("Output tensor shape size must match the sum of input tensor shape sizes for concat");
            }
            int dimC = axis + 1;
            result.shape.rangeParallel(dimC, [&](const int idx, const std::vector<int> &indices)
                                       {
                        int concatIdxCurrentTensor=indices[axis];;
                        int tensorIdx=0;
                        while (tensorIdx < tensors.size()  ) {
                            if (concatIdxCurrentTensor<tensors[tensorIdx]->shape[axis]) {
                                break;
                            }else{
                                concatIdxCurrentTensor-=tensors[tensorIdx]->shape[axis];
                                tensorIdx++;
                            }
                        }
                        
                        std::vector<int> currentTensorIndices=indices;
                        currentTensorIndices[axis]=concatIdxCurrentTensor;

                        int idxCurrentTensor=tensors[tensorIdx]->shape.linearat(currentTensorIndices);
                        int copylen=tensors[tensorIdx]->shape.strides[axis];
                        std::copy(tensors[tensorIdx]->data+idxCurrentTensor,tensors[tensorIdx]->data+idxCurrentTensor+copylen,result.data+idx); });
        }
    };

    vector<int> fromBroadcastIndices(const vector<BroadcastMap> &broadcastMap, const vector<int> &broadcastIndices)
    {
        vector<int> srcindices;
        for (int i = 0, j = 0; i < broadcastMap.size(); ++i)
        {
            switch (broadcastMap[i])
            {
            case xTox:
                srcindices.push_back(broadcastIndices[i]);
                break;
            case nullTo1:
                break;
            case xTo1:
                srcindices.push_back(0);
                break;
            }
        }
        return srcindices;
    }

    template <typename T>
    struct broadcastToDispatcher<miaobyte, T>
    {
        static void broadcastTo(const Tensor<T> &A, const vector<int> &new_shape, Tensor<T> &B)
        {
            auto A_broadcastShape = broadcastShape(A.shape.shape, new_shape);
            if (A_broadcastShape.empty() || A_broadcastShape != new_shape)
            {
                throw TensorShapeError("Broadcast shape mismatch");
            }
            auto bmap = broadcastMap(A.shape.shape, new_shape);

            B.shape.rangeParallel(B.shape.dim, [&](const int idx, const std::vector<int> &bindices)
                                  {
                        vector<int> aindices=fromBroadcastIndices(bmap, bindices);
                        B.data[idx] = A.data[A.shape.linearat(aindices)]; });
        }
    };

    // indexselect
    // output_indices,index,index_indices,gatheraxis->input_indices
    template <typename GatherAxisT>
    void fromIndexselectIndices(const vector<int> &output_indices, const Tensor<GatherAxisT> &index,vector<int> &index_indices, const int gatherAxis, vector<int> &input_indices)
    {
 
        std::copy(output_indices.begin(), output_indices.begin()+gatherAxis, input_indices.begin());
        std::copy(output_indices.begin()+gatherAxis,output_indices.begin()+gatherAxis+index_indices.size(), index_indices.begin());
        int index_idx=index.shape.linearat(index_indices);
        input_indices[gatherAxis] = index.data[index_idx];
        std::copy(output_indices.begin()+gatherAxis+index_indices.size(),output_indices.begin()+output_indices.size(), input_indices.begin()+gatherAxis+1);

    }

    template <typename T, typename GatherAxisT>
    struct indexselectDispatcher<miaobyte, T, GatherAxisT>
    {
        static void indexselect(const Tensor<T> &input, const Tensor<GatherAxisT> &index, const int axis, Tensor<T> &output)
        {
            int gatherAxis = axis < 0 ? input.shape.dim + axis : axis;
            if (gatherAxis < 0 || gatherAxis >= input.shape.dim)
            {
                throw std::invalid_argument("Axis is out of bounds");
            }

            vector<int>  gatherShape =  indexselectShape(input.shape.shape,index.shape.shape,gatherAxis);
            if (gatherShape.empty() || gatherShape != output.shape.shape)
            {
                throw TensorShapeError("Indexselect shape mismatch");
            }
            output.shape.rangeParallel(output.shape.dim, [&](const int idx, const std::vector<int> &output_indices, ThreadLocalVectors &tlv)
                                       {  
                            fromIndexselectIndices(output_indices, index,tlv.get(1), gatherAxis, tlv.get(0));
                            output.data[idx] = input.data[input.shape.linearat(tlv.get(0))]; 
                        },
                    {input.shape.dim,index.shape.dim});
        }
    };

    // template <typename T>
    // void split(const Tensor<T> &tensor, const int axis, std::vector<Tensor<T> *> &results)
    // {
    //     tensor.shape.rangeParallel(axis, [&](const int idx, const std::vector<int> &indices)
    //                                {
    //                     int splitIdxCurrentTensor=indices[axis];
    //                     int tensorIdx=0;
    //                     while (tensorIdx < results.size()  ) {
    //                         if (splitIdxCurrentTensor<results[tensorIdx]->shape[axis]) {
    //                             break;
    //                         }else{
    //                             splitIdxCurrentTensor-=results[tensorIdx]->shape[axis];
    //                             tensorIdx++;
    //                         }
    //                     }
    //                     std::vector<int> currentTensorIndices=indices;
    //                     currentTensorIndices[axis]=splitIdxCurrentTensor;
    //                     results[tensorIdx]->shape.linearat(currentTensorIndices);
    //                     int idxCurrentTensor=results[tensorIdx]->shape.linearat(currentTensorIndices);
    //                     int copylen=results[tensorIdx]->shape.strides[axis];
    //                     std::copy(tensor.data+idxCurrentTensor,tensor.data+idxCurrentTensor+copylen,results[tensorIdx]->data+idx); });
    // }

    // // 扩展张量维度 - 将形状中为1的维度扩展到目标维度
    // template <typename T>
    // void expand(const Tensor<T> &input, Tensor<T> &output)
    // {
    //     // 检查输入和目标形状的兼容性
    //     if (input.shape.dim != output.shape.dim)
    //     {
    //         throw std::invalid_argument("expand维度不匹配: 输入维度 " +
    //                                     std::to_string(input.shape.dim) +
    //                                     ", 目标维度 " +
    //                                     std::to_string(output.shape.dim) +
    //                                     "请先前dim补1的方式reshape");
    //     }

    //     for (size_t i = 0; i < input.shape.dim; ++i)
    //     {
    //         if (input.shape[i] != output.shape[i] && input.shape[i] != 1)
    //         {
    //             throw std::invalid_argument("维度 " + std::to_string(i) +
    //                                         " 不能被扩展: " +
    //                                         std::to_string(input.shape[i]) +
    //                                         " 到 " +
    //                                         std::to_string(output.shape[i]));
    //         }
    //     }

    //     // 创建扩展映射
    //     std::vector<BroadcastMap> bm = broadcastMap(input.shape.shape, output.shape.shape);

    //     // 找到最后一个需要扩展的维度
    //     int last_expand_dim = -1;
    //     for (int i = input.shape.dim - 1; i >= 0; --i)
    //     {
    //         if (input.shape[i] != output.shape.shape[i])
    //         {
    //             last_expand_dim = i;
    //             break;
    //         }
    //     }

    //     // 如果最后几个维度不需要扩展，可以连续复制
    //     if (last_expand_dim < output.shape.dim - 1)
    //     {
    //         int copy_len = output.shape.strides[last_expand_dim + 1];
    //         output.shape.rangeParallel(last_expand_dim + 1, [&bm, &output, &input, copy_len](int idx_linear, const std::vector<int> &indices, std::vector<int> &oldIndices)
    //                                    {
    //                 fromBroadcastIndices(bm, indices, oldIndices);
    //                 int idx_old = input.shape.linearat(oldIndices);
    //                 std::copy(input.data + idx_old,
    //                          input.data + idx_old + copy_len,
    //                          output.data + idx_linear); }, input.shape.dim);
    //     }
    //     else
    //     {
    //         output.shape.rangeParallel(output.shape.dim, [&bm, &output, &input](int idx_linear, const std::vector<int> &indices, std::vector<int> &oldIndices)
    //                                    {
    //                 fromBroadcastIndices(bm, indices, oldIndices);
    //                 int idx_old = input.shape.linearat(oldIndices);
    //                 output.data[idx_linear] = input.data[idx_old]; }, input.shape.dim);
    //     }
    // }
}
#endif // DEEPX_TENSORFUNC_CHANGESHAPE_MIAOBYTE_HPP