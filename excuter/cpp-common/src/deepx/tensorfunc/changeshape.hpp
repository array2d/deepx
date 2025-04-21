#ifndef DEEPX_TENSORFUNC_CHANGESHAPE_HPP
#define DEEPX_TENSORFUNC_CHANGESHAPE_HPP

#include <vector>
#include "deepx/tensor.hpp"
#include "stdutil/error.hpp"

namespace deepx::tensorfunc
{
    using namespace std;

    // reshape
    template <typename Author, typename T>
    struct reshapeDispatcher
    {
        static void reshape(const Tensor<T> &tensor, const std::vector<int> &new_shape, Tensor<T> &output) = delete;
    };

    template <typename Author, typename T>
    void reshape(const Tensor<T> &tensor, const std::vector<int> &new_shape, Tensor<T> &output)
    {
        reshapeDispatcher<Author, T>::reshape(tensor, new_shape, output);
    }

    // transpose
    template <typename Author, typename T>
    struct transposeDispatcher
    {
        static void transpose(const Tensor<T> &tensor, const std::vector<int> &dim_order, Tensor<T> &output) = delete;
    };

    template <typename Author, typename T>
    void transpose(const Tensor<T> &tensor, const std::vector<int> &dim_order, Tensor<T> &output)
    {
        transposeDispatcher<Author, T>::transpose(tensor, dim_order, output);
    }

    // concat
    template <typename Author, typename T>
    struct concatDispatcher
    {
        static void concat(const vector<Tensor<T> *> tensors, const int axis, Tensor<T> &C) = delete;
    };

    template <typename Author, typename T>
    void concat(const vector<Tensor<T> *> tensors, const int axis, Tensor<T> &C)
    {
        concatDispatcher<Author, T>::concat(tensors, axis, C);
    }

    // broadcastTo
    template <typename Author, typename T>
    struct broadcastToDispatcher
    {
        static void broadcastTo(const Tensor<T> &A, const vector<int> &new_shape, Tensor<T> &B) = delete;
    };

    template <typename Author, typename T>
    void broadcastTo(const Tensor<T> &A, const vector<int> &new_shape, Tensor<T> &B)
    {
        broadcastToDispatcher<Author, T>::broadcastTo(A, new_shape, B);
    }

    // gather
    template <typename Author, typename T, typename GatherAxisT>
    struct gatherDispatcher
    {
        static void gather(const Tensor<T> &input, const Tensor<GatherAxisT> &indices, const int axis, Tensor<T> &output) = delete;
    };

    template <typename Author, typename T, typename GatherAxisT>
    void gather(const Tensor<T> &input, const Tensor<GatherAxisT> &indices, const int axis, Tensor<T> &output)
    {
        gatherDispatcher<Author, T, GatherAxisT>::gather(input, indices, axis, output);
    }

    // // split
    // //  https://onnx.ai/onnx/operators/onnx__Split.html
    // template <typename Author, typename T>
    // struct splitDispatcher
    // {
    //     static void split(const Tensor<T> &A, const int axis, const std::vector<int> &splits, Tensor<T> *&B) = delete;
    //     static void split(const Tensor<T> &A, const int axis, const int num_outputs, Tensor<T> *&B) = delete;
    // };
    // template <typename Author, typename T>
    // void split(const Tensor<T> &A, const int axis, const std::vector<int> &splits, Tensor<T> *&B)
    // {
    //     splitDispatcher<Author, T>::split(A, axis, splits, B);
    // }

    // // split(tensor,axis,num_outputs)=>tensors
    // template <typename Author, typename T>
    // void split(const Tensor<T> &A, const int axis, const int num_outputs, Tensor<T> *&B)
    // {
    //     splitDispatcher<Author, T>::split(A, axis, num_outputs, B);
    // }

    // template <typename Author, typename T>
    // struct expandDispatcher
    // {
    //     static void expand(const Tensor<T> &A, const Shape &new_shape, Tensor<T> &B) = delete;
    // };

    // template <typename Author, typename T>
    // void expand(const Tensor<T> &A, const Shape &new_shape, Tensor<T> &B)
    // {
    //     expandDispatcher<Author, T>::expand(A, new_shape, B);
    // }

    // template <typename Author, typename T>
    // struct squeezeDispatcher
    // {
    //     static void squeeze(Tensor<T> &tensor) = delete;
    // };

    // template <typename Author, typename T>
    // void squeeze(Tensor<T> &tensor)
    // {
    //     squeezeDispatcher<Author, T>::squeeze(tensor);
    // }

    // template <typename Author, typename T>
    // struct unsqueezeDispatcher
    // {
    //     static void unsqueeze(Tensor<T> &tensor, const int axis) = delete;
    // };

    // template <typename Author, typename T>
    // void unsqueeze(Tensor<T> &tensor, const int axis)
    // {
    //     unsqueezeDispatcher<Author, T>::unsqueeze(tensor, axis);
    // }

    // template <typename Author, typename T>
    // struct flattenDispatcher
    // {
    //     static void flatten(Tensor<T> &tensor) = delete;
    // };

    // template <typename Author, typename T>
    // void flatten(Tensor<T> &tensor)
    // {
    //     flattenDispatcher<Author, T>::flatten(tensor);
    // }

    // template <typename Author, typename T>
    // struct paddingDispatcher
    // {
    //     static void padding(Tensor<T> &tensor, const Shape &new_shape) = delete;
    // };

    // template <typename Author, typename T>
    // void padding(Tensor<T> &tensor, const Shape &new_shape)
    // {
    //     paddingDispatcher<Author, T>::padding(tensor, new_shape);
    // }
}

#endif
