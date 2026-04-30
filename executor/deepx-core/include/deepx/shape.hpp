#ifndef DEEPX_SHAPE_HPP
#define DEEPX_SHAPE_HPP

#include <vector>
#include <string>
#include <functional>
#include <fstream>
#include <utility>
#include <stdexcept>

#include "stdutil/fs.hpp"
#include "deepx/precision.hpp"

namespace deepx
{
    struct Shape
    {
        Precision dtype;
        std::vector<int> shape;
        std::vector<int> strides;
        int64_t size;
        int64_t bytes() const;

        Shape() = default;
        Shape(const std::vector<int> &shape);
        Shape(const std::initializer_list<int> &shape);
        Shape(const int *shape, int dim);
        void setshape(const int *shape, int dim);
        int dim() const;
        int operator[](int index) const;
        int &operator[](int index);
        bool operator==(const Shape &shape) const { return shape.shape == shape.shape; }
        void print() const;
        void range(int dimCount, std::function<void(const std::vector<int> &indices)> func) const;
        void range(int dimCount, std::function<void(const int idx_linear, const std::vector<int> &indices)> func) const;
        void range(int dimCount, std::function<void(const int idx_linear)> func) const;

        int linearat(const std::vector<int> &indices) const;
        std::vector<int> linearto(int idx_linear) const;

        std::string toYaml() const;
        void fromYaml(const std::string &yaml);

        void saveShape(const std::string &tensorPath) const;

        static std::pair<std::string, Shape> loadShape(const std::string &path);
    };

// ── range() serial, inline ──
inline void Shape::range(int dimCount, std::function<void(const std::vector<int> &indices)> func) const
{
    if (dimCount < 0) dimCount += dim();
    if (dimCount > dim()) throw std::invalid_argument("dimCount exceeds the number of dimensions");
    int totalSize = 1;
    for (int i = 0; i < dimCount; ++i) totalSize *= shape[i];
    std::vector<int> indices(dimCount, 0);
    for (int idx = 0; idx < totalSize; idx++)
    {
        int idx_ = idx;
        for (int dim = dimCount - 1; dim >= 0; --dim)
        {
            indices[dim] = idx_ % shape[dim];
            idx_ /= shape[dim];
        }
        func(indices);
    }
}

inline void Shape::range(int dimCount, std::function<void(const int idx_linear, const std::vector<int> &indices)> func) const
{
    if (dimCount < 0) dimCount += dim();
    if (dimCount > dim()) throw std::invalid_argument("dimCount exceeds the number of dimensions");
    int totalSize = 1, stride = 1;
    for (int i = 0; i < dimCount; ++i) totalSize *= shape[i];
    for (int i = dimCount; i < dim(); ++i) stride *= shape[i];
    std::vector<int> indices(dimCount, 0);
    for (int idx = 0; idx < totalSize; idx++)
    {
        int idx_ = idx;
        for (int dim = dimCount - 1; dim >= 0; --dim)
        {
            indices[dim] = idx_ % shape[dim];
            idx_ /= shape[dim];
        }
        func(idx * stride, indices);
    }
}

inline void Shape::range(int dimCount, std::function<void(const int idx_linear)> func) const
{
    if (dimCount < 0) dimCount += dim();
    if (dimCount > dim()) throw std::invalid_argument("dimCount exceeds the number of dimensions");
    int totalSize = 1, stride = 1;
    for (int i = 0; i < dimCount; ++i) totalSize *= shape[i];
    for (int i = dimCount; i < dim(); ++i) stride *= shape[i];
    for (int idx = 0; idx < totalSize; idx++)
        func(idx * stride);
}

}

#endif // DEEPX_SHAPE_HPP
