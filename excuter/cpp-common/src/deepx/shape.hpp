#ifndef DEEPX_SHAPE_HPP
#define DEEPX_SHAPE_HPP

#include <vector>
#include <string>
#include <functional>

#include "deepx/dtype.hpp"
namespace deepx
{
    //omp内线程局部变量
    class ThreadLocalVectors
    {
    private:
        std::vector<std::vector<int>> vectors;

    public:
        // 构造函数接收向量大小数组
        explicit ThreadLocalVectors(const std::vector<int> &sizes)
        {
            vectors.resize(sizes.size());
            for (size_t i = 0; i < sizes.size(); ++i)
            {
                vectors[i].resize(sizes[i], 0);
            }
        }

        // 获取指定索引的向量引用
        std::vector<int> &get(size_t index)
        {
            return vectors[index];
        }

        // 获取所有向量
        std::vector<std::vector<int>> &getAll()
        {
            return vectors;
        }
    };

    struct Shape
    {
        Precision dtype;
        std::vector<int> shape;
        std::vector<int> strides;
        int dim;
        int64_t size;
        int64_t bytes() const;

        Shape() = default;
        Shape(const std::vector<int> &shape);
        Shape(const std::initializer_list<int> &shape);
        Shape(const int *shape, int dim);
        void setshape(const int *shape, int dim);
        int operator[](int index) const;
        int &operator[](int index);
        bool operator==(const Shape &shape) const { return shape.shape == shape.shape; }
        void print() const;
        // range 不支持omp
        void range(int dimCount, std::function<void(const std::vector<int> &indices)> func) const;
        void range(int dimCount, std::function<void(const int idx_linear, const std::vector<int> &indices)> func) const;
        void range(int dimCount, std::function<void(const int idx_linear)> func) const;

        // rangeParallel 支持omp,但omp内无需线程local变量
        void rangeParallel(int dimCount, std::function<void(const std::vector<int> &indices)> func) const;
        void rangeParallel(int dimCount, std::function<void(const int idx_linear)> func) const;
        void rangeParallel(int dimCount, std::function<void(const int idx_linear, const std::vector<int> &indices)> func) const;

        // 支持omp,但omp内需要线程local变量
        void rangeParallel(int dimCount, std::function<void(const std::vector<int> &indices, ThreadLocalVectors &tlv)> func,const vector<int> tlv_sizes) const;
        void rangeParallel(int dimCount, std::function<void(const int idx_linear, ThreadLocalVectors &tlv)> func,const vector<int> tlv_sizes) const;
        void rangeParallel(int dimCount, std::function<void(const int idx_linear, const std::vector<int> &indices, ThreadLocalVectors &tlv)> func,const vector<int> tlv_sizes) const;
        int linearat(const std::vector<int> &indices) const;
        std::vector<int> linearto(int idx_linear) const;

        std::string toYaml() const;
        void fromYaml(const std::string &yaml);
    };

}

#endif // DEEPX_SHAPE_HPP
