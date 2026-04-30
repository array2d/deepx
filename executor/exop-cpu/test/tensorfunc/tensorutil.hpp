#ifndef TENSORUTIL_HPP
#define TENSORUTIL_HPP

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <random>

#include "deepx/tensor.hpp"

using namespace deepx;
 
/*
    dimlen_min:shape.size()的最小维度长度
    dimlen_max:shape.size()的最大维度长度
    shape_min:shape[i]的最小维度数量
    shape_max:shape[i]的最大维度数量
*/
 
std::vector<int> randomshape(size_t dimlen_min, size_t dimlen_max, size_t shape_min, size_t shape_max) {
    // 初始化随机数种子
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // 随机生成维度长度
    size_t dimlen = dimlen_min + std::rand() % (dimlen_max - dimlen_min + 1);

    // 创建存储形状的向量
    std::vector<int> shape(dimlen);

    // 为每个维度随机生成形状值
    for (size_t i = 0; i < dimlen; ++i) {
        shape[i] = static_cast<int>(shape_min + std::rand() % (shape_max - shape_min + 1));
    }

    return shape;
}

std::vector<int> randomshape2(size_t dimlen_min, size_t dimlen_max, size_t dim_min, size_t dim_max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // 生成维度数量
    std::uniform_int_distribution<> dim_dist(dimlen_min, dimlen_max);
    int dims = dim_dist(gen);
    
    // 生成每个维度的长度
    std::uniform_int_distribution<> len_dist(dim_min, dim_max);
    std::vector<int> shape;
    shape.reserve(dims);
    
    for (int i = 0; i < dims; ++i) {
        shape.push_back(len_dist(gen));
    }
    
    return shape;
}


#endif // TENSORUTIL_HPP
