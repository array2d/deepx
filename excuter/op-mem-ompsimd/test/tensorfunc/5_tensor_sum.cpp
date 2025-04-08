#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

#include <omp.h>
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/reduce_miaobyte.hpp"
#include "stdutil/vector.hpp"
#include "deepx/vector_combination.hpp"
#include "deepx/shape_reduce.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/init_miaobyte.hpp"
#include "deepx/tensorfunc/io_miaobyte.hpp"
#include "deepx/tensorfunc/authors.hpp"

#include <omp.h>

using namespace deepx;
using namespace deepx::tensorfunc;
void test_sum()
{
    omp_set_num_threads(1); 

    Shape shape({2, 3, 4});
    deepx::Tensor<float> tensor= New<float>(shape.shape);
    constant<miaobyte,float>(tensor,float(1));
    print<miaobyte>(tensor);
    cout<<""<<endl;
    std::vector<std::vector<int>> result = combination(3);
    for (const auto &comb : result)
    {
        std::cout <<"sum(t,"<< comb <<")"<< std::endl;
        Shape sumshape=reduceShape(shape,comb);
        Tensor<float> r = New<float>(sumshape.shape);
        sum<miaobyte,float>(tensor, comb,r);
        print<miaobyte>(r);
    }
/*
[]=>[2, 3, 4]
[0]=>[3, 4]
[1]=>[2, 4]
[2]=>[2, 3]
[0, 1]=>[4]
[0, 2]=>[3]
[1, 2]=>[2]
[0, 1, 2]=>[1]
*/
}

void benchmark_sum(int i){
    Shape shape({i,i,i});
    deepx::Tensor<float> tensor= New<float>(shape.shape);
    std::iota(tensor.data ,tensor.data+tensor.shape.size,0);
    std::vector<std::vector<int>> result = combination(3);
     std::cout<<"sum "<<shape.shape<<"=>";
     auto start = std::chrono::high_resolution_clock::now();
    for (const auto &comb : result)
    {
        Shape sShape = reduceShape(shape, comb);
        Tensor<float> r=New<float>(sShape.shape);
        sum<miaobyte,float>(tensor, comb,r);
        save<miaobyte>(r,"5_tensor_sum"+std::to_string(i)+"result");
    }
    auto end=std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "time:" << duration.count() << " seconds" << std::endl;
}

int main(int arvc,char **argv)
{   
    omp_set_num_threads(1);
    int i=0;
    if (arvc>1){
        i=std::atoi(argv[1]);
    }
    switch (i)
    {
    case 0:
        test_sum();
        break;
    default:
         benchmark_sum(i);
    }
    return 0;
}