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
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/io_miaobyte.hpp"
#include "deepx/tensorfunc/authors.hpp"

#include <omp.h>

using namespace deepx;
using namespace deepx::tensorfunc;
void test_sum()
{
    omp_set_num_threads(1); 

    std::vector<int> shape={2, 3, 4};
    Tensor<float> tensor= New<float>(shape);
    constant<miaobyte,float>(tensor,float(1));
    print<miaobyte>(tensor,"%.0f");
    cout<<""<<endl;
    std::vector<std::vector<int>> result = combination(3);
    for (const auto &comb : result)
    {
        std::cout <<"sum(t,"<< comb <<")"<< std::endl;
        std::vector<int> checkeddims=checkedDims(shape,comb);
        std::vector<int> sumshape=reducedShape(shape,checkeddims);
        Tensor<float> r = New<float>(sumshape);
        sum<miaobyte,float>(tensor, checkeddims,false,r);
        print<miaobyte>(r,"%.0f");
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
    std::vector<int> shape={i,i,i};
    deepx::Tensor<float> tensor= New<float>(shape);
    std::iota(tensor.data ,tensor.data+tensor.shape.size,0);
    std::vector<std::vector<int>> result = combination(3);
    std::cout<<"sum "<<shape<<"=>";
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto &comb : result)
    {
        std::cout <<"sum(t,"<< comb <<")"<< std::endl;
        std::vector<int> checkeddims=checkedDims(shape,comb);
        std::vector<int> sumshape=reducedShape(shape,checkeddims);
        Tensor<float> r=New<float>(sumshape);
        sum<miaobyte,float>(tensor, checkeddims,false,r);
        string combstr="";
        for (const auto &c : comb)
        {
            combstr+=std::to_string(c)+"_";
        }
        save<miaobyte>(r,"5_tensor_sum."+ combstr);
        print<miaobyte>(r,"%.0f");
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