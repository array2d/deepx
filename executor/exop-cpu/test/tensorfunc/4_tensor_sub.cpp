#include <numeric>
#include <cstdint>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/io_miaobyte.hpp"
#include "deepx/tensorfunc/tensorlife_miaobyte.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/elementwise_miaobyte.hpp"
#include "tensorutil.hpp"
using namespace deepx;
using namespace deepx::tensorfunc;


void test_sub(){
    std::vector<int> shape=randomshape(1,3,1,10);
    Tensor<int16_t> a_int16=New<int16_t>(shape);
    Tensor<int16_t> b_int16=New<int16_t>(shape);
    std::iota(a_int16.data,a_int16.data+a_int16.shape.size,1);
    std::iota(b_int16.data,b_int16.data+b_int16.shape.size,2);
    print<miaobyte>(a_int16,"%d");
    print<miaobyte>(b_int16,"%d");
    sub<tensorfunc::miaobyte,int16_t>(a_int16, b_int16,a_int16);  
    print<miaobyte>(a_int16,"%d");

    Tensor<float> a_float=New<float>(shape);
    Tensor<float> b_float=New<float>(shape);
    std::iota(a_float.data,a_float.data+a_float.shape.size,1.0f);
    std::iota(b_float.data,b_float.data+b_float.shape.size,2.0f);
    print<miaobyte>(a_float);
    print<miaobyte>(b_float);
    sub<tensorfunc::miaobyte,float>(a_float, b_float,a_float);  
    print<miaobyte>(a_float);
}
void test_sub_1(){
    std::vector<int> shape=randomshape(1,1,1,100);
    Tensor<float> a=New<float>(shape);
    Tensor<float> b=New<float>(shape);
    std::iota(a.data,a.data+a.shape.size,1.0f);
    std::iota(b.data,b.data+b.shape.size,101.0f);
    print<miaobyte>(a);
    print<miaobyte>(b);
    sub<tensorfunc::miaobyte,float>(a, b,a);  
    print<miaobyte>(a);
}
void test_sub_scalar(){
    std::vector<int> shape=randomshape(1,1,1,100);
    Tensor<float> a=New<float>(shape);
    std::iota(a.data,a.data+a.shape.size,1.0f);
    print<miaobyte>(a);
    subscalar<tensorfunc::miaobyte,float>(a, 100.0f,a);
    print<miaobyte>(a);
}
int main(int argc, char** argv){
    if (argc!=2){
        std::cerr<<"Usage: "<<argv[0]<<" <case> 1,2,3"<<std::endl;
        return 1;
    }
    int case_=atoi(argv[1]); 
    switch(case_){
        case 1:
            std::cout<<"test_sub"<<std::endl;
            test_sub();
            break;
        case 2:
            std::cout<<"test_sub_1"<<std::endl;
            test_sub_1();
            break;
        case 3:
            std::cout<<"test_sub_scalar"<<std::endl;
            test_sub_scalar();
            break;
    }
    return 0;
}