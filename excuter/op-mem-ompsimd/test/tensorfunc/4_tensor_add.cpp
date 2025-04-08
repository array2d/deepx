#include <numeric>
#include <cstdint>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/elementwise_miaobyte.hpp"
#include "deepx/tensorfunc/io_miaobyte.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "tensorutil.hpp"
#include "deepx/tensorfunc/authors.hpp"
using namespace deepx;
using namespace deepx::tensorfunc;
using namespace std;

void test_add_inplace(){
    std::vector<int> shape=randomshape(1,3,1,55);
    Tensor<int16_t> a_int16=New<int16_t>(shape);
    Tensor<int16_t> b_int16=New<int16_t>(shape);
    std::iota(a_int16.data,a_int16.data+a_int16.shape.size,1);
    std::iota(b_int16.data,b_int16.data+b_int16.shape.size,2);
    print<miaobyte>(a_int16,"%d");
    print<miaobyte>(b_int16,"%d");
    add<tensorfunc::miaobyte,int16_t>(a_int16, b_int16,a_int16);  
    print<miaobyte>(a_int16,"%d");

    Tensor<int8_t> a_int8=New<int8_t>(shape);   
    Tensor<int8_t> b_int8=New<int8_t>(shape);
    std::iota(a_int8.data,a_int8.data+a_int8.shape.size,1);
    std::iota(b_int8.data,b_int8.data+b_int8.shape.size,2);
    print<miaobyte>(a_int8,"%d");
    print<miaobyte>(b_int8,"%d");
    add<tensorfunc::miaobyte,int8_t>(a_int8, b_int8,a_int8);
    print<miaobyte>(a_int8,"%d");
}
void test_add_inplace_1(){
    Tensor<float> a=New<float>({101});
    Tensor<float> b=New<float>({101});
    std::iota(a.data,a.data+a.shape.size,1.0f);
    std::iota(b.data,b.data+b.shape.size,101.0f);
    print<miaobyte>(a);
    print<miaobyte>(b);
    add<tensorfunc::miaobyte,float>(a, b,a);  
    print<miaobyte>(a);
}
void test_add_inplace_scalar(){
    Tensor<float> a=New<float>({101});
    std::iota(a.data,a.data+a.shape.size,1.0f);
    cout<<"a"<<endl;
    print<miaobyte>(a);
    addscalar<tensorfunc::miaobyte,float>(a, 100.0f,a);
    cout<<"a"<<endl;
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
            std::cout<<"test_add_inplace"<<std::endl;
            test_add_inplace();
            break;
        case 2:
            std::cout<<"test_add_inplace_1"<<std::endl;
            test_add_inplace_1();
            break;
        case 3:
            std::cout<<"test_add_inplace_scalar"<<std::endl;
            test_add_inplace_scalar();
            break;
    }
    return 0;
}