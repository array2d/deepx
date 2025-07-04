
#include <iostream>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/tensorlife_miaobyte.hpp"
#include "deepx/tensorfunc/init_miaobyte.hpp"
#include "deepx/tensorfunc/io_miaobyte.hpp"
#include "deepx/tensorfunc/authors.hpp"


using namespace deepx;
using namespace deepx::tensorfunc;
void test_tensor_range(){
    Tensor<float> tensor=New<float>({2, 3});
    constant<miaobyte,float>(tensor,1);
    print<miaobyte>(tensor);
 
    Tensor<float> tensor2=New<float>({2, 3});
    constant<miaobyte,float>(tensor2,2);
    print<miaobyte>(tensor2);
 
}
 
int main(){
    test_tensor_range();
 
    return 0;
}