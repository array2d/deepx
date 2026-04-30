#include <iostream>
#include <numeric>

#include "deepx/tensorfunc/equal.hpp"
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/tensorlife_miaobyte.hpp"

using namespace deepx;
using namespace deepx::tensorfunc;
void test_equal(){
    Tensor<float> tensor1=New<float>({4096,4096});
    std::iota(tensor1.data,tensor1.data+tensor1.shape.size,0);
    Tensor<float> tensor2=New<float>({4096,4096});
    std::iota(tensor2.data,tensor2.data+tensor2.shape.size,0);
    std::cout<<equal(tensor1,tensor2)<<std::endl;
}

int main(){
    test_equal();
    return 0;
}