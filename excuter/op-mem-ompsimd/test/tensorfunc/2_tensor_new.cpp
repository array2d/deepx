
#include <iostream>

#include "deepx/tensor.hpp"

#include "deepx/tensorfunc/tensorlife_miaobyte.hpp"
#include "deepx/tensorfunc/init_miaobyte.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/io_miaobyte.hpp"

using namespace deepx;
using namespace deepx::tensorfunc;
void test_tensor_new(){
    Tensor<float> tensor=New<float>({2, 3});
    constant<miaobyte,float>(tensor,1);
    print<miaobyte>(tensor);
    tensor.save("tensor");
    Tensor<float> tensor2=New<float>({2, 3});
    constant<miaobyte,float>(tensor2,2);
    print<miaobyte>(tensor2);
    tensor2.save("tensor2");
}

void test_arange() {
    Tensor<float> tensor=New<float>({2, 3});
    arange<miaobyte,float>(tensor,float(0),float(1));
    print<miaobyte>(tensor);
}
 
int main(int argc,char **argv){
    int i=0;
    if (argc>1){
        
        i=std::atoi(argv[1]);
    }
    switch (i) {
        case 1:
            test_tensor_new();
        case 0:
            test_arange();
    }
    return 0;
}