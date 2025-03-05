
#include <iostream>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "deepx/tensorfunc/print.hpp"
 
#include "deepx/tensorfunc/file.hpp"

using namespace deepx;
using namespace deepx::tensorfunc;
void test_tensor_new(){
    Tensor<float> tensor=New<float>({2, 3});
    constant<float>(tensor,1);
    print(tensor);
    save(tensor,"tensor");
    Tensor<float> tensor2=New<float>({2, 3});
    constant<float>(tensor2,2);
    print(tensor2);
    save(tensor2,"tensor2");
}

void test_arange() {
    Tensor<float> tensor=New<float>({2, 3});
    arange(tensor,float(0),float(1));
    print(tensor);
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