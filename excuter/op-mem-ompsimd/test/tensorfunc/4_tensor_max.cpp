#include <numeric>
#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/elementwise_miaobyte.hpp"
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/init_miaobyte.hpp"
#include "deepx/tensorfunc/io_miaobyte.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "tensorutil.hpp"
    
using namespace deepx;
using namespace deepx::tensorfunc;

void test_max(){
    std::vector<int> shape=randomshape(1,3,1,19);
    Tensor<float> A=New<float>(shape);
    std::iota(A.data,A.data+A.shape.size,0);
    print<miaobyte>(A)  ;
    Tensor<float> B=New<float>(shape);
    constant<miaobyte,float>(B,float(55));
    print<miaobyte>(B);
    Tensor<float> C=New<float>(shape);
    Tensor<float> D=New<float>(shape);
    max<tensorfunc::miaobyte,float>(A,B,C);
    print<miaobyte>(C);
    min<tensorfunc::miaobyte,float>(A,B,D);
    print<miaobyte>(D);
}
int main(){
    test_max();
}