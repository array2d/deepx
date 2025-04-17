#include <numeric>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/io_miaobyte.hpp"
#include "deepx/tensorfunc/tensorlife_miaobyte.hpp"
#include "deepx/tensorfunc/authors.hpp"

using namespace deepx::tensorfunc;
int main(){
    deepx::Tensor<float> t=New<float>({2, 3,4});
    std::iota(t.data, t.data+t.shape.size, 0);
    print<miaobyte>(t);
    save<miaobyte>(t,"3_tensor_print");
    return 0;
}