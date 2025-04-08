#include <numeric>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/io_miaobyte.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/authors.hpp"
int main(){
    deepx::Tensor<float> t=deepx::tensorfunc::New<float>({2, 3,4});
    std::iota(t.data, t.data+t.shape.size, 0);
    deepx::tensorfunc::print<deepx::tensorfunc::miaobyte>(t);
    deepx::tensorfunc::save<deepx::tensorfunc::miaobyte>(t,"3_tensor_print");
    return 0;
}