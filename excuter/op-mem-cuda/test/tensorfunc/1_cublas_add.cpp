#include "deepx/tensorfunc/init_miaobyte.hpp"
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/io_miaobyte.hpp"
#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/elementwise_cublas_basic.hpp"
using namespace deepx::tensorfunc;
using namespace deepx;
void test_add()
{
    Tensor<float> a=New<float>({10, 10});
    arange<miaobyte,float>(a, 1.0f, 0.1f);
    Tensor<float> b=New<float>({10, 10});
    arange<miaobyte,float>(b, 2.0f, 0.2f);
    Tensor<float> c=New<float>({10, 10});
    constant<miaobyte,float>(c, 0.0f);

    add<cublas,float>(a, b, c);
    print<miaobyte>(c,"%.2f");
}

int main()
{
    test_add();
}