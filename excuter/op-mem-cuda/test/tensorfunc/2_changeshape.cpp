#include "deepx/tensorfunc/init_miaobyte.hpp"
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/print_miaobyte.hpp"
#include "deepx/tensorfunc/changeshape_miaobyte.hpp"
using namespace deepx::tensorfunc;
using namespace deepx;
void test_transpose()
{
    Tensor<float> a=New<float>({3,4,6});
    arange<miaobyte,float>(a, 1.0f, 1.0f);
    print<miaobyte>(a,"%.0f");
    Tensor<float> b=New<float>({3,6,4});
    transpose<miaobyte,float>(a, {0,2,1}, b);
    print<miaobyte>(b,"%.0f");
}

void test_concat()
{
    Tensor<float> a=New<float>({3,2,6});
    arange<miaobyte,float>(a, 1.0f, 1.0f);
    print<miaobyte>(a,"%.0f");
    Tensor<float> b=New<float>({3,4,6});
    constant<miaobyte,float>(b, 2.0f);
    print<miaobyte>(b,"%.0f");
    Tensor<float> c=New<float>({3,6,6});
    constant<miaobyte,float>(c, 3.0f);
    print<miaobyte>(c,"%.0f");
    Tensor<float> d=New<float>({3,12,6});
    concat<miaobyte,float>({&a,&b,&c},1,d);
    print<miaobyte>(d,"%.0f");
}

void test_broadcastTo()
{
    Tensor<float> a=New<float>({3,2});
    arange<miaobyte,float>(a, 1.0f, 1.0f);
    Tensor<float> b=New<float>({4,3,2});
    broadcastTo<miaobyte,float>(a, b.shape.shape, b);
    print<miaobyte>(b,"%.0f");
}
int main(int argc, char **argv)
{      
    int casearg=atoi(argv[1]);
    switch (casearg)
    {
    case 0:
        test_transpose();
        break;
    case 1:
        test_concat();
        break;
    case 2:
        test_broadcastTo();
        break;
    }
    return 0;
}