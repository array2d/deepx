#include <iostream>
#include <numeric>
#include <vector>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/changeshape_miaobyte.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/print_miaobyte.hpp"
#include "stdutil/vector.hpp"
#include "tensorutil.hpp"
#include "deepx/shape_transpose.hpp"

using namespace deepx::tensorfunc;
using namespace deepx;
using namespace std;
void test_transpose()
{
    std::vector<int> shape = randomshape(2, 4, 1, 6);
    Tensor tensor = New<float>(shape);
    std::iota(tensor.data, tensor.data + tensor.shape.size, 1);
    print<miaobyte>(tensor);

    vector<int> dimOrder = swaplastTwoDimOrder(shape);

    std::vector<int> resultshape = transposeShape(tensor.shape.shape, dimOrder);
    Tensor result = New<float>(resultshape);
    transpose<miaobyte,float>(tensor, dimOrder, result);
    print<miaobyte>(result);
}

int main()
{
    test_transpose();
}