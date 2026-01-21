#include <cassert>
#include <vector>

#include "deepx/tensor.hpp"
#include "deepx/dtype.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/tensorlife_miaobyte.hpp"
#include "deepx/tensorfunc/init_miaobyte.hpp"
#include "deepx/tensorfunc/io_miaobyte.hpp"

using namespace deepx;
using namespace deepx::tensorfunc;

int main()
{
    auto t = New<float>({2, 3});
    constant<miaobyte, float>(t, 3.5f);
    print<miaobyte>(t, "%2.3f");

    return 0;
}
