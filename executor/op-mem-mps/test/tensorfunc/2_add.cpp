#include <cassert>
#include <vector>

#include "deepx/tensor.hpp"
#include "deepx/dtype.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/tensorlife_miaobyte.hpp"
#include "deepx/tensorfunc/init_miaobyte.hpp"
#include "deepx/tensorfunc/elementwise_miaobyte.hpp"
#include "deepx/tensorfunc/io_miaobyte.hpp"

using namespace deepx;
using namespace deepx::tensorfunc;

int main()
{
    auto a = New<float>({2, 2});
    auto b = New<float>({2, 2});
    auto c = New<float>({2, 2});

    constant<miaobyte, float>(a, 1.25f);
    constant<miaobyte, float>(b, 2.5f);
    add<miaobyte, float>(a, b, c);
    print<miaobyte>(c, "%2.3f");
    return 0;
}
