#include <memory>
#include "deepx/mem/mem_ompsimd.hpp"
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/tensorlife_miaobyte.hpp"
#include "deepx/tensorfunc/init_miaobyte.hpp"
#include "deepx/tensorfunc/io_miaobyte.hpp"
#include "deepx/tensorfunc/authors.hpp"

using namespace deepx::mem;
using namespace deepx;
using namespace deepx::tensorfunc;
using namespace std;
int main()
{
    shared_ptr<MemBase> mem=make_shared<Mem>();
    for (int i = 0; i < 10; i++)
    {
        Tensor<float> tensor = New<float>({1, 2, 3});
        uniform<miaobyte>(tensor,0.0f,1.0f);
        mem->addtensor("tensor" + std::to_string(i),  tensor );
    }
 
    cout << mem->existstensor(string("tensor0")) << endl;
    print<miaobyte>(*(mem->gettensor<float>(string("tensor0")).get()));
    mem->clear();
 
    return 0;
}