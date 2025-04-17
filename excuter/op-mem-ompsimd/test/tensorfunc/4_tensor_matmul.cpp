
#include <iostream>
#include <numeric>
#include <chrono>

#include "deepx/tensorfunc/io_miaobyte.hpp"
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/tensorlife_miaobyte.hpp"
    
#include "deepx/tensorfunc/matmul.hpp"
#include "deepx/tensorfunc/matmul_miaobyte.hpp"
#include "deepx/tensorfunc/init_miaobyte.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/shape_matmul.hpp"
 
using namespace deepx;
using namespace deepx::tensorfunc;
/*
import torch

# 创建两个张量
tensor1 = torch.arange(0, 24, dtype=torch.float32).reshape(2, 3, 4)  # 形状 (2, 3, 4)
tensor2 = torch.arange(0, 40, dtype=torch.float32).reshape(2, 4, 5)  # 形状 (2, 4, 5)

# 执行矩阵乘法
tensor3 = torch.matmul(tensor1, tensor2)  # 结果形状 (2, 3, 5)

print(tensor3)
tensor([[[  70.,   76.,   82.,   88.,   94.],
         [ 190.,  212.,  234.,  256.,  278.],
         [ 310.,  348.,  386.,  424.,  462.]],

        [[1510., 1564., 1618., 1672., 1726.],
         [1950., 2020., 2090., 2160., 2230.],
         [2390., 2476., 2562., 2648., 2734.]]])
         
*/ 
void test_tensor_matmul(){
    Tensor<float> tensor= New<float>({2, 3,4});
    std::iota(tensor.data, tensor.data+tensor.shape.size, 0);
    Tensor<float> tensor2= New<float>({2, 4,5});
    std::iota(tensor2.data, tensor2.data+tensor2.shape.size, 0);
    Tensor<float> tensor3= New<float>(matmul_shape(tensor.shape, tensor2.shape).shape);
    matmul<tensorfunc::miaobyte,float>(tensor, tensor2, tensor3);

    print<miaobyte>(tensor3);
}

void bench_tensor_matmul(int i) {
    Tensor<float> tensor= New<float>({i,i});
    uniform<miaobyte,float>(tensor,0,1);
    save<miaobyte>(tensor,"4_tensor_matmul"+std::to_string(i)+"tensor");
    Tensor<float> tensor2= New<float>({i,i});
    uniform<miaobyte,float>(tensor2,0,1);
    save<miaobyte>(tensor2,"4_tensor_matmul"+std::to_string(i)+"tensor2");
    Tensor<float> tensor3= New<float>(matmul_shape(tensor.shape, tensor2.shape).shape);
    std::cout<<("matmul ", i, "x", i);
    auto start = std::chrono::high_resolution_clock::now();

    matmul<tensorfunc::miaobyte,float>(tensor, tensor2, tensor3);
    auto end=std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    save<miaobyte>(tensor3,"4_tensor_matmul"+std::to_string(i)+"result");
    std::cout << "time:" << duration.count() << " seconds" << std::endl;
}
 
int main(){
    test_tensor_matmul();
    // //test_tensor_matmul_cuda();
    // for (int i = 64; i <=4096*1024; i*=2) {
    //     bench_tensor_matmul(i);
    // }
    return 0;
}