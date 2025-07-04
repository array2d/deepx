#include "deepx/tensorfunc/init_miaobyte.hpp"
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/tensorlife_miaobyte.hpp"
#include "deepx/tensorfunc/io_miaobyte.hpp"
#include "deepx/tensorfunc/matmul.hpp"
#include "deepx/tensorfunc/matmul_cublas.hpp"

using namespace deepx::tensorfunc;
using namespace deepx;

void test_matmul()
{
    // 创建矩阵 A (2x3)
    Tensor<float> a = New<float>({2, 3}); 
    arange<miaobyte,float>(a, 1.0f, 1.0f);  // 1,2,3
                                            // 4,5,6
    
    // 创建矩阵 B (3x2)
    Tensor<float> b = New<float>({3, 2});
    arange<miaobyte,float>(b, 1.0f, 1.0f);  // 1,2
                                            // 3,4
                                            // 5,6
    
    // 创建结果矩阵 C (2x2) 
    Tensor<float> c = New<float>({2, 2});
    constant<miaobyte,float>(c, 0.0f);

    // 打印输入矩阵
    print<miaobyte>(a, "%.2f");
    print<miaobyte>(b, "%.2f");

    // 执行矩阵乘法 C = A × B
    matmul<deepx::tensorfunc::cublas,float>(a, b, c);

    // 打印结果
    print<miaobyte>(c, "%.2f");
}

void test_matmul_batch()
{
    // 创建矩阵 A 
    Tensor<float> a = New<float>({2, 3,4,5}); 
    arange<miaobyte,float>(a, 1.0f, 1.0f); 

    // 创建矩阵 B 
    Tensor<float> b = New<float>({2,3,5,6});
    arange<miaobyte,float>(b, 1.0f, 1.0f);  

    // 创建结果矩阵 C  
    Tensor<float> c = New<float>({2, 3,4,6});
    constant<miaobyte,float>(c, 0.0f);

    // 打印输入矩阵
    print<miaobyte>(a, "%.2f");
    print<miaobyte>(b, "%.2f");

    // 执行矩阵乘法 C = A × B
    matmul<deepx::tensorfunc::cublas,float>(a, b, c);

    // 打印结果
    print<miaobyte>(c, "%.2f");
}

int main(int argc, char **argv)
{ 
    int casei = 0;
    if (argc > 1) {
        casei = atoi(argv[1]);
    }
    switch (casei) {
        case 0:
            test_matmul();
            break;
        case 1:
            test_matmul_batch();
            break;  
    }
    return 0;
}