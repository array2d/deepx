#include "deepx/tf/tf.hpp"
#include "deepx/dtype.hpp"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace deepx::tf;
using namespace deepx;

void test_1() {
    unordered_map<string, TypeDef> dtype_map = {
        {"tensor<any>", make_dtype(DataCategory::Tensor, Precision::Any)},
        {"tensor<int>", make_dtype(DataCategory::Tensor, Precision::Int)},
        {"tensor<float64>", make_dtype(DataCategory::Tensor, Precision::Float64)},
        {"tensor<float32>", make_dtype(DataCategory::Tensor, Precision::Float32)},
        {"tensor<float16>", make_dtype(DataCategory::Tensor, Precision::Float16)},
        {"tensor<bfloat16>", make_dtype(DataCategory::Tensor, Precision::BFloat16)},
        {"tensor<float8e5m2>", make_dtype(DataCategory::Tensor, Precision::Float8E5M2)},
        {"tensor<float8e4m3>", make_dtype(DataCategory::Tensor, Precision::Float8E4M3)},
        {"tensor<float4e2m1>", make_dtype(DataCategory::Tensor, Precision::Float4E2M1)},
        {"tensor<int32>", make_dtype(DataCategory::Tensor, Precision::Int32)},
        {"vector<float64>", make_dtype(DataCategory::Vector, Precision::Float64)},
        {"var<int32>", make_dtype(DataCategory::Var, Precision::Int32)},
        {"var<float32>", make_dtype(DataCategory::Var, Precision::Float32)},
        {"var<bool>", make_dtype(DataCategory::Var, Precision::Bool)},

         {"tensor", make_dtype(DataCategory::Tensor, Precision::Any)},
        {"vector", make_dtype(DataCategory::Vector, Precision::Any)},
        {"var", make_dtype(DataCategory::Var, Precision::Any)},
    };

    // 打印表头
    cout << string(80, '=') << endl;
    cout << setw(25) << left << "Original Type"
         << setw(15) << "Status"
         << "Converted Back" << endl;
    cout << string(80, '-') << endl;

    // 遍历所有类型进行测试
    for (const auto &[type_str1, dtype1] : dtype_map)
    {
        // 将type_str1转换为DataType
        TypeDef converted1 = dtype(type_str1);
        // 检查转换后的DataType是否与原始值相等
        bool equal1 = (converted1 == dtype1);
        // 将转换后的DataType转回字符串
        string back_str1 = dtype_str(converted1);

        // 输出测试结果
        cout << setw(25) << left << type_str1
             << setw(15) << (equal1 ? "[MATCH]" : "[MISMATCH]")
             << back_str1 << endl;
    }

    cout << string(80, '=') << endl;
}

// test to tensor type
void test_2() {
    if (typeid(tensor_t<Precision::Float64>)== typeid(double)) {
        std::cout<<"it's ok"<<std::endl;
    } else {
        std::cout<<"it's wrong"<<std::endl;
    }

    if (typeid(tensor_t<Precision::Float32>)== typeid(float)) {
        std::cout<<"it's ok"<<std::endl;
    } else {
        std::cout<<"it's wrong"<<std::endl;
    }
}

int main(int argc, char **argv)
{
    // test_1();
    test_2();
    return 0;
}