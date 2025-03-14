#include "deepx/tf/arg.hpp"
#include "deepx/tf/tf.hpp"
#include "deepx/tf/new.hpp"
#include "deepx/tf/print.hpp"
#include "deepx/tf/changeshape.hpp"
#include "deepx/dtype.hpp"
#include "deepx/tf/tffactory.hpp"

namespace deepx::tf
{
    string TfFactory::print_markdown() const
    {
        std::stringstream ss;
        ss << "## excuter/op-mem-ompsimd 支持算子列表 \n\n";
        ss << "本页面由 `excuter/op-mem-ompsimd/src/deepx/tf/tffactory.hpp` 生成，请勿手动修改 \n\n";
        ss << "| Operation | Author | Func Def | Math Formula | IR Instruction |\n";
        ss << "|-----------|--------|------------|--------------|----------------|\n";

        // 输出每个操作及其信息
        for (auto &[name, tf_family] : tf_families)
        {
            for (auto &[author, tf_author] : tf_family->tf_authors)
            {              
                // 为每个数据类型创建一行
                for (const auto &[funcdef, tf] : tf_author->tfs)
                {
                    ss << "| " << name << " | ";
                    if (author.empty()){
                        ss << " none ";
                    }else{
                        ss << author;
                    }
                    ss << " | " << funcdef << " | ";
                    // 获取当前数据类型的TF实例
                    ss << tf->math_formula() << " | ";
                    ss << tf->to_string() << " |\n";
                }
            }
        }
        return ss.str();
    }
    // tensor
    void register_lifecycle(TfFactory &tffactory)
    {
        tffactory.add_tf(std::make_shared<ArgSet>());
        tffactory.add_tf(std::make_shared<VecSet>());
        tffactory.add_tf(std::make_shared<NewTensor>(0));
        tffactory.add_tf(std::make_shared<NewTensor>(1));
        //opfactory.add_op(DelTensor<float>());
    }

    // // init
    // void register_init(OpFactory &opfactory)
    // {
    //     opfactory.add_op(Uniform<float>());
    //     opfactory.add_op(Uniform<double>());

    //     opfactory.add_op(Constant<float>());
    //     opfactory.add_op(Constant<double>());

    //     opfactory.add_op(Arange<float>());
    //     opfactory.add_op(Arange<double>());
    // }
    //io
    void register_util(TfFactory &opfactory)
    {
        opfactory.add_tf(std::make_shared<Print>());
    }

    // // elementwise
    // void register_elementwise(OpFactory &opfactory)
    // {
    //     opfactory.add_op(Add_miaobyte<float>());
    //     opfactory.add_op(Add_miaobyte<double>());
    //     opfactory.add_op(Add_miaobyte<int8_t>());
    //     opfactory.add_op(Add_miaobyte<int16_t>());
    //     opfactory.add_op(Add_miaobyte<int32_t>());
    //     opfactory.add_op(Add_miaobyte<int64_t>());

    //     opfactory.add_op(Add_cblas<float>());
    //     opfactory.add_op(Add_cblas<double>());
   

    //     opfactory.add_op(Addscalar_miaobyte<float>());
    //     opfactory.add_op(Addscalar_miaobyte<double>());
 
    //     opfactory.add_op(Sub_miaobyte<float>());
    //     opfactory.add_op(Sub_miaobyte<double>());
 
    //     opfactory.add_op(Sub_cblas<float>());
    //     opfactory.add_op(Sub_cblas<double>());

    //     opfactory.add_op(Mul_miaobyte<float>());
    //     opfactory.add_op(Mul_miaobyte<double>());
 
    //     opfactory.add_op(Mulscalar_miaobyte<float>());
    //     opfactory.add_op(Mulscalar_miaobyte<double>());

    //     opfactory.add_op(Div_miaobyte<float>());
    //     opfactory.add_op(Div_miaobyte<double>());
 
    //     opfactory.add_op(Divscalar_miaobyte<float>());
    //     opfactory.add_op(Divscalar_miaobyte<double>());

    //     opfactory.add_op(RDivscalar_miaobyte<float>());
    //     opfactory.add_op(RDivscalar_miaobyte<double>());

    //     opfactory.add_op(Sqrt_miaobyte<float>());
    //     opfactory.add_op(Sqrt_miaobyte<double>());

    //     opfactory.add_op(Exp_miaobyte<float>());
    //     opfactory.add_op(Exp_miaobyte<double>());

    //     opfactory.add_op(Pow_miaobyte<float>());
    //     opfactory.add_op(Pow_miaobyte<double>());

    //     opfactory.add_op(Powscalar_miaobyte<float>());
    //     opfactory.add_op(Powscalar_miaobyte<double>());
    // }
    // // matmul
    // void register_matmul(OpFactory &opfactory)
    // {
    //     opfactory.add_op(MatMul<float>());
    //     opfactory.add_op(MatMul<double>());
    // }
    // // changeshape
    void register_changeshape(TfFactory &tffactory)
    {
    //     opfactory.add_op(Transpose<float>());
    //     opfactory.add_op(Reshape<float>());
    //     opfactory.add_op(Expand<float>());
        tffactory.add_tf(std::make_shared<Concat>());
        
    }
    // // reduce
    // void register_reduce(OpFactory &opfactory)
    // {
    //     opfactory.add_op(Max<float>());
    //     opfactory.add_op(Max<double>());
    //     opfactory.add_op(Maxscalar<float>());
    //     opfactory.add_op(Maxscalar<double>());
    //     opfactory.add_op(Min<float>());
    //     opfactory.add_op(Min<double>());
    //     opfactory.add_op(Minscalar<float>());
    //     opfactory.add_op(Minscalar<double>());
    //     opfactory.add_op(Sum<float>());
    //     opfactory.add_op(Sum<double>());
    // }
    int register_all(TfFactory &tffactory)
    {
        register_lifecycle(tffactory);
        // register_init(opfactory);
        // register_util(opfactory);
        // register_elementwise(opfactory);
        // register_matmul(opfactory);
        register_changeshape(tffactory);
        // register_reduce(opfactory);
        return 0;
    }
}