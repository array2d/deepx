#include "deepx/tf/arg.hpp"
#include "deepx/tf/tf.hpp"
#include "deepx/tf/new.hpp"
#include "deepx/tf/print.hpp"
#include "deepx/tf/init.hpp"
#include "deepx/tf/elementwise_basic.hpp"
#include "deepx/dtype.hpp"
#include "deepx/tf/tffactory.hpp"
#include "deepx/tensorfunc/authors.hpp"
namespace deepx::tf
{
    using namespace deepx::tensorfunc;
    // tensor
    void register_lifecycle(TfFactory &tffactory)
    {
        tffactory.add_tf(std::make_shared<ArgSet>(vector<Param>(
                                                      {
                                                          Param("value", DataCategory::Var, Precision::Any),
                                                      }),
                                                  vector<Param>(
                                                      {
                                                          Param("name", DataCategory::Var, Precision::Any),
                                                      })));
        tffactory.add_tf(std::make_shared<VecSet>(
            vector<Param>(
                {
                    Param("value", DataCategory::Vector, Precision::Any),
                }),
            vector<Param>(
                {
                    Param("name", DataCategory::Vector, Precision::Any),
                })));
        tffactory.add_tf(std::make_shared<NewTensor>(vector<Param>(
                                                         {
                                                             Param("shape", DataCategory::Vector, Precision::Int32),
                                                         }),
                                                     vector<Param>(
                                                         {
                                                             Param("tensor1", DataCategory::Tensor, Precision::Any),
                                                         })));
        tffactory.add_tf(std::make_shared<NewTensor>(vector<Param>(
                                                         {
                                                             Param("shape", DataCategory::Var, Precision::String),
                                                         }),
                                                     vector<Param>(
                                                         {
                                                             Param("tensor1", DataCategory::Tensor, Precision::Any),
                                                         })));
        // opfactory.add_op(DelTensor<float>());
    }

    // init
    void register_init(TfFactory &tffactory)
    {

        tffactory.add_tf(std::make_shared<Constant<miaobyte>>(vector<Param>(
                                                                  {
                                                                      Param("t", DataCategory::Tensor, Precision::Any),
                                                                      Param("value", DataCategory::Var, Precision::Any),
                                                                  }),
                                                              vector<Param>()));

        tffactory.add_tf(std::make_shared<Arange<miaobyte>>(vector<Param>(
                                                                {
                                                                    Param("t", DataCategory::Tensor, Precision::Any),
                                                                    Param("start", DataCategory::Var, Precision::Any),
                                                                    Param("step", DataCategory::Var, Precision::Any),
                                                                }),
                                                            vector<Param>()));
        tffactory.add_tf(std::make_shared<Uniform<miaobyte>>(vector<Param>(
                                                                 {
                                                                     Param("t", DataCategory::Tensor, Precision::Any),
                                                                     Param("low", DataCategory::Var, Precision::Any),
                                                                     Param("high", DataCategory::Var, Precision::Any),
                                                                     Param("seed", DataCategory::Var, Precision::Int32),
                                                                 }),
                                                             vector<Param>()));
    }
    // io
    void register_util(TfFactory &opfactory)
    {
        opfactory.add_tf(std::make_shared<Print<miaobyte>>(vector<Param>(
                                                               {
                                                                   Param("", DataCategory::Tensor, Precision::Any),
                                                               }),
                                                           vector<Param>()));

        opfactory.add_tf(std::make_shared<Print<miaobyte>>(vector<Param>(
                                                               {
                                                                   Param("", DataCategory::Tensor, Precision::Any),
                                                                   Param("", DataCategory::Var, Precision::String),
                                                               }),
                                                           vector<Param>()));
    }

    // elementwise
    void register_elementwise(TfFactory &tffactory)
    {
        tffactory.add_tf(std::make_shared<Add<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("a", DataCategory::Tensor, Precision::Any),
                                                                 Param("b", DataCategory::Tensor, Precision::Any),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("c", DataCategory::Tensor, Precision::Any),
                                                             })));
        tffactory.add_tf(std::make_shared<Add<cublas>>(vector<Param>(
                                                             {
                                                                 Param("a", DataCategory::Tensor, Precision::Any),
                                                                 Param("b", DataCategory::Tensor, Precision::Any),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("c", DataCategory::Tensor, Precision::Any),
                                                             })));
        tffactory.add_tf(std::make_shared<AddScalar<miaobyte>>(vector<Param>(
                                                                   {
                                                                       Param("A", DataCategory::Tensor, Precision::Any),
                                                                       Param("b", DataCategory::Var, Precision::Any),
                                                                   }),
                                                               vector<Param>(
                                                                   {
                                                                       Param("C", DataCategory::Tensor, Precision::Any),
                                                                   })));

        tffactory.add_tf(std::make_shared<Sub<miaobyte>>(vector<Param>(
                                                                   {
                                                                       Param("A", DataCategory::Tensor, Precision::Any),
                                                                       Param("B", DataCategory::Tensor, Precision::Any),
                                                                   }),
                                                               vector<Param>(
                                                                   {
                                                                       Param("C", DataCategory::Tensor, Precision::Any),
                                                                   })));
        tffactory.add_tf(std::make_shared<SubScalar<miaobyte>>(vector<Param>(
                                                                   {
                                                                       Param("A", DataCategory::Tensor, Precision::Any),
                                                                       Param("b", DataCategory::Var, Precision::Any),
                                                                   }),
                                                               vector<Param>(
                                                                   {
                                                                       Param("C", DataCategory::Tensor, Precision::Any),
                                                                   })));

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
    }
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
        // tffactory.add_tf(std::make_shared<Concat>());
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
        register_init(tffactory);
        register_util(tffactory);
        register_elementwise(tffactory);
        // register_matmul(opfactory);
        register_changeshape(tffactory);
        // register_reduce(opfactory);
        return 0;
    }
}