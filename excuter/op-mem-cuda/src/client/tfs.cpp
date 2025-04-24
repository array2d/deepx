#include "deepx/tf/arg.hpp"
#include "deepx/tf/tf.hpp"
#include "deepx/tf/tensorlife.hpp"
#include "deepx/tf/io.hpp"
#include "deepx/tf/init.hpp"
#include "deepx/tf/elementwise_basic.hpp"
#include "deepx/tf/elementwise_sqrt.hpp"
#include "deepx/tf/elementwise_sin.hpp"
#include "deepx/tf/elementwise_compare.hpp"
#include "deepx/tf/matmul.hpp"
#include "deepx/tf/changeshape.hpp"
#include "deepx/tf/reduce.hpp"
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
        // copytensor
        tffactory.add_tf(std::make_shared<CopyTensor>(vector<Param>(
                                                          {
                                                              Param("src", DataCategory::Tensor, Precision::Any),
                                                          }),
                                                      vector<Param>(
                                                          {
                                                              Param("dst", DataCategory::Tensor, Precision::Any),
                                                          })));
        // deltensor
        tffactory.add_tf(std::make_shared<DelTensor>(vector<Param>(),
                                                     vector<Param>(
                                                         {
                                                             Param("t", DataCategory::Tensor, Precision::Any),
                                                         })));
        // renametensor
        tffactory.add_tf(std::make_shared<RenameTensor>(vector<Param>(
                                                            {

                                                                Param("new_name", DataCategory::Var, Precision::String),
                                                            }),
                                                        vector<Param>(
                                                            {
                                                                Param("t", DataCategory::Tensor, Precision::Any),
                                                            })));
    }

    // init
    void register_init(TfFactory &tffactory)
    {

        tffactory.add_tf(std::make_shared<Constant<miaobyte>>(vector<Param>(
                                                                  {

                                                                      Param("value", DataCategory::Var, Precision::Any),
                                                                  }),
                                                              vector<Param>({
                                                                  Param("t", DataCategory::Tensor, Precision::Any),
                                                              })));

        tffactory.add_tf(std::make_shared<Arange<miaobyte>>(vector<Param>(
                                                                {

                                                                    Param("start", DataCategory::Var, Precision::Any),
                                                                    Param("step", DataCategory::Var, Precision::Any),
                                                                }),
                                                            vector<Param>({
                                                                Param("t", DataCategory::Tensor, Precision::Any),
                                                            })));
        tffactory.add_tf(std::make_shared<Uniform<miaobyte>>(vector<Param>(
                                                                 {

                                                                     Param("low", DataCategory::Var, Precision::Any),
                                                                     Param("high", DataCategory::Var, Precision::Any),
                                                                     Param("seed", DataCategory::Var, Precision::Int32),
                                                                 }),
                                                             vector<Param>({
                                                                 Param("t", DataCategory::Tensor, Precision::Any),
                                                             })));
        tffactory.add_tf(std::make_shared<Normal<miaobyte>>(vector<Param>(
                                                                {

                                                                    Param("mean", DataCategory::Var, Precision::Any),
                                                                    Param("stddev", DataCategory::Var, Precision::Any),
                                                                    Param("seed", DataCategory::Var, Precision::Int32),
                                                                }),
                                                            vector<Param>({
                                                                Param("t", DataCategory::Tensor, Precision::Any),
                                                            })));
    }
    // io
    void register_io(TfFactory &opfactory)
    {
        opfactory.add_tf(std::make_shared<Print<miaobyte>>(vector<Param>(
                                                               {
                                                                   Param("t", DataCategory::Tensor, Precision::Any),
                                                               }),
                                                           vector<Param>()));

        opfactory.add_tf(std::make_shared<Print<miaobyte>>(vector<Param>(
                                                               {
                                                                   Param("t", DataCategory::Tensor, Precision::Any),
                                                                   Param("format", DataCategory::Var, Precision::String),
                                                               }),
                                                           vector<Param>()));

        opfactory.add_tf(std::make_shared<Save>(vector<Param>(
                                                    {
                                                        Param("t", DataCategory::Tensor, Precision::Any),
                                                        Param("path", DataCategory::Var, Precision::String),
                                                    }),
                                                vector<Param>()));

        opfactory.add_tf(std::make_shared<Load>(vector<Param>(
                                                    {
                                                        Param("path", DataCategory::Var, Precision::String),
                                                    }),
                                                vector<Param>()));
        // loadtensordata
        opfactory.add_tf(std::make_shared<LoadTensorData>(vector<Param>(
                                                              {
                                                                  Param("path", DataCategory::Var, Precision::String),
                                                              }),
                                                          vector<Param>(
                                                              {
                                                                  Param("t", DataCategory::Tensor, Precision::Any),
                                                              })));
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
        tffactory.add_tf(std::make_shared<Mul<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("A", DataCategory::Tensor, Precision::Any),
                                                                 Param("B", DataCategory::Tensor, Precision::Any),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("C", DataCategory::Tensor, Precision::Any),
                                                             })));
        tffactory.add_tf(std::make_shared<MulScalar<miaobyte>>(vector<Param>(
                                                                   {
                                                                       Param("A", DataCategory::Tensor, Precision::Any),
                                                                       Param("b", DataCategory::Var, Precision::Any),
                                                                   }),
                                                               vector<Param>(
                                                                   {
                                                                       Param("C", DataCategory::Tensor, Precision::Any),
                                                                   })));
        tffactory.add_tf(std::make_shared<Div<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("A", DataCategory::Tensor, Precision::Any),
                                                                 Param("B", DataCategory::Tensor, Precision::Any),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("C", DataCategory::Tensor, Precision::Any),
                                                             })));
        tffactory.add_tf(std::make_shared<DivScalar<miaobyte>>(vector<Param>(
                                                                   {
                                                                       Param("A", DataCategory::Tensor, Precision::Any),
                                                                       Param("scalar", DataCategory::Var, Precision::Any),
                                                                   }),
                                                               vector<Param>(
                                                                   {
                                                                       Param("C", DataCategory::Tensor, Precision::Any),
                                                                   })));
        tffactory.add_tf(std::make_shared<RDivScalar<miaobyte>>(vector<Param>(
                                                                    {
                                                                        Param("scalar", DataCategory::Var, Precision::Any),
                                                                        Param("A", DataCategory::Tensor, Precision::Any),
                                                                    }),
                                                                vector<Param>(
                                                                    {
                                                                        Param("C", DataCategory::Tensor, Precision::Any),
                                                                    })));
        // invert
        tffactory.add_tf(std::make_shared<Invert<miaobyte>>(vector<Param>(
                                                                {
                                                                    Param("A", DataCategory::Tensor, Precision::Int64 | Precision::Int32 | Precision::Int16 | Precision::Int8),
                                                                }),
                                                            vector<Param>(
                                                                {
                                                                    Param("C", DataCategory::Tensor, Precision::Int64 | Precision::Int32 | Precision::Int16 | Precision::Int8),
                                                                })));

        tffactory.add_tf(std::make_shared<Sqrt<miaobyte>>(vector<Param>(
                                                              {
                                                                  Param("A", DataCategory::Tensor, Precision::Float64 | Precision::Float32 | Precision::Float16 | Precision::BFloat16),
                                                              }),
                                                          vector<Param>(
                                                              {
                                                                  Param("C", DataCategory::Tensor, Precision::Float64 | Precision::Float32 | Precision::Float16 | Precision::BFloat16),
                                                              })));

        tffactory.add_tf(std::make_shared<Pow<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("A", DataCategory::Tensor, Precision::Float64 | Precision::Float32),
                                                                 Param("B", DataCategory::Tensor, Precision::Float64 | Precision::Float32),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("C", DataCategory::Tensor, Precision::Float64 | Precision::Float32),
                                                             })));
        tffactory.add_tf(std::make_shared<PowScalar<miaobyte>>(vector<Param>(
                                                                   {
                                                                       Param("A", DataCategory::Tensor, Precision::Float64 | Precision::Float32),
                                                                       Param("scalar", DataCategory::Var, Precision::Float64 | Precision::Int32),
                                                                   }),
                                                               vector<Param>(
                                                                   {
                                                                       Param("C", DataCategory::Tensor, Precision::Float64 | Precision::Float32),
                                                                   })));
        // rpowscalar
        tffactory.add_tf(std::make_shared<RpowScalar<miaobyte>>(vector<Param>(
                                                                    {
                                                                        Param("scalar", DataCategory::Var, Precision::Float64 | Precision::Int32),
                                                                        Param("A", DataCategory::Tensor, Precision::Float64 | Precision::Float32),
                                                                    }),
                                                                vector<Param>(
                                                                    {
                                                                        Param("C", DataCategory::Tensor, Precision::Float64 | Precision::Float32),
                                                                    })));
        // log
        tffactory.add_tf(std::make_shared<Log<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("A", DataCategory::Tensor, Precision::Float64 | Precision::Float32 | Precision::Float16 | Precision::BFloat16),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("C", DataCategory::Tensor, Precision::Float64 | Precision::Float32 | Precision::Float16 | Precision::BFloat16),
                                                             })));
        tffactory.add_tf(std::make_shared<Exp<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("A", DataCategory::Tensor, Precision::Float64 | Precision::Float32 | Precision::Float16 | Precision::BFloat16),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("C", DataCategory::Tensor, Precision::Float64 | Precision::Float32 | Precision::Float16 | Precision::BFloat16),
                                                             })));
        tffactory.add_tf(std::make_shared<Sin<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("A", DataCategory::Tensor, Precision::Float64 | Precision::Float32 | Precision::Float16 | Precision::BFloat16),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("C", DataCategory::Tensor, Precision::Float64 | Precision::Float32 | Precision::Float16 | Precision::BFloat16),
                                                             })));
        tffactory.add_tf(std::make_shared<Cos<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("A", DataCategory::Tensor, Precision::Float64 | Precision::Float32 | Precision::Float16 | Precision::BFloat16),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("C", DataCategory::Tensor, Precision::Float64 | Precision::Float32 | Precision::Float16 | Precision::BFloat16),
                                                             })));
        tffactory.add_tf(std::make_shared<Tan<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("A", DataCategory::Tensor, Precision::Float64 | Precision::Float32),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("C", DataCategory::Tensor, Precision::Float64 | Precision::Float32),
                                                             })));
        tffactory.add_tf(std::make_shared<Max<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("A", DataCategory::Tensor, Precision::Any),
                                                                 Param("B", DataCategory::Tensor, Precision::Any),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("C", DataCategory::Tensor, Precision::Any),
                                                             })));
        tffactory.add_tf(std::make_shared<MaxScalar<miaobyte>>(vector<Param>(
                                                                   {
                                                                       Param("A", DataCategory::Tensor, Precision::Any),
                                                                       Param("scalar", DataCategory::Var, Precision::Any),
                                                                   }),
                                                               vector<Param>(
                                                                   {
                                                                       Param("C", DataCategory::Tensor, Precision::Any),
                                                                   })));
        tffactory.add_tf(std::make_shared<Min<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("A", DataCategory::Tensor, Precision::Any),
                                                                 Param("B", DataCategory::Tensor, Precision::Any),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("C", DataCategory::Tensor, Precision::Any),
                                                             })));
        tffactory.add_tf(std::make_shared<MinScalar<miaobyte>>(vector<Param>(
                                                                   {
                                                                       Param("A", DataCategory::Tensor, Precision::Any),
                                                                       Param("scalar", DataCategory::Var, Precision::Any),
                                                                   }),
                                                               vector<Param>(
                                                                   {
                                                                       Param("C", DataCategory::Tensor, Precision::Any),
                                                                   })));
        // equal
        tffactory.add_tf(std::make_shared<Equal<miaobyte>>(vector<Param>(
                                                               {
                                                                   Param("A", DataCategory::Tensor, Precision::Any),
                                                                   Param("B", DataCategory::Tensor, Precision::Any),
                                                                   Param("epsilon", DataCategory::Var, Precision::Float64),
                                                               }),
                                                           vector<Param>(
                                                               {
                                                                   Param("mask", DataCategory::Tensor, Precision::Bool),
                                                               })));
        tffactory.add_tf(std::make_shared<EqualScalar<miaobyte>>(vector<Param>(
                                                                     {
                                                                         Param("A", DataCategory::Tensor, Precision::Any),
                                                                         Param("scalar", DataCategory::Var, Precision::Any),
                                                                         Param("epsilon", DataCategory::Var, Precision::Float64),
                                                                     }),
                                                                 vector<Param>(
                                                                     {
                                                                         Param("mask", DataCategory::Tensor, Precision::Bool),
                                                                     })));
        // less
        tffactory.add_tf(std::make_shared<Less<miaobyte>>(vector<Param>(
                                                              {
                                                                  Param("A", DataCategory::Tensor, Precision::Any),
                                                                  Param("B", DataCategory::Tensor, Precision::Any),
                                                              }),
                                                          vector<Param>(
                                                              {
                                                                  Param("mask", DataCategory::Tensor, Precision::Bool),
                                                              })));
        // lessscalar
        tffactory.add_tf(std::make_shared<LessScalar<miaobyte>>(vector<Param>(
                                                                    {
                                                                        Param("A", DataCategory::Tensor, Precision::Any),
                                                                        Param("scalar", DataCategory::Var, Precision::Any),
                                                                    }),
                                                                vector<Param>(
                                                                    {
                                                                        Param("mask", DataCategory::Tensor, Precision::Bool),
                                                                    })));
        // greater
        tffactory.add_tf(std::make_shared<Greater<miaobyte>>(vector<Param>(
                                                                 {
                                                                     Param("A", DataCategory::Tensor, Precision::Any),
                                                                     Param("B", DataCategory::Tensor, Precision::Any),
                                                                 }),
                                                             vector<Param>(
                                                                 {
                                                                     Param("mask", DataCategory::Tensor, Precision::Bool),
                                                                 })));
        // greaterscalar
        tffactory.add_tf(std::make_shared<GreaterScalar<miaobyte>>(vector<Param>(
                                                                       {
                                                                           Param("A", DataCategory::Tensor, Precision::Any),
                                                                           Param("scalar", DataCategory::Var, Precision::Any),
                                                                       }),
                                                                   vector<Param>(
                                                                       {
                                                                           Param("mask", DataCategory::Tensor, Precision::Bool),
                                                                       })));
        // switch
        tffactory.add_tf(std::make_shared<Switch<miaobyte>>(vector<Param>(
                                                                {
                                                                    Param("tensors", DataCategory::ListTensor, Precision::Any),
                                                                    Param("cases", DataCategory::Tensor, Precision::Int8),
                                                                }),
                                                            vector<Param>(
                                                                {
                                                                    Param("result", DataCategory::Tensor, Precision::Any),
                                                                })));
    }
    // matmul
    void register_matmul(TfFactory &tffactory)
    {
        tffactory.add_tf(std::make_shared<MatMul<cublas>>(vector<Param>(
                                                              {
                                                                  Param("A", DataCategory::Tensor, Precision::Any),
                                                                  Param("B", DataCategory::Tensor, Precision::Any),
                                                              }),
                                                          vector<Param>(
                                                              {
                                                                  Param("C", DataCategory::Tensor, Precision::Any),
                                                              })));
    }
    // changeshape
    void register_changeshape(TfFactory &tffactory)
    {
        // reshape
        tffactory.add_tf(std::make_shared<Reshape<miaobyte>>(vector<Param>(
                                                                 {
                                                                     Param("A", DataCategory::Tensor, Precision::Any),
                                                                     Param("shape", DataCategory::Vector, Precision::Int32),
                                                                 }),
                                                             vector<Param>(
                                                                 {
                                                                     Param("B", DataCategory::Tensor, Precision::Any),
                                                                 })));
        // transpose
        tffactory.add_tf(std::make_shared<Transpose<miaobyte>>(vector<Param>(
                                                                   {
                                                                       Param("A", DataCategory::Tensor, Precision::Any),
                                                                       Param("dim_order", DataCategory::Vector, Precision::Int32),
                                                                   }),
                                                               vector<Param>(
                                                                   {
                                                                       Param("C", DataCategory::Tensor, Precision::Any),
                                                                   })));
        // concat
        tffactory.add_tf(std::make_shared<Concat<miaobyte>>(vector<Param>(
                                                                {
                                                                    Param("tensors", DataCategory::ListTensor, Precision::Any),
                                                                    Param("dim", DataCategory::Var, Precision::Int32),
                                                                }),
                                                            vector<Param>(
                                                                {
                                                                    Param("result", DataCategory::Tensor, Precision::Any),
                                                                })));
        // broadcastTo
        tffactory.add_tf(std::make_shared<BroadcastTo<miaobyte>>(vector<Param>(
                                                                     {
                                                                         Param("A", DataCategory::Tensor, Precision::Any),
                                                                         Param("new_shape", DataCategory::Vector, Precision::Int32),
                                                                     }),
                                                                 vector<Param>(
                                                                     {
                                                                         Param("B", DataCategory::Tensor, Precision::Any),
                                                                     })));
        // indexselect
        tffactory.add_tf(std::make_shared<IndexSelect<miaobyte>>(vector<Param>(
                                                                     {
                                                                         Param("A", DataCategory::Tensor, Precision::Any),
                                                                         Param("indices", DataCategory::Tensor, Precision::Int64 | Precision::Int32),
                                                                         Param("axis", DataCategory::Var, Precision::Int32),
                                                                     }),
                                                                 vector<Param>(
                                                                     {
                                                                         Param("B", DataCategory::Tensor, Precision::Any),
                                                                     })));
    }
    // reduce
    void register_reduce(TfFactory &tffactory)
    {
        // sum
        tffactory.add_tf(std::make_shared<Sum<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("A", DataCategory::Tensor, Precision::Any),
                                                                 Param("dims", DataCategory::Vector, Precision::Int32),
                                                                 Param("keepdims", DataCategory::Var, Precision::Bool),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("B", DataCategory::Tensor, Precision::Any),
                                                             })));
        // prod
        tffactory.add_tf(std::make_shared<Prod<miaobyte>>(vector<Param>(
                                                              {
                                                                  Param("A", DataCategory::Tensor, Precision::Any),
                                                                  Param("dims", DataCategory::Vector, Precision::Int32),
                                                                  Param("keepdims", DataCategory::Var, Precision::Bool),
                                                              }),
                                                          vector<Param>(
                                                              {
                                                                  Param("B", DataCategory::Tensor, Precision::Any),
                                                              })));

        // max
        tffactory.add_tf(std::make_shared<ReduceMax<miaobyte>>(vector<Param>(
                                                                   {
                                                                       Param("A", DataCategory::Tensor, Precision::Any),
                                                                       Param("dims", DataCategory::Vector, Precision::Int32),
                                                                       Param("keepdims", DataCategory::Var, Precision::Bool),
                                                                   }),
                                                               vector<Param>(
                                                                   {
                                                                       Param("B", DataCategory::Tensor, Precision::Any),
                                                                   })));
        // min
        tffactory.add_tf(std::make_shared<ReduceMin<miaobyte>>(vector<Param>(
                                                                   {
                                                                       Param("A", DataCategory::Tensor, Precision::Any),
                                                                       Param("dims", DataCategory::Vector, Precision::Int32),
                                                                       Param("keepdims", DataCategory::Var, Precision::Bool),
                                                                   }),
                                                               vector<Param>(
                                                                   {
                                                                       Param("B", DataCategory::Tensor, Precision::Any),
                                                                   })));
    }

    int register_all(TfFactory &tffactory)
    {
        register_lifecycle(tffactory);
        register_init(tffactory);
        register_io(tffactory);
        register_elementwise(tffactory);
        register_matmul(tffactory);
        register_changeshape(tffactory);
        register_reduce(tffactory);
        return 0;
    }
}