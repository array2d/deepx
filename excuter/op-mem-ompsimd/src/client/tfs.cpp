#include "client/tfs.hpp"

#include "deepx/dtype.hpp"
#include "deepx/tf/arg.hpp"
#include "deepx/tf/tensorlife.hpp"
#include "deepx/tf/init.hpp"
#include "deepx/tf/io.hpp"
#include "deepx/tf/changeshape.hpp"
#include "deepx/tf/elementwise.hpp"
#include "deepx/tf/reduce.hpp"
#include "deepx/tf/tffactory.hpp"
#include "deepx/tf/matmul.hpp"
#include "deepx/tensorfunc/authors.hpp"
namespace deepx::tf
{
    using namespace std;
    using namespace deepx::tensorfunc;
    // tensor
    void register_lifecycle(TfFactory &tffactory)
    {
        // argset author=miaobyte
        tffactory.add_tf(std::make_shared<ArgSet>(vector<Param>(
                                                      {
                                                          Param("value", DataCategory::Var, Precision::Any),
                                                      }),
                                                  vector<Param>(
                                                      {
                                                          Param("name", DataCategory::Var, Precision::Any),
                                                      })));
        // vecset author=miaobyte
        tffactory.add_tf(std::make_shared<VecSet>(
            vector<Param>(
                {
                    Param("value", DataCategory::Vector, Precision::Any),
                }),
            vector<Param>(
                {
                    Param("name", DataCategory::Vector, Precision::Any),
                })));
        // newtensor author=miaobyte
        tffactory.add_tf(std::make_shared<NewTensor>(vector<Param>(
                                                         {
                                                             Param("shape", DataCategory::Vector, Precision::Int32),
                                                         }),
                                                     vector<Param>(
                                                         {
                                                             Param("tensor1", DataCategory::Tensor, Precision::Any),
                                                         })));
        // newtensor author=miaobyte
        tffactory.add_tf(std::make_shared<NewTensor>(vector<Param>(
                                                         {
                                                             Param("shape", DataCategory::Var, Precision::String),
                                                         }),
                                                     vector<Param>(
                                                         {
                                                             Param("t", DataCategory::Tensor, Precision::Any),
                                                         })));
        //copytensor
        tffactory.add_tf(std::make_shared<CopyTensor>(vector<Param>(
                                                         {
                                                             Param("src", DataCategory::Tensor, Precision::Any),
                                                             Param("dst", DataCategory::Tensor, Precision::Any),
                                                         }),
                                                     vector<Param>()));
        //deltensor
        tffactory.add_tf(std::make_shared<DelTensor>(vector<Param>(
                                                         {
                                                             Param("t", DataCategory::Tensor, Precision::Any),
                                                         }),
                                                     vector<Param>()));
    }

    // init
    void register_init(TfFactory &tffactory)
    {
        // constant author=miaobyte
        tffactory.add_tf(std::make_shared<Constant<miaobyte>>(vector<Param>(
                                                                  {
                                                                      Param("t", DataCategory::Tensor, Precision::Any),
                                                                      Param("value", DataCategory::Var, Precision::Any),
                                                                  }),
                                                              vector<Param>()));
        // arange author=miaobyte
        tffactory.add_tf(std::make_shared<Arange<miaobyte>>(vector<Param>(
                                                                {
                                                                    Param("t", DataCategory::Tensor, Precision::Any),
                                                                    Param("start", DataCategory::Var, Precision::Any),
                                                                    Param("step", DataCategory::Var, Precision::Any),
                                                                }),
                                                            vector<Param>()));
        // uniform author=miaobyte
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
        // print author=miaobyte
        opfactory.add_tf(std::make_shared<Print<miaobyte>>(vector<Param>(
                                                               {
                                                                   Param("", DataCategory::Tensor, Precision::Any),
                                                               }),
                                                           vector<Param>()));
        // print author=miaobyte
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
        // add author=cblas
        tffactory.add_tf(std::make_shared<Add<cblas>>(vector<Param>(
                                                          {
                                                              Param("a", DataCategory::Tensor, Precision::Float64 | Precision::Float32),
                                                              Param("b", DataCategory::Tensor, Precision::Float64 | Precision::Float32),
                                                          }),
                                                      vector<Param>(
                                                          {
                                                              Param("c", DataCategory::Tensor, Precision::Float64 | Precision::Float32),
                                                          })));

        // add scalar author=miaobyte
        tffactory.add_tf(std::make_shared<AddScalar<miaobyte>>(vector<Param>(
                                                                   {
                                                                       Param("a", DataCategory::Tensor, Precision::Any),
                                                                       Param("scalar", DataCategory::Var, Precision::Any),
                                                                   }),
                                                               vector<Param>(
                                                                   {
                                                                       Param("c", DataCategory::Tensor, Precision::Any),
                                                                   })));
        // sub author=miaobyte
        tffactory.add_tf(std::make_shared<Sub<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("a", DataCategory::Tensor, Precision::Any),
                                                                 Param("b", DataCategory::Tensor, Precision::Any),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("c", DataCategory::Tensor, Precision::Any),
                                                             })));
        // sub scalar author=miaobyte
        tffactory.add_tf(std::make_shared<SubScalar<miaobyte>>(vector<Param>(
                                                                   {
                                                                       Param("a", DataCategory::Tensor, Precision::Any),
                                                                       Param("scalar", DataCategory::Var, Precision::Any),
                                                                   }),
                                                               vector<Param>(
                                                                   {
                                                                       Param("c", DataCategory::Tensor, Precision::Any),
                                                                   })));
        // mul author=miaobyte
        tffactory.add_tf(std::make_shared<Mul<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("A", DataCategory::Tensor, Precision::Any),
                                                                 Param("B", DataCategory::Tensor, Precision::Any),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("C", DataCategory::Tensor, Precision::Any),
                                                             })));
        // mul scalar author=miaobyte
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
        // div scalar author=miaobyte
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
        // invert author=miaobyte
        tffactory.add_tf(std::make_shared<Invert<miaobyte>>(vector<Param>(
                                                                 {
                                                                     Param("A", DataCategory::Tensor, Precision::Int64 | Precision::Int32 | Precision::Int16 | Precision::Int8),
                                                                 }),
                                                                 vector<Param>(
                                                                     {
                                                                         Param("C", DataCategory::Tensor, Precision::Int64 | Precision::Int32 | Precision::Int16 | Precision::Int8),
                                                                     })));
        // sqrt author=miaobyte
        tffactory.add_tf(std::make_shared<Sqrt<miaobyte>>(vector<Param>(
                                                              {
                                                                  Param("A", DataCategory::Tensor, Precision::Any),
                                                              }),
                                                          vector<Param>(
                                                              {
                                                                  Param("C", DataCategory::Tensor, Precision::Any),
                                                              })));

        // pow author=miaobyte
        tffactory.add_tf(std::make_shared<Pow<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("A", DataCategory::Tensor, Precision::Any),
                                                                 Param("B", DataCategory::Tensor, Precision::Any),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("C", DataCategory::Tensor, Precision::Any),
                                                             })));
        // pow scalar author=miaobyte
        tffactory.add_tf(std::make_shared<PowScalar<miaobyte>>(vector<Param>(
                                                                   {
                                                                       Param("A", DataCategory::Tensor, Precision::Any),
                                                                       Param("scalar", DataCategory::Var, Precision::Any),
                                                                   }),
                                                               vector<Param>(
                                                                   {
                                                                       Param("C", DataCategory::Tensor, Precision::Any),
                                                                   })));
        // rpowscalar author=miaobyte
        tffactory.add_tf(std::make_shared<RpowScalar<miaobyte>>(vector<Param>(
                                                                   {
                                                                       Param("scalar", DataCategory::Var, Precision::Any),
                                                                       Param("A", DataCategory::Tensor, Precision::Any),
                                                                   }),
                                                               vector<Param>(
                                                                   {
                                                                       Param("C", DataCategory::Tensor, Precision::Any),
                                                                   })));

        // log author=miaobyte
        tffactory.add_tf(std::make_shared<Log<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("A", DataCategory::Tensor, Precision::Any),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("C", DataCategory::Tensor, Precision::Any),
                                                             })));
        // exp author=miaobyte
        tffactory.add_tf(std::make_shared<Exp<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("A", DataCategory::Tensor, Precision::Any),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("C", DataCategory::Tensor, Precision::Any),
                                                             })));
        // max author=miaobyte
        tffactory.add_tf(std::make_shared<Max<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("A", DataCategory::Tensor, Precision::Any),
                                                                 Param("B", DataCategory::Tensor, Precision::Any),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("C", DataCategory::Tensor, Precision::Any),
                                                             })));
        // max scalar author=miaobyte
        tffactory.add_tf(std::make_shared<MaxScalar<miaobyte>>(vector<Param>(
                                                                   {
                                                                       Param("A", DataCategory::Tensor, Precision::Any),
                                                                       Param("scalar", DataCategory::Var, Precision::Any),
                                                                   }),
                                                               vector<Param>(
                                                                   {
                                                                       Param("C", DataCategory::Tensor, Precision::Any),
                                                                   })));
        // min author=miaobyte
        tffactory.add_tf(std::make_shared<Min<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("A", DataCategory::Tensor, Precision::Any),
                                                                 Param("B", DataCategory::Tensor, Precision::Any),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("C", DataCategory::Tensor, Precision::Any),
                                                             })));
        // min scalar author=miaobyte
        tffactory.add_tf(std::make_shared<MinScalar<miaobyte>>(vector<Param>(
                                                                   {
                                                                       Param("A", DataCategory::Tensor, Precision::Any),
                                                                       Param("scalar", DataCategory::Var, Precision::Any),
                                                                   }),
                                                               vector<Param>(
                                                                   {
                                                                       Param("C", DataCategory::Tensor, Precision::Any),
                                                                   })));
        // compare author=miaobyte
        tffactory.add_tf(std::make_shared<Compare<miaobyte>>(vector<Param>(
                                                                 {
                                                                     Param("A", DataCategory::Tensor, Precision::Any),
                                                                     Param("B", DataCategory::Tensor, Precision::Any),

                                                                 }),
                                                             vector<Param>(
                                                                 {
                                                                     Param("mask", DataCategory::Tensor, Precision::Float32),
                                                                 })));
        // compare scalar author=miaobyte
        tffactory.add_tf(std::make_shared<CompareScalar<miaobyte>>(vector<Param>(
                                                                       {
                                                                           Param("A", DataCategory::Tensor, Precision::Any),
                                                                           Param("scalar", DataCategory::Var, Precision::Any),
                                                                       }),
                                                                   vector<Param>(
                                                                       {
                                                                           Param("mask", DataCategory::Tensor, Precision::Float32),
                                                                       })));
    }
    // matmul
    void register_matmul(TfFactory &tffactory)
    {
        // matmul author=miaobyte
        tffactory.add_tf(std::make_shared<MatMul<miaobyte>>(vector<Param>(
                                                                {
                                                                    Param("A", DataCategory::Tensor, Precision::Any),
                                                                    Param("B", DataCategory::Tensor, Precision::Any),
                                                                }),
                                                            vector<Param>(
                                                                {
                                                                    Param("C", DataCategory::Tensor, Precision::Any),
                                                                })));
        // matmul author=cblas
        tffactory.add_tf(std::make_shared<MatMul<cblas>>(vector<Param>(
                                                             {
                                                                 Param("A", DataCategory::Tensor, Precision::Float64 | Precision::Float32),
                                                                 Param("B", DataCategory::Tensor, Precision::Float64 | Precision::Float32),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("C", DataCategory::Tensor, Precision::Float64 | Precision::Float32),
                                                             })));
    }
    // // changeshape
    void register_changeshape(TfFactory &tffactory)
    {
        // reshape author=miaobyte
        tffactory.add_tf(std::make_shared<Reshape<miaobyte>>(vector<Param>(
                                                                 {
                                                                     Param("A", DataCategory::Tensor, Precision::Any),
                                                                     Param("shape", DataCategory::Vector, Precision::Int32),
                                                                 }),
                                                             vector<Param>(
                                                                 {
                                                                     Param("B", DataCategory::Tensor, Precision::Any),
                                                                 })));

        tffactory.add_tf(std::make_shared<Transpose<miaobyte>>(vector<Param>(
                                                                   {
                                                                       Param("A", DataCategory::Tensor, Precision::Any),
                                                                       Param("dim_order", DataCategory::Vector, Precision::Int32),
                                                                   }),
                                                               vector<Param>(
                                                                   {
                                                                       Param("C", DataCategory::Tensor, Precision::Any),
                                                                   })));
        // concat author=miaobyte
        tffactory.add_tf(std::make_shared<Concat<miaobyte>>(vector<Param>(
                                                                {
                                                                    Param("tensors", DataCategory::ListTensor, Precision::Any),
                                                                    Param("dim", DataCategory::Var, Precision::Int32),
                                                                }),
                                                            vector<Param>(
                                                                {
                                                                    Param("result", DataCategory::Tensor, Precision::Any),
                                                                })));
        // broadcastto author=miaobyte
        tffactory.add_tf(std::make_shared<BroadcastTo<miaobyte>>(vector<Param>(
                                                                     {
                                                                         Param("A", DataCategory::Tensor, Precision::Any),
                                                                         Param("new_shape", DataCategory::Vector, Precision::Int32),
                                                                     }),
                                                                 vector<Param>(
                                                                     {
                                                                         Param("B", DataCategory::Tensor, Precision::Any),
                                                                     })));
    }
    // // reduce
    void register_reduce(TfFactory &tffactory)
    {
        // sum author=miaobyte
        tffactory.add_tf(std::make_shared<Sum<miaobyte>>(vector<Param>(
                                                             {
                                                                 Param("A", DataCategory::Tensor, Precision::Any),
                                                                 Param("axis", DataCategory::Vector, Precision::Int32),
                                                                 Param("keepdims", DataCategory::Var, Precision::Bool),
                                                             }),
                                                         vector<Param>(
                                                             {
                                                                 Param("B", DataCategory::Tensor, Precision::Any),
                                                             })));
        // prod author=miaobyte
        tffactory.add_tf(std::make_shared<Prod<miaobyte>>(vector<Param>(
                                                              {
                                                                  Param("A", DataCategory::Tensor, Precision::Any),
                                                                  Param("axis", DataCategory::Vector, Precision::Int32),
                                                                  Param("keepdims", DataCategory::Var, Precision::Bool),
                                                              }),
                                                          vector<Param>(
                                                              {
                                                                  Param("B", DataCategory::Tensor, Precision::Any),
                                                                 })));
        // reducemax author=miaobyte
        tffactory.add_tf(std::make_shared<ReduceMax<miaobyte>>(vector<Param>(
                                                                 {
                                                                     Param("A", DataCategory::Tensor, Precision::Any),
                                                                     Param("axis", DataCategory::Vector, Precision::Int32),
                                                                     Param("keepdims", DataCategory::Var, Precision::Bool),
                                                                 }),
                                                                 vector<Param>(
                                                                     {
                                                                         Param("B", DataCategory::Tensor, Precision::Any),
                                                                     })));
        // reducemin author=miaobyte
        tffactory.add_tf(std::make_shared<ReduceMin<miaobyte>>(vector<Param>(
                                                                 {
                                                                     Param("A", DataCategory::Tensor, Precision::Any),
                                                                     Param("axis", DataCategory::Vector, Precision::Int32),
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
        register_util(tffactory);
        register_elementwise(tffactory);
        register_matmul(tffactory);
        register_changeshape(tffactory);
        register_reduce(tffactory);
        return 0;
    }
}