#ifndef DEEPX_TF_REGISTER_MIAOBYTE_HPP
#define DEEPX_TF_REGISTER_MIAOBYTE_HPP

#include <memory>
#include "deepx/tf/tffactory.hpp"
#include "deepx/tf/elementwise.hpp"
#include "deepx/tf/changeshape.hpp"
#include "deepx/tf/reduce.hpp"
#include "deepx/tf/io.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tf
{
    // ═══════════════════════════════════════════════════════════
    // register_miaobyte — registers all miaobyte-authored Metal ops
    // into the provided TfFactory.
    //
    // This is called by the scheduler/dispatcher binary (not main.mm
    // directly, which uses its own Redis-queue dispatch).
    // ═══════════════════════════════════════════════════════════

    inline void register_miaobyte(TfFactory &factory)
    {
        using Author = tensorfunc::miaobyte;

        // ── elementwise: binary ──
        factory.add_tf(std::make_shared<Add<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Tensor, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<AddScalar<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Scalar, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<Sub<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Tensor, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<SubScalar<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Scalar, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<RSubScalar<Author>>(
            vector<Param>{{"", DataCategory::Scalar, Precision::Float},
                          {"", DataCategory::Tensor, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<Mul<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Tensor, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<MulScalar<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Scalar, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<Div<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Tensor, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<DivScalar<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Scalar, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<RDivScalar<Author>>(
            vector<Param>{{"", DataCategory::Scalar, Precision::Float},
                          {"", DataCategory::Tensor, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<Max<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Tensor, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<MaxScalar<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Scalar, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<Min<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Tensor, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<MinScalar<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Scalar, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<Pow<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Tensor, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<PowScalar<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Scalar, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));

        // ── elementwise: unary ──
        factory.add_tf(std::make_shared<ReLU<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<Invert<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Int}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Int}}));

        // ── elementwise: comparison ──
        factory.add_tf(std::make_shared<Equal<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Tensor, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Bool}}));
        factory.add_tf(std::make_shared<NotEqual<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Tensor, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Bool}}));
        factory.add_tf(std::make_shared<Less<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Tensor, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Bool}}));
        factory.add_tf(std::make_shared<Greater<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Tensor, Precision::Float}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Bool}}));

        // ── changeshape ──
        factory.add_tf(std::make_shared<Reshape<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Vector}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<Transpose<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Vector}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<Concat<Author>>(
            vector<Param>{{"", DataCategory::Vector},
                          {"", DataCategory::Scalar}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<BroadcastTo<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Vector}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<IndexSelect<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Tensor, Precision::Int},
                          {"", DataCategory::Scalar}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<Repeat<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Vector}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));

        // ── reduce ──
        factory.add_tf(std::make_shared<Sum<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Vector},
                          {"", DataCategory::Scalar, Precision::Bool}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<Prod<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Vector},
                          {"", DataCategory::Scalar, Precision::Bool}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<ReduceMax<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Vector},
                          {"", DataCategory::Scalar, Precision::Bool}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
        factory.add_tf(std::make_shared<ReduceMin<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::Vector},
                          {"", DataCategory::Scalar, Precision::Bool}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));

        // ── io ──
        factory.add_tf(std::make_shared<Print<Author>>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}},
            vector<Param>{}));
        factory.add_tf(std::make_shared<Save>(
            vector<Param>{{"", DataCategory::Tensor, Precision::Float},
                          {"", DataCategory::String}},
            vector<Param>{}));
        factory.add_tf(std::make_shared<Load>(
            vector<Param>{{"", DataCategory::String}},
            vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
    }

} // namespace deepx::tf

#endif // DEEPX_TF_REGISTER_MIAOBYTE_HPP
