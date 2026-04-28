# 基础算子
算术运算
一元运算：Abs, Acos, Acosh, Asin, Asinh, Atan, Atanh, Ceil, Cos, Cosh, Erf, Exp, Floor, Log, Neg, Reciprocal, Sign, Sin, Sinh, Sqrt, Tan, Tanh
二元运算：Add, Div, Mul, Pow, Sub
比较运算：Equal, Greater, GreaterOrEqual, Less, LessOrEqual, Not
逻辑运算：And, Or, Xor, BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, BitShift
激活函数：Elu, Gelu, HardSigmoid, HardSwish, Hardmax, LeakyRelu, Mish, PRelu, Relu, Selu, Sigmoid, Softmax, Softplus, Softsign, ThresholdedRelu
数据变换
形状变换：Cast, CastLike, Flatten, Reshape, Squeeze, Transpose, Unsqueeze
元素选择与索引：ArgMax, ArgMin, Gather, GatherElements, GatherND, Scatter, ScatterElements, ScatterND, Slice, TopK
数据生成：Constant, ConstantOfShape, EyeLike, Range, RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike
池化操作
普通池化：AveragePool, GlobalAveragePool, GlobalLpPool, GlobalMaxPool, LpPool, MaxPool, Mean, Min
特殊池化：MaxRoiPool, MaxUnpool, SpaceToDepth, DepthToSpace
归一化操作：BatchNormalization, GroupNormalization, InstanceNormalization, LayerNormalization, LpNormalization, MeanVarianceNormalization
统计运算：CumSum, ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum, ReduceSumSquare
张量操作：Concat, ConcatFromSequence, Split, SplitToSequence, Expand, Pad, Resize, ReverseSequence, Shrink, Tile, Where
类型判断：IsInf, IsNaN
其他：Identity, OneHot, SequenceAt, SequenceConstruct, SequenceEmpty, SequenceErase, SequenceInsert, SequenceLength, SequenceMap, Shape, Size, StringConcat, StringNormalizer, StringSplit
融合算子
神经网络层：Conv, ConvInteger, ConvTranspose, DeformConv, GRU, LSTM, RNN, QLinearConv, QLinearMatMul
损失函数：NegativeLogLikelihoodLoss, SoftmaxCrossEntropyLoss
其他融合操作：AffineGrid, CenterCropPad, Col2Im, Compress, DFT, ImageDecoder, Loop, NonMaxSuppression, Optional, OptionalGetElement, OptionalHasElement, RegexFullMatch, Scan, TfIdfVectorizer, Upsample










