# mypy: 允许无类型定义的函数
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter

from .module import Module


__all__ = ["Embedding", "EmbeddingBag"]


class Embedding(Module):
    r"""一个存储固定字典和大小的嵌入向量的简单查找表。

    该模块常用于存储词嵌入并通过索引检索它们。
    模块的输入是索引列表，输出是对应的词嵌入向量。

    参数:
        num_embeddings (int): 嵌入字典的大小（词汇表大小）
        embedding_dim (int): 每个嵌入向量的维度
        padding_idx (int, 可选): 如果指定，该索引位置的条目不参与梯度计算；
                                  因此，该位置的嵌入向量在训练中不会更新，保持为固定的"填充"向量。
                                  对于新创建的嵌入层，该位置的嵌入向量默认全零，但可更新为其他值作为填充向量。
        max_norm (float, 可选): 如果指定，范数超过此值的嵌入向量会被重新归一化到该范数
        norm_type (float, 可选): 计算max_norm时使用的p范数（默认L2范数，p=2）
        scale_grad_by_freq (bool, 可选): 如果为True，梯度会按mini-batch中词的频率倒数缩放（默认False）
        sparse (bool, 可选): 如果为True，权重矩阵的梯度将是稀疏张量（详见注释）

    属性:
        weight (Tensor): 模块的可学习权重，形状为(num_embeddings, embedding_dim)，
                         初始化为正态分布N(0, 1)

    形状:
        - 输入: :math:`(*)`, 任意形状的IntTensor或LongTensor，包含要提取的索引
        - 输出: :math:`(*, H)`, 其中*是输入形状，H=embedding_dim

    .. 注意::
        注意只有部分优化器支持稀疏梯度：目前支持的有SGD（CPU和CUDA）、SparseAdam（CPU和CUDA）、Adagrad（CPU）

    .. 注意::
        当max_norm不为None时，嵌入层的前向传播会原地修改weight张量。
        由于梯度计算所需的张量不能被原地修改，因此在调用前向传播前对weight进行可微操作时，
        若max_norm不为None则需要克隆weight。例如::

            n, d, m = 3, 5, 7
            embedding = nn.Embedding(n, d, max_norm=1.0)
            W = torch.randn((m, d), requires_grad=True)
            idx = torch.tensor([1, 2])
            a = embedding.weight.clone() @ W.t()  # weight必须克隆以保证可微性
            b = embedding(idx) @ W.t()  # 原地修改weight
            out = (a.unsqueeze(0) + b.unsqueeze(1))
            loss = out.sigmoid().prod()
            loss.backward()

    示例::

        >>> # 包含10个3维张量的嵌入层
        >>> embedding = nn.Embedding(10, 3)
        >>> # 2个样本，每个包含4个索引的批次
        >>> input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> embedding(input)
        tensor([[[-0.0251, -1.6902,  0.7172],
                 [-0.6431,  0.0748,  0.6969],
                 [ 1.4970,  1.3448, -0.9685],
                 [-0.3677, -2.7265, -0.1685]],

                [[ 1.4970,  1.3448, -0.9685],
                 [ 0.4362, -0.4004,  0.9400],
                 [-0.6431,  0.0748,  0.6969],
                 [ 0.9124, -2.3616,  1.1151]]])


        >>> # 带padding_idx的示例
        >>> embedding = nn.Embedding(10, 3, padding_idx=0)
        >>> input = torch.LongTensor([[0, 2, 0, 5]])
        >>> embedding(input)
        tensor([[[ 0.0000,  0.0000,  0.0000],
                 [ 0.1535, -2.0309,  0.9315],
                 [ 0.0000,  0.0000,  0.0000],
                 [-0.1655,  0.9897,  0.0635]]])

        >>> # 修改填充向量的示例
        >>> padding_idx = 0
        >>> embedding = nn.Embedding(3, 3, padding_idx=padding_idx)
        >>> embedding.weight
        Parameter containing:
        tensor([[ 0.0000,  0.0000,  0.0000],
                [-0.7895, -0.7089, -0.0364],
                [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
        >>> with torch.no_grad():
        ...     embedding.weight[padding_idx] = torch.ones(3)
        >>> embedding.weight
        Parameter containing:
        tensor([[ 1.0000,  1.0000,  1.0000],
                [-0.7895, -0.7089, -0.0364],
                [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    freeze: bool
    sparse: bool

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx必须在num_embeddings范围内"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx必须在num_embeddings范围内"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = Parameter(
                torch.empty((num_embeddings, embedding_dim), **factory_kwargs),
                requires_grad=not _freeze,
            )
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [
                num_embeddings,
                embedding_dim,
            ], "权重形状与num_embeddings和embedding_dim不匹配"
            self.weight = Parameter(_weight, requires_grad=not _freeze)

        self.sparse = sparse

    def reset_parameters(self) -> None:
        init.normal_(self.weight)  # 正态分布初始化权重
        self._fill_padding_idx_with_zero()  # 填充索引位置归零

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():  # 不计算梯度
                self.weight[self.padding_idx].fill_(0)  # 填充位置设为0

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(
        cls,
        embeddings,
        freeze=True,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
    ):
        r"""从给定的2维FloatTensor创建Embedding实例。

        参数:
            embeddings (Tensor): 包含嵌入权重的FloatTensor，
                第一维作为num_embeddings，第二维作为embedding_dim。
            freeze (bool, 可选): 若为True，张量在学习过程中不更新，
                相当于embedding.weight.requires_grad = False。默认True。
            padding_idx (int, 可选): 同模块初始化文档说明。
            max_norm (float, 可选): 同模块初始化文档说明。
            norm_type (float, 可选): 同模块初始化文档说明，默认2。
            scale_grad_by_freq (bool, 可选): 同模块初始化文档说明，默认False。
            sparse (bool, 可选): 同模块初始化文档说明。

        示例::

            >>> # 包含预训练权重的FloatTensor
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embedding = nn.Embedding.from_pretrained(weight)
            >>> # 获取索引1的嵌入
            >>> input = torch.LongTensor([1])
            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> embedding(input)
            tensor([[ 4.0000,  5.1000,  6.3000]])
        """
        assert (
            embeddings.dim() == 2
        ), "Embeddings参数应为2维张量"
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            _freeze=freeze,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        return embedding


class EmbeddingBag(Module):
    r"""计算嵌入"袋"的和或均值，无需实例化中间嵌入。

    对于固定长度的袋、无per_sample_weights、无等于padding_idx的索引，且输入为2D时，
    该类的行为如下：
        * mode="sum"等价于Embedding层后接torch.sum(dim=1)
        * mode="mean"等价于Embedding层后接torch.mean(dim=1)
        * mode="max"等价于Embedding层后接torch.max(dim=1)

    但EmbeddingBag比链式操作更节省时间和内存。

    EmbeddingBag还支持在正向传播时传入样本权重，
    这会在按mode指定的方式进行加权归约前缩放嵌入输出。
    若传入per_sample_weights，仅支持mode="sum"，即按权重计算加权和。

    参数:
        num_embeddings (int): 嵌入字典的大小（词汇表大小）
        embedding_dim (int): 每个嵌入向量的维度
        max_norm (float, 可选): 若指定，范数超过此值的嵌入向量会被重新归一化到该范数
        norm_type (float, 可选): 计算max_norm时使用的p范数（默认L2范数，p=2）
        scale_grad_by_freq (bool, 可选): 若为True，梯度会按mini-batch中词的频率倒数缩放（默认False）。
                                         注意：mode="max"时不支持此选项。
        mode (str, 可选): "sum"、"mean"或"max"，指定袋的归约方式。
                           "sum"计算加权和（考虑per_sample_weights），
                           "mean"计算袋内平均值，"max"计算袋内最大值。默认"mean"。
        sparse (bool, 可选): 若为True，权重矩阵的梯度将是稀疏张量（详见注释）。
                             注意：mode="max"时不支持此选项。
        include_last_offset (bool, 可选): 若为True，offsets包含一个额外元素，
                                          其值等于indices的长度，符合CSR格式。
        padding_idx (int, 可选): 若指定，该索引位置的条目不参与梯度计算；
                                 因此，该位置的嵌入向量在训练中不会更新，保持为固定的"填充"向量。
                                 对于新创建的EmbeddingBag，该位置的嵌入向量默认全零，
                                 但可更新为其他值作为填充向量。注意该位置的嵌入向量会被排除在归约之外。

    属性:
        weight (Tensor): 模块的可学习权重，形状为(num_embeddings, embedding_dim)，
                         初始化为正态分布N(0, 1)。

    示例::

        >>> # 包含10个3维张量的EmbeddingBag（求和模式）
        >>> embedding_sum = nn.EmbeddingBag(10, 3, mode='sum')
        >>> # 2个样本，每个包含4个索引的输入（展平为1D）
        >>> input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long)
        >>> offsets = torch.tensor([0, 4], dtype=torch.long)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> embedding_sum(input, offsets)
        tensor([[-0.8861, -5.4350, -0.0523],
                [ 1.1306, -2.5798, -1.0044]])

        >>> # 带padding_idx的示例
        >>> embedding_sum = nn.EmbeddingBag(10, 3, mode='sum', padding_idx=2)
        >>> input = torch.tensor([2, 2, 2, 2, 4, 3, 2, 9], dtype=torch.long)
        >>> offsets = torch.tensor([0, 4], dtype=torch.long)
        >>> embedding_sum(input, offsets)
        tensor([[ 0.0000,  0.0000,  0.0000],
                [-0.7082,  3.2145, -2.6251]])

        >>> # 从Embedding加载EmbeddingBag的示例
        >>> embedding = nn.Embedding(10, 3, padding_idx=2)
        >>> embedding_sum = nn.EmbeddingBag.from_pretrained(
                embedding.weight,
                padding_idx=embedding.padding_idx,
                mode='sum')
    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "mode",
        "sparse",
        "include_last_offset",
        "padding_idx",
    ]

    num_embeddings: int
    embedding_dim: int
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    mode: str
    sparse: bool
    include_last_offset: bool
    padding_idx: Optional[int]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        include_last_offset: bool = False,
        padding_idx: Optional[int] = None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "padding_idx必须在num_embeddings范围内"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "padding_idx必须在num_embeddings范围内"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        if _weight is None:
            self.weight = Parameter(
                torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
            )
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [
                num_embeddings,
                embedding_dim,
            ], "权重形状与num_embeddings和embedding_dim不匹配"
            self.weight = Parameter(_weight)
        self.mode = mode
        self.sparse = sparse
        self.include_last_offset = include_last_offset

    def reset_parameters(self) -> None:
        init.normal_(self.weight)  # 正态分布初始化权重
        self._fill_padding_idx_with_zero()  # 填充索引位置归零

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():  # 不计算梯度
                self.weight[self.padding_idx].fill_(0)  # 填充位置设为0

    def forward(
        self,
        input: Tensor,
        offsets: Optional[Tensor] = None,
        per_sample_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """EmbeddingBag的正向传播。

        参数:
            input (Tensor): 包含嵌入矩阵索引袋的张量。
            offsets (Tensor, 可选): 仅当input为1D时使用，确定input中每个袋（序列）的起始索引位置。
            per_sample_weights (Tensor, 可选): 浮点/双精度权重张量，None表示所有权重为1。
                若指定，形状必须与input相同，且在offsets非None时使用相同的偏移量。仅支持mode='sum'。

        返回:
            形状为(B, embedding_dim)的张量。

        .. 注意::

            关于input和offsets的说明：
            - input和offsets必须同类型（int或long）
            - 若input为2D形状(B, N)，视为B个固定长度N的袋，返回B个按mode聚合的值，此时offsets被忽略且必须为None。
            - 若input为1D形状(N)，视为多个袋（序列）的拼接，offsets必须为1D张量，包含每个袋在input中的起始索引位置。
              因此，对于形状(B)的offsets，input视为B个袋，空袋（长度为0）返回全零向量。
        """
        return F.embedding_bag(
            input,
            self.weight,
            offsets,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.mode,
            self.sparse,
            per_sample_weights,
            self.include_last_offset,
            self.padding_idx,
        )

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        s += ", mode={mode}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        return s.format(**{k: repr(v) for k, v in self.__dict__.items()})

    @classmethod
    def from_pretrained(
        cls,
        embeddings: Tensor,
        freeze: bool = True,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        include_last_offset: bool = False,
        padding_idx: Optional[int] = None,
    ) -> "EmbeddingBag":
        r"""从给定的2维FloatTensor创建EmbeddingBag实例。

        参数:
            embeddings (Tensor): 包含EmbeddingBag权重的FloatTensor，
                第一维作为num_embeddings，第二维作为embedding_dim。
            freeze (bool, 可选): 若为True，张量在学习过程中不更新，
                相当于embeddingbag.weight.requires_grad = False。默认True。
            max_norm (float, 可选): 同模块初始化文档说明，默认None。
            norm_type (float, 可选): 同模块初始化文档说明，默认2。
            scale_grad_by_freq (bool, 可选): 同模块初始化文档说明，默认False。
            mode (str, 可选): 同模块初始化文档说明，默认"mean"。
            sparse (bool, 可选): 同模块初始化文档说明，默认False。
            include_last_offset (bool, 可选): 同模块初始化文档说明，默认False。
            padding_idx (int, 可选): 同模块初始化文档说明，默认None。

        示例::

            >>> # 包含预训练权重的FloatTensor
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embeddingbag = nn.EmbeddingBag.from_pretrained(weight)
            >>> # 获取索引1和0的嵌入袋（2D输入）
            >>> input = torch.LongTensor([[1, 0]])
            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> embeddingbag(input)
            tensor([[ 2.5000,  3.7000,  4.6500]])
        """
        assert (
            embeddings.dim() == 2
        ), "Embeddings参数应为2维张量"
        rows, cols = embeddings.shape
        embeddingbag = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            mode=mode,
            sparse=sparse,
            include_last_offset=include_last_offset,
            padding_idx=padding_idx,
        )
        embeddingbag.weight.requires_grad = not freeze
        return embeddingbag