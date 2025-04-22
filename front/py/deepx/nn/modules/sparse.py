from  .module import Module
from deepx.tensor import Tensor

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


    def __init__(self, 
                 num_embeddings:int, #嵌入字典的大小（词汇表大小）vocab_size，llama=128256
                 embedding_dim:int, #每个嵌入向量的维度,隐藏层大小hidden_size，llama=4096
                #  padding_idx:int=None,
                #  max_norm:float=None, 
                #  norm_type:float=2.0, 
                #  scale_grad_by_freq:bool=False, 
                 weight:Tensor=None,dtype='float32',
                #  sparse:bool=False
                 ):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
       
        # if padding_idx is not None:
        #     if padding_idx > 0:
        #         assert (
        #             padding_idx < self.num_embeddings
        #         ), "Padding_idx必须在num_embeddings范围内"
        #     elif padding_idx < 0:
        #         assert (
        #             padding_idx >= -self.num_embeddings
        #         ), "Padding_idx必须在num_embeddings范围内"
        #         padding_idx = self.num_embeddings + padding_idx
        # self.padding_idx = padding_idx
        # self.max_norm = max_norm
        # self.norm_type = norm_type
        # self.scale_grad_by_freq = scale_grad_by_freq
        if weight is None:
            self.weight = Tensor(shape=(num_embeddings, embedding_dim),dtype=dtype)
            self.register_parameter('weight', self.weight)
            self.reset_parameters()
        else:
            assert list(weight.shape) == [
                num_embeddings,
                embedding_dim,
            ], "权重形状与num_embeddings和embedding_dim不匹配"
            self.weight = weight
        
        # self.sparse = sparse
        
        # if padding_idx is not None:
        #     self.weight[padding_idx] = 0
    def reset_parameters(self) -> None:
        self.weight.normal_()  # 正态分布初始化权重
        self._fill_padding_idx_with_zero()  # 填充索引位置归零

    def _fill_padding_idx_with_zero(self) -> None:
        #TODO
        pass
        # if self.padding_idx is not None:
        #    self.weight[self.padding_idx].fill_(0)
    def forward(self, input:Tensor)->Tensor:
        return self.weight[input]
    
    def backward(self, grad:Tensor)->Tensor:
        self.weight.grad = grad
        return None
    
    def __str__(self)->str:
        return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"
    
    def __repr__(self)->str:
        return self.__str__()
    
    def __len__(self)->int:
        return self.num_embeddings
    
