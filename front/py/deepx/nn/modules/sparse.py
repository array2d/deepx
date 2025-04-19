from  .module import Module
from deepx.tensor import Tensor

class Embedding(Module):
    def __init__(self, 
                 num_embeddings:int, 
                 embedding_dim:int, 
                 padding_idx:int=None, 
                 max_norm:float=None, 
                 norm_type:float=2.0, 
                 scale_grad_by_freq:bool=False, 
                 sparse:bool=False):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.weight = Tensor(num_embeddings, embedding_dim)
        self.weight.uniform_(-0.01, 0.01)
        if padding_idx is not None:
            self.weight[padding_idx] = 0

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
    
