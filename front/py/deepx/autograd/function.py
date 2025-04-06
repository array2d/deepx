from deepx.autograd import Graph
class Context:
    def __init__(self,requires_grad=False):
        self._requires_grad = requires_grad
        self._saved_tensors = []
        self._non_tensor_data = {}

    def save_tensors(self, *tensors):
        self._saved_tensors.extend(tensors)

    @property
    def get_tensor(self):
        return tuple(self._saved_tensors)

    def save_data(self, key, value):
        self._non_tensor_data[key] = value

    def get_data(self, key):
        return self._non_tensor_data.get(key)

    @property
    def requires_grad(self):
        return self._requires_grad

class Function:
    @staticmethod
    def forward(ctx:Context, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx:Context, *grad_outputs):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        requires_grad = kwargs.pop('requires_grad', False)
        ctx = Context(requires_grad=requires_grad)
        result = cls.forward(ctx, *args, **kwargs)
        return result
    
