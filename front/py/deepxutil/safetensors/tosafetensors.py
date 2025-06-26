
class SafeTensorSaver:
    def __init__(self, tensors, metadata=None):
        self.tensors = tensors
        self.metadata = metadata or {}

    def save(self, save_path):
        """保存模型到safetensor格式"""
        tensor_dict = {}
        for name, tensor in self.tensors.items():
            if isinstance(tensor, Tensor):
                tensor_dict[name] = tensor.data
            else:
                tensor_dict[name] = tensor

        from safetensors.numpy import save_file
        save_file(tensor_dict, save_path, metadata=self.metadata)

    # 使用示例
