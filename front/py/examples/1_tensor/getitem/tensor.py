class Tensor:
    def __init__(self, data):
        # 假设 data 是嵌套 list 或 numpy ndarray
        import numpy as np
        self.data = np.array(data)

    def __getitem__(self, idx):
        # 支持 None, int, slice, tuple 等
        # 重点：遇到 None 时，插入新轴
        import numpy as np

        if not isinstance(idx, tuple):
            idx = (idx,)

        # 统计原始索引和 None 的位置，组装成新的索引
        new_idx = []
        expand_axes = []
        for i, ix in enumerate(idx):
            if ix is None:
                expand_axes.append(len(new_idx))
            else:
                new_idx.append(ix)

        # 先索引
        result = self.data[tuple(new_idx)]
        # 再插入新轴
        for ax in expand_axes:
            result = np.expand_dims(result, axis=ax)

        # 返回新 Tensor
        ret = Tensor(result)
        return ret

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data=\n{self.data})"


# 测试代码
t = Tensor([[1, 2, 3], [4, 5, 6]])
print("原始shape:", t.shape)  # (2, 3)

t2 = t[None]
print("t[None].shape:", t2.shape)  # (1, 2, 3)

t3 = t[:, None]
print("t[:, None].shape:", t3.shape)  # (2, 1, 3)

t4 = t[None, :, None]
print("t[None, :, None].shape:", t4.shape)  # (1, 2, 1, 3)