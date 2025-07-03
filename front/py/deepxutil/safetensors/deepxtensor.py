class TensorInfo:
    def __init__(self, dtype, ndim, shape, size, strides=None):
        self.dtype = dtype  # 数据精度类型，如"float32"
        self.ndim = ndim  # 维度数
        self.shape = shape  # 形状元组
        self.size = size  # 总元素数量
        self.strides = strides  # 步长数组（可选）


class Tensor:
    def __init__(self, data, tensorinfo: TensorInfo):
        assert isinstance(tensorinfo, TensorInfo),"tensorinfo必须是TensorInfo实例"
        self.data = data
        self.tensorinfo = tensorinfo
