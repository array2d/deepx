package funcs

type TensorShape struct {
	dims    []int // 形状的维度
	strides []int // 步长
}
type TensorData struct {
	LocatypeType string // 数据类型
	LocatypeUrl  int    // 数据类型大小
}

type Tensor struct {
	Var
	Shape TensorShape // 张量的形状

}
