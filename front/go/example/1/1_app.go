package main

import (
	"os"

	"deepx/front/go/deepx"
)

type Module1 struct {
	g *deepx.Graph
}

func (m *Module1) Linear(input *deepx.Tensor) *deepx.Tensor {
	// 创建输入节点
	w_node := m.g.AddTensor("W", deepx.DtypeFloat32, []int{3, 4, 5}, true)

	// 自动构建计算图
	y := input.Matmul(w_node.Tensor())

	b_node := m.g.AddTensor("b", deepx.DtypeFloat32, []int{1, 4, 5}, true)
	z := y.Add(b_node.Tensor())
	return z
}
func (m *Module1) Forward() (z *deepx.Tensor) {
	x_node := m.g.AddTensor("Input", deepx.DtypeFloat32, []int{1, 2, 3}, true)
	z = x_node.Tensor()
	for i := 0; i < 2; i++ {
		z = m.Linear(z)
	}

	return z
}

func main() {
	module := &Module1{
		g: deepx.NewGraph(),
	}
	module.Forward()

	dot := module.g.ToDOT()
	os.WriteFile("1_app.dot", []byte(dot), 0644)
}
