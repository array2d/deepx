package main

import (
	"os"

	"deepx/front/go/deepx"
)

func main() {
	// 创建计算图
	g := deepx.NewGraph()

	// 创建 Transformer 配置
	config := struct {
		hidden_size int
		num_heads   int
		num_layers  int
		mlp_ratio   int
		dtype       deepx.Dtype
	}{
		hidden_size: 256,
		num_heads:   4,
		num_layers:  2,
		mlp_ratio:   4,
		dtype:       deepx.DtypeFloat32,
	}

	// 创建 Transformer 模型
	transformer := deepx.NewTransformer(
		"transformer",
		config.num_layers,
		config.hidden_size,
		config.num_heads,
		config.mlp_ratio,
		config.dtype,
		g,
	)

	// 创建输入张量
	batch_size := 1
	seq_len := 32
	input := g.AddTensor(
		"input",
		config.dtype,
		[]int{batch_size, seq_len, config.hidden_size},
		true,
	)

	// 前向计算，构建计算图
	transformer.Forward(input.Tensor())

	// 将计算图导出为 DOT 格式
	dot := g.ToDOT()
	os.WriteFile("transformer.dot", []byte(dot), 0644)
}
