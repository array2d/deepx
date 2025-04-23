package tensor

import "fmt"

func (t *Tensor[T]) Range(dimCount int, f func(indices []int)) {
	Shape := t.Shape
	if dimCount > len(Shape.Shape) {
		panic("dimCount exceeds the number of dimensions in the Tensor.")
	}

	totalSize := 1

	// 计算总的循环次数
	for i := 0; i < dimCount; i++ {
		totalSize *= Shape.At(i)
	}
	indices := make([]int, dimCount) // 初始化索引向量
	// 遍历所有可能的索引组合
	for idx := 0; idx < totalSize; idx++ {
		// 反算出 indices 数组
		idx_ := idx
		for dim := dimCount - 1; dim >= 0; dim-- {
			indices[dim] = idx_ % Shape.At(dim) // 计算当前维度的索引
			idx_ /= Shape.At(dim)               // 更新 idx
		}
		f(indices) // 调用传入的函数
	}
}

func AutoFormat(dtype string) string {
	switch dtype {
	case "bool":
		return "%v"
	case "int8":
		return "%d"
	case "int16":
		return "%d"
	case "int32":
		return "%d"
	case "int64":
		return "%d"
	case "float16":
		return "%f"
	case "float32":
		return "%f"
	case "float64":
		return "%f"
	default:
		return "%v"
	}
}

// Print 打印Tensor的值
func (t *Tensor[T]) Print(format_ ...string) {
	Shape := t.Shape
	format := AutoFormat(t.Dtype)
	if len(format_) > 0 {
		format = format_[0]
	}
	fmt.Print("shape:[")
	for i := 0; i < Shape.Dim; i++ {
		fmt.Print(Shape.At(i))
		if i < Shape.Dim-1 {
			fmt.Print(", ")
		}
	}
	fmt.Println("]")
	if Shape.Dim == 1 {
		fmt.Print("[")
		for i := 0; i < Shape.At(0); i++ {
			if i > 0 {
				fmt.Print(" ")
			}
			fmt.Printf(format, t.Get(i))
		}
		fmt.Println("]")
	} else if Shape.Dim == 2 {
		fmt.Println("[")
		for i := 0; i < Shape.At(0); i++ {
			fmt.Print(" [")
			for j := 0; j < Shape.At(1); j++ {
				if j > 0 {
					fmt.Print(" ")
				}
				fmt.Printf(format, t.Get(i, j))
			}

			fmt.Print("]")
			if i < Shape.At(0)-1 {
				fmt.Print(",")
			}
			fmt.Println()
		}
		fmt.Println("]")
	} else {
		t.Range(Shape.Dim-2, func(indices []int) {
			fmt.Print(indices)
			m, n := Shape.At(Shape.Dim-2), Shape.At(Shape.Dim-1)
			fmt.Print([]int{m, n})
			fmt.Println("=")

			fmt.Println("[")
			for i := 0; i < m; i++ {
				fmt.Print(" [")
				for j := 0; j < n; j++ {
					if j > 0 {
						fmt.Print(" ")
					}
					fmt.Printf(format, t.Get(append(indices, i, j)...))
				}

				fmt.Print("]")
				if i < m-1 {
					fmt.Print(",")
				}
				fmt.Println()
			}
			fmt.Println("]")
		})
	}
}
