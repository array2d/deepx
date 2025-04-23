package tensor

import (
	"fmt"
)

type Shape struct {
	Shape  []int  `json:"shape"`
	Stride []int  `json:"stride"`
	Dim    int    `json:"ndim"`
	Size   int    `json:"size"`
	Dtype  string `json:"dtype"`
}

func NewTensorShape(shape []int) (s Shape) {
	s.Dim = len(shape)
	s.Shape = make([]int, len(shape))
	copy(s.Shape, shape)
	s.Stride = make([]int, len(shape))
	s.Stride[len(shape)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		s.Stride[i] = s.Stride[i+1] * shape[i+1]
	}
	s.Size = s.Stride[0] * shape[0]
	return s
}
func (s Shape) String() string {
	return fmt.Sprintf("%v", s.Shape)
}

func (s Shape) At(i int) int {
	return s.Shape[i]
}

func (s Shape) LinearAt(indices []int) int {
	idx := 0
	for i := 0; i < len(indices); i++ {
		idx += indices[i] * s.Stride[i]
	}
	return idx
}
func (s Shape) LinearTo(idx int) (indices []int) {
	linearIndex := idx
	indices = make([]int, s.Dim)
	for i := 0; i < s.Dim; i++ {
		indices[i] = linearIndex / s.Stride[i]
		linearIndex %= s.Stride[i]
	}
	return indices
}

func BitSize(Dtype string) int {
	switch Dtype {
	case "bool":
		return 8
	case "int8":
		return 8
	case "int16":
		return 16
	case "int32":
		return 32
	case "int64":
		return 64
	case "float16":
		return 16
	case "float32":
		return 32
	case "float64":
		return 64
	default:
		return 0
	}
}

type Number interface {
	comparable
	float64 | float32 | int64 | int32 | int16 | int8 | bool
}

type Tensor[T Number] struct {
	Data []T
	Shape
}

// Get 获取Tensor的值
func (t *Tensor[T]) Get(indices ...int) T {
	idx := t.Shape.LinearAt(indices)
	return t.Data[idx]

}
