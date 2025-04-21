package tensor

import (
	"encoding/binary"
	"fmt"
	"math"
)

type Shape struct {
	Shape  []int  `json:"shape"`
	Stride []int  `json:"stride"`
	Ndim   int    `json:"ndim"`
	Size   int    `json:"size"`
	Dtype  string `json:"dtype"`
}

func NewTensorShape(shape []int) (s Shape) {
	s.Ndim = len(shape)
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
	indices = make([]int, s.Ndim)
	for i := 0; i < s.Ndim; i++ {
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

type Tensor struct {
	Data []byte
	Shape
}

func (t *Tensor) GetLinear(idx int) interface{} {
	byteSize := BitSize(t.Dtype) / 8
	start := idx
	end := start + byteSize

	if end > len(t.Data) {
		// 处理索引越界，这里简单返回nil，实际应根据需求处理
		return nil
	}

	data := t.Data[start:end]

	switch t.Dtype {
	case "bool":
		return data[0] != 0
	case "int8":
		return int8(data[0])
	case "int16":
		return int16(binary.BigEndian.Uint16(data))
	case "int32":
		return int32(binary.BigEndian.Uint32(data))
	case "int64":
		return binary.BigEndian.Uint64(data)
	case "float16":
		// Float16需要特殊处理，这里假设使用IEEE 754格式，实际可能需要转换函数
		return Byte2ToFloat16(data)
	case "float32":
		return math.Float32frombits(binary.BigEndian.Uint32(data))
	case "float64":
		return math.Float64frombits(binary.BigEndian.Uint64(data))
	default:
		return nil
	}
}

// Get 获取Tensor的值
func (t *Tensor) Get(indices ...int) interface{} {
	idx := t.Shape.LinearAt(indices)
	return t.GetLinear(idx)

}
