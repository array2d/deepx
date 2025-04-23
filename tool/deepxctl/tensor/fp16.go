package tensor

import (
	"encoding/binary"
	"math"
)

func Byte2ToFloat16(value []byte) float32 {
	bits := binary.BigEndian.Uint16(value)
	// 这里需要实现float16到float32的转换
	// 简化实现，实际项目中需要更完整的实现
	sign := float32(1)
	if bits&0x8000 != 0 {
		sign = -1
	}
	exp := int((bits & 0x7C00) >> 10)
	frac := float32(bits&0x03FF) / 1024.0

	if exp == 0 {
		return sign * frac * float32(1.0/16384.0) // 非规格化数
	} else if exp == 31 {
		if frac == 0 {
			return sign * float32(math.Inf(1)) // 无穷大
		}
		return float32(math.NaN()) // NaN
	}
	return sign * float32(math.Pow(2, float64(exp-15))) * (1.0 + frac) // 规格化数
}
