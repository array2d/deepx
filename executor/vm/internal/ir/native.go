package ir

// nativeOps 定义 VM 原生求值的算子集合。
// 包含符号算子 (中缀) 和 built-in 风格函数 (前缀)。
// 这些算子直接在 VM 内求值，不需要分发到 op-plat。
var nativeOps = map[string]bool{
	// 算术 (符号)
	"+": true, "-": true, "*": true, "/": true, "%": true,
	// 比较 (符号)
	"==": true, "!=": true, "<": true, ">": true, "<=": true, ">=": true,
	// 逻辑 (符号)
	"&&": true, "||": true, "!": true,
	// 位运算 (符号)
	"&": true, "|": true, "^": true, "<<": true, ">>": true,

	// 数学 (built-in 命名)
	"abs":  true, // abs(x) 绝对值
	"pow":  true, // pow(x, y) 幂运算
	"min":  true, // min(x, y) 取最小值
	"max":  true, // max(x, y) 取最大值
	"sqrt": true, // sqrt(x) 平方根
	"exp":  true, // exp(x) e^x
	"log":  true, // log(x) 自然对数
	"neg":  true, // neg(x) 取反
	"sign": true, // sign(x) 符号函数 (-1/0/1)

	// 类型转换 (built-in 命名)
	"int":   true, // int(x)  转整数 (截断)
	"float": true, // float(x)  转浮点
	"bool":  true, // bool(x) 转布尔
}

// IsNativeOp 判断是否为 VM 原生求值的符号算子。
func IsNativeOp(opcode string) bool {
	return nativeOps[opcode]
}

// IsUnaryNativeOp 判断是否为单目原生算子。
func IsUnaryNativeOp(opcode string) bool {
	switch opcode {
	case "!", "-", "abs", "sqrt", "exp", "log", "neg", "sign",
		"int", "float", "bool":
		return true
	}
	return false
}
