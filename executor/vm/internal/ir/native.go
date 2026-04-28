package ir

// nativeOps 定义 VM 原生求值的符号算子集合。
// 这些算子直接在 VM 内求值，不需要分发到 op-plat。
var nativeOps = map[string]bool{
	// 算术
	"+": true, "-": true, "*": true, "/": true, "%": true,
	// 比较
	"==": true, "!=": true, "<": true, ">": true, "<=": true, ">=": true,
	// 逻辑
	"&&": true, "||": true, "!": true,
	// 位运算
	"&": true, "|": true, "^": true, "<<": true, ">>": true,
}

// IsNativeOp 判断是否为 VM 原生求值的符号算子。
func IsNativeOp(opcode string) bool {
	return nativeOps[opcode]
}

// IsUnaryNativeOp 判断是否为单目原生算子。
func IsUnaryNativeOp(opcode string) bool {
	return opcode == "!" || opcode == "-"
}
