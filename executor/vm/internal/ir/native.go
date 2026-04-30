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

	// IO (built-in 命名)
	"print": true, // print(x, ...) → stdout
	"cerr":  true, // cerr(x, ...)  → stderr
	"input": true, // input([prompt]) → stdin
}

// IsNativeOp 判断是否为 VM 原生求值的符号算子。
func IsNativeOp(opcode string) bool {
	return nativeOps[opcode]
}

// NativeOpList 返回所有 VM 原生算子的有序列表 (仅 opcode)。
func NativeOpList() []string {
	ops := make([]string, 0, len(nativeOps))
	for op := range nativeOps {
		ops = append(ops, op)
	}
	return ops
}

// nativeSigs 定义每个内置算子的 def 签名: 参数读写 + 类型。
// 格式: "def op(reads...) -> (writes...)"
// 类型: num(int/float), int, float, bool, any
var nativeSigs = map[string]string{
	// 算术 (符号) — 双目, num → num
	"+": "def +(A:num, B:num) -> (C:num)",
	"-": "def -(A:num, B:num) -> (C:num)",
	"*": "def *(A:num, B:num) -> (C:num)",
	"/": "def /(A:num, B:num) -> (C:float)",
	"%": "def %(A:int, B:int) -> (C:int)",

	// 比较 (符号) — 双目, num → bool
	"==": "def ==(A:num, B:num) -> (C:bool)",
	"!=": "def !=(A:num, B:num) -> (C:bool)",
	"<":  "def <(A:num, B:num) -> (C:bool)",
	">":  "def >(A:num, B:num) -> (C:bool)",
	"<=": "def <=(A:num, B:num) -> (C:bool)",
	">=": "def >=(A:num, B:num) -> (C:bool)",

	// 逻辑 (符号) — bool → bool
	"&&": "def &&(A:bool, B:bool) -> (C:bool)",
	"||": "def ||(A:bool, B:bool) -> (C:bool)",
	"!":  "def !(A:bool) -> (C:bool)",

	// 位运算 (符号) — 双目, int → int
	"&":  "def &(A:int, B:int) -> (C:int)",
	"|":  "def |(A:int, B:int) -> (C:int)",
	"^":  "def ^(A:int, B:int) -> (C:int)",
	"<<": "def <<(A:int, B:int) -> (C:int)",
	">>": "def >>(A:int, B:int) -> (C:int)",

	// 数学 (built-in)
	"abs":  "def abs(A:num) -> (C:num)",
	"neg":  "def neg(A:num) -> (C:num)",
	"sqrt": "def sqrt(A:num) -> (C:float)",
	"exp":  "def exp(A:num) -> (C:float)",
	"log":  "def log(A:num) -> (C:float)",
	"pow":  "def pow(A:num, B:num) -> (C:float)",
	"min":  "def min(A:num, B:num) -> (C:num)",
	"max":  "def max(A:num, B:num) -> (C:num)",
	"sign": "def sign(A:num) -> (C:int)",

	// 类型转换 (built-in)
	"int":   "def int(A:any) -> (C:int)",
	"float": "def float(A:any) -> (C:float)",
	"bool":  "def bool(A:any) -> (C:bool)",

	// IO (built-in) — 可变参数输出 / 单参数输入
	"print": "def print(A:any, ...) -> ()",
	"cerr":  "def cerr(A:any, ...) -> ()",
	"input": "def input(prompt:string?) -> (C:string)",
}

// OpDefs 返回格式化后的算子定义文本列表 (按 opcode 排序)。
// 每个元素为 "def opcode(reads...) -> (writes...)" 格式。
func OpDefs() []string {
	defs := make([]string, 0, len(nativeSigs))
	for op := range nativeOps {
		if sig, ok := nativeSigs[op]; ok {
			defs = append(defs, sig)
		} else {
			defs = append(defs, "def "+op+"() -> ()")
		}
	}
	return defs
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
