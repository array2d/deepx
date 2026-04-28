package engine

import (
	"context"
	"fmt"
	"log"
	"strconv"
	"strings"

	"git.array2d.com/ai/deepx/executor/vm/internal/ir"
	"github.com/redis/go-redis/v9"
)

// nativeValue 表示 VM 原生求值中的值，支持 bool / int / float / string。
type nativeValue struct {
	kind string // "bool" | "int" | "float" | "string"
	raw  string
	b    bool
	i    int64
	f    float64
}

// parseNativeValue 将 Redis 中存储的字符串解析为类型化值。
// 解析顺序: bool → int → float → string (兜底)
func parseNativeValue(raw string) nativeValue {
	v := nativeValue{raw: raw}
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "true":
		v.kind = "bool"
		v.b = true
		return v
	case "false":
		v.kind = "bool"
		v.b = false
		return v
	}
	if i, err := strconv.ParseInt(raw, 10, 64); err == nil {
		v.kind = "int"
		v.i = i
		return v
	}
	if f, err := strconv.ParseFloat(raw, 64); err == nil {
		v.kind = "float"
		v.f = f
		return v
	}
	v.kind = "string"
	return v
}

func (v nativeValue) String() string {
	switch v.kind {
	case "bool":
		if v.b {
			return "true"
		}
		return "false"
	case "int":
		return strconv.FormatInt(v.i, 10)
	case "float":
		// 避免多余的小数位
		s := strconv.FormatFloat(v.f, 'f', -1, 64)
		if !strings.Contains(s, ".") {
			s += ".0"
		}
		return s
	default:
		return v.raw
	}
}

func (v nativeValue) asFloat() float64 {
	switch v.kind {
	case "int":
		return float64(v.i)
	case "float":
		return v.f
	default:
		return 0
	}
}

func (v nativeValue) asInt() int64 {
	switch v.kind {
	case "int":
		return v.i
	case "float":
		return int64(v.f)
	default:
		return 0
	}
}

func (v nativeValue) asBool() bool {
	switch v.kind {
	case "bool":
		return v.b
	default:
		return v.raw != "" && v.raw != "0"
	}
}

// dispatchNative 直接求值基础类型运算指令，不经过 op-plat。
func dispatchNative(ctx context.Context, rdb *redis.Client, vtid string, pc string, inst *ir.Instruction) error {
	// 1. 读取输入
	inputs := make([]nativeValue, 0, len(inst.Reads))
	for _, r := range inst.Reads {
		var raw string
		if isRelative(r) {
			key := "/vthread/" + vtid + "/" + r[2:]
			val, err := rdb.Get(ctx, key).Result()
			if err != nil {
				msg := fmt.Sprintf("native read %s: %v", key, err)
				setError(ctx, rdb, vtid, pc, msg)
				return fmt.Errorf("%s", msg)
			}
			raw = val
		} else {
			raw = r
		}
		inputs = append(inputs, parseNativeValue(raw))
	}

	// 2. 求值
	result, err := evalNative(inst.Opcode, inputs)
	if err != nil {
		setError(ctx, rdb, vtid, pc, err.Error())
		return err
	}

	// 3. 写回输出
	if len(inst.Writes) > 0 {
		outKey := resolveWriteKey(vtid, inst.Writes[0])
		if err := rdb.Set(ctx, outKey, result.String(), 0).Err(); err != nil {
			msg := fmt.Sprintf("native write %s: %v", outKey, err)
			setError(ctx, rdb, vtid, pc, msg)
			return fmt.Errorf("%s", msg)
		}
	}

	log.Printf("[%s] NATIVE %s %v = %s", vtid, inst.Opcode, inputs, result.String())
	setState(ctx, rdb, vtid, ir.NextPC(pc), "running")
	return nil
}

func resolveWriteKey(vtid, param string) string {
	if isRelative(param) {
		return "/vthread/" + vtid + "/" + param[2:]
	}
	return param
}

// evalNative 执行符号算子的类型感知求值。
func evalNative(op string, inputs []nativeValue) (nativeValue, error) {
	switch op {
	// ── 算术 ──
	case "+":
		return evalBinaryArith(inputs, func(a, b float64) float64 { return a + b })
	case "-":
		if len(inputs) == 1 {
			v := inputs[0]
			return evalNeg(v)
		}
		return evalBinaryArith(inputs, func(a, b float64) float64 { return a - b })
	case "*":
		return evalBinaryArith(inputs, func(a, b float64) float64 { return a * b })
	case "/":
		return evalDiv(inputs)
	case "%":
		return evalMod(inputs)

	// ── 比较 ──
	case "==":
		return evalCmp(inputs, func(a, b float64) bool { return a == b },
			func(a, b string) bool { return a == b })
	case "!=":
		return evalCmp(inputs, func(a, b float64) bool { return a != b },
			func(a, b string) bool { return a != b })
	case "<":
		return evalCmpNum(inputs, func(a, b float64) bool { return a < b })
	case ">":
		return evalCmpNum(inputs, func(a, b float64) bool { return a > b })
	case "<=":
		return evalCmpNum(inputs, func(a, b float64) bool { return a <= b })
	case ">=":
		return evalCmpNum(inputs, func(a, b float64) bool { return a >= b })

	// ── 逻辑 ──
	case "&&":
		return evalLogic(inputs, func(a, b bool) bool { return a && b })
	case "||":
		return evalLogic(inputs, func(a, b bool) bool { return a || b })
	case "!":
		return evalNot(inputs)

	// ── 位运算 (int only) ──
	case "&":
		return evalBinaryInt(inputs, func(a, b int64) int64 { return a & b })
	case "|":
		return evalBinaryInt(inputs, func(a, b int64) int64 { return a | b })
	case "^":
		return evalBinaryInt(inputs, func(a, b int64) int64 { return a ^ b })
	case "<<":
		return evalBinaryInt(inputs, func(a, b int64) int64 { return a << uint64(b) })
	case ">>":
		return evalBinaryInt(inputs, func(a, b int64) int64 { return a >> uint64(b) })

	default:
		return nativeValue{}, fmt.Errorf("unknown native op: %s", op)
	}
}

// ── evaluator helpers ──

func requireBinary(inputs []nativeValue) error {
	if len(inputs) != 2 {
		return fmt.Errorf("binary op requires 2 inputs, got %d", len(inputs))
	}
	return nil
}

func requireUnary(inputs []nativeValue) error {
	if len(inputs) != 1 {
		return fmt.Errorf("unary op requires 1 input, got %d", len(inputs))
	}
	return nil
}

// Binary arithmetic: promote to float, compute, demote to int if both inputs int.
func evalBinaryArith(inputs []nativeValue, fn func(float64, float64) float64) (nativeValue, error) {
	if err := requireBinary(inputs); err != nil {
		return nativeValue{}, err
	}
	a, b := inputs[0], inputs[1]
	result := fn(a.asFloat(), b.asFloat())
	if a.kind == "int" && b.kind == "int" {
		return nativeValue{kind: "int", i: int64(result)}, nil
	}
	return nativeValue{kind: "float", f: result}, nil
}

func evalNeg(v nativeValue) (nativeValue, error) {
	switch v.kind {
	case "int":
		return nativeValue{kind: "int", i: -v.i}, nil
	case "float":
		return nativeValue{kind: "float", f: -v.f}, nil
	default:
		return nativeValue{}, fmt.Errorf("cannot negate %s", v.kind)
	}
}

func evalDiv(inputs []nativeValue) (nativeValue, error) {
	if err := requireBinary(inputs); err != nil {
		return nativeValue{}, err
	}
	a, b := inputs[0], inputs[1]
	bf := b.asFloat()
	if bf == 0 {
		return nativeValue{}, fmt.Errorf("division by zero")
	}
	result := a.asFloat() / bf
	return nativeValue{kind: "float", f: result}, nil
}

func evalMod(inputs []nativeValue) (nativeValue, error) {
	if err := requireBinary(inputs); err != nil {
		return nativeValue{}, err
	}
	a, b := inputs[0], inputs[1]
	if b.asInt() == 0 {
		return nativeValue{}, fmt.Errorf("modulo by zero")
	}
	return nativeValue{kind: "int", i: a.asInt() % b.asInt()}, nil
}

// evalCmp: 优先数值比较，回退字符串比较
func evalCmp(inputs []nativeValue, numCmp func(float64, float64) bool, strCmp func(string, string) bool) (nativeValue, error) {
	if err := requireBinary(inputs); err != nil {
		return nativeValue{}, err
	}
	a, b := inputs[0], inputs[1]
	// 两边都是数值类型 → 数值比较
	if (a.kind == "int" || a.kind == "float") && (b.kind == "int" || b.kind == "float") {
		return nativeValue{kind: "bool", b: numCmp(a.asFloat(), b.asFloat())}, nil
	}
	return nativeValue{kind: "bool", b: strCmp(a.raw, b.raw)}, nil
}

func evalCmpNum(inputs []nativeValue, fn func(float64, float64) bool) (nativeValue, error) {
	return evalCmp(inputs, fn, func(a, b string) bool { return a < b })
}

func evalLogic(inputs []nativeValue, fn func(bool, bool) bool) (nativeValue, error) {
	if err := requireBinary(inputs); err != nil {
		return nativeValue{}, err
	}
	a, b := inputs[0], inputs[1]
	return nativeValue{kind: "bool", b: fn(a.asBool(), b.asBool())}, nil
}

func evalNot(inputs []nativeValue) (nativeValue, error) {
	if err := requireUnary(inputs); err != nil {
		return nativeValue{}, err
	}
	return nativeValue{kind: "bool", b: !inputs[0].asBool()}, nil
}

func evalBinaryInt(inputs []nativeValue, fn func(int64, int64) int64) (nativeValue, error) {
	if err := requireBinary(inputs); err != nil {
		return nativeValue{}, err
	}
	return nativeValue{kind: "int", i: fn(inputs[0].asInt(), inputs[1].asInt())}, nil
}
