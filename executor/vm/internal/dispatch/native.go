package dispatch

import (
	"context"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"deepx/executor/vm/internal/ir"
	"deepx/executor/vm/internal/logx"
	"deepx/executor/vm/internal/state"
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

// Native 直接求值基础类型运算指令，不经过 op-plat。
func Native(ctx context.Context, rdb *redis.Client, vtid string, pc string, inst *ir.Instruction) error {
	inputs := make([]nativeValue, 0, len(inst.Reads))
	for _, r := range inst.Reads {
		var raw string
		if IsRelative(r) {
			key := "/vthread/" + vtid + "/" + r[2:]
			val, err := rdb.Get(ctx, key).Result()
			if err != nil {
				msg := fmt.Sprintf("native read %s: %v", key, err)
				state.SetError(ctx, rdb, vtid, pc, msg)
				return fmt.Errorf("%s", msg)
			}
			raw = val
		} else {
			raw = r
		}
		inputs = append(inputs, parseNativeValue(raw))
	}

	result, err := evalNative(inst.Opcode, inputs)
	if err != nil {
		state.SetError(ctx, rdb, vtid, pc, err.Error())
		return err
	}

	// print 输出到 io writer (文件、网络等)，由 /sys/term/default/stdout 指向
	if inst.Opcode == "print" {
		parts := make([]string, len(inputs))
		for i, v := range inputs {
			parts[i] = v.String()
		}
		line := strings.Join(parts, " ")
		logx.Debug("[%s] PRINT %s", vtid, line)

		// 解析 io writer 并写入
		if err := writeToStdout(ctx, rdb, vtid, line); err != nil {
			logx.Debug("[%s] PRINT write error: %v", vtid, err)
		}

		state.Set(ctx, rdb, vtid, ir.NextPC(pc), "running")
		return nil
	}

	if len(inst.Writes) > 0 {
		outKey := ResolveWriteKey(vtid, inst.Writes[0])
		if err := rdb.Set(ctx, outKey, result.String(), 0).Err(); err != nil {
			msg := fmt.Sprintf("native write %s: %v", outKey, err)
			state.SetError(ctx, rdb, vtid, pc, msg)
			return fmt.Errorf("%s", msg)
		}
	}

	logx.Debug("[%s] NATIVE %s %v = %s", vtid, inst.Opcode, inputs, result.String())
	state.Set(ctx, rdb, vtid, ir.NextPC(pc), "running")
	return nil
}

func evalNative(op string, inputs []nativeValue) (nativeValue, error) {
	switch op {
	case "+":
		return evalBinaryArith(inputs, func(a, b float64) float64 { return a + b })
	case "-":
		if len(inputs) == 1 {
			return evalNeg(inputs[0])
		}
		return evalBinaryArith(inputs, func(a, b float64) float64 { return a - b })
	case "*":
		return evalBinaryArith(inputs, func(a, b float64) float64 { return a * b })
	case "/":
		return evalDiv(inputs)
	case "%":
		return evalMod(inputs)
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
	case "&&":
		return evalLogic(inputs, func(a, b bool) bool { return a && b })
	case "||":
		return evalLogic(inputs, func(a, b bool) bool { return a || b })
	case "!":
		return evalNot(inputs)
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

	// ── 数学 built-in ──
	case "abs":
		return evalAbs(inputs)
	case "pow":
		return evalPow(inputs)
	case "min":
		return evalMin(inputs)
	case "max":
		return evalMax(inputs)
	case "sqrt":
		return evalSqrt(inputs)
	case "exp":
		return evalExp(inputs)
	case "log":
		return evalLog(inputs)
	case "neg":
		return evalUnaryArith(inputs, func(a float64) float64 { return -a })
	case "sign":
		return evalSign(inputs)

	// ── 类型转换 built-in ──
	case "int":
		return evalToInt(inputs)
	case "float":
		return evalToFloat(inputs)
	case "bool":
		return evalToBool(inputs)

	// ── 输出 built-in ──
	case "print":
		return evalPrint(inputs)

	default:
		return nativeValue{}, fmt.Errorf("unknown native op: %s", op)
	}
}

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

func evalCmp(inputs []nativeValue, numCmp func(float64, float64) bool, strCmp func(string, string) bool) (nativeValue, error) {
	if err := requireBinary(inputs); err != nil {
		return nativeValue{}, err
	}
	a, b := inputs[0], inputs[1]
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

// ── 数学 built-in evaluators ──

func evalAbs(inputs []nativeValue) (nativeValue, error) {
	if err := requireUnary(inputs); err != nil {
		return nativeValue{}, err
	}
	v := inputs[0]
	switch v.kind {
	case "int":
		if v.i < 0 {
			return nativeValue{kind: "int", i: -v.i}, nil
		}
		return nativeValue{kind: "int", i: v.i}, nil
	case "float":
		return nativeValue{kind: "float", f: math.Abs(v.f)}, nil
	default:
		return nativeValue{}, fmt.Errorf("abs requires numeric, got %s", v.kind)
	}
}

func evalPow(inputs []nativeValue) (nativeValue, error) {
	if err := requireBinary(inputs); err != nil {
		return nativeValue{}, err
	}
	result := math.Pow(inputs[0].asFloat(), inputs[1].asFloat())
	return nativeValue{kind: "float", f: result}, nil
}

func evalMin(inputs []nativeValue) (nativeValue, error) {
	if err := requireBinary(inputs); err != nil {
		return nativeValue{}, err
	}
	a, b := inputs[0], inputs[1]
	if (a.kind == "int" || a.kind == "float") && (b.kind == "int" || b.kind == "float") {
		af, bf := a.asFloat(), b.asFloat()
		result := math.Min(af, bf)
		if a.kind == "int" && b.kind == "int" {
			return nativeValue{kind: "int", i: int64(result)}, nil
		}
		return nativeValue{kind: "float", f: result}, nil
	}
	// 字符串回退
	if a.raw < b.raw {
		return a, nil
	}
	return b, nil
}

func evalMax(inputs []nativeValue) (nativeValue, error) {
	if err := requireBinary(inputs); err != nil {
		return nativeValue{}, err
	}
	a, b := inputs[0], inputs[1]
	if (a.kind == "int" || a.kind == "float") && (b.kind == "int" || b.kind == "float") {
		af, bf := a.asFloat(), b.asFloat()
		result := math.Max(af, bf)
		if a.kind == "int" && b.kind == "int" {
			return nativeValue{kind: "int", i: int64(result)}, nil
		}
		return nativeValue{kind: "float", f: result}, nil
	}
	if a.raw > b.raw {
		return a, nil
	}
	return b, nil
}

func evalSqrt(inputs []nativeValue) (nativeValue, error) {
	if err := requireUnary(inputs); err != nil {
		return nativeValue{}, err
	}
	x := inputs[0].asFloat()
	if x < 0 {
		return nativeValue{}, fmt.Errorf("sqrt of negative number: %v", x)
	}
	return nativeValue{kind: "float", f: math.Sqrt(x)}, nil
}

func evalExp(inputs []nativeValue) (nativeValue, error) {
	if err := requireUnary(inputs); err != nil {
		return nativeValue{}, err
	}
	return nativeValue{kind: "float", f: math.Exp(inputs[0].asFloat())}, nil
}

func evalLog(inputs []nativeValue) (nativeValue, error) {
	if err := requireUnary(inputs); err != nil {
		return nativeValue{}, err
	}
	x := inputs[0].asFloat()
	if x <= 0 {
		return nativeValue{}, fmt.Errorf("log of non-positive number: %v", x)
	}
	return nativeValue{kind: "float", f: math.Log(x)}, nil
}

func evalSign(inputs []nativeValue) (nativeValue, error) {
	if err := requireUnary(inputs); err != nil {
		return nativeValue{}, err
	}
	v := inputs[0]
	f := v.asFloat()
	if f > 0 {
		return nativeValue{kind: "int", i: 1}, nil
	} else if f < 0 {
		return nativeValue{kind: "int", i: -1}, nil
	}
	return nativeValue{kind: "int", i: 0}, nil
}

func evalUnaryArith(inputs []nativeValue, fn func(float64) float64) (nativeValue, error) {
	if err := requireUnary(inputs); err != nil {
		return nativeValue{}, err
	}
	v := inputs[0]
	result := fn(v.asFloat())
	if v.kind == "int" {
		return nativeValue{kind: "int", i: int64(result)}, nil
	}
	return nativeValue{kind: "float", f: result}, nil
}

// ── 类型转换 built-in evaluators ──

func evalToInt(inputs []nativeValue) (nativeValue, error) {
	if err := requireUnary(inputs); err != nil {
		return nativeValue{}, err
	}
	v := inputs[0]
	switch v.kind {
	case "int":
		return v, nil
	case "float":
		return nativeValue{kind: "int", i: int64(v.f)}, nil
	case "bool":
		if v.b {
			return nativeValue{kind: "int", i: 1}, nil
		}
		return nativeValue{kind: "int", i: 0}, nil
	default:
		return nativeValue{kind: "int", i: v.asInt()}, nil
	}
}

func evalToFloat(inputs []nativeValue) (nativeValue, error) {
	if err := requireUnary(inputs); err != nil {
		return nativeValue{}, err
	}
	v := inputs[0]
	switch v.kind {
	case "float":
		return v, nil
	case "int":
		return nativeValue{kind: "float", f: float64(v.i)}, nil
	case "bool":
		if v.b {
			return nativeValue{kind: "float", f: 1.0}, nil
		}
		return nativeValue{kind: "float", f: 0.0}, nil
	default:
		return nativeValue{kind: "float", f: v.asFloat()}, nil
	}
}

func evalToBool(inputs []nativeValue) (nativeValue, error) {
	if err := requireUnary(inputs); err != nil {
		return nativeValue{}, err
	}
	return nativeValue{kind: "bool", b: inputs[0].asBool()}, nil
}

// ── 输出 built-in evaluators ──

// evalPrint 打印所有输入的原生值到日志，返回空字符串。
// print 可以接受任意数量参数（0~N），无返回值。
func evalPrint(inputs []nativeValue) (nativeValue, error) {
	if len(inputs) == 0 {
		return nativeValue{kind: "string", raw: ""}, nil
	}
	parts := make([]string, len(inputs))
	for i, v := range inputs {
		parts[i] = v.String()
	}
	logx.Debug("PRINT %s", strings.Join(parts, " "))
	return nativeValue{kind: "string", raw: strings.Join(parts, " ")}, nil
}

// ── io writer: print 输出的目标 ──

// sysTermDefaultStdout 系统级默认 stdout writer 注册 key。
// 由启动时注册，VM 遇到 print 指令时读取该 key 获取 writer URI。
// Writer 类型: file://path 等。
const sysTermDefaultStdout = "/sys/term/default/stdout"

// stdoutWriterBase 默认 stdout writer 文件根目录。
const stdoutWriterBase = "/tmp/deepx-stdout"

// getStdoutWriter 获取系统注册的 stdout io writer URI。
// 如果 /sys/term/default/stdout 未注册，创建默认文件 writer 并注册。
func getStdoutWriter(ctx context.Context, rdb *redis.Client) (string, error) {
	uri, err := rdb.Get(ctx, sysTermDefaultStdout).Result()
	if err == nil && uri != "" {
		// 验证 URI 格式
		if strings.HasPrefix(uri, "file://") || strings.HasPrefix(uri, "tcp://") {
			return uri, nil
		}
		logx.Warn("invalid stdout writer URI at %s: %s, recreating", sysTermDefaultStdout, uri)
		rdb.Del(ctx, sysTermDefaultStdout)
	} else if err != nil && err != redis.Nil {
		// 类型错误 (如 WRONGTYPE) → 删除重建
		logx.Warn("stdout writer error at %s: %v, recreating", sysTermDefaultStdout, err)
		rdb.Del(ctx, sysTermDefaultStdout)
	}

	// 创建默认文件 writer (file 类型)
	os.MkdirAll(stdoutWriterBase, 0755)
	filePath := stdoutWriterBase + "/default.log"
	uri = "file://" + filePath

	os.WriteFile(filePath, nil, 0644)

	rdb.Set(ctx, sysTermDefaultStdout, uri, 0)
	logx.Debug("registered default stdout writer: %s", uri)

	return uri, nil
}

// writeToStdout 将一行输出写入系统 stdout io writer。
func writeToStdout(ctx context.Context, rdb *redis.Client, vtid string, line string) error {
	uri, err := getStdoutWriter(ctx, rdb)
	if err != nil {
		return fmt.Errorf("get stdout writer: %w", err)
	}

	// 解析 URI，目前支持 file://
	if !strings.HasPrefix(uri, "file://") {
		return fmt.Errorf("unsupported stdout writer: %s", uri)
	}
	filePath := strings.TrimPrefix(uri, "file://")

	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("open stdout file %s: %w", filePath, err)
	}
	defer f.Close()

	if _, err := fmt.Fprintln(f, line); err != nil {
		return fmt.Errorf("write stdout file %s: %w", filePath, err)
	}
	return nil
}
