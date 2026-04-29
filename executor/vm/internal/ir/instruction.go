package ir

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/redis/go-redis/v9"
)

// Instruction 表示执行层 [addr0, addr1] 解码后的一条指令
type Instruction struct {
	Opcode string   // [addr0, 0] = "+" | "call" | "return" | ...
	Reads  []string // [addr0, -1], [addr0, -2], ...
	Writes []string // [addr0, 1], [addr0, 2], ...
	PC     string   // 当前指令坐标, e.g., "[3,0]" 或 "[2,0]/[1,0]"
}

const maxParams = 10

// Decode 从 Redis 执行层 key 解码指令
func Decode(ctx context.Context, rdb *redis.Client, vtid string, pc string) (*Instruction, error) {
	prefix, addr0 := parsePC(pc)
	keyBase := fmt.Sprintf("/vthread/%s/%s", vtid, prefix)

	keys := make([]string, 0, 1+maxParams*2)
	keys = append(keys, fmt.Sprintf("%s[%d,0]", keyBase, addr0))
	for i := 1; i <= maxParams; i++ {
		keys = append(keys, fmt.Sprintf("%s[%d,-%d]", keyBase, addr0, i))
		keys = append(keys, fmt.Sprintf("%s[%d,%d]", keyBase, addr0, i))
	}

	vals, err := rdb.MGet(ctx, keys...).Result()
	if err != nil {
		return nil, fmt.Errorf("decode MGET: %w", err)
	}

	inst := &Instruction{PC: pc}

	if s, ok := vals[0].(string); ok {
		inst.Opcode = s
	}

	for i := 1; i <= maxParams; i++ {
		readIdx := (i-1)*2 + 1
		writeIdx := readIdx + 1
		if readIdx < len(vals) {
			if s, ok := vals[readIdx].(string); ok && s != "" {
				inst.Reads = append(inst.Reads, s)
			}
		}
		if writeIdx < len(vals) {
			if s, ok := vals[writeIdx].(string); ok && s != "" {
				inst.Writes = append(inst.Writes, s)
			}
		}
	}

	return inst, nil
}

// DecodeFromCache 从本地缓存 map 解码 (子栈场景, 零 Redis 访问)
func DecodeFromCache(cache map[string]string, pc string) *Instruction {
	_, addr0 := parsePC(pc)
	inst := &Instruction{PC: pc}
	inst.Opcode = cache[fmt.Sprintf("[%d,0]", addr0)]

	for i := 1; i <= maxParams; i++ {
		key := fmt.Sprintf("[%d,-%d]", addr0, i)
		if v, ok := cache[key]; ok && v != "" {
			inst.Reads = append(inst.Reads, v)
		}
		key = fmt.Sprintf("[%d,%d]", addr0, i)
		if v, ok := cache[key]; ok && v != "" {
			inst.Writes = append(inst.Writes, v)
		}
	}
	return inst
}

func parsePC(pc string) (prefix string, addr0 int) {
	idx := strings.LastIndex(pc, "/")
	if idx >= 0 {
		prefix = pc[:idx+1]
		addr0 = extractAddr0(pc[idx+1:])
	} else {
		addr0 = extractAddr0(pc)
	}
	return
}

func IsComputeOp(opcode string) bool {
	return !isLifecycleOrControl(opcode)
}

func IsLifecycleOp(opcode string) bool {
	return opcode == "newtensor" || opcode == "deltensor" || opcode == "clonetensor"
}

func IsControlOp(opcode string) bool {
	switch opcode {
	case "call", "return", "if", "for":
		return true
	}
	return false
}

func isLifecycleOrControl(opcode string) bool {
	switch opcode {
	case "call", "return", "if", "for",
		"newtensor", "deltensor", "clonetensor":
		return true
	}
	return false
}

func NextPC(pc string) string {
	parts := strings.Split(pc, "/")
	last := parts[len(parts)-1]
	num := extractAddr0(last)
	parts[len(parts)-1] = fmt.Sprintf("[%d,0]", num+1)
	return strings.Join(parts, "/")
}

func ParentPC(pc string) string {
	idx := strings.LastIndex(pc, "/")
	if idx < 0 {
		return pc
	}
	return NextPC(pc[:idx])
}

func extractAddr0(coord string) int {
	s := strings.Trim(coord, "[]")
	parts := strings.Split(s, ",")
	if len(parts) > 0 {
		n, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil {
			return 0
		}
		return n
	}
	return 0
}

// ParseDxlang 解析 dxlang 指令字符串为 Instruction。
//
// 支持三种赋值风格:
//
//	前缀 (命名函数):  add(A, B) -> ./C
//	中缀 (符号算子):  A + B -> ./C
//	                  !A -> ./C
//	C风格 (左箭头):   ./C <- A + B
//	                  ./C <- add(A, B)
func ParseDxlang(line string) (*Instruction, error) {
	line = strings.TrimSpace(line)
	if line == "" {
		return nil, fmt.Errorf("empty dxlang line")
	}

	inst := &Instruction{}

	// 1. 分离输出 (支持 -> 和 <- 两种箭头)
	var expr string
	if larrow := findArrow(line, "<-"); larrow >= 0 {
		// C风格: ./C <- A + B  → 输出在左, 表达式在右
		writesStr := strings.TrimSpace(line[:larrow])
		expr = strings.TrimSpace(line[larrow+2:])
		if strings.HasPrefix(writesStr, "(") && strings.HasSuffix(writesStr, ")") {
			writesStr = writesStr[1 : len(writesStr)-1]
		}
		inst.Writes = parseParamList(writesStr)
	} else if arrow := strings.Index(line, "->"); arrow >= 0 {
		// 传统风格: add(A, B) -> ./C  → 表达式在左, 输出在右
		expr = strings.TrimSpace(line[:arrow])
		writesStr := strings.TrimSpace(line[arrow+2:])
		if strings.HasPrefix(writesStr, "(") && strings.HasSuffix(writesStr, ")") {
			writesStr = writesStr[1 : len(writesStr)-1]
		}
		inst.Writes = parseParamList(writesStr)
	} else {
		expr = line
	}

	// 2. 尝试中缀解析: "A + B", "!A", "A == B"
	if op, left, right, ok := parseInfix(expr); ok {
		inst.Opcode = op
		if left != "" {
			inst.Reads = append(inst.Reads, left)
		}
		if right != "" {
			inst.Reads = append(inst.Reads, right)
		}
		return inst, nil
	}

	// 3. 回退到前缀解析: "add(A, B)"
	if idx := strings.Index(expr, "("); idx >= 0 {
		inst.Opcode = strings.TrimSpace(expr[:idx])
		rest := expr[idx+1:]

		parenDepth := 1
		closeIdx := -1
		for i, c := range rest {
			if c == '(' {
				parenDepth++
			} else if c == ')' {
				parenDepth--
				if parenDepth == 0 {
					closeIdx = i
					break
				}
			}
		}
		if closeIdx < 0 {
			return nil, fmt.Errorf("unmatched paren in: %s", line)
		}

		inst.Reads = parseParamList(rest[:closeIdx])
	}

	return inst, nil
}

// findArrow 查找左箭头 <- (区别于 <= 和 <<)。
// 返回 <- 中 < 的位置, 未找到返回 -1。
func findArrow(s, arrow string) int {
	for i := 0; i < len(s)-1; i++ {
		if s[i] == arrow[0] {
			// 排除 <=, <<, <>, < 后不是 - 的情况
			if s[i+1] == arrow[1] {
				return i
			}
		}
	}
	return -1
}

// parseInfix 尝试中缀表达式解析。支持二元和单目符号算子。
// 如果表达式含 '(' 则跳过 (回退到前缀解析)。
func parseInfix(expr string) (op, left, right string, ok bool) {
	expr = strings.TrimSpace(expr)
	if expr == "" {
		return
	}

	// 含 '(' → 前缀格式 (如 add(A, B))，跳过中缀避免 / 在 ./a 中被误判
	if strings.IndexByte(expr, '(') >= 0 {
		return
	}

	// 多字符算子 (先匹配长的避免子串误匹配)
	multiOps := []string{"==", "!=", "<=", ">=", "&&", "||", "<<", ">>"}
	for _, op := range multiOps {
		if idx := strings.Index(expr, op); idx > 0 {
			return op, strings.TrimSpace(expr[:idx]), strings.TrimSpace(expr[idx+len(op):]), true
		}
	}

	// 单字符二元算子
	singleOps := []string{"+", "*", "/", "%", "<", ">", "&", "|", "^"}
	for _, op := range singleOps {
		if idx := strings.Index(expr, op); idx > 0 {
			return op, strings.TrimSpace(expr[:idx]), strings.TrimSpace(expr[idx+1:]), true
		}
	}

	// '-' 二元 (idx>0, 不是 unary)
	if idx := strings.Index(expr, "-"); idx > 0 {
		return "-", strings.TrimSpace(expr[:idx]), strings.TrimSpace(expr[idx+1:]), true
	}

	// 单目算子 (位置 0)
	if len(expr) > 0 {
		if expr[0] == '!' {
			return "!", strings.TrimSpace(expr[1:]), "", true
		}
		if expr[0] == '-' {
			return "-", strings.TrimSpace(expr[1:]), "", true
		}
	}

	return
}

func parseParamList(s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	// Bracket-aware split: respect nested [], (), {}
	var params []string
	depth := 0
	start := 0
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '[', '(', '{':
			depth++
		case ']', ')', '}':
			if depth > 0 {
				depth--
			}
		case ',':
			if depth == 0 {
				p := strings.TrimSpace(s[start:i])
				if p != "" {
					params = append(params, stripQuotes(p))
				}
				start = i + 1
			}
		}
	}
	// Last param
	if start < len(s) {
		p := strings.TrimSpace(s[start:])
		if p != "" {
			params = append(params, stripQuotes(p))
		}
	}
	return params
}

// stripQuotes removes surrounding double quotes.
func stripQuotes(s string) string {
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
		return s[1 : len(s)-1]
	}
	return s
}
