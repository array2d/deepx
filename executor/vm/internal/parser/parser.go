// Package parser 提供 dxlang 语法分析：Token 流 → AST。
//
// 两个入口:
//   - ParseFile(path) → *ast.File      (文件级)
//   - ParseLine(line) → *ast.Instruction (行级, 原 ir.ParseDxlang)
package parser

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"deepx/executor/vm/internal/ast"
)

// ── 文件级解析 ──

// ParseFile 解析 .dx 源文件，返回所有函数定义和顶层调用。
func ParseFile(path string) (*ast.File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	var lines []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		lines = append(lines, line)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(lines) == 0 {
		return nil, fmt.Errorf("%s: empty", path)
	}

	df := &ast.File{}
	i := 0
	for i < len(lines) {
		if !strings.HasPrefix(lines[i], "def ") {
			i++
			continue
		}
		defLine := lines[i]
		name := extractFuncName(defLine)
		if name == "" {
			return nil, fmt.Errorf("%s: cannot extract name from: %s", path, defLine)
		}

		var body []string
		bodyEnd := len(lines)
		if strings.HasSuffix(defLine, "{") {
			for j := i + 1; j < len(lines); j++ {
				if lines[j] == "}" {
					bodyEnd = j
					break
				}
				body = append(body, lines[j])
			}
			if bodyEnd == len(lines) {
				return nil, fmt.Errorf("%s: unclosed brace in %s", path, name)
			}
			i = bodyEnd + 1
		} else {
			for j := i + 1; j < len(lines); j++ {
				if strings.HasPrefix(lines[j], "def ") {
					bodyEnd = j
					break
				}
				if looksLikeCall(lines[j]) {
					bodyEnd = j
					break
				}
				body = append(body, lines[j])
			}
			i = bodyEnd
		}
		if len(body) == 0 {
			return nil, fmt.Errorf("%s: empty body for %s", path, name)
		}
		df.Funcs = append(df.Funcs, ast.Func{
			Name:      name,
			Signature: strings.TrimSuffix(defLine, " {"),
			Body:      body,
		})
	}
	if len(df.Funcs) == 0 {
		return nil, fmt.Errorf("%s: no 'def' found", path)
	}

	// 顶层调用
	for _, line := range lines {
		if strings.HasPrefix(line, "def ") || line == "}" {
			continue
		}
		inBody := false
		for _, fn := range df.Funcs {
			for _, bl := range fn.Body {
				if bl == line {
					inBody = true
					break
				}
			}
			if inBody {
				break
			}
		}
		if inBody {
			continue
		}
		if tc, ok := parseTopLevelCall(line); ok {
			df.TopLevelCalls = append(df.TopLevelCalls, tc)
		}
	}
	return df, nil
}

// ── 行级解析 (原 ir.ParseDxlang) ──

// ParseLine 解析单条 dxlang 指令字符串。
//
// 支持三种赋值风格:
//
//	前缀 (命名函数):  add(A, B) -> ./C
//	中缀 (符号算子):  A + B -> ./C, !A -> ./C
//	C风格 (左箭头):   ./C <- A + B, ./C <- add(A, B)
func ParseLine(line string) (*ast.Instruction, error) {
	line = strings.TrimSpace(line)
	if line == "" {
		return nil, fmt.Errorf("empty dxlang line")
	}

	inst := &ast.Instruction{}

	// 1. 分离输出
	var expr string
	if larrow := findArrow(line, "<-"); larrow >= 0 {
		writesStr := strings.TrimSpace(line[:larrow])
		expr = strings.TrimSpace(line[larrow+2:])
		if strings.HasPrefix(writesStr, "(") && strings.HasSuffix(writesStr, ")") {
			writesStr = writesStr[1 : len(writesStr)-1]
		}
		if err := validateKeyRefs(writesStr, "write", line); err != nil {
			return nil, err
		}
		inst.Writes = parseParamList(writesStr)
	} else if arrow := strings.Index(line, "->"); arrow >= 0 {
		expr = strings.TrimSpace(line[:arrow])
		writesStr := strings.TrimSpace(line[arrow+2:])
		if strings.HasPrefix(writesStr, "(") && strings.HasSuffix(writesStr, ")") {
			writesStr = writesStr[1 : len(writesStr)-1]
		}
		if err := validateKeyRefs(writesStr, "write", line); err != nil {
			return nil, err
		}
		inst.Writes = parseParamList(writesStr)
	} else {
		expr = line
	}

	// 2. 中缀解析
	if op, left, right, ok := parseInfix(expr); ok {
		inst.Opcode = op
		if left != "" {
			if err := validateRef(left, line); err != nil {
				return nil, err
			}
			inst.Reads = append(inst.Reads, stripQuotes(left))
		}
		if right != "" {
			if err := validateRef(right, line); err != nil {
				return nil, err
			}
			inst.Reads = append(inst.Reads, stripQuotes(right))
		}
		return inst, nil
	}

	// 3. 前缀解析: add(A, B)
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
		readsStr := rest[:closeIdx]
		if err := validateKeyRefs(readsStr, "read", line); err != nil {
			return nil, err
		}
		inst.Reads = parseParamList(readsStr)
	}

	return inst, nil
}

// ── 签名解析 ──

// ParseSignature 解析函数签名，提取形参。
func ParseSignature(sig string) ast.FormalParams {
	var fp ast.FormalParams
	sig = strings.TrimSpace(sig)
	if strings.HasPrefix(sig, "def ") {
		sig = strings.TrimSpace(sig[4:])
	}
	if len(sig) >= 2 && sig[0] == '(' && sig[len(sig)-1] == ')' {
		sig = sig[1 : len(sig)-1]
	}
	arrow := strings.Index(sig, "->")
	if arrow < 0 {
		return fp
	}
	left := strings.TrimSpace(sig[:arrow])
	right := strings.TrimSpace(sig[arrow+2:])
	if lp := strings.Index(left, "("); lp >= 0 {
		rp := strings.LastIndex(left, ")")
		if rp > lp {
			fp.Reads = extractParamNames(left[lp+1 : rp])
		}
	}
	right = strings.TrimSpace(right)
	if len(right) >= 2 && right[0] == '(' && right[len(right)-1] == ')' {
		fp.Writes = extractParamNames(right[1 : len(right)-1])
	} else {
		fp.Writes = extractParamNames(right)
	}
	return fp
}

// ── helpers ──

func extractFuncName(sig string) string {
	sig = strings.TrimSpace(sig)
	if strings.HasPrefix(sig, "def ") {
		sig = strings.TrimSpace(sig[4:])
	}
	if len(sig) >= 2 && sig[0] == '(' && sig[len(sig)-1] == ')' {
		sig = sig[1 : len(sig)-1]
	}
	sig = strings.TrimSuffix(sig, " {")
	sig = strings.TrimSpace(sig)
	left := sig
	if idx := strings.Index(sig, "->"); idx >= 0 {
		left = strings.TrimSpace(sig[:idx])
	}
	if idx := strings.Index(left, "("); idx >= 0 {
		return strings.TrimSpace(left[:idx])
	}
	return left
}

func looksLikeCall(line string) bool {
	if strings.HasPrefix(line, "def ") || line == "}" || line == "{" {
		return false
	}
	return strings.Contains(line, "->")
}

func parseTopLevelCall(line string) (ast.TopLevelCall, bool) {
	arrowIdx := strings.Index(line, "->")
	if arrowIdx < 0 {
		return ast.TopLevelCall{}, false
	}
	left := strings.TrimSpace(line[:arrowIdx])
	right := strings.TrimSpace(line[arrowIdx+2:])
	open := strings.Index(left, "(")
	if open < 0 {
		return ast.TopLevelCall{}, false
	}
	close := strings.LastIndex(left, ")")
	if close < 0 || close <= open {
		return ast.TopLevelCall{}, false
	}
	funcName := strings.TrimSpace(left[:open])
	if funcName == "" {
		return ast.TopLevelCall{}, false
	}
	var args []string
	s := strings.TrimSpace(left[open+1 : close])
	if s != "" {
		for _, a := range strings.Split(s, ",") {
			a = strings.TrimSpace(a)
			if a != "" {
				args = append(args, a)
			}
		}
	}
	var outputs []string
	right = strings.Trim(right, "()")
	if right != "" {
		for _, o := range strings.Split(right, ",") {
			o = strings.TrimSpace(o)
			o = strings.Trim(o, `"`)
			if o != "" {
				outputs = append(outputs, o)
			}
		}
	}
	return ast.TopLevelCall{FuncName: funcName, Args: args, Outputs: outputs}, true
}

func findArrow(s, arrow string) int {
	for i := 0; i < len(s)-1; i++ {
		if s[i] == arrow[0] && s[i+1] == arrow[1] {
			return i
		}
	}
	return -1
}

func parseInfix(expr string) (op, left, right string, ok bool) {
	expr = strings.TrimSpace(expr)
	if expr == "" {
		return
	}
	if strings.IndexByte(expr, '(') >= 0 {
		return // 含括号 → 前缀格式
	}
	multiOps := []string{"==", "!=", "<=", ">=", "&&", "||", "<<", ">>"}
	for _, o := range multiOps {
		if idx := strings.Index(expr, o); idx > 0 {
			return o, strings.TrimSpace(expr[:idx]), strings.TrimSpace(expr[idx+len(o):]), true
		}
	}
	singleOps := []string{"+", "*", "/", "%", "<", ">", "&", "|", "^"}
	for _, o := range singleOps {
		if idx := strings.Index(expr, o); idx > 0 {
			return o, strings.TrimSpace(expr[:idx]), strings.TrimSpace(expr[idx+1:]), true
		}
	}
	if idx := strings.Index(expr, "-"); idx > 0 {
		return "-", strings.TrimSpace(expr[:idx]), strings.TrimSpace(expr[idx+1:]), true
	}
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
	var params []string
	depth := 0
	inQuote := false
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '"' {
			inQuote = !inQuote
			continue
		}
		if inQuote {
			continue
		}
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
	if start < len(s) {
		p := strings.TrimSpace(s[start:])
		if p != "" {
			params = append(params, stripQuotes(p))
		}
	}
	return params
}

func parseParamListRaw(s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	var params []string
	depth := 0
	inQuote := false
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '"' {
			inQuote = !inQuote
			continue
		}
		if inQuote {
			continue
		}
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
					params = append(params, p)
				}
				start = i + 1
			}
		}
	}
	if start < len(s) {
		p := strings.TrimSpace(s[start:])
		if p != "" {
			params = append(params, p)
		}
	}
	return params
}

func stripQuotes(s string) string {
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
		return s[1 : len(s)-1]
	}
	return s
}

func validateRef(raw string, line string) error {
	unquoted := stripQuotes(raw)
	if (strings.HasPrefix(unquoted, "/") || strings.HasPrefix(unquoted, "./")) &&
		!(len(raw) >= 2 && raw[0] == '"' && raw[len(raw)-1] == '"') {
		return fmt.Errorf("path %q must be quoted (e.g. %q) in: %s", unquoted, `"`+unquoted+`"`, line)
	}
	return nil
}

func validateKeyRefs(rawExpr string, role string, line string) error {
	for _, raw := range parseParamListRaw(rawExpr) {
		unquoted := stripQuotes(raw)
		if (strings.HasPrefix(unquoted, "/") || strings.HasPrefix(unquoted, "./")) &&
			!(len(raw) >= 2 && raw[0] == '"' && raw[len(raw)-1] == '"') {
			return fmt.Errorf("%s %q must be quoted (e.g. %s) in: %s", role, unquoted, `"`+unquoted+`"`, line)
		}
	}
	return nil
}

func extractParamNames(s string) []string {
	var names []string
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if colon := strings.Index(p, ":"); colon >= 0 {
			p = p[:colon]
		}
		p = strings.TrimSpace(p)
		if p != "" {
			names = append(names, p)
		}
	}
	return names
}
