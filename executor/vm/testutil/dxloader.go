// Package testutil provides helpers for VM integration testing,
// including loading .dx function source files and registering them in Redis.
package testutil

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

// DxFunc represents a parsed dxlang function from a .dx file.
type DxFunc struct {
	Name      string   // e.g., "add_test"
	Signature string   // e.g., "def add_test(A:int, B:int) -> (C:int)"
	Body      []string // dxlang instruction lines
}

// TopLevelCall represents a function call at the file's outermost scope
// (outside any def { } block). When present, the loader writes /func/main
// to trigger VM execution.
type TopLevelCall struct {
	FuncName string   // e.g., "add_test"
	Args     []string // e.g., ["./a", "./b"]
	Outputs  []string // e.g., ["./c"]
}

// DxFile represents a fully parsed .dx file with function definitions
// and optional top-level call expressions.
type DxFile struct {
	Funcs         []DxFunc
	TopLevelCalls []TopLevelCall
}

// LoadDxFile reads a .dx file and returns the first parsed function.
// Deprecated: Use ParseDxFile for multi-function files with top-level call support.
func LoadDxFile(path string) (*DxFunc, error) {
	df, err := ParseDxFile(path)
	if err != nil {
		return nil, err
	}
	if len(df.Funcs) == 0 {
		return nil, fmt.Errorf("%s: no function definitions found", path)
	}
	return &df.Funcs[0], nil
}

// ParseDxFile reads a .dx file and returns all function definitions and
// any top-level call expressions.
//
// File format:
//
//	# comment lines (ignored)
//	def funcName(param1:type, ...) -> (out1:type, ...) {
//	    instruction1
//	    instruction2
//	}
//	# more def blocks ...
//
//	# optional top-level calls (outside any def { } block):
//	funcName(arg1, arg2) -> "./output"
//
// Top-level calls trigger automatic vthread creation via /func/main.
func ParseDxFile(path string) (*DxFile, error) {
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
		return nil, fmt.Errorf("%s: file is empty (no content lines)", path)
	}

	df := &DxFile{}

	// ── Pass 1: Parse all def blocks ──
	i := 0
	for i < len(lines) {
		if !strings.HasPrefix(lines[i], "def ") {
			i++
			continue
		}

		defLine := lines[i]
		name := extractFuncName(defLine)
		if name == "" {
			return nil, fmt.Errorf("%s: cannot extract function name from: %s", path, defLine)
		}

		// Extract body
		var body []string
		bodyEnd := len(lines) // default: rest of file

		if strings.HasSuffix(defLine, "{") {
			// Braced format: find matching '}'
			for j := i + 1; j < len(lines); j++ {
				if lines[j] == "}" {
					bodyEnd = j
					break
				}
				body = append(body, lines[j])
			}
			if bodyEnd == len(lines) {
				return nil, fmt.Errorf("%s: unclosed brace in function %s", path, name)
			}
			i = bodyEnd + 1 // skip past '}'
		} else {
			// No-brace format: scan until next 'def' or top-level call
			for j := i + 1; j < len(lines); j++ {
				if strings.HasPrefix(lines[j], "def ") {
					bodyEnd = j
					break
				}
				// Stop at top-level call expressions (detected by `->` outside braces)
				if isTopLevelCall(lines[j]) {
					bodyEnd = j
					break
				}
				body = append(body, lines[j])
			}
			if bodyEnd == len(lines) {
				// consumed rest of file
				i = len(lines)
			} else {
				i = bodyEnd
			}
		}

		if len(body) == 0 {
			return nil, fmt.Errorf("%s: function %s body is empty", path, name)
		}

		df.Funcs = append(df.Funcs, DxFunc{
			Name:      name,
			Signature: strings.TrimSuffix(defLine, " {"),
			Body:      body,
		})
	}

	if len(df.Funcs) == 0 {
		return nil, fmt.Errorf("%s: no 'def' function definition found", path)
	}

	// ── Pass 2: Parse top-level calls (lines not consumed by def blocks) ──
	for _, line := range lines {
		// Skip lines that are part of def blocks (they start with def or are inside braces)
		if strings.HasPrefix(line, "def ") || line == "}" {
			continue
		}
		// Check if this line is inside a known function body
		inBody := false
		for _, fn := range df.Funcs {
			for _, bodyLine := range fn.Body {
				if bodyLine == line {
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

		// Try to parse as top-level call
		if tc, ok := parseTopLevelCall(line); ok {
			df.TopLevelCalls = append(df.TopLevelCalls, tc)
		}
	}

	return df, nil
}

// parseTopLevelCall attempts to parse a top-level call expression like:
//
//	funcName(arg1, arg2) -> "./output"
//	funcName() -> ("./a", "./b")
//
// Returns the parsed call and true if successful.
func parseTopLevelCall(line string) (TopLevelCall, bool) {
	// Must contain '->' (call operator)
	arrowIdx := strings.Index(line, "->")
	if arrowIdx < 0 {
		return TopLevelCall{}, false
	}

	left := strings.TrimSpace(line[:arrowIdx])
	right := strings.TrimSpace(line[arrowIdx+2:])

	// Left side: funcName(args)
	parenOpen := strings.Index(left, "(")
	if parenOpen < 0 {
		return TopLevelCall{}, false
	}
	parenClose := strings.LastIndex(left, ")")
	if parenClose < 0 || parenClose <= parenOpen {
		return TopLevelCall{}, false
	}

	funcName := strings.TrimSpace(left[:parenOpen])
	if funcName == "" {
		return TopLevelCall{}, false
	}

	// Parse args
	argsStr := strings.TrimSpace(left[parenOpen+1 : parenClose])
	var args []string
	if argsStr != "" {
		for _, a := range strings.Split(argsStr, ",") {
			a = strings.TrimSpace(a)
			if a != "" {
				args = append(args, a)
			}
		}
	}

	// Parse outputs (right side)
	var outputs []string
	right = strings.Trim(right, "()")
	if right != "" {
		for _, o := range strings.Split(right, ",") {
			o = strings.TrimSpace(o)
			// Strip quotes if present
			o = strings.Trim(o, `"`)
			if o != "" {
				outputs = append(outputs, o)
			}
		}
	}

	return TopLevelCall{
		FuncName: funcName,
		Args:     args,
		Outputs:  outputs,
	}, true
}

// isTopLevelCall checks if a line looks like a function call expression
// (contains '->' and is not a def line or brace).
func isTopLevelCall(line string) bool {
	if strings.HasPrefix(line, "def ") || line == "}" || line == "{" {
		return false
	}
	return strings.Contains(line, "->")
}

// extractFuncName extracts the function name from a def line like
// "def add_test(A:int, B:int) -> (C:int) {" or legacy "(add_test(A, B) -> (C))".
func extractFuncName(sig string) string {
	sig = strings.TrimSpace(sig)
	// Strip "def " prefix
	if strings.HasPrefix(sig, "def ") {
		sig = strings.TrimSpace(sig[4:])
	}
	// Strip outer parens (legacy format)
	if len(sig) >= 2 && sig[0] == '(' && sig[len(sig)-1] == ')' {
		sig = sig[1 : len(sig)-1]
	}
	// Strip trailing " {" (braced def)
	sig = strings.TrimSuffix(sig, " {")
	sig = strings.TrimSpace(sig)

	// Isolate left side of "->"
	left := sig
	if idx := strings.Index(sig, "->"); idx >= 0 {
		left = strings.TrimSpace(sig[:idx])
	}

	// "add_test(A, B)" → "add_test"
	if idx := strings.Index(left, "("); idx >= 0 {
		return strings.TrimSpace(left[:idx])
	}
	return left
}

// RegisterFunc registers a DxFunc in Redis at /src/func/<name>.
func (f *DxFunc) RegisterFunc(ctx context.Context, rdb *redis.Client) error {
	if err := rdb.Set(ctx, "/src/func/"+f.Name, f.Signature, 0).Err(); err != nil {
		return fmt.Errorf("register sig: %w", err)
	}
	for i, line := range f.Body {
		key := fmt.Sprintf("/src/func/%s/%d", f.Name, i)
		if err := rdb.Set(ctx, key, line, 0).Err(); err != nil {
			return fmt.Errorf("register body[%d]: %w", i, err)
		}
	}
	return nil
}

// VThreadState mirrors state.VThreadState for test usage.
type VThreadState struct {
	PC     string `json:"pc"`
	Status string `json:"status"`
	Mode   string `json:"mode,omitempty"`
}

// CreateVThread creates a new vthread with initial state and entry instruction.
// The entry instruction uses the function name as the opcode directly:
//
//	opcode = funcName (e.g., "add_test"), reads = args, writes = outputs
//
// The engine detects function names at runtime via isFunctionCall() and
// converts them to internal CALL instructions automatically.
func CreateVThread(ctx context.Context, rdb *redis.Client, funcName string, reads, writes []string) (string, error) {
	vtid := fmt.Sprintf("test-%d", time.Now().UnixNano())

	st := VThreadState{PC: "[0,0]", Status: "init", Mode: "single"}
	data, _ := json.Marshal(st)
	if err := rdb.Set(ctx, "/vthread/"+vtid, data, 0).Err(); err != nil {
		return "", fmt.Errorf("set state: %w", err)
	}

	pipe := rdb.Pipeline()
	// 直接使用函数名作为 opcode (不再使用 "call" 关键字)
	pipe.Set(ctx, "/vthread/"+vtid+"/[0,0]", funcName, 0)
	for i, r := range reads {
		pipe.Set(ctx, fmt.Sprintf("/vthread/%s/[0,-%d]", vtid, i+1), r, 0)
	}
	for i, w := range writes {
		pipe.Set(ctx, fmt.Sprintf("/vthread/%s/[0,%d]", vtid, i+1), w, 0)
	}
	if _, err := pipe.Exec(ctx); err != nil {
		return "", fmt.Errorf("pipeline: %w", err)
	}

	return vtid, nil
}
