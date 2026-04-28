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

// LoadDxFile reads a .dx file and returns the parsed function.
//
// File format:
//
//	# comment lines (ignored)
//	def funcName(param1:type, ...) -> (out1:type, ...) {
//	    instruction1
//	    instruction2
//	}
//
// The first non-empty, non-comment line must start with 'def'.
// Body lines are inside braces '{' ... '}'.
func LoadDxFile(path string) (*DxFunc, error) {
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

	if len(lines) < 2 {
		return nil, fmt.Errorf("%s: need at least def line + 1 body instruction, got %d lines", path, len(lines))
	}

	// Parse def line
	defLine := lines[0]
	if !strings.HasPrefix(defLine, "def ") {
		return nil, fmt.Errorf("%s: first content line must start with 'def', got: %s", path, defLine)
	}

	name := extractFuncName(defLine)
	if name == "" {
		return nil, fmt.Errorf("%s: cannot extract function name from: %s", path, defLine)
	}

	// Extract body — inside { } or remaining lines
	var body []string
	if strings.HasSuffix(defLine, "{") {
		// Braced format: body between { and }
		for i := 1; i < len(lines); i++ {
			if lines[i] == "}" {
				break
			}
			body = append(body, lines[i])
		}
	} else {
		// No-brace format: remaining lines are body
		body = lines[1:]
	}

	if len(body) == 0 {
		return nil, fmt.Errorf("%s: function body is empty", path)
	}

	return &DxFunc{
		Name:      name,
		Signature: strings.TrimSuffix(defLine, " {"),
		Body:      body,
	}, nil
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
