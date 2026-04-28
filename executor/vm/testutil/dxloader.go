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
	Signature string   // e.g., "(add_test(A, B) -> (C))"
	Body      []string // dxlang instruction lines
}

// LoadDxFile reads a .dx file and returns the parsed function.
//
// File format:
//
//	# comment lines (ignored)
//	(funcName(param1:type, ...) -> (out1:type, ...))
//	instruction1
//	instruction2
//
// The first non-empty, non-comment line is the signature.
// All subsequent non-empty, non-comment lines form the body.
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
		return nil, fmt.Errorf("%s: need at least signature + 1 body instruction, got %d lines", path, len(lines))
	}

	sig := lines[0]
	body := lines[1:]

	name := extractFuncName(sig)
	if name == "" {
		return nil, fmt.Errorf("%s: cannot extract function name from signature: %s", path, sig)
	}

	return &DxFunc{
		Name:      name,
		Signature: sig,
		Body:      body,
	}, nil
}

// extractFuncName extracts the function name from a signature like
// "(add_test(A:tensor, B:tensor) -> (C:tensor))" or "add_test -> (C)".
func extractFuncName(sig string) string {
	sig = strings.TrimSpace(sig)
	// strip outer parens
	if len(sig) >= 2 && sig[0] == '(' && sig[len(sig)-1] == ')' {
		sig = sig[1 : len(sig)-1]
	}
	sig = strings.TrimSpace(sig)

	// isolate left side of "->"
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

// RegisterFunc 将 DxFunc 注册到 Redis 的 /src/func/<name> 路径。
// 签名写入 key="/src/func/<name>"，每条 body 指令写入 "/src/func/<name>/0", "/src/func/<name>/1", ...
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

// VThreadState mirrors engine.VThreadState for test usage.
type VThreadState struct {
	PC     string `json:"pc"`
	Status string `json:"status"`
	Mode   string `json:"mode,omitempty"`
}

// CreateVThread 创建一个新的 vthread，写入初始状态和入口 CALL 指令。
//
// 参数:
//   - funcName: 被调函数名
//   - reads:    实参列表 (不含函数名), 如 ["./a", "./b"]
//   - writes:   输出列表, 如 ["./c"]
//
// 返回生成的 vtid。
func CreateVThread(ctx context.Context, rdb *redis.Client, funcName string, reads, writes []string) (string, error) {
	vtid := fmt.Sprintf("test-%d", time.Now().UnixNano())

	// 构建 CALL 指令
	allReads := append([]string{funcName}, reads...)

	// 写入 vthread 状态
	state := VThreadState{PC: "[0,0]", Status: "init", Mode: "single"}
	data, _ := json.Marshal(state)
	if err := rdb.Set(ctx, "/vthread/"+vtid, data, 0).Err(); err != nil {
		return "", fmt.Errorf("set state: %w", err)
	}

	// 写入指令 slot
	pipe := rdb.Pipeline()
	pipe.Set(ctx, "/vthread/"+vtid+"/[0,0]", "call", 0)
	for i, r := range allReads {
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
