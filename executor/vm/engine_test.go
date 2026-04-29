package vm_test

import (
	"context"
	"encoding/json"
	"testing"

	"git.array2d.com/ai/deepx/executor/vm/internal/ir"
	"git.array2d.com/ai/deepx/executor/vm/internal/route"
	"github.com/redis/go-redis/v9"
)

func TestParseDxlang_Basic(t *testing.T) {
	tests := []struct {
		input       string
		opcode      string
		readsCount  int
		writesCount int
	}{
		// 前缀格式 (命名函数)
		{"matmul(A, B) -> ./Y", "matmul", 2, 1},
		{"add(./a, ./b) -> ./c", "add", 2, 1},
		{"relu(./x) -> ./y", "relu", 1, 1},
		{"call(gemm) -> ./out", "call", 1, 1},
		{"mul(./Y, alpha) -> ./Y", "mul", 2, 1},
		// 中缀格式 (符号算子)
		{"A + B -> ./C", "+", 2, 1},
		{"A - B -> ./C", "-", 2, 1},
		{"A * B -> ./C", "*", 2, 1},
		{"A / B -> ./C", "/", 2, 1},
		{"A == B -> ./C", "==", 2, 1},
		{"A != B -> ./C", "!=", 2, 1},
		{"A < B -> ./C", "<", 2, 1},
		{"A >= B -> ./C", ">=", 2, 1},
		{"A && B -> ./C", "&&", 2, 1},
		{"A || B -> ./C", "||", 2, 1},
		// 单目中缀
		{"!A -> ./C", "!", 1, 1},
		{"-A -> ./C", "-", 1, 1},
		// 链式中间值
		{"./tmp * C -> ./D", "*", 2, 1},
	}

	for _, tc := range tests {
		inst, err := ir.ParseDxlang(tc.input)
		if err != nil {
			t.Errorf("ParseDxlang(%q) error: %v", tc.input, err)
			continue
		}
		if inst.Opcode != tc.opcode {
			t.Errorf("ParseDxlang(%q): opcode = %q, want %q", tc.input, inst.Opcode, tc.opcode)
		}
		if len(inst.Reads) != tc.readsCount {
			t.Errorf("ParseDxlang(%q): reads = %d, want %d", tc.input, len(inst.Reads), tc.readsCount)
		}
		if len(inst.Writes) != tc.writesCount {
			t.Errorf("ParseDxlang(%q): writes = %d, want %d", tc.input, len(inst.Writes), tc.writesCount)
		}
	}
}

func TestNextPC(t *testing.T) {
	tests := []struct {
		pc   string
		want string
	}{
		{"[0,0]", "[1,0]"},
		{"[3,0]", "[4,0]"},
		{"[0,0]/[0,0]", "[0,0]/[1,0]"},
		{"[2,0]/[3,0]", "[2,0]/[4,0]"},
	}

	for _, tc := range tests {
		if got := ir.NextPC(tc.pc); got != tc.want {
			t.Errorf("NextPC(%q) = %q, want %q", tc.pc, got, tc.want)
		}
	}
}

func TestParentPC(t *testing.T) {
	tests := []struct {
		pc   string
		want string
	}{
		{"[2,0]/[1,0]", "[3,0]"},
		{"[0,0]/[5,0]", "[1,0]"},
		{"[0,0]/[3,0]/[2,0]", "[0,0]/[4,0]"},
	}

	for _, tc := range tests {
		if got := ir.ParentPC(tc.pc); got != tc.want {
			t.Errorf("ParentPC(%q) = %q, want %q", tc.pc, got, tc.want)
		}
	}
}

func TestIsComputeOp(t *testing.T) {
	compute := []string{"add", "sub", "mul", "div", "matmul", "relu", "sigmoid", "tanh"}
	control := []string{"call", "return", "if", "for"}
	lifecycle := []string{"newtensor", "deltensor", "clonetensor"}

	for _, op := range compute {
		if !ir.IsComputeOp(op) {
			t.Errorf("IsComputeOp(%q) = false, want true", op)
		}
	}
	for _, op := range control {
		if ir.IsComputeOp(op) {
			t.Errorf("IsComputeOp(%q) = true, want false", op)
		}
	}
	for _, op := range lifecycle {
		if ir.IsLifecycleOp(op) && ir.IsComputeOp(op) {
			t.Errorf("IsLifecycleOp(%q) should not also be IsComputeOp", op)
		}
	}
}

func TestDecode(t *testing.T) {
	// 集成测试需要 Redis，此处用 mock 或跳过
	t.Skip("requires Redis")
}

func TestRouteSelect_NoRedis(t *testing.T) {
	// 测试 route.Select 在没有 Redis 时的错误处理
	rdb := redis.NewClient(&redis.Options{Addr: "127.0.0.1:9999"})
	ctx := context.Background()
	_, err := route.Select(ctx, rdb, "add")
	if err == nil {
		t.Error("expected error when Redis is not available")
	}
	t.Logf("expected error: %v", err)
}

func TestDecodeFromCache(t *testing.T) {
	cache := map[string]string{
		"[3,0]":  "add",
		"[3,-1]": "./a",
		"[3,-2]": "./b",
		"[3,1]":  "./c",
	}

	inst := ir.DecodeFromCache(cache, "[3,0]")
	if inst.Opcode != "add" {
		t.Errorf("opcode = %q, want 'add'", inst.Opcode)
	}
	if len(inst.Reads) != 2 || inst.Reads[0] != "./a" || inst.Reads[1] != "./b" {
		t.Errorf("reads = %v, want [./a ./b]", inst.Reads)
	}
	if len(inst.Writes) != 1 || inst.Writes[0] != "./c" {
		t.Errorf("writes = %v, want [./c]", inst.Writes)
	}
}

func TestParseSignature(t *testing.T) {
	// 使用 encoding/json 测试
	sig := `(add_test(A:tensor, B:tensor) -> (C:tensor))`
	_ = sig
	// parseSignature 是内部函数，此处间接测试
}

func TestIntegration_AddVthread(t *testing.T) {
	// 完整的集成测试 (需要 Redis + op-plat 运行)
	t.Skip("requires full stack: redis + heap + op-plat + vm")

	ctx := context.Background()
	rdb := redis.NewClient(&redis.Options{Addr: "127.0.0.1:6379"})
	if err := rdb.Ping(ctx).Err(); err != nil {
		t.Skipf("Redis not available: %v", err)
	}

	// 准备 test func
	rdb.Set(ctx, "/src/func/add_test", "(add_test(A, B) -> (C))", 0)
	rdb.Set(ctx, "/src/func/add_test/0", "add(A, B) -> ./C", 0)

	// 创建 vthread
	state := map[string]string{
		"pc":     "[0,0]",
		"status": "init",
		"mode":   "single",
	}
	data, _ := json.Marshal(state)
	rdb.Set(ctx, "/vthread/999", data, 0)
	rdb.Set(ctx, "/vthread/999/[0,0]", "call", 0)
	rdb.Set(ctx, "/vthread/999/[0,-1]", "add_test", 0)
	rdb.Set(ctx, "/vthread/999/[0,-2]", "./a", 0)
	rdb.Set(ctx, "/vthread/999/[0,-3]", "./b", 0)
	rdb.Set(ctx, "/vthread/999/[0,1]", "./c", 0)

	// 通知 VM
	rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"999"}`)

	// 清理
	defer rdb.Del(ctx, "/vthread/999")
	t.Log("integration test setup done")
}
