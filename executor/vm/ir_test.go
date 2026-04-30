package vm_test

import (
	"context"
	"testing"

	"deepx/executor/vm/internal/ir"
	"deepx/executor/vm/internal/route"
	"github.com/redis/go-redis/v9"
)

// ── PC navigation (salvaged from deleted engine_test.go) ──

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

// ── Route: error handling (no live Redis needed) ──

func TestRouteSelect_NoRedis(t *testing.T) {
	rdb := redis.NewClient(&redis.Options{Addr: "127.0.0.1:9999"})
	ctx := context.Background()
	_, err := route.Select(ctx, rdb, "add")
	if err == nil {
		t.Error("expected error when Redis is not available")
	}
	t.Logf("expected error: %v", err)
}
