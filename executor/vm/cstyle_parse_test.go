package vm_test

import (
	"testing"

	"deepx/executor/vm/internal/parser"
)

func TestParseDxlang_CstyleArrow(t *testing.T) {
	tests := []struct {
		name   string
		line   string
		op     string
		reads  []string
		writes []string
	}{
		// C-style infix (keys must be single-quoted)
		{"infix_add", "'./C' <- A + B", "+", []string{"A", "B"}, []string{"./C"}},
		{"infix_sub", "'./out' <- X - Y", "-", []string{"X", "Y"}, []string{"./out"}},
		{"infix_mul", "'./R' <- P * Q", "*", []string{"P", "Q"}, []string{"./R"}},
		// C-style prefix (function call)
		{"prefix_call", "'./C' <- add(A, B)", "add", []string{"A", "B"}, []string{"./C"}},
		{"prefix_relu", "'./Y' <- relu(X)", "relu", []string{"X"}, []string{"./Y"}},
		// C-style unary
		{"unary_neg", "'./C' <- -A", "-", []string{"A"}, []string{"./C"}},
		{"unary_not", "'./C' <- !A", "!", []string{"A"}, []string{"./C"}},
		// C-style with absolute paths and string literals
		{"newtensor", "'/data/x' <- newtensor(\"f32\", \"[4]\")", "newtensor", []string{"f32", "[4]"}, []string{"/data/x"}},
		// Multi-write (parens) — less common but legal
		{"multi_write", "('./a', './b') <- split(X)", "split", []string{"X"}, []string{"./a", "./b"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inst, err := parser.ParseLine(tt.line)
			if err != nil {
				t.Fatalf("ParseDxlang(%q): %v", tt.line, err)
			}
			if inst.Opcode != tt.op {
				t.Errorf("opcode=%q, want %q", inst.Opcode, tt.op)
			}
			if !strSliceEq(inst.Reads, tt.reads) {
				t.Errorf("reads=%v, want %v", inst.Reads, tt.reads)
			}
			if !strSliceEq(inst.Writes, tt.writes) {
				t.Errorf("writes=%v, want %v", inst.Writes, tt.writes)
			}
		})
	}

	// Ensure traditional -> still works with single-quoted keys
	t.Run("traditional_still_works", func(t *testing.T) {
		inst, err := parser.ParseLine("add(A, B) -> './C'")
		if err != nil {
			t.Fatal(err)
		}
		if inst.Opcode != "add" {
			t.Errorf("opcode=%q, want add", inst.Opcode)
		}
		if len(inst.Writes) != 1 || inst.Writes[0] != "./C" {
			t.Errorf("writes=%v, want [./C]", inst.Writes)
		}
	})

	// Edge: <- embedded in comparison should not match
	t.Run("less_than_with_neg", func(t *testing.T) {
		inst, err := parser.ParseLine("A < -B -> './C'")
		if err != nil {
			t.Fatal(err)
		}
		if inst.Opcode != "<" {
			t.Errorf("opcode=%q, want <", inst.Opcode)
		}
		if !strSliceEq(inst.Writes, []string{"./C"}) {
			t.Errorf("writes=%v, want [./C]", inst.Writes)
		}
	})

	// Edge: <= should not match <-
	t.Run("less_or_equal", func(t *testing.T) {
		inst, err := parser.ParseLine("A <= B -> './C'")
		if err != nil {
			t.Fatal(err)
		}
		if inst.Opcode != "<=" {
			t.Errorf("opcode=%q, want <=", inst.Opcode)
		}
		if !strSliceEq(inst.Writes, []string{"./C"}) {
			t.Errorf("writes=%v, want [./C]", inst.Writes)
		}
	})
}

func strSliceEq(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
