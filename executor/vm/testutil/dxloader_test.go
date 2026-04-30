package testutil

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadDxFile(t *testing.T) {
	dxPath := filepath.Join("..", "..", "..", "example", "dxlang", "builtin", "call", "add_test.dx")

	fn, err := LoadDxFile(dxPath)
	if err != nil {
		t.Fatalf("LoadDxFile: %v", err)
	}

	if fn.Name != "add_test" {
		t.Errorf("Name = %q, want 'add_test'", fn.Name)
	}
	if fn.Signature != "def add_test(A:tensor, B:tensor) -> (C:tensor)" {
		t.Errorf("Signature = %q", fn.Signature)
	}
	if len(fn.Body) != 1 {
		t.Fatalf("Body len = %d, want 1", len(fn.Body))
	}
	if fn.Body[0] != `add(A, B) -> "./C"` {
		t.Errorf("Body[0] = %q, want `add(A, B) -> \"./C\"`", fn.Body[0])
	}
}

func TestExtractFuncName(t *testing.T) {
	tests := []struct {
		sig  string
		want string
	}{
		// def format
		{"def add_test(A:int, B:int) -> (C:int) {", "add_test"},
		{"def add_test(A:int, B:int) -> (C:int)", "add_test"},
		{"def gemm(A, B, alpha, beta, C) -> (Y)", "gemm"},
		{"def relu(X:tensor) -> (Y:tensor)", "relu"},
		// legacy format
		{"(add_test(A:tensor, B:tensor) -> (C:tensor))", "add_test"},
		{"(gemm(A, B, alpha, beta, C) -> (Y))", "gemm"},
		{"add_test -> (C)", "add_test"},
		{"simple_func", "simple_func"},
	}

	for _, tc := range tests {
		got := extractFuncName(tc.sig)
		if got != tc.want {
			t.Errorf("extractFuncName(%q) = %q, want %q", tc.sig, got, tc.want)
		}
	}
}

func TestLoadDxFile_Errors(t *testing.T) {
	_, err := LoadDxFile("/nonexistent/path.dx")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}

	tmpDir := t.TempDir()

	// Only comments
	emptyPath := filepath.Join(tmpDir, "empty.dx")
	os.WriteFile(emptyPath, []byte("# only comments\n"), 0644)
	_, err = LoadDxFile(emptyPath)
	if err == nil {
		t.Error("expected error for file with only comments")
	}

	// No def prefix
	noDef := filepath.Join(tmpDir, "nodef.dx")
	os.WriteFile(noDef, []byte("(foo(A) -> (B))\nadd(A) -> \"./B\"\n"), 0644)
	_, err = LoadDxFile(noDef)
	if err == nil {
		t.Error("expected error for file without 'def' prefix")
	}

	// Def with no body
	noBody := filepath.Join(tmpDir, "nobody.dx")
	os.WriteFile(noBody, []byte("def foo(A) -> (B) {\n}\n"), 0644)
	_, err = LoadDxFile(noBody)
	if err == nil {
		t.Error("expected error for function with empty body")
	}
}
