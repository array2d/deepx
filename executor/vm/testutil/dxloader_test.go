package testutil

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadDxFile(t *testing.T) {
	// 相对于 testutil/ 目录的 testdata
	dxPath := filepath.Join("..", "testdata", "add_test.dx")

	fn, err := LoadDxFile(dxPath)
	if err != nil {
		t.Fatalf("LoadDxFile: %v", err)
	}

	if fn.Name != "add_test" {
		t.Errorf("Name = %q, want 'add_test'", fn.Name)
	}
	if fn.Signature != "(add_test(A:tensor, B:tensor) -> (C:tensor))" {
		t.Errorf("Signature = %q", fn.Signature)
	}
	if len(fn.Body) != 1 {
		t.Fatalf("Body len = %d, want 1", len(fn.Body))
	}
	if fn.Body[0] != "add(A, B) -> ./C" {
		t.Errorf("Body[0] = %q, want 'add(A, B) -> ./C'", fn.Body[0])
	}
}

func TestExtractFuncName(t *testing.T) {
	tests := []struct {
		sig  string
		want string
	}{
		{"(add_test(A:tensor, B:tensor) -> (C:tensor))", "add_test"},
		{"(gemm(A, B, alpha, beta, C) -> (Y))", "gemm"},
		{"(relu(X:tensor) -> (Y:tensor))", "relu"},
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
	// 不存在的文件
	_, err := LoadDxFile("/nonexistent/path.dx")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}

	// 签名行太少
	tmpDir := t.TempDir()
	emptyPath := filepath.Join(tmpDir, "empty.dx")
	os.WriteFile(emptyPath, []byte("# only comments\n"), 0644)
	_, err = LoadDxFile(emptyPath)
	if err == nil {
		t.Error("expected error for file with only comments")
	}

	// 只有签名没有 body
	sigOnly := filepath.Join(tmpDir, "sigonly.dx")
	os.WriteFile(sigOnly, []byte("(foo(A) -> (B))\n"), 0644)
	_, err = LoadDxFile(sigOnly)
	if err == nil {
		t.Error("expected error for signature-only file")
	}
}
