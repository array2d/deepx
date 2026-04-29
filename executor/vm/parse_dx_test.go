package vm_test

import (
	"testing"
	"git.array2d.com/ai/deepx/executor/vm/internal/ir"
	"git.array2d.com/ai/deepx/executor/vm/testutil"
)

func TestParseDxlang_Lifecycle(t *testing.T) {
	tests := []struct {
		dxFile string
		wantLines []struct {
			op     string
			reads  []string
			writes []string
		}
	}{
		{
			dxFile: "testdata/lifecycle_newtensor.dx",
			wantLines: []struct {
				op     string
				reads  []string
				writes []string
			}{
				{op: "newtensor", reads: []string{"f32", "[16]"}, writes: []string{"/data/x"}},
			},
		},
		{
			dxFile: "testdata/lifecycle_full.dx",
			wantLines: []struct {
				op     string
				reads  []string
				writes []string
			}{
				{op: "newtensor", reads: []string{"f32", "[4]"}, writes: []string{"/data/a"}},
				{op: "newtensor", reads: []string{"f32", "[4]"}, writes: []string{"/data/b"}},
				{op: "newtensor", reads: []string{"f32", "[4]"}, writes: []string{"/data/c"}},
				{op: "add", reads: []string{"/data/a", "/data/b"}, writes: []string{"/data/c"}},
				{op: "deltensor", reads: []string{"/data/a"}, writes: nil},
				{op: "deltensor", reads: []string{"/data/b"}, writes: nil},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.dxFile, func(t *testing.T) {
			fn, err := testutil.LoadDxFile(tt.dxFile)
			if err != nil {
				t.Fatalf("LoadDxFile: %v", err)
			}
			if len(fn.Body) != len(tt.wantLines) {
				t.Fatalf("body has %d lines, want %d", len(fn.Body), len(tt.wantLines))
			}
			for i, want := range tt.wantLines {
				inst, err := ir.ParseDxlang(fn.Body[i])
				if err != nil {
					t.Errorf("line[%d] parse error: %v", i, err)
					continue
				}
				if inst.Opcode != want.op {
					t.Errorf("line[%d] opcode=%s, want %s", i, inst.Opcode, want.op)
				}
				if !strSliceEq(inst.Reads, want.reads) {
					t.Errorf("line[%d] reads=%v, want %v", i, inst.Reads, want.reads)
				}
				if !strSliceEq(inst.Writes, want.writes) {
					t.Errorf("line[%d] writes=%v, want %v", i, inst.Writes, want.writes)
				}
			}
		})
	}
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
