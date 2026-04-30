package vm_test

import (
	"testing"

	"deepx/executor/vm/internal/ir"
	"deepx/executor/vm/testutil"
)

// wantInst 定义期望的指令结构
type wantInst struct {
	op     string
	reads  []string
	writes []string
}

// verifyInst 验证一条指令的解析结果
func verifyInst(t *testing.T, dxFile string, lineIdx int, inst *ir.Instruction, want wantInst) {
	t.Helper()
	if inst.Opcode != want.op {
		t.Errorf("[%s] line[%d] opcode=%s, want %s", dxFile, lineIdx, inst.Opcode, want.op)
	}
	if len(inst.Reads) != len(want.reads) {
		t.Errorf("[%s] line[%d] reads len=%d, want %d (%v vs %v)", dxFile, lineIdx, len(inst.Reads), len(want.reads), inst.Reads, want.reads)
		return
	}
	for i := range inst.Reads {
		if inst.Reads[i] != want.reads[i] {
			t.Errorf("[%s] line[%d] reads[%d]=%s, want %s", dxFile, lineIdx, i, inst.Reads[i], want.reads[i])
		}
	}
	if len(inst.Writes) != len(want.writes) {
		t.Errorf("[%s] line[%d] writes len=%d, want %d (%v vs %v)", dxFile, lineIdx, len(inst.Writes), len(want.writes), inst.Writes, want.writes)
		return
	}
	for i := range inst.Writes {
		if inst.Writes[i] != want.writes[i] {
			t.Errorf("[%s] line[%d] writes[%d]=%s, want %s", dxFile, lineIdx, i, inst.Writes[i], want.writes[i])
		}
	}
}

// checkDx 加载 .dx 文件并逐行验证解析结果
func checkDx(t *testing.T, dxFile string, wants []wantInst) {
	t.Helper()
	fn, err := testutil.LoadDxFile(dxFile)
	if err != nil {
		t.Fatalf("LoadDxFile(%s): %v", dxFile, err)
	}
	if len(fn.Body) != len(wants) {
		t.Fatalf("[%s] body has %d lines, want %d:\n  got:  %v\n  want: %v", dxFile, len(fn.Body), len(wants), fn.Body, wants)
	}
	for i, w := range wants {
		inst, err := ir.ParseDxlang(fn.Body[i])
		if err != nil {
			t.Errorf("[%s] line[%d] parse error: %v", dxFile, i, err)
			continue
		}
		verifyInst(t, dxFile, i, inst, w)
	}
}

// ── Lifecycle ───────────────────────────────────────────────

func TestParse_Lifecycle(t *testing.T) {
	t.Run("newtensor", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/tensor/lifecycle/newtensor.dx", []wantInst{
			{op: "newtensor", reads: []string{"f32", "[16]"}, writes: []string{"/data/x"}},
		})
	})
	t.Run("del", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/tensor/lifecycle/del.dx", []wantInst{
			{op: "newtensor", reads: []string{"f32", "[8]"}, writes: []string{"/data/tmp"}},
			{op: "deltensor", reads: []string{"/data/tmp"}, writes: nil},
		})
	})
	t.Run("compute_small", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/tensor/lifecycle/compute.dx", []wantInst{
			{op: "newtensor", reads: []string{"f32", "[8]"}, writes: []string{"/data/a"}},
			{op: "newtensor", reads: []string{"f32", "[8]"}, writes: []string{"/data/b"}},
			{op: "newtensor", reads: []string{"f32", "[8]"}, writes: []string{"/data/c"}},
			{op: "zeros", reads: nil, writes: []string{"/data/a"}},
			{op: "zeros", reads: nil, writes: []string{"/data/b"}},
			{op: "add", reads: []string{"/data/a", "/data/b"}, writes: []string{"/data/c"}},
			{op: "deltensor", reads: []string{"/data/a"}, writes: nil},
			{op: "deltensor", reads: []string{"/data/b"}, writes: nil},
		})
	})

}

// ── Call / Function Nesting ─────────────────────────────────

func TestParse_Call(t *testing.T) {
	t.Run("add_test", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/call/add_test.dx", []wantInst{
			{op: "add", reads: []string{"A", "B"}, writes: []string{"./C"}},
		})
	})
	t.Run("callee", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/call/callee.dx", []wantInst{
			{op: "+", reads: []string{"X", "Y"}, writes: []string{"./Z"}},
		})
	})
	t.Run("caller", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/call/caller.dx", []wantInst{
			{op: "callee", reads: []string{"A", "B"}, writes: []string{"./C"}},
		})
	})
	t.Run("middle", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/call/middle.dx", []wantInst{
			{op: "leaf", reads: []string{"X"}, writes: []string{"./tmp"}},
			{op: "+", reads: []string{"./tmp", "1"}, writes: []string{"./Y"}},
		})
	})
	t.Run("deep3", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/call/deep3.dx", []wantInst{
			{op: "middle", reads: []string{"X"}, writes: []string{"./Y"}},
		})
	})
	t.Run("diamond", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/call/diamond.dx", []wantInst{
			{op: "double", reads: []string{"A"}, writes: []string{"./d"}},
			{op: "triple", reads: []string{"A"}, writes: []string{"./t"}},
			{op: "+", reads: []string{"./d", "./t"}, writes: []string{"./R"}},
		})
	})
	t.Run("double", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/call/double.dx", []wantInst{
			{op: "*", reads: []string{"X", "2"}, writes: []string{"./Y"}},
		})
	})
	t.Run("triple", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/call/triple.dx", []wantInst{
			{op: "*", reads: []string{"X", "3"}, writes: []string{"./Y"}},
		})
	})
}

// ── Native: Arithmetic ──────────────────────────────────────

func TestParse_NativeArith(t *testing.T) {
	tests := []struct {
		name string
		file string
		op   string
	}{
		{"add", "../../example/dxlang/builtin/native/arith/add.dx", "+"},
		{"sub", "../../example/dxlang/builtin/native/arith/sub.dx", "-"},
		{"mul", "../../example/dxlang/builtin/native/arith/mul.dx", "*"},
		{"div", "../../example/dxlang/builtin/native/arith/div.dx", "/"},
		{"neg", "../../example/dxlang/builtin/native/arith/neg.dx", "neg"},
		{"abs", "../../example/dxlang/builtin/native/arith/abs.dx", "abs"},
		{"sign", "../../example/dxlang/builtin/native/arith/sign.dx", "sign"},
		{"pow", "../../example/dxlang/builtin/native/arith/pow.dx", "pow"},
		{"exp", "../../example/dxlang/builtin/native/arith/exp.dx", "exp"},
		{"log", "../../example/dxlang/builtin/native/arith/log.dx", "log"},
		{"sqrt", "../../example/dxlang/builtin/native/arith/sqrt.dx", "sqrt"},
		{"max", "../../example/dxlang/builtin/native/arith/max.dx", "max"},
		{"min", "../../example/dxlang/builtin/native/arith/min.dx", "min"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fn, err := testutil.LoadDxFile(tt.file)
			if err != nil {
				t.Fatal(err)
			}
			if len(fn.Body) == 0 {
				t.Fatal("empty body")
			}
			inst, err := ir.ParseDxlang(fn.Body[0])
			if err != nil {
				t.Fatal(err)
			}
			if inst.Opcode != tt.op {
				t.Errorf("opcode=%s, want %s", inst.Opcode, tt.op)
			}
		})
	}
}

// ── Native: Compare / Logic / Cast / Chain ──────────────────

func TestParse_NativeOther(t *testing.T) {
	t.Run("compare", func(t *testing.T) {
		t.Run("eq", func(t *testing.T) {
			checkDx(t, "../../example/dxlang/builtin/native/compare/eq.dx", []wantInst{
				{op: "==", reads: []string{"A", "B"}, writes: []string{"./C"}},
			})
		})
		t.Run("lt", func(t *testing.T) {
			checkDx(t, "../../example/dxlang/builtin/native/compare/lt.dx", []wantInst{
				{op: "<", reads: []string{"A", "B"}, writes: []string{"./C"}},
			})
		})
	})
	t.Run("logic", func(t *testing.T) {
		t.Run("and", func(t *testing.T) {
			checkDx(t, "../../example/dxlang/builtin/native/logic/and.dx", []wantInst{
				{op: "&&", reads: []string{"A", "B"}, writes: []string{"./C"}},
			})
		})
		t.Run("not", func(t *testing.T) {
			checkDx(t, "../../example/dxlang/builtin/native/logic/not.dx", []wantInst{
				{op: "!", reads: []string{"A"}, writes: []string{"./C"}},
			})
		})
		t.Run("bool", func(t *testing.T) {
			checkDx(t, "../../example/dxlang/builtin/native/logic/bool.dx", []wantInst{
				{op: "bool", reads: []string{"A"}, writes: []string{"./C"}},
			})
		})
	})
	t.Run("cast", func(t *testing.T) {
		t.Run("int", func(t *testing.T) {
			checkDx(t, "../../example/dxlang/builtin/native/cast/int.dx", []wantInst{
				{op: "int", reads: []string{"A"}, writes: []string{"./C"}},
			})
		})
		t.Run("float", func(t *testing.T) {
			checkDx(t, "../../example/dxlang/builtin/native/cast/float.dx", []wantInst{
				{op: "float", reads: []string{"A"}, writes: []string{"./C"}},
			})
		})
	})
	t.Run("chain", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/native/chain/chain.dx", []wantInst{
			{op: "+", reads: []string{"A", "B"}, writes: []string{"./tmp"}},
			{op: "*", reads: []string{"./tmp", "C"}, writes: []string{"./D"}},
		})
	})
}
