package vm_test

import (
	"fmt"
	"testing"

	"deepx/executor/vm/internal/parser"
	"deepx/executor/vm/internal/ast"
)

// wantInst 定义期望的指令结构

// loadFirstFunc parses a .dx file and returns its first function.
func loadFirstFunc(path string) (*ast.Func, error) {
	df, err := parser.ParseFile(path)
	if err != nil {
		return nil, err
	}
	if len(df.Funcs) == 0 {
		return nil, fmt.Errorf("no functions in %s", path)
	}
	return &df.Funcs[0], nil
}

type wantInst struct {
	op     string
	reads  []string
	writes []string
}

// verifyInst 验证一条指令的解析结果
func verifyInst(t *testing.T, dxFile string, lineIdx int, inst *ast.Instruction, want wantInst) {
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
	fn, err := loadFirstFunc(dxFile)
	if err != nil {
		t.Fatalf("LoadDxFile(%s): %v", dxFile, err)
	}
	if len(fn.Body) != len(wants) {
		t.Fatalf("[%s] body has %d lines, want %d:\n  got:  %v\n  want: %v", dxFile, len(fn.Body), len(wants), fn.Body, wants)
	}
	for i, w := range wants {
		inst, err := parser.ParseLine(fn.Body[i])
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
		{"add", "../../example/dxlang/builtin/arith/add.dx", "+"},
		{"sub", "../../example/dxlang/builtin/arith/sub.dx", "-"},
		{"mul", "../../example/dxlang/builtin/arith/mul.dx", "*"},
		{"div", "../../example/dxlang/builtin/arith/div.dx", "/"},
		{"neg", "../../example/dxlang/builtin/arith/neg.dx", "neg"},
		{"abs", "../../example/dxlang/builtin/arith/abs.dx", "abs"},
		{"sign", "../../example/dxlang/builtin/arith/sign.dx", "sign"},
		{"pow", "../../example/dxlang/builtin/arith/pow.dx", "pow"},
		{"exp", "../../example/dxlang/builtin/arith/exp.dx", "exp"},
		{"log", "../../example/dxlang/builtin/arith/log.dx", "log"},
		{"sqrt", "../../example/dxlang/builtin/arith/sqrt.dx", "sqrt"},
		{"max", "../../example/dxlang/builtin/arith/max.dx", "max"},
		{"min", "../../example/dxlang/builtin/arith/min.dx", "min"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fn, err := loadFirstFunc(tt.file)
			if err != nil {
				t.Fatal(err)
			}
			if len(fn.Body) == 0 {
				t.Fatal("empty body")
			}
			// 每个 buildin 示例现在都以 print 语句开头和结尾
			// 首行应为 print (输入打印)
			firstInst, err := parser.ParseLine(fn.Body[0])
			if err != nil {
				t.Fatal(err)
			}
			if firstInst.Opcode != "print" {
				t.Errorf("[%s] first opcode=%s, want print", tt.name, firstInst.Opcode)
			}
			// 核心计算在倒数第二行 (最后一行是 print("./C"))
			computeLine := fn.Body[len(fn.Body)-2]
			inst, err := parser.ParseLine(computeLine)
			if err != nil {
				t.Fatal(err)
			}
			if inst.Opcode != tt.op {
				t.Errorf("[%s] opcode=%s, want %s", tt.name, inst.Opcode, tt.op)
			}
		})
	}
}

// ── Native: Compare / Logic / Cast / Chain ──────────────────

func TestParse_NativeOther(t *testing.T) {
	t.Run("compare", func(t *testing.T) {
		t.Run("eq", func(t *testing.T) {
			checkDx(t, "../../example/dxlang/builtin/compare/eq.dx", []wantInst{
				{op: "print", reads: []string{"A"}, writes: nil},
				{op: "print", reads: []string{"B"}, writes: nil},
				{op: "==", reads: []string{"A", "B"}, writes: []string{"./C"}},
				{op: "print", reads: []string{"./C"}, writes: nil},
			})
		})
		t.Run("lt", func(t *testing.T) {
			checkDx(t, "../../example/dxlang/builtin/compare/lt.dx", []wantInst{
				{op: "print", reads: []string{"A"}, writes: nil},
				{op: "print", reads: []string{"B"}, writes: nil},
				{op: "<", reads: []string{"A", "B"}, writes: []string{"./C"}},
				{op: "print", reads: []string{"./C"}, writes: nil},
			})
		})
	})
	t.Run("logic", func(t *testing.T) {
		t.Run("and", func(t *testing.T) {
			checkDx(t, "../../example/dxlang/builtin/logic/and.dx", []wantInst{
				{op: "print", reads: []string{"A"}, writes: nil},
				{op: "print", reads: []string{"B"}, writes: nil},
				{op: "&&", reads: []string{"A", "B"}, writes: []string{"./C"}},
				{op: "print", reads: []string{"./C"}, writes: nil},
			})
		})
		t.Run("not", func(t *testing.T) {
			checkDx(t, "../../example/dxlang/builtin/logic/not.dx", []wantInst{
				{op: "print", reads: []string{"A"}, writes: nil},
				{op: "!", reads: []string{"A"}, writes: []string{"./C"}},
				{op: "print", reads: []string{"./C"}, writes: nil},
			})
		})
		t.Run("bool", func(t *testing.T) {
			checkDx(t, "../../example/dxlang/builtin/logic/bool.dx", []wantInst{
				{op: "print", reads: []string{"A"}, writes: nil},
				{op: "bool", reads: []string{"A"}, writes: []string{"./C"}},
				{op: "print", reads: []string{"./C"}, writes: nil},
			})
		})
	})
	t.Run("cast", func(t *testing.T) {
		t.Run("int", func(t *testing.T) {
			checkDx(t, "../../example/dxlang/builtin/cast/int.dx", []wantInst{
				{op: "print", reads: []string{"A"}, writes: nil},
				{op: "int", reads: []string{"A"}, writes: []string{"./C"}},
				{op: "print", reads: []string{"./C"}, writes: nil},
			})
		})
		t.Run("float", func(t *testing.T) {
			checkDx(t, "../../example/dxlang/builtin/cast/float.dx", []wantInst{
				{op: "print", reads: []string{"A"}, writes: nil},
				{op: "float", reads: []string{"A"}, writes: []string{"./C"}},
				{op: "print", reads: []string{"./C"}, writes: nil},
			})
		})
	})
	t.Run("chain", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/chain/chain.dx", []wantInst{
			{op: "print", reads: []string{"A"}, writes: nil},
			{op: "print", reads: []string{"B"}, writes: nil},
			{op: "print", reads: []string{"C"}, writes: nil},
			{op: "+", reads: []string{"A", "B"}, writes: []string{"./tmp"}},
			{op: "print", reads: []string{"./tmp"}, writes: nil},
			{op: "*", reads: []string{"./tmp", "C"}, writes: []string{"./D"}},
			{op: "print", reads: []string{"./D"}, writes: nil},
		})
	})
}

// ── New Examples: Multi-read/write, C-style (<-) ──────────

func TestParse_NewExamples(t *testing.T) {
	t.Run("double_op", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/arith/double_op.dx", []wantInst{
			{op: "print", reads: []string{"A"}, writes: nil},
			{op: "print", reads: []string{"B"}, writes: nil},
			{op: "+", reads: []string{"A", "B"}, writes: []string{"./S"}},
			{op: "print", reads: []string{"./S"}, writes: nil},
			{op: "-", reads: []string{"A", "B"}, writes: []string{"./D"}},
			{op: "print", reads: []string{"./D"}, writes: nil},
		})
	})
	t.Run("double_op_cstyle", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/arith/double_op_cstyle.dx", []wantInst{
			{op: "print", reads: []string{"A"}, writes: nil},
			{op: "print", reads: []string{"B"}, writes: nil},
			{op: "+", reads: []string{"A", "B"}, writes: []string{"./S"}},
			{op: "print", reads: []string{"./S"}, writes: nil},
			{op: "-", reads: []string{"A", "B"}, writes: []string{"./D"}},
			{op: "print", reads: []string{"./D"}, writes: nil},
		})
	})
	t.Run("three_add", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/arith/three_add.dx", []wantInst{
			{op: "print", reads: []string{"A"}, writes: nil},
			{op: "print", reads: []string{"B"}, writes: nil},
			{op: "+", reads: []string{"A", "B"}, writes: []string{"./t"}},
			{op: "print", reads: []string{"./t"}, writes: nil},
			{op: "print", reads: []string{"C"}, writes: nil},
			{op: "+", reads: []string{"./t", "C"}, writes: []string{"./R"}},
			{op: "print", reads: []string{"./R"}, writes: nil},
		})
	})
	t.Run("poly3", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/arith/poly3.dx", []wantInst{
			{op: "*", reads: []string{"A", "X"}, writes: []string{"./t1"}},
			{op: "print", reads: []string{"./t1"}, writes: nil},
			{op: "*", reads: []string{"./t1", "X"}, writes: []string{"./t2"}},
			{op: "print", reads: []string{"./t2"}, writes: nil},
			{op: "*", reads: []string{"B", "X"}, writes: []string{"./t3"}},
			{op: "print", reads: []string{"./t3"}, writes: nil},
			{op: "+", reads: []string{"./t2", "./t3"}, writes: []string{"./t4"}},
			{op: "print", reads: []string{"./t4"}, writes: nil},
			{op: "+", reads: []string{"./t4", "C"}, writes: []string{"./Y"}},
			{op: "print", reads: []string{"./Y"}, writes: nil},
		})
	})
	t.Run("multi_io", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/call/multi_io.dx", []wantInst{
			{op: "+", reads: []string{"X", "Y"}, writes: []string{"./t"}},
			{op: "+", reads: []string{"./t", "Z"}, writes: []string{"./R"}},
		})
	})
	t.Run("multi_ret", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/call/multi_ret.dx", []wantInst{
			{op: "+", reads: []string{"A", "B"}, writes: []string{"./S"}},
			{op: "-", reads: []string{"A", "B"}, writes: []string{"./D"}},
			{op: "*", reads: []string{"A", "B"}, writes: []string{"./P"}},
		})
	})
	t.Run("add_sub", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/tensor/lifecycle/add_sub.dx", []wantInst{
			{op: "newtensor", reads: []string{"f32", "[64]"}, writes: []string{"/data/a"}},
			{op: "newtensor", reads: []string{"f32", "[64]"}, writes: []string{"/data/b"}},
			{op: "newtensor", reads: []string{"f32", "[64]"}, writes: []string{"/data/sum"}},
			{op: "newtensor", reads: []string{"f32", "[64]"}, writes: []string{"/data/diff"}},
			{op: "zeros", reads: nil, writes: []string{"/data/a"}},
			{op: "zeros", reads: nil, writes: []string{"/data/b"}},
			{op: "add", reads: []string{"/data/a", "/data/b"}, writes: []string{"./sum"}},
			{op: "sub", reads: []string{"/data/a", "/data/b"}, writes: []string{"./diff"}},
			{op: "deltensor", reads: []string{"/data/a"}, writes: nil},
			{op: "deltensor", reads: []string{"/data/b"}, writes: nil},
		})
	})
	t.Run("compute_cstyle", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/tensor/lifecycle/compute_cstyle.dx", []wantInst{
			{op: "newtensor", reads: []string{"f32", "[8]"}, writes: []string{"/data/a"}},
			{op: "newtensor", reads: []string{"f32", "[8]"}, writes: []string{"/data/b"}},
			{op: "newtensor", reads: []string{"f32", "[8]"}, writes: []string{"/data/c"}},
			{op: "zeros", reads: nil, writes: []string{"/data/a"}},
			{op: "zeros", reads: nil, writes: []string{"/data/b"}},
			{op: "add", reads: []string{"/data/a", "/data/b"}, writes: []string{"./c"}},
			{op: "deltensor", reads: []string{"/data/a"}, writes: nil},
			{op: "deltensor", reads: []string{"/data/b"}, writes: nil},
		})
	})
}

// ── Native: Print ────────────────────────────────────────────

func TestParse_Print(t *testing.T) {
	t.Run("print_int", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/print/print_int.dx", []wantInst{
			{op: "print", reads: []string{"X"}, writes: nil},
			{op: "+", reads: []string{"X", "0"}, writes: []string{"./R"}},
			{op: "print", reads: []string{"./R"}, writes: nil},
		})
	})
	t.Run("print_multi", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/print/print_multi.dx", []wantInst{
			{op: "print", reads: []string{"A"}, writes: nil},
			{op: "print", reads: []string{"B"}, writes: nil},
			{op: "print", reads: []string{"C"}, writes: nil},
			{op: "+", reads: []string{"A", "B"}, writes: []string{"./t"}},
			{op: "print", reads: []string{"./t"}, writes: nil},
			{op: "+", reads: []string{"./t", "C"}, writes: []string{"./R"}},
			{op: "print", reads: []string{"./R"}, writes: nil},
		})
	})
	t.Run("print_bool", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/print/print_bool.dx", []wantInst{
			{op: "bool", reads: []string{"A"}, writes: []string{"./C"}},
			{op: "print", reads: []string{"./C"}, writes: nil},
		})
	})
	t.Run("print_chain", func(t *testing.T) {
		checkDx(t, "../../example/dxlang/builtin/print/print_chain.dx", []wantInst{
			{op: "print", reads: []string{"A"}, writes: nil},
			{op: "+", reads: []string{"A", "B"}, writes: []string{"./tmp"}},
			{op: "print", reads: []string{"./tmp"}, writes: nil},
			{op: "print", reads: []string{"C"}, writes: nil},
			{op: "*", reads: []string{"./tmp", "C"}, writes: []string{"./D"}},
			{op: "print", reads: []string{"./D"}, writes: nil},
		})
	})
}
