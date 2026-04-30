//go:build integration

package vm_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"deepx/executor/vm/internal/engine"
	"deepx/executor/vm/internal/ir"
	"deepx/executor/vm/testutil"
)

// ═══════════════════════════════════════════════════════════════
// Phase 1: 所有 .dx 文件语法解析正确性 (零 Redis)
// ═══════════════════════════════════════════════════════════════

func TestAllDxFilesParse(t *testing.T) {
	root := filepath.Join("..", "..", "example", "dxlang")

	var files []string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(path, ".dx") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk example/dxlang: %v", err)
	}
	if len(files) == 0 {
		t.Fatal("no .dx files found")
	}
	t.Logf("found %d .dx files", len(files))

	loaded := 0
	lines := 0
	for _, f := range files {
		fn, err := testutil.LoadDxFile(f)
		if err != nil {
			t.Errorf("LoadDxFile(%s): %v", f, err)
			continue
		}
		if fn.Name == "" {
			t.Errorf("LoadDxFile(%s): empty function name", f)
			continue
		}
		if len(fn.Body) == 0 {
			t.Errorf("LoadDxFile(%s): empty body", f)
			continue
		}

		// 验证每行指令可解析
		for i, line := range fn.Body {
			inst, err := ir.ParseDxlang(line)
			if err != nil {
				t.Errorf("[%s] body[%d]=%q parse error: %v", f, i, line, err)
				continue
			}
			if inst.Opcode == "" {
				t.Errorf("[%s] body[%d]=%q empty opcode", f, i, line)
			}
		}

		loaded++
		lines += len(fn.Body)
	}
	t.Logf("all %d files parsed (%d total body lines)", loaded, lines)
}

// ═══════════════════════════════════════════════════════════════
// 针对新复杂示例的专项解析验证
// ═══════════════════════════════════════════════════════════════

func TestParse_ComplexExamples(t *testing.T) {
	cases := []struct {
		file     string
		funcName string
		minBody  int // 最少指令行数
	}{
		{"lifecycle/batch_ops.dx", "batch_ops", 10},
		{"lifecycle/clone_and_use.dx", "clone_and_use", 8},
		{"nn/mlp_small.dx", "mlp_small", 16},
		{"nn/polynomial.dx", "polynomial", 12},
		{"nn/elemwise_long.dx", "elemwise_long", 12},
		{"nn/normalize.dx", "normalize", 7},
		{"math/dist2.dx", "dist2", 7},
		{"math/hadamard3.dx", "hadamard3", 8},
		{"math/max_abs.dx", "max_abs", 8},
		{"call/tensor_pipeline.dx", "producer", 4}, // 多函数文件中测试第一个
		{"mixed/native_and_gpu.dx", "native_and_gpu", 8},
	}

	root := filepath.Join("..", "..", "example", "dxlang")
	for _, tc := range cases {
		t.Run(tc.funcName, func(t *testing.T) {
			fn, err := testutil.LoadDxFile(filepath.Join(root, tc.file))
			if err != nil {
				t.Fatalf("LoadDxFile: %v", err)
			}
			if fn.Name != tc.funcName {
				t.Errorf("func name: got %q, want %q", fn.Name, tc.funcName)
			}
			if len(fn.Body) < tc.minBody {
				t.Errorf("body lines: got %d, want >= %d", len(fn.Body), tc.minBody)
			}

			// 验证每条指令的关键字
			for i, line := range fn.Body {
				inst, err := ir.ParseDxlang(line)
				if err != nil {
					t.Errorf("body[%d] %q: %v", i, line, err)
				}
				_ = inst
			}
			t.Logf("%s: %d body lines OK", tc.funcName, len(fn.Body))
		})
	}
}

// ═══════════════════════════════════════════════════════════════
// Phase 2: 端到端集成测试 (需要 Redis)
// Native Scalar / Cross-Call (纯 VM, 无需 plats)
// ═══════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════
// Integration: Native Scalar (VM only, no plats needed)
// ═══════════════════════════════════════════════════════════════

func TestIntegration_NativeScalar(t *testing.T) {
	rdb, ctx := connectRedisIntegration(t)
	defer rdb.Close()

	vmCtx, vmCancel := context.WithCancel(ctx)
	defer vmCancel()
	go engine.RunWorker(vmCtx, rdb, 0)
	time.Sleep(150 * time.Millisecond)

	type testCase struct {
		name    string
		dxFile  string
		reads   []string
		writes  []string
		inputs  map[string]string
		wantKey string
		wantVal string
	}

	root := filepath.Join("..", "..", "example", "dxlang")
	cases := []testCase{
		// 算术
		{name: "add", dxFile: "native/arith/add.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "2", "b": "3"}, wantKey: "c", wantVal: "5"},
		{name: "mul", dxFile: "native/arith/mul.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "6", "b": "7"}, wantKey: "c", wantVal: "42"},
		{name: "div", dxFile: "native/arith/div.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "15", "b": "2"}, wantKey: "c", wantVal: "7.5"},
		{name: "sub", dxFile: "native/arith/sub.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "10", "b": "3"}, wantKey: "c", wantVal: "7"},
		// 比较
		{name: "eq_true", dxFile: "native/compare/eq.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "5", "b": "5"}, wantKey: "c", wantVal: "true"},
		{name: "eq_false", dxFile: "native/compare/eq.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "2", "b": "9"}, wantKey: "c", wantVal: "false"},
		// 链式
		{name: "chain", dxFile: "native/chain/chain.dx", reads: []string{"./a", "./b", "./c"}, writes: []string{"./d"},
			inputs: map[string]string{"a": "2", "b": "3", "c": "4"}, wantKey: "d", wantVal: "20"},
		// built-in
		{name: "abs", dxFile: "native/arith/abs.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "-5"}, wantKey: "c", wantVal: "5"},
		{name: "pow", dxFile: "native/arith/pow.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "2", "b": "3"}, wantKey: "c", wantVal: "8.0"},
		{name: "max", dxFile: "native/arith/max.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "7", "b": "3"}, wantKey: "c", wantVal: "7"},
		{name: "min", dxFile: "native/arith/min.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "-2", "b": "5"}, wantKey: "c", wantVal: "-2"},
		{name: "sqrt", dxFile: "native/arith/sqrt.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "16"}, wantKey: "c", wantVal: "4.0"},
		{name: "neg", dxFile: "native/arith/neg.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "5"}, wantKey: "c", wantVal: "-5"},
		{name: "sign_pos", dxFile: "native/arith/sign.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "5"}, wantKey: "c", wantVal: "1"},
		{name: "sign_neg", dxFile: "native/arith/sign.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "-8"}, wantKey: "c", wantVal: "-1"},
		// cast
		{name: "int", dxFile: "native/cast/int.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "3.7"}, wantKey: "c", wantVal: "3"},
		{name: "float", dxFile: "native/cast/float.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "42"}, wantKey: "c", wantVal: "42.0"},
	}

	for i, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			funcName := fmt.Sprintf("native_%s_%d", tc.name, i)
			fp := filepath.Join(root, tc.dxFile)

			fn, err := testutil.LoadDxFile(fp)
			if err != nil {
				t.Fatalf("LoadDxFile: %v", err)
			}
			fn.Name = funcName
			if err := fn.RegisterFunc(ctx, rdb); err != nil {
				t.Fatalf("RegisterFunc: %v", err)
			}

			vtid, err := testutil.CreateVThread(ctx, rdb, funcName, tc.reads, tc.writes)
			if err != nil {
				t.Fatalf("CreateVThread: %v", err)
			}
			for slot, val := range tc.inputs {
				rdb.Set(ctx, "/vthread/"+vtid+"/"+slot, val, 0)
			}
			rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"`+vtid+`"}`)

			outputs, done := waitVthreadDone(t, rdb, vtid, 10*time.Second)
			if !done {
				t.Fatal("vthread did not complete")
			}
			got := outputs[tc.wantKey]
			if got != tc.wantVal {
				t.Errorf("%s: got %q, want %q", tc.wantKey, got, tc.wantVal)
			} else {
				t.Logf("  %s = %s ✓", tc.wantKey, got)
			}
		})
	}
}

// ═══════════════════════════════════════════════════════════════
// Integration: Cross-Call (多函数链)
// ═══════════════════════════════════════════════════════════════

func TestIntegration_CrossCall(t *testing.T) {
	rdb, ctx := connectRedisIntegration(t)
	defer rdb.Close()

	root := filepath.Join("..", "..", "example", "dxlang")

	// 加载 double, triple, diamond
	double, _ := testutil.LoadDxFile(filepath.Join(root, "call/double.dx"))
	double.Name = "double"
	double.RegisterFunc(ctx, rdb)

	triple, _ := testutil.LoadDxFile(filepath.Join(root, "call/triple.dx"))
	triple.Name = "triple"
	triple.RegisterFunc(ctx, rdb)

	diamond, _ := testutil.LoadDxFile(filepath.Join(root, "call/diamond.dx"))
	diamond.Name = "diamond"
	diamond.RegisterFunc(ctx, rdb)

	// Start VM worker
	vmCtx, vmCancel := context.WithCancel(ctx)
	defer vmCancel()
	go engine.RunWorker(vmCtx, rdb, 0)
	time.Sleep(150 * time.Millisecond)

	// diamond(A=5) → double(5)=10, triple(5)=15, R=25
	vtid, _ := testutil.CreateVThread(ctx, rdb, "diamond", []string{"./a"}, []string{"./r"})
	rdb.Set(ctx, "/vthread/"+vtid+"/a", "5", 0)
	rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"`+vtid+`"}`)

	outputs, done := waitVthreadDone(t, rdb, vtid, 15*time.Second)
	if !done {
		t.Fatal("vthread did not complete")
	}
	if outputs["r"] != "25" {
		t.Errorf("diamond(5): r=%q, want '25'", outputs["r"])
	} else {
		t.Log("diamond(5) = 25 ✓")
	}
}

