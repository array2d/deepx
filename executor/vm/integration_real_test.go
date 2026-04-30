//go:build integration_real

// Package vm_test provides end-to-end integration tests using real
// heap-metal and op-metal binaries as subprocesses.
//
// Architecture
//
//	VM (Go test) ──RPUSH──> cmd:heap-metal:0 ──BLPOP──> heap-metal (C++ binary)
//	VM (Go test) ──RPUSH──> cmd:op-metal:0   ──BLPOP──> op-metal   (C++ binary)
//	heap-metal   ──LPUSH──> done:<vtid>       ──BLPOP──> VM
//	op-metal     ──LPUSH──> done:<vtid>       ──BLPOP──> VM
//
// Prerequisites
//
//	go test -tags=integration_real -run TestIntegrationReal ./...
//
// Environment variables:
//
//	REDIS_ADDR           Redis address (default "127.0.0.1:6379")
//	HEAP_METAL_BIN       heap-metal binary path (default "../build-heap/deepx-heap-metal")
//	OP_METAL_BIN         op-metal binary path (default "../build/deepx-op-metal")
package vm_test

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"git.array2d.com/ai/deepx/executor/vm/internal/engine"
	"git.array2d.com/ai/deepx/executor/vm/testutil"
	"github.com/redis/go-redis/v9"
)

// ═══════════════════════════════════════════════════════════════
// Subprocess Manager
// ═══════════════════════════════════════════════════════════════

// platProc manages a platform binary (heap-metal or op-metal) as subprocess.
type platProc struct {
	Name string
	Cmd  *exec.Cmd
	mu   sync.Mutex
	done chan struct{}
}

// startPlat starts a platform binary and waits for it to register in Redis.
func startPlat(t *testing.T, name, binPath, redisAddr string, rdb *redis.Client) *platProc {
	t.Helper()

	// Resolve redis host:port
	host, port := "127.0.0.1", "6379"
	if redisAddr != "" {
		parts := strings.Split(redisAddr, ":")
		if len(parts) >= 1 {
			host = parts[0]
		}
		if len(parts) >= 2 {
			port = parts[1]
		}
	}

	// Build log file paths
	logDir := "/tmp/deepx-test-logs"
	os.MkdirAll(logDir, 0755)
	logPath := filepath.Join(logDir, fmt.Sprintf("%s-%d.log", name, time.Now().UnixNano()))

	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, binPath, host, port)

	logFile, err := os.Create(logPath)
	if err != nil {
		cancel()
		t.Fatalf("create log file %s: %v", logPath, err)
	}
	cmd.Stdout = logFile
	cmd.Stderr = logFile

	if err := cmd.Start(); err != nil {
		cancel()
		logFile.Close()
		t.Fatalf("start %s (%s): %v", name, binPath, err)
	}

	p := &platProc{
		Name: name,
		Cmd:  cmd,
		done: make(chan struct{}),
	}

	// Background: wait for process exit, then close log
	go func() {
		cmd.Wait()
		logFile.Close()
		close(p.done)
	}()

	// Background: kill subprocess when context is cancelled
	go func() {
		<-ctx.Done()
		p.Kill()
	}()

	t.Cleanup(func() {
		cancel()
		select {
		case <-p.done:
		case <-time.After(3 * time.Second):
			t.Logf("[%s] force killing after timeout", name)
			p.Kill()
		}
	})

	// ── Wait for registration ──
	regKey := fmt.Sprintf("/sys/%s-plat/%s:0", name, name)
	t.Logf("[%s] waiting for registration at %s (log: %s)", name, regKey, logPath)

	ctx2, waitCancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer waitCancel()

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx2.Done():
			t.Fatalf("[%s] registration timeout (key: %s, log: %s)", name, regKey, logPath)
		case <-ticker.C:
			val, err := rdb.Get(context.Background(), regKey).Result()
			if err == redis.Nil {
				continue
			}
			if err != nil {
				continue
			}
			var reg map[string]interface{}
			if err := json.Unmarshal([]byte(val), &reg); err != nil {
				continue
			}
			if status, ok := reg["status"].(string); ok && status == "running" {
				t.Logf("[%s] registered (pid=%v, log=%s)", name, reg["pid"], logPath)
				return p
			}
		}
	}
}

// Kill forcefully terminates the subprocess.
func (p *platProc) Kill() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.Cmd.Process != nil {
		p.Cmd.Process.Signal(syscall.SIGKILL)
	}
}

func (p *platProc) IsRunning() bool {
	select {
	case <-p.done:
		return false
	default:
		return true
	}
}

// ═══════════════════════════════════════════════════════════════
// Test Helpers
// ═══════════════════════════════════════════════════════════════

// getRedisAddr returns Redis address from env or default.
func getRedisAddr() string {
	if addr := os.Getenv("REDIS_ADDR"); addr != "" {
		return addr
	}
	return "127.0.0.1:6379"
}

// getBinaryPath returns binary path from env or default relative path.
func getBinaryPath(envKey, defaultRelPath string) string {
	if p := os.Getenv(envKey); p != "" {
		return p
	}
	return filepath.Join("..", "..", defaultRelPath)
}

// cleanTestFiles removes leftover test output files.
func cleanTestFiles(paths ...string) {
	for _, p := range paths {
		os.Remove(p + ".shape")
		os.Remove(p + ".data")
	}
}

// ═══════════════════════════════════════════════════════════════
// Integration: Lifecycle (heap-metal)
// ═══════════════════════════════════════════════════════════════

func TestIntegrationReal_Lifecycle(t *testing.T) {
	rdb, ctx := connectRedisIntegration(t)
	defer rdb.Close()

	redisAddr := getRedisAddr()
	heapBin := getBinaryPath("HEAP_METAL_BIN", "build-heap/deepx-heap-metal")

	// Start heap-metal
	heap := startPlat(t, "heap", heapBin, redisAddr, rdb)

	// Start VM worker
	vmCtx, vmCancel := context.WithCancel(ctx)
	defer vmCancel()
	go engine.RunWorker(vmCtx, rdb, 0)
	_ = heap
	time.Sleep(150 * time.Millisecond)

	root := filepath.Join("..", "..", "example", "dxlang")

	// ── Test newtensor ──
	t.Run("newtensor", func(t *testing.T) {
		fn, err := testutil.LoadDxFile(filepath.Join(root, "lifecycle", "newtensor.dx"))
		if err != nil {
			t.Fatalf("LoadDxFile: %v", err)
		}
		fn.Name = "test_newtensor"
		if err := fn.RegisterFunc(ctx, rdb); err != nil {
			t.Fatalf("RegisterFunc: %v", err)
		}

		vtid, err := testutil.CreateVThread(ctx, rdb, "test_newtensor", nil, []string{"/data/x"})
		if err != nil {
			t.Fatalf("CreateVThread: %v", err)
		}
		rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"`+vtid+`"}`)

		_, done := waitVthreadDone(t, rdb, vtid, 10*time.Second)
		if !done {
			t.Fatal("vthread did not complete")
		}

		// Verify tensor exists in Redis
		val, err := rdb.Get(ctx, "/data/x").Result()
		if err != nil {
			t.Fatalf("/data/x not found: %v", err)
		}
		var meta map[string]interface{}
		json.Unmarshal([]byte(val), &meta)

		if dtype, ok := meta["dtype"].(string); !ok || dtype != "f32" {
			t.Errorf("dtype: got %v, want f32", meta["dtype"])
		}
		if addr, ok := meta["address"].(map[string]interface{}); !ok || addr["shm_name"] == nil {
			t.Error("tensor missing shm address")
		}
		t.Logf("newtensor: /data/x created (dtype=%v, shape=%v, shm=%v)",
			meta["dtype"], meta["shape"], meta["address"].(map[string]interface{})["shm_name"])
	})

	// ── Test deltensor ──
	t.Run("deltensor", func(t *testing.T) {
		// First create a tensor
		fnNew, _ := testutil.LoadDxFile(filepath.Join(root, "lifecycle", "newtensor.dx"))
		// Use a temporary tensor key to avoid conflicts
		fnNew.Name = "test_deltensor_new"
		fnNew.Body = []string{`newtensor("f32", "[8]") -> "/data/del_test"`}
		fnNew.RegisterFunc(ctx, rdb)

		vtid1, _ := testutil.CreateVThread(ctx, rdb, "test_deltensor_new", nil, []string{"/data/del_test"})
		rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"`+vtid1+`"}`)
		waitVthreadDone(t, rdb, vtid1, 10*time.Second)

		// Verify it exists
		if _, err := rdb.Get(ctx, "/data/del_test").Result(); err != nil {
			t.Fatalf("/data/del_test should exist before delete: %v", err)
		}

		// Now delete it
		fnDel, _ := testutil.LoadDxFile(filepath.Join(root, "lifecycle", "del.dx"))
		fnDel.Name = "test_deltensor_del"
		fnDel.Body = []string{`deltensor("/data/del_test")`}
		fnDel.RegisterFunc(ctx, rdb)

		vtid2, _ := testutil.CreateVThread(ctx, rdb, "test_deltensor_del", nil, nil)
		rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"`+vtid2+`"}`)
		_, done := waitVthreadDone(t, rdb, vtid2, 10*time.Second)
		if !done {
			t.Fatal("deltensor vthread did not complete")
		}

		// Verify deleted
		if _, err := rdb.Get(ctx, "/data/del_test").Result(); err != redis.Nil {
			t.Error("/data/del_test should be deleted")
		}
		t.Log("deltensor: /data/del_test deleted ✓")
	})
}

// ═══════════════════════════════════════════════════════════════
// Integration: Compute (op-metal)
// ═══════════════════════════════════════════════════════════════

func TestIntegrationReal_Compute(t *testing.T) {
	rdb, ctx := connectRedisIntegration(t)
	defer rdb.Close()

	redisAddr := getRedisAddr()
	heapBin := getBinaryPath("HEAP_METAL_BIN", "build-heap/deepx-heap-metal")
	opBin := getBinaryPath("OP_METAL_BIN", "build/deepx-op-metal")

	// Start plats
	heap := startPlat(t, "heap", heapBin, redisAddr, rdb)
	op := startPlat(t, "op", opBin, redisAddr, rdb)
	_ = heap
	_ = op

	// Start VM worker
	vmCtx, vmCancel := context.WithCancel(ctx)
	defer vmCancel()
	go engine.RunWorker(vmCtx, rdb, 0)
	time.Sleep(150 * time.Millisecond)

	root := filepath.Join("..", "..", "example", "dxlang")

	// ── Test: full lifecycle (create → compute → delete) ──
	t.Run("compute_full", func(t *testing.T) {
		fn, err := testutil.LoadDxFile(filepath.Join(root, "lifecycle", "full.dx"))
		if err != nil {
			t.Fatalf("LoadDxFile: %v", err)
		}
		fn.Name = "test_compute_full"
		if err := fn.RegisterFunc(ctx, rdb); err != nil {
			t.Fatalf("RegisterFunc: %v", err)
		}

		vtid, err := testutil.CreateVThread(ctx, rdb, "test_compute_full", nil, []string{"/data/c"})
		if err != nil {
			t.Fatalf("CreateVThread: %v", err)
		}
		rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"`+vtid+`"}`)

		_, done := waitVthreadDone(t, rdb, vtid, 15*time.Second)
		if !done {
			t.Fatal("vthread did not complete")
		}

		// Verify output tensor /data/c exists
		val, err := rdb.Get(ctx, "/data/c").Result()
		if err != nil {
			t.Fatalf("/data/c not found: %v", err)
		}
		var meta map[string]interface{}
		json.Unmarshal([]byte(val), &meta)
		t.Logf("/data/c: dtype=%v, shape=%v", meta["dtype"], meta["shape"])

		// Verify input tensors are deleted
		for _, key := range []string{"/data/a", "/data/b"} {
			if _, err := rdb.Get(ctx, key).Result(); err != redis.Nil {
				t.Errorf("%s should be deleted", key)
			}
		}
		t.Log("compute_full: newtensor → add → deltensor ✓")
	})

	// ── Test: compute (separate tensors, elementwise add) ──
	t.Run("compute_separate", func(t *testing.T) {
		fn, err := testutil.LoadDxFile(filepath.Join(root, "lifecycle", "compute.dx"))
		if err != nil {
			t.Fatalf("LoadDxFile: %v", err)
		}
		fn.Name = "test_compute_sep"
		if err := fn.RegisterFunc(ctx, rdb); err != nil {
			t.Fatalf("RegisterFunc: %v", err)
		}

		vtid, err := testutil.CreateVThread(ctx, rdb, "test_compute_sep", nil, []string{"/data/c"})
		if err != nil {
			t.Fatalf("CreateVThread: %v", err)
		}
		rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"`+vtid+`"}`)

		_, done := waitVthreadDone(t, rdb, vtid, 15*time.Second)
		if !done {
			t.Fatal("vthread did not complete")
		}

		// Verify result
		if _, err := rdb.Get(ctx, "/data/c").Result(); err != nil {
			t.Fatalf("/data/c not found: %v", err)
		}
		for _, key := range []string{"/data/a", "/data/b"} {
			if _, err := rdb.Get(ctx, key).Result(); err != redis.Nil {
				t.Errorf("%s should be deleted", key)
			}
		}
		t.Log("compute_separate: newtensor × 3 → add → deltensor × 2 ✓")
	})
}

// ═══════════════════════════════════════════════════════════════
// Integration: IO (op-metal print/save/load)
// ═══════════════════════════════════════════════════════════════

func TestIntegrationReal_IO(t *testing.T) {
	rdb, ctx := connectRedisIntegration(t)
	defer rdb.Close()

	redisAddr := getRedisAddr()
	heapBin := getBinaryPath("HEAP_METAL_BIN", "build-heap/deepx-heap-metal")
	opBin := getBinaryPath("OP_METAL_BIN", "build/deepx-op-metal")

	// Start plats
	heap := startPlat(t, "heap", heapBin, redisAddr, rdb)
	op := startPlat(t, "op", opBin, redisAddr, rdb)
	_ = heap
	_ = op

	// Start VM worker
	vmCtx, vmCancel := context.WithCancel(ctx)
	defer vmCancel()
	go engine.RunWorker(vmCtx, rdb, 0)
	time.Sleep(150 * time.Millisecond)

	root := filepath.Join("..", "..", "example", "dxlang", "io")

	// ── Test: print_tensor ──
	t.Run("print_tensor", func(t *testing.T) {
		fn, err := testutil.LoadDxFile(filepath.Join(root, "print_tensor.dx"))
		if err != nil {
			t.Fatalf("LoadDxFile: %v", err)
		}
		fn.Name = "test_print_tensor"
		if err := fn.RegisterFunc(ctx, rdb); err != nil {
			t.Fatalf("RegisterFunc: %v", err)
		}

		vtid, err := testutil.CreateVThread(ctx, rdb, "test_print_tensor", nil, []string{"/data/x"})
		if err != nil {
			t.Fatalf("CreateVThread: %v", err)
		}
		rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"`+vtid+`"}`)

		_, done := waitVthreadDone(t, rdb, vtid, 10*time.Second)
		if !done {
			t.Fatal("vthread did not complete")
		}

		// Verify tensor still exists (print doesn't delete)
		if _, err := rdb.Get(ctx, "/data/x").Result(); err != nil {
			t.Error("/data/x should still exist after print")
		}
		t.Log("print_tensor ✓")
	})

	// ── Test: save_load ──
	t.Run("save_load", func(t *testing.T) {
		savePath := "/tmp/test_save_load_real"
		cleanTestFiles(savePath)

		fn, err := testutil.LoadDxFile(filepath.Join(root, "save_load.dx"))
		if err != nil {
			t.Fatalf("LoadDxFile: %v", err)
		}
		fn.Name = "test_save_load"
		fn.Body = []string{
			`newtensor("f32", "[10]") -> "/data/src"`,
			`save("/data/src", "/tmp/test_save_load_real")`,
			`newtensor("f32", "[10]") -> "/data/loaded"`,
			`load("/tmp/test_save_load_real") -> "/data/loaded"`,
			`print("/data/loaded")`,
			`deltensor("/data/src")`,
		}
		if err := fn.RegisterFunc(ctx, rdb); err != nil {
			t.Fatalf("RegisterFunc: %v", err)
		}

		vtid, err := testutil.CreateVThread(ctx, rdb, "test_save_load", nil, []string{"/data/loaded"})
		if err != nil {
			t.Fatalf("CreateVThread: %v", err)
		}
		rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"`+vtid+`"}`)

		_, done := waitVthreadDone(t, rdb, vtid, 15*time.Second)
		if !done {
			t.Fatal("vthread did not complete")
		}

		// Verify save files exist
		if _, err := os.Stat(savePath + ".shape"); os.IsNotExist(err) {
			t.Error("save file not created: " + savePath + ".shape")
		}
		if _, err := os.Stat(savePath + ".data"); os.IsNotExist(err) {
			t.Error("save file not created: " + savePath + ".data")
		}

		// Verify loaded tensor exists
		val, err := rdb.Get(ctx, "/data/loaded").Result()
		if err != nil {
			t.Fatalf("/data/loaded not found: %v", err)
		}
		var meta map[string]interface{}
		json.Unmarshal([]byte(val), &meta)
		t.Logf("loaded tensor: dtype=%v, shape=%v", meta["dtype"], meta["shape"])

		// Verify src deleted
		if _, err := rdb.Get(ctx, "/data/src").Result(); err != redis.Nil {
			t.Error("/data/src should be deleted")
		}
		t.Log("save_load: save → load → cleanup ✓")
	})

	// ── Test: io_pipeline (full: create → compute → save → load → cleanup) ──
	t.Run("io_pipeline", func(t *testing.T) {
		savePath := "/tmp/io_pipeline_z_real"
		cleanTestFiles(savePath)

		fn, err := testutil.LoadDxFile(filepath.Join(root, "io_pipeline.dx"))
		if err != nil {
			t.Fatalf("LoadDxFile: %v", err)
		}
		fn.Name = "test_io_pipeline"
		fn.Body = []string{
			`newtensor("f32", "[64]") -> "/data/x"`,
			`newtensor("f32", "[64]") -> "/data/y"`,
			`newtensor("f32", "[64]") -> "/data/z"`,
			`newtensor("f32", "[64]") -> "/data/temp"`,
			`add("/data/x", "/data/y") -> "/data/temp"`,
			`mul("/data/temp", "/data/z") -> "/data/z"`,
			`print("/data/z")`,
			`save("/data/z", "/tmp/io_pipeline_z_real")`,
			`newtensor("f32", "[64]") -> "/data/final"`,
			`load("/tmp/io_pipeline_z_real") -> "/data/final"`,
			`print("/data/final")`,
			`deltensor("/data/x")`,
			`deltensor("/data/y")`,
			`deltensor("/data/z")`,
			`deltensor("/data/temp")`,
		}
		if err := fn.RegisterFunc(ctx, rdb); err != nil {
			t.Fatalf("RegisterFunc: %v", err)
		}

		vtid, err := testutil.CreateVThread(ctx, rdb, "test_io_pipeline", nil, []string{"/data/final"})
		if err != nil {
			t.Fatalf("CreateVThread: %v", err)
		}
		rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"`+vtid+`"}`)

		_, done := waitVthreadDone(t, rdb, vtid, 30*time.Second)
		if !done {
			t.Fatal("vthread did not complete")
		}

		// Verify save files
		if _, err := os.Stat(savePath + ".shape"); os.IsNotExist(err) {
			t.Error("save file not created: " + savePath + ".shape")
		}
		if _, err := os.Stat(savePath + ".data"); os.IsNotExist(err) {
			t.Error("save file not created: " + savePath + ".data")
		}

		// Verify loaded tensor
		if _, err := rdb.Get(ctx, "/data/final").Result(); err != nil {
			t.Fatalf("/data/final not found: %v", err)
		}

		// Verify intermediates deleted
		for _, key := range []string{"/data/x", "/data/y", "/data/z", "/data/temp"} {
			if _, err := rdb.Get(ctx, key).Result(); err != redis.Nil {
				t.Errorf("%s should be deleted", key)
			}
		}
		t.Log("io_pipeline: save → load → cleanup ✓")
	})
}

// ═══════════════════════════════════════════════════════════════
// Integration: Cross-Call (native function composition)
// ═══════════════════════════════════════════════════════════════

func TestIntegrationReal_CrossCall(t *testing.T) {
	rdb, ctx := connectRedisIntegration(t)
	defer rdb.Close()

	root := filepath.Join("..", "..", "example", "dxlang")

	// Register call-chain functions (native scalar, no plats needed)
	double, _ := testutil.LoadDxFile(filepath.Join(root, "call", "double.dx"))
	double.Name = "double"
	double.RegisterFunc(ctx, rdb)

	triple, _ := testutil.LoadDxFile(filepath.Join(root, "call", "triple.dx"))
	triple.Name = "triple"
	triple.RegisterFunc(ctx, rdb)

	diamond, _ := testutil.LoadDxFile(filepath.Join(root, "call", "diamond.dx"))
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
		t.Log("cross_call diamond(5) = 25 ✓")
	}
}

// ═══════════════════════════════════════════════════════════════
// Integration: End-to-End Stress (all components)
// ═══════════════════════════════════════════════════════════════

func TestIntegrationReal_E2E(t *testing.T) {
	rdb, ctx := connectRedisIntegration(t)
	defer rdb.Close()

	redisAddr := getRedisAddr()
	heapBin := getBinaryPath("HEAP_METAL_BIN", "build-heap/deepx-heap-metal")
	opBin := getBinaryPath("OP_METAL_BIN", "build/deepx-op-metal")

	// Start all plats
	heap := startPlat(t, "heap", heapBin, redisAddr, rdb)
	op := startPlat(t, "op", opBin, redisAddr, rdb)
	_ = heap
	_ = op

	// Start VM worker
	vmCtx, vmCancel := context.WithCancel(ctx)
	defer vmCancel()
	go engine.RunWorker(vmCtx, rdb, 0)
	time.Sleep(150 * time.Millisecond)

	root := filepath.Join("..", "..", "example", "dxlang")

	// Register functions
	register := func(name, relPath string) {
		fn, err := testutil.LoadDxFile(filepath.Join(root, relPath))
		if err != nil {
			t.Fatalf("LoadDxFile %s: %v", name, err)
		}
		fn.Name = name
		if err := fn.RegisterFunc(ctx, rdb); err != nil {
			t.Fatalf("RegisterFunc %s: %v", name, err)
		}
	}

	register("e2e_double", "call/double.dx")
	register("e2e_triple", "call/triple.dx")

	// ── Sequential execution: double(A=3) then triple(double result) ──
	// Step 1: double(3) = 6
	t.Run("sequential_calls", func(t *testing.T) {
		vtid1, _ := testutil.CreateVThread(ctx, rdb, "e2e_double", []string{"./input"}, []string{"./output"})
		rdb.Set(ctx, "/vthread/"+vtid1+"/input", "3", 0)
		rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"`+vtid1+`"}`)

		out1, done1 := waitVthreadDone(t, rdb, vtid1, 10*time.Second)
		if !done1 {
			t.Fatal("double vthread did not complete")
		}
		if out1["output"] != "6" {
			t.Errorf("double(3): got %q, want '6'", out1["output"])
		}
		t.Logf("double(3) = %s ✓", out1["output"])

		// Step 2: triple(6) = 18
		vtid2, _ := testutil.CreateVThread(ctx, rdb, "e2e_triple", []string{"./input"}, []string{"./output"})
		rdb.Set(ctx, "/vthread/"+vtid2+"/input", out1["output"], 0)
		rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"`+vtid2+`"}`)

		out2, done2 := waitVthreadDone(t, rdb, vtid2, 10*time.Second)
		if !done2 {
			t.Fatal("triple vthread did not complete")
		}
		if out2["output"] != "18" {
			t.Errorf("triple(6): got %q, want '18'", out2["output"])
		}
		t.Logf("triple(6) = %s ✓", out2["output"])
	})

	// ── GPU compute + native scalar in same flow ──
	t.Run("gpu_native_mixed", func(t *testing.T) {
		// Load mixed/native_and_gpu.dx but adapt it
		// The original file references ops that may not be registered; we inline a simpler version
		fn := &testutil.DxFunc{
			Name:      "e2e_gpu_native",
			Signature: "def e2e_gpu_native() -> (\"/data/result\")",
			Body: []string{
				`newtensor("f32", "[16]") -> "/data/gpu_a"`,
				`newtensor("f32", "[16]") -> "/data/gpu_b"`,
				`newtensor("f32", "[16]") -> "/data/result"`,
				`add("/data/gpu_a", "/data/gpu_b") -> "/data/result"`,
				`deltensor("/data/gpu_a")`,
				`deltensor("/data/gpu_b")`,
			},
		}
		if err := fn.RegisterFunc(ctx, rdb); err != nil {
			t.Fatalf("RegisterFunc: %v", err)
		}

		vtid, _ := testutil.CreateVThread(ctx, rdb, "e2e_gpu_native", nil, []string{"/data/result"})
		rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"`+vtid+`"}`)

		_, done := waitVthreadDone(t, rdb, vtid, 15*time.Second)
		if !done {
			t.Fatal("vthread did not complete")
		}

		if _, err := rdb.Get(ctx, "/data/result").Result(); err != nil {
			t.Fatalf("/data/result not found: %v", err)
		}
		// Verify cleanup
		for _, key := range []string{"/data/gpu_a", "/data/gpu_b"} {
			if _, err := rdb.Get(ctx, key).Result(); err != redis.Nil {
				t.Errorf("%s should be deleted", key)
			}
		}
		t.Log("gpu_native_mixed: heap → op-metal compute → heap cleanup ✓")
	})
}

func init() {
	// Suppress verbose engine logs during tests
	log.SetFlags(log.Ltime | log.Lshortfile)
}