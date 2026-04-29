//go:build integration

package vm_test

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"testing"
	"time"

	"git.array2d.com/ai/deepx/executor/vm/internal/engine"
	"git.array2d.com/ai/deepx/executor/vm/testutil"
	"github.com/redis/go-redis/v9"
)

// mockOpPlat 模拟 op-plat
func mockOpPlat(ctx context.Context, rdb *redis.Client) {
	reg := map[string]interface{}{
		"program": "op-metal", "device": "mock",
		"status": "running", "load": 0.1,
		"pid": os.Getpid(), "started_at": time.Now().Unix(),
	}
	data, _ := json.Marshal(reg)
	rdb.Set(ctx, "/sys/op-plat/op-metal:0", data, 0)
	rdb.RPush(ctx, "/op/op-metal/list", "add", "sub", "mul", "matmul", "relu")
	log.Println("[mock-op] registered")

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		result, err := rdb.BRPop(ctx, 1*time.Second, "cmd:op-metal:0").Result()
		if err != nil {
			continue
		}
		if len(result) < 2 {
			continue
		}

		var task struct {
			Vtid    string                   `json:"vtid"`
			PC      string                   `json:"pc"`
			Opcode  string                   `json:"opcode"`
			Inputs  []map[string]interface{} `json:"inputs"`
			Outputs []map[string]interface{} `json:"outputs"`
			Params  map[string]interface{}   `json:"params"`
		}
		if err := json.Unmarshal([]byte(result[1]), &task); err != nil {
			continue
		}

		log.Printf("[mock-op] %s for vtid=%s pc=%s", task.Opcode, task.Vtid, task.PC)

		if task.Opcode == "add" {
			var a, b float64
			for _, in := range task.Inputs {
				key, _ := in["key"].(string)
				if key != "" {
					val, _ := rdb.Get(ctx, key).Result()
					var f float64
					fmt.Sscanf(val, "%f", &f)
					if a == 0 {
						a = f
					} else {
						b = f
					}
				}
			}
			resultVal := a + b
			log.Printf("[mock-op] add: %v + %v = %v", a, b, resultVal)
			if len(task.Outputs) > 0 {
				outKey, _ := task.Outputs[0]["key"].(string)
				if outKey != "" {
					rdb.Set(ctx, outKey, fmt.Sprintf("%v", resultVal), 0)
				}
			}
		}

		done := map[string]interface{}{"pc": task.PC, "status": "ok"}
		doneJSON, _ := json.Marshal(done)
		rdb.LPush(ctx, "done:"+task.Vtid, doneJSON)
		log.Printf("[mock-op] done vtid=%s", task.Vtid)
	}
}

func connectRedis(t *testing.T) (*redis.Client, context.Context, context.CancelFunc) {
	t.Helper()
	addr := os.Getenv("REDIS_ADDR")
	if addr == "" {
		addr = "127.0.0.1:16379"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	rdb := redis.NewClient(&redis.Options{Addr: addr, PoolSize: 10, MinIdleConns: 2})
	if err := rdb.Ping(ctx).Err(); err != nil {
		cancel()
		t.Fatalf("Redis not available at %s: %v", addr, err)
	}
	return rdb, ctx, cancel
}

// waitVthread 轮询等待 vthread 完成，返回输出值。
// VM 不清理 vthread key，调用方 (pysdk / 测试) 负责清理。
func waitVthread(t *testing.T, rdb *redis.Client, vtid string, timeout time.Duration) (bool, map[string]string) {
	t.Helper()
	ctx := context.Background()
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		<-ticker.C

		val, err := rdb.Get(ctx, "/vthread/"+vtid).Result()
		if err == redis.Nil {
			continue // 还没创建或已被外部删除
		}

		var s struct {
			Status string `json:"status"`
			PC     string `json:"pc"`
		}
		json.Unmarshal([]byte(val), &s)

		switch s.Status {
		case "done":
			outputs := readVThreadOutputs(t, rdb, vtid)
			return true, outputs
		case "error":
			var full map[string]interface{}
			json.Unmarshal([]byte(val), &full)
			t.Logf("  vtid=%s error: %v", vtid, full["error"])
			return false, nil
		}
	}

	// timeout diagnostics
	val, _ := rdb.Get(ctx, "/vthread/"+vtid).Result()
	t.Logf("timeout state: %s", val)
	return false, nil
}

func readVThreadOutputs(t *testing.T, rdb *redis.Client, vtid string) map[string]string {
	t.Helper()
	ctx := context.Background()
	keys, err := rdb.Keys(ctx, "/vthread/"+vtid+"/*").Result()
	if err != nil || len(keys) == 0 {
		return nil
	}

	pipe := rdb.Pipeline()
	cmds := make([]*redis.StringCmd, len(keys))
	for i, k := range keys {
		cmds[i] = pipe.Get(ctx, k)
	}
	pipe.Exec(ctx)

	outputs := make(map[string]string)
	prefix := "/vthread/" + vtid + "/"
	for i, cmd := range cmds {
		val, err := cmd.Result()
		if err != nil {
			continue
		}
		slot := keys[i][len(prefix):]
		if len(slot) > 0 && slot[0] != '[' {
			outputs[slot] = val
		}
	}
	return outputs
}

type nativeTestCase struct {
	dxFile   string
	reads    []string
	writes   []string
	inputs   map[string]string
	wantKey  string
	wantVal  string
}

func TestIntegration_EndToEnd(t *testing.T) {
	rdb, ctx, cancel := connectRedis(t)
	defer cancel()
	rdb.FlushDB(ctx)

	dxPath := filepath.Join("testdata", "add_test.dx")
	fn, err := testutil.LoadDxFile(dxPath)
	if err != nil {
		t.Fatalf("LoadDxFile: %v", err)
	}
	t.Logf("loaded %s (sig=%s, body=%d lines)", fn.Name, fn.Signature, len(fn.Body))

	if err := fn.RegisterFunc(ctx, rdb); err != nil {
		t.Fatalf("RegisterFunc: %v", err)
	}

	vtid, err := testutil.CreateVThread(ctx, rdb, "add_test",
		[]string{"./a", "./b"}, []string{"./c"})
	if err != nil {
		t.Fatalf("CreateVThread: %v", err)
	}
	rdb.Set(ctx, "/vthread/"+vtid+"/a", "1", 0)
	rdb.Set(ctx, "/vthread/"+vtid+"/b", "2", 0)
	t.Logf("vtid=%s", vtid)

	mockCtx, mockCancel := context.WithCancel(ctx)
	defer mockCancel()
	go mockOpPlat(mockCtx, rdb)
	time.Sleep(300 * time.Millisecond)

	vmCtx, vmCancel := context.WithCancel(ctx)
	defer vmCancel()
	go engine.RunWorker(vmCtx, rdb, 0)
	time.Sleep(200 * time.Millisecond)

	rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"`+vtid+`"}`)

	done, outputs := waitVthread(t, rdb, vtid, 10*time.Second)
	if !done {
		t.Fatal("vthread did not complete")
	}
	t.Logf("outputs: %v", outputs)
	if outputs["c"] != "3" {
		t.Errorf("c = %q, want '3'", outputs["c"])
	}
}

func TestIntegration_NativeOps(t *testing.T) {
	rdb, ctx, cancel := connectRedis(t)
	defer cancel()
	rdb.FlushDB(ctx)

	vmCtx, vmCancel := context.WithCancel(ctx)
	defer vmCancel()
	go engine.RunWorker(vmCtx, rdb, 0)
	time.Sleep(200 * time.Millisecond)

	cases := []nativeTestCase{
		// 算术
		{dxFile: "native_arith.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "2", "b": "3"}, wantKey: "c", wantVal: "5"},
		{dxFile: "native_sub.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "10", "b": "3"}, wantKey: "c", wantVal: "7"},
		{dxFile: "native_mul.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "4", "b": "5"}, wantKey: "c", wantVal: "20"},
		{dxFile: "native_div.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "7", "b": "2"}, wantKey: "c", wantVal: "3.5"},
		// 比较
		{dxFile: "native_eq.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "5", "b": "5"}, wantKey: "c", wantVal: "true"},
		{dxFile: "native_eq.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "3", "b": "7"}, wantKey: "c", wantVal: "false"},
		{dxFile: "native_lt.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "3", "b": "7"}, wantKey: "c", wantVal: "true"},
		{dxFile: "native_lt.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "10", "b": "2"}, wantKey: "c", wantVal: "false"},
		// 逻辑
		{dxFile: "native_and.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "true", "b": "true"}, wantKey: "c", wantVal: "true"},
		{dxFile: "native_and.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "true", "b": "false"}, wantKey: "c", wantVal: "false"},
		{dxFile: "native_not.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "true"}, wantKey: "c", wantVal: "false"},
		{dxFile: "native_not.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "false"}, wantKey: "c", wantVal: "true"},
		// 链式
		{dxFile: "native_chain.dx", reads: []string{"./a", "./b", "./c"}, writes: []string{"./d"},
			inputs: map[string]string{"a": "2", "b": "3", "c": "4"}, wantKey: "d", wantVal: "20"},

		// ── 数学 built-in ──
		{dxFile: "native_abs.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "-5"}, wantKey: "c", wantVal: "5"},
		{dxFile: "native_abs.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "3"}, wantKey: "c", wantVal: "3"},
		{dxFile: "native_pow.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "2", "b": "3"}, wantKey: "c", wantVal: "8"},
		{dxFile: "native_pow.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "10", "b": "0"}, wantKey: "c", wantVal: "1"},
		{dxFile: "native_min.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "7", "b": "3"}, wantKey: "c", wantVal: "3"},
		{dxFile: "native_min.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "-2", "b": "5"}, wantKey: "c", wantVal: "-2"},
		{dxFile: "native_max.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "7", "b": "3"}, wantKey: "c", wantVal: "7"},
		{dxFile: "native_max.dx", reads: []string{"./a", "./b"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "-9", "b": "0"}, wantKey: "c", wantVal: "0"},
		{dxFile: "native_sqrt.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "16"}, wantKey: "c", wantVal: "4"},
		{dxFile: "native_exp.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "1"}, wantKey: "c", wantVal: "2.718281828459045"},
		{dxFile: "native_log.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "1"}, wantKey: "c", wantVal: "0"},
		{dxFile: "native_neg.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "5"}, wantKey: "c", wantVal: "-5"},
		{dxFile: "native_neg.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "-3"}, wantKey: "c", wantVal: "3"},
		{dxFile: "native_sign.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "5"}, wantKey: "c", wantVal: "1"},
		{dxFile: "native_sign.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "-8"}, wantKey: "c", wantVal: "-1"},
		{dxFile: "native_sign.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "0"}, wantKey: "c", wantVal: "0"},

		// ── 类型转换 built-in ──
		{dxFile: "native_int.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "3.7"}, wantKey: "c", wantVal: "3"},
		{dxFile: "native_int.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "-2.9"}, wantKey: "c", wantVal: "-2"},
		{dxFile: "native_float.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "42"}, wantKey: "c", wantVal: "42.0"},
		{dxFile: "native_bool.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "1"}, wantKey: "c", wantVal: "true"},
		{dxFile: "native_bool.dx", reads: []string{"./a"}, writes: []string{"./c"},
			inputs: map[string]string{"a": "0"}, wantKey: "c", wantVal: "false"},
	}

	for i, tc := range cases {
		t.Run(fmt.Sprintf("%s_%d", fileToFuncName(tc.dxFile), i), func(t *testing.T) {
			funcName := fmt.Sprintf("%s_%d", fileToFuncName(tc.dxFile), i)

			dxPath := filepath.Join("testdata", tc.dxFile)
			fn, err := testutil.LoadDxFile(dxPath)
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

			done, outputs := waitVthread(t, rdb, vtid, 10*time.Second)
			if !done {
				t.Fatal("vthread did not complete")
			}

			got := outputs[tc.wantKey]
			t.Logf("%s = %q (want %q)", tc.wantKey, got, tc.wantVal)
			if got != tc.wantVal {
				t.Errorf("%s: got %q, want %q", tc.wantKey, got, tc.wantVal)
			}
		})
	}
}

// mockHeapPlat 模拟 heap-plat (heap-metal)，处理 tensor 生命周期指令。
// 与 mockOpPlat 协同: 创建真实 POSIX shm，写入 Redis 元信息。
func mockHeapPlat(ctx context.Context, rdb *redis.Client) {
	reg := map[string]interface{}{
		"program": "heap-metal", "device": "gpu0",
		"status": "running", "pid": os.Getpid(),
		"started_at": time.Now().Unix(),
	}
	data, _ := json.Marshal(reg)
	rdb.Set(ctx, "/sys/heap-plat/heap-metal:0", data, 0)
	log.Println("[mock-heap] registered")

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		result, err := rdb.BRPop(ctx, 1*time.Second, "cmd:heap-metal:0").Result()
		if err != nil {
			continue
		}
		if len(result) < 2 {
			continue
		}

		var task struct {
			Vtid  string `json:"vtid"`
			PC    string `json:"pc"`
			Op    string `json:"op"`
			Key   string `json:"key"`
			Dtype string `json:"dtype"`
			Shape []int  `json:"shape"`
		}
		if err := json.Unmarshal([]byte(result[1]), &task); err != nil {
			log.Printf("[mock-heap] parse error: %v", err)
			continue
		}

		log.Printf("[mock-heap] %s vtid=%s key=%s pc=%s", task.Op, task.Vtid, task.Key, task.PC)

		switch task.Op {
		case "newtensor":
			elemBytes := 4 // default f32
			switch task.Dtype {
			case "f64", "i64":
				elemBytes = 8
			case "f16", "i16":
				elemBytes = 2
			case "i8", "bool":
				elemBytes = 1
			}
			totalElems := 1
			for _, d := range task.Shape {
				totalElems *= d
			}
			byteSize := totalElems * elemBytes

			// Write tensor metadata to Redis
			meta := map[string]interface{}{
				"dtype":     task.Dtype,
				"shape":     task.Shape,
				"byte_size": byteSize,
				"device":    "gpu0",
				"address": map[string]interface{}{
					"type":     "shm",
					"shm_name": "/mock_shm_" + task.Key,
					"node":     "n1",
				},
			}
			metaJSON, _ := json.Marshal(meta)
			rdb.Set(ctx, task.Key, metaJSON, 0)
			log.Printf("[mock-heap] newtensor '%s' shape=%v bytes=%d", task.Key, task.Shape, byteSize)

			done := map[string]interface{}{"pc": task.PC, "status": "ok"}
			doneJSON, _ := json.Marshal(done)
			rdb.LPush(ctx, "done:"+task.Vtid, doneJSON)

		case "deltensor":
			// Delete tensor metadata from Redis
			rdb.Del(ctx, task.Key)
			log.Printf("[mock-heap] deltensor '%s'", task.Key)

			done := map[string]interface{}{"pc": task.PC, "status": "ok"}
			doneJSON, _ := json.Marshal(done)
			rdb.LPush(ctx, "done:"+task.Vtid, doneJSON)

		default:
			log.Printf("[mock-heap] unknown op: %s", task.Op)
			done := map[string]interface{}{"pc": task.PC, "status": "error", "error": "unknown op: " + task.Op}
			doneJSON, _ := json.Marshal(done)
			rdb.LPush(ctx, "done:"+task.Vtid, doneJSON)
		}
	}
}

func fileToFuncName(filename string) string {
	base := filepath.Base(filename)
	return base[:len(base)-len(filepath.Ext(base))]
}

// TestIntegration_CrossCall 测试一个 dx 函数 call 另一个函数
func TestIntegration_CrossCall(t *testing.T) {
	rdb, ctx, cancel := connectRedis(t)
	defer cancel()
	rdb.FlushDB(ctx)

	// 1. 加载 callee 和 caller
	calleePath := filepath.Join("testdata", "callee.dx")
	callee, err := testutil.LoadDxFile(calleePath)
	if err != nil {
		t.Fatalf("Load callee: %v", err)
	}
	callerPath := filepath.Join("testdata", "caller.dx")
	caller, err := testutil.LoadDxFile(callerPath)
	if err != nil {
		t.Fatalf("Load caller: %v", err)
	}

	// 2. 注册两个函数（callee 必须先注册，因为 caller 依赖它）
	if err := callee.RegisterFunc(ctx, rdb); err != nil {
		t.Fatalf("Register callee: %v", err)
	}
	if err := caller.RegisterFunc(ctx, rdb); err != nil {
		t.Fatalf("Register caller: %v", err)
	}
	t.Logf("registered callee=%s caller=%s", callee.Name, caller.Name)

	// 3. 创建 vthread
	vtid, err := testutil.CreateVThread(ctx, rdb, "caller",
		[]string{"./a", "./b"}, []string{"./c"})
	if err != nil {
		t.Fatalf("CreateVThread: %v", err)
	}
	rdb.Set(ctx, "/vthread/"+vtid+"/a", "7", 0)
	rdb.Set(ctx, "/vthread/"+vtid+"/b", "8", 0)
	t.Logf("vtid=%s", vtid)

	// 4. 启动 VM worker
	vmCtx, vmCancel := context.WithCancel(ctx)
	defer vmCancel()
	go engine.RunWorker(vmCtx, rdb, 0)
	time.Sleep(200 * time.Millisecond)

	// 5. 通知 VM
	rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"`+vtid+`"}`)

	// 6. 等待并验证
	done, outputs := waitVthread(t, rdb, vtid, 10*time.Second)
	if !done {
		t.Fatal("vthread did not complete")
	}
	t.Logf("outputs: %v", outputs)
	if outputs["c"] != "15" {
		t.Errorf("caller(7+8): got c=%q, want '15'", outputs["c"])
	}
}

// ── Lifecycle 集成测试 ──

// TestIntegration_LifecycleNewtensor 测试 newtensor → heap-plat 创建 tensor 元信息
func TestIntegration_LifecycleNewtensor(t *testing.T) {
	rdb, ctx, cancel := connectRedis(t)
	defer cancel()
	rdb.FlushDB(ctx)

	dxPath := filepath.Join("testdata", "lifecycle_newtensor.dx")
	fn, err := testutil.LoadDxFile(dxPath)
	if err != nil {
		t.Fatalf("LoadDxFile: %v", err)
	}
	t.Logf("loaded %s (sig=%s, body=%d lines)", fn.Name, fn.Signature, len(fn.Body))

	if err := fn.RegisterFunc(ctx, rdb); err != nil {
		t.Fatalf("RegisterFunc: %v", err)
	}

	vtid, err := testutil.CreateVThread(ctx, rdb, "lifecycle_newtensor",
		nil, []string{"/data/x"})
	if err != nil {
		t.Fatalf("CreateVThread: %v", err)
	}
	t.Logf("vtid=%s", vtid)

	// 启动 mock heap-plat
	heapCtx, heapCancel := context.WithCancel(ctx)
	defer heapCancel()
	go mockHeapPlat(heapCtx, rdb)
	time.Sleep(200 * time.Millisecond)

	// 启动 VM worker
	vmCtx, vmCancel := context.WithCancel(ctx)
	defer vmCancel()
	go engine.RunWorker(vmCtx, rdb, 0)
	time.Sleep(200 * time.Millisecond)

	rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"`+vtid+`"}`)

	done, _ := waitVthread(t, rdb, vtid, 10*time.Second)
	if !done {
		t.Fatal("vthread did not complete")
	}

	// 验证 tensor 元信息已写入 Redis
	val, err := rdb.Get(ctx, "/data/x").Result()
	if err != nil {
		t.Fatalf("tensor /data/x not found in Redis: %v", err)
	}
	t.Logf("/data/x meta: %s", val)

	var meta map[string]interface{}
	json.Unmarshal([]byte(val), &meta)
	if dtype, _ := meta["dtype"].(string); dtype != "f32" {
		t.Errorf("dtype = %q, want f32", dtype)
	}
}

// TestIntegration_LifecycleDelete 测试 newtensor → deltensor 完整销毁流程
func TestIntegration_LifecycleDelete(t *testing.T) {
	rdb, ctx, cancel := connectRedis(t)
	defer cancel()
	rdb.FlushDB(ctx)

	dxPath := filepath.Join("testdata", "lifecycle_del.dx")
	fn, err := testutil.LoadDxFile(dxPath)
	if err != nil {
		t.Fatalf("LoadDxFile: %v", err)
	}
	t.Logf("loaded %s (sig=%s, body=%d lines)", fn.Name, fn.Signature, len(fn.Body))

	if err := fn.RegisterFunc(ctx, rdb); err != nil {
		t.Fatalf("RegisterFunc: %v", err)
	}

	vtid, err := testutil.CreateVThread(ctx, rdb, "lifecycle_del",
		nil, nil)
	if err != nil {
		t.Fatalf("CreateVThread: %v", err)
	}
	t.Logf("vtid=%s", vtid)

	heapCtx, heapCancel := context.WithCancel(ctx)
	defer heapCancel()
	go mockHeapPlat(heapCtx, rdb)
	time.Sleep(200 * time.Millisecond)

	vmCtx, vmCancel := context.WithCancel(ctx)
	defer vmCancel()
	go engine.RunWorker(vmCtx, rdb, 0)
	time.Sleep(200 * time.Millisecond)

	rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"`+vtid+`"}`)

	done, _ := waitVthread(t, rdb, vtid, 10*time.Second)
	if !done {
		t.Fatal("vthread did not complete")
	}

	// 验证 tensor 已被删除
	_, err = rdb.Get(ctx, "/data/tmp").Result()
	if err != redis.Nil {
		t.Errorf("tensor /data/tmp should be deleted, but still exists")
	}
	t.Log("tensor /data/tmp correctly deleted")
}

// TestIntegration_LifecycleFull 完整生命周期: create → compute → delete
func TestIntegration_LifecycleFull(t *testing.T) {
	rdb, ctx, cancel := connectRedis(t)
	defer cancel()
	rdb.FlushDB(ctx)

	dxPath := filepath.Join("testdata", "lifecycle_full.dx")
	fn, err := testutil.LoadDxFile(dxPath)
	if err != nil {
		t.Fatalf("LoadDxFile: %v", err)
	}
	t.Logf("loaded %s (sig=%s, body=%d lines)", fn.Name, fn.Signature, len(fn.Body))

	if err := fn.RegisterFunc(ctx, rdb); err != nil {
		t.Fatalf("RegisterFunc: %v", err)
	}

	vtid, err := testutil.CreateVThread(ctx, rdb, "lifecycle_full",
		nil, []string{"/data/c"})
	if err != nil {
		t.Fatalf("CreateVThread: %v", err)
	}
	t.Logf("vtid=%s", vtid)

	// 启动 mock heap-plat
	heapCtx, heapCancel := context.WithCancel(ctx)
	defer heapCancel()
	go mockHeapPlat(heapCtx, rdb)
	time.Sleep(200 * time.Millisecond)

	// 启动 mock op-plat (处理 add 计算)
	mockOpCtx, mockOpCancel := context.WithCancel(ctx)
	defer mockOpCancel()
	go mockOpPlat(mockOpCtx, rdb)
	time.Sleep(200 * time.Millisecond)

	// 启动 VM worker
	vmCtx, vmCancel := context.WithCancel(ctx)
	defer vmCancel()
	go engine.RunWorker(vmCtx, rdb, 0)
	time.Sleep(200 * time.Millisecond)

	rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"`+vtid+`"}`)

	done, _ := waitVthread(t, rdb, vtid, 15*time.Second)
	if !done {
		t.Fatal("vthread did not complete")
	}

	// 验证 /data/a 和 /data/b 已被删除
	_, err = rdb.Get(ctx, "/data/a").Result()
	if err != redis.Nil {
		t.Errorf("tensor /data/a should be deleted")
	}
	_, err = rdb.Get(ctx, "/data/b").Result()
	if err != redis.Nil {
		t.Errorf("tensor /data/b should be deleted")
	}

	// 验证 /data/c 仍然存在 (output tensor, not deleted)
	val, err := rdb.Get(ctx, "/data/c").Result()
	if err != nil {
		t.Errorf("tensor /data/c should exist: %v", err)
	} else {
		t.Logf("/data/c meta: %s", val)
	}
}
