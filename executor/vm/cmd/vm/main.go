// VM 命令入口：生产级 server 模式 + 可选的 single-run 调试模式。
//
//	server 模式:  ./vm [redis_addr]                 → worker pool, 信号管理, 优雅退出
//	single 模式:  ./vm run <vtid> [redis_addr]       → 执行单个 vthread 后退出 (调试用)
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"deepx/executor/vm/internal/engine"
	"deepx/executor/vm/internal/state"
	"github.com/redis/go-redis/v9"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// ── single-run 模式: ./vm run <vtid> [redis_addr] ──
	if len(os.Args) >= 2 && os.Args[1] == "run" {
		vtid := os.Args[2]
		redisAddr := "127.0.0.1:6379"
		if len(os.Args) >= 4 {
			redisAddr = os.Args[3]
		}
		rdb := redis.NewClient(&redis.Options{Addr: redisAddr})
		defer rdb.Close()
		if err := rdb.Ping(ctx).Err(); err != nil {
			log.Printf("redis connect failed: %v", err)
			os.Exit(1)
		}
		singleRun(ctx, rdb, vtid)
		return
	}

	// ── server 模式: ./vm [redis_addr] ──
	redisAddr := "127.0.0.1:6379"
	if len(os.Args) >= 2 {
		redisAddr = os.Args[1]
	}
	vmID := os.Getenv("VM_ID")
	if vmID == "" {
		vmID = "0"
	}

	workers := runtime.GOMAXPROCS(0)
	log.Printf("VM-%s starting with %d workers, redis=%s", vmID, workers, redisAddr)

	// 连接 Redis (生产级连接池)
	rdb := redis.NewClient(&redis.Options{
		Addr:         redisAddr,
		PoolSize:     workers * 2,
		MinIdleConns: workers,
		PoolTimeout:  10 * time.Second,
		ReadTimeout:  3 * time.Second,
		WriteTimeout: 3 * time.Second,
	})
	defer rdb.Close()

	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Printf("VM-%s redis connect failed: %v", vmID, err)
		os.Exit(1)
	}

	// 注册 VM 实例到 /sys/vm/<id>
	reg := map[string]interface{}{
		"status":     "running",
		"pid":        os.Getpid(),
		"started_at": time.Now().Unix(),
	}
	data, err := json.Marshal(reg)
	if err != nil {
		log.Printf("VM-%s register marshal failed: %v", vmID, err)
		os.Exit(1)
	}
	if err := rdb.Set(ctx, "/sys/vm/"+vmID, data, 0).Err(); err != nil {
		log.Printf("VM-%s register SET failed: %v", vmID, err)
		os.Exit(1)
	}
	log.Printf("VM-%s registered at /sys/vm/%s", vmID, vmID)

	// 启动 worker pool
	for i := 0; i < workers; i++ {
		go engine.RunWorker(ctx, rdb, i)
	}
	log.Printf("VM-%s %d workers started", vmID, workers)

	// ── 心跳上报 ──
	heartbeatKey := fmt.Sprintf("/sys/heartbeat/vm:%s", vmID)
	go func() {
		updateVMHeartbeat(ctx, rdb, heartbeatKey, "running") // 初始心跳
		ticker := time.NewTicker(2 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				updateVMHeartbeat(context.Background(), rdb, heartbeatKey, "stopped")
				log.Printf("VM-%s final heartbeat: stopped", vmID)
				return
			case <-ticker.C:
				updateVMHeartbeat(ctx, rdb, heartbeatKey, "running")
			}
		}
	}()
	log.Printf("VM-%s heartbeat → %s (every 2s)", vmID, heartbeatKey)

	// ── /func/main 监听 ──
	// 自动检测 loader 写入的入口点，创建 vthread 并执行。
	// 如果没有 /func/main，则休息等待。
	go func() {
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				watchFuncMain(ctx, rdb, vmID)
			}
		}
	}()
	log.Printf("VM-%s /func/main watcher started (every 1s)", vmID)

	// ── 系统命令监听 (Redis) ──
	// VM 同时监听 OS 信号和 Redis 系统命令队列，二者任一触发即优雅退出。
	sysQueue := fmt.Sprintf("sys:cmd:vm:%s", vmID)
	go func() {
		for {
			result, err := rdb.BLPop(ctx, 5*time.Second, sysQueue).Result()
			if err != nil {
				// ctx cancelled 或 Redis 断连 → 退出监听
				if ctx.Err() != nil {
					return
				}
				continue
			}
			// result[0]=key, result[1]=value
			var sysCmd struct {
				Cmd string `json:"cmd"`
			}
			if err := json.Unmarshal([]byte(result[1]), &sysCmd); err != nil {
				log.Printf("VM-%s sys cmd parse error: %v", vmID, err)
				continue
			}
			if sysCmd.Cmd == "shutdown" {
				log.Printf("VM-%s received sys shutdown via Redis, shutting down...", vmID)
				cancel()
				return
			}
			log.Printf("VM-%s unknown sys cmd: %s", vmID, sysCmd.Cmd)
		}
	}()

	// ── OS 信号监听 (安全兜底) ──
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)

	select {
	case sigName := <-sig:
		log.Printf("VM-%s received %s, shutting down...", vmID, sigName)
	case <-ctx.Done():
		log.Printf("VM-%s context cancelled, shutting down...", vmID)
	}

	// 取消 context → 所有 worker 退出
	cancel()

	// 注销
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer shutdownCancel()
	if err := rdb.Del(shutdownCtx, "/sys/vm/"+vmID).Err(); err != nil {
		log.Printf("VM-%s deregister failed: %v", vmID, err)
	}
	log.Printf("VM-%s shutdown complete", vmID)
}

// updateVMHeartbeat writes a heartbeat to Redis.
func updateVMHeartbeat(ctx context.Context, rdb *redis.Client, key, status string) {
	hb := map[string]interface{}{
		"ts":     time.Now().Unix(),
		"status": status,
		"pid":    os.Getpid(),
	}
	data, _ := json.Marshal(hb)
	rdb.Set(ctx, key, data, 0)
}

// singleRun 执行单个 vthread 后退出 (调试/单步执行用)。
func singleRun(ctx context.Context, rdb *redis.Client, vtid string) {
	vs := state.Get(ctx, rdb, vtid)
	if vs.Status != "init" {
		log.Printf("vthread %s status=%s (expect init)", vtid, vs.Status)
		os.Exit(1)
	}

	log.Printf("[single] executing vthread %s", vtid)
	engine.Execute(ctx, rdb, vtid)

	// 等待异步任务完成
	time.Sleep(3 * time.Second)

	vs = state.Get(ctx, rdb, vtid)
	fmt.Printf("\n=== VThread %s ===\n", vtid)
	fmt.Printf("  PC:     %s\n", vs.PC)
	fmt.Printf("  Status: %s\n", vs.Status)
	if vs.Error != nil {
		fmt.Printf("  Error:  %v\n", vs.Error)
	}
}

// watchFuncMain polls /func/main and auto-creates vthreads when an entry is present.
//
// Protocol:
//
//	Loader writes:  SET /func/main {"entry":"funcName","reads":["...","..."],"writes":["...","..."]}
//	VM detects:     GET /func/main → DEL /func/main (claim) → create vthread
//	                → SET /func/main {"vtid":"<id>","status":"executing"} → LPUSH notify:vm
//	After execution: SET /func/main {"vtid":"<id>","status":"done"} or {"status":"error",...}
//
// deepxctl polls /func/main for vtid → polls vthread → reads result → DEL /func/main.
func watchFuncMain(ctx context.Context, rdb *redis.Client, vmID string) {
	const key = "/func/main"

	val, err := rdb.Get(ctx, key).Result()
	if err != nil {
		// Key doesn't exist — nothing to do, VM rests
		return
	}

	var entry struct {
		Entry  string   `json:"entry"`
		Reads  []string `json:"reads"`
		Writes []string `json:"writes"`
		Vtid   string   `json:"vtid"`
		Status string   `json:"status"`
	}
	if err := json.Unmarshal([]byte(val), &entry); err != nil {
		log.Printf("VM-%s /func/main parse error: %v", vmID, err)
		return
	}

	switch {
	case entry.Entry != "":
		// Phase 1: Loader wrote an entry point → claim and create vthread
		log.Printf("VM-%s /func/main detected entry=%s (reads=%v writes=%v)", vmID, entry.Entry, entry.Reads, entry.Writes)

		// Claim ownership (atomic DEL)
		if err := rdb.Del(ctx, key).Err(); err != nil {
			log.Printf("VM-%s failed to claim /func/main: %v", vmID, err)
			return
		}

		// Allocate vtid
		vtid, err := rdb.Incr(ctx, "/sys/vtid_counter").Result()
		if err != nil {
			log.Printf("VM-%s INCR vtid_counter failed: %v", vmID, err)
			return
		}
		vtidStr := fmt.Sprintf("%d", vtid)

		// Create vthread (same format as redis.CreateVThread)
		base := fmt.Sprintf("/vthread/%d", vtid)
		initState := `{"pc":"[0,0]","status":"init"}`
		pipe := rdb.Pipeline()
		pipe.Set(ctx, base, initState, 0)
		pipe.Set(ctx, base+"/[0,0]", entry.Entry, 0)
		pipe.Set(ctx, base+"/[0,1]", "./ret", 0)
		if _, err := pipe.Exec(ctx); err != nil {
			log.Printf("VM-%s create vthread %d failed: %v", vmID, vtid, err)
			return
		}

		// Inform deepxctl of the vtid
		statusData, _ := json.Marshal(map[string]string{
			"vtid":   vtidStr,
			"status": "executing",
		})
		rdb.Set(ctx, key, statusData, 0)

		// Wake workers
		notify, _ := json.Marshal(map[string]interface{}{
			"event": "new_vthread",
			"vtid":  vtidStr,
		})
		rdb.LPush(ctx, "notify:vm", notify)
		log.Printf("VM-%s /func/main → vthread %d created, workers notified", vmID, vtid)

	case entry.Vtid != "" && entry.Status == "executing":
		// Phase 2: VThread is executing — check if it completed
		vtidStr := entry.Vtid
		vstate, err := rdb.Get(ctx, "/vthread/"+vtidStr).Result()
		if err != nil {
			return // vthread not yet created or already cleaned up
		}

		var vs struct {
			Status string `json:"status"`
		}
		if err := json.Unmarshal([]byte(vstate), &vs); err != nil {
			return
		}

		if vs.Status == "done" || vs.Status == "error" {
			statusData, _ := json.Marshal(map[string]string{
				"vtid":   vtidStr,
				"status": vs.Status,
			})
			rdb.Set(ctx, key, statusData, 0)
			log.Printf("VM-%s /func/main vtid=%s → status=%s", vmID, vtidStr, vs.Status)
		}
	}
}
