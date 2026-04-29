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

	"git.array2d.com/ai/deepx/executor/vm/internal/engine"
	"git.array2d.com/ai/deepx/executor/vm/internal/state"
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

	// 优雅退出
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	sigName := <-sig
	log.Printf("VM-%s received %s, shutting down...", vmID, sigName)

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
