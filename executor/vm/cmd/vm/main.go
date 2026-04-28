package main

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"git.array2d.com/ai/deepx/executor/vm/internal/engine"
	"github.com/redis/go-redis/v9"
)

func main() {
	ctx := context.Background()

	// 配置
	redisAddr := os.Getenv("REDIS_ADDR")
	if redisAddr == "" {
		redisAddr = "127.0.0.1:6379"
	}

	vmID := os.Getenv("VM_ID")
	if vmID == "" {
		vmID = "0"
	}

	workers := runtime.GOMAXPROCS(0)
	log.Printf("Starting VM-%s with %d workers, redis=%s", vmID, workers, redisAddr)

	// 1. 连接 Redis
	rdb := redis.NewClient(&redis.Options{
		Addr:         redisAddr,
		PoolSize:     workers * 2,
		MinIdleConns: workers,
		PoolTimeout:  10 * time.Second,
		ReadTimeout:  3 * time.Second,
		WriteTimeout: 3 * time.Second,
	})
	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Fatalf("redis connect failed: %v", err)
	}
	defer rdb.Close()

	// 2. 注册 VM 实例
	reg := map[string]interface{}{
		"status":     "running",
		"pid":        os.Getpid(),
		"started_at": time.Now().Unix(),
	}
	data, _ := json.Marshal(reg)
	rdb.Set(ctx, "/sys/vm/"+vmID, data, 0)
	log.Printf("VM-%s registered at /sys/vm/%s", vmID, vmID)

	// 3. 启动 worker pool
	for i := 0; i < workers; i++ {
		go engine.RunWorker(ctx, rdb, i)
	}

	// 4. 等待信号
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig

	// 5. 注销
	rdb.Del(context.Background(), "/sys/vm/"+vmID)
	log.Println("VM shutdown")
}
