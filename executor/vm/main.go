package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"deepx/executor/vm/internal/parser"
	"deepx/executor/vm/internal/state"
	"deepx/executor/vm/internal/vm"
	"deepx/executor/vm/internal/logx"
	"github.com/redis/go-redis/v9"
)

func main() {
	redisAddr := "127.0.0.1:16379"
	if v := os.Getenv("REDIS_ADDR"); v != "" {
		redisAddr = v
	}

	rdb := redis.NewClient(&redis.Options{Addr: redisAddr})
	ctx := context.Background()
	if err := rdb.Ping(ctx).Err(); err != nil {
		logx.Fatal("Redis %s 不可用: %v", redisAddr, err)
	}
	defer rdb.Close()
	rdb.FlushDB(ctx)

	// 1. 加载 .dx
	dxPath := filepath.Join(os.Getenv("PWD"), "test_print.dx")
	df, err := parser.ParseFile(dxPath)
	if err != nil {
		logx.Fatal("加载失败: %v", err)
	}
	if len(df.Funcs) == 0 {
		logx.Fatal("文件中没有函数定义")
	}
	fn := &df.Funcs[0]
	if err != nil {
		logx.Fatal("加载失败: %v", err)
	}
	fn.Name = "print_demo"
	if err := fn.Register(ctx, rdb); err != nil {
		logx.Fatal("注册失败: %v", err)
	}
	logx.Info("✅ 已加载: %s (body %d 行)", fn.Name, len(fn.Body))

	// 2. 启动 VM
	vmCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	go vm.RunWorker(vmCtx, rdb, 0)
	time.Sleep(100 * time.Millisecond)

	// 3. 创建虚线程
	vtid, err := state.CreateVThread(ctx, rdb, "print_demo",
		[]string{"./a", "./b", "./c"}, []string{"./r"})
	if err != nil {
		logx.Fatal("创建 vthread 失败: %v", err)
	}
	rdb.Set(ctx, "/vthread/"+vtid+"/a", "2", 0)
	rdb.Set(ctx, "/vthread/"+vtid+"/b", "3", 0)
	rdb.Set(ctx, "/vthread/"+vtid+"/c", "4", 0)
	rdb.RPush(ctx, "notify:vm", `{"event":"new_vthread","vtid":"`+vtid+`"}`)

	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("📝 dx 代码:")
	for _, line := range fn.Body {
		fmt.Printf("   %s\n", line)
	}
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("🚀 执行: print_demo(a=2, b=3, c=4)\n\n")

	// 4. 等待完成
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	deadline := time.Now().Add(10 * time.Second)
	done := false
	for time.Now().Before(deadline) {
		<-ticker.C
		val, err := rdb.Get(ctx, "/vthread/"+vtid).Result()
		if err != nil {
			continue
		}
		var s struct {
			Status string `json:"status"`
			PC     string `json:"pc"`
		}
		json.Unmarshal([]byte(val), &s)
		if s.Status == "done" || s.Status == "error" {
			done = true
			break
		}
	}

	if !done {
		logx.Fatal("❌ 超时")
	}

	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	// 读取结果
	result, _ := rdb.Get(ctx, "/vthread/"+vtid+"/r").Result()
	fmt.Printf("✅ 计算结果: r = %s\n", result)
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("👆 上方 [PRINT] 即为 native print 指令输出")
}
