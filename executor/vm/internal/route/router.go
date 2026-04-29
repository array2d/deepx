package route

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"strings"

	"github.com/redis/go-redis/v9"
)

// Select 根据 opcode 选择负载最低的 op-plat 实例
// 返回实例标识符, e.g., "metal:0", "cuda:1"
func Select(ctx context.Context, rdb *redis.Client, opcode string) (string, error) {
	// 1. 找到支持该算子的所有程序
	programs, err := rdb.Keys(ctx, "/op/*/list").Result()
	if err != nil {
		return "", fmt.Errorf("list op programs: %w", err)
	}

	var chosenProgram string
	for _, progKey := range programs {
		list, err := rdb.LRange(ctx, progKey, 0, -1).Result()
		if err != nil {
			continue
		}
		for _, op := range list {
			if op == opcode {
				parts := strings.Split(progKey, "/")
				// "/op/op-cuda/list" → "op-cuda"
				if len(parts) >= 3 {
					chosenProgram = parts[2]
				}
				break
			}
		}
		if chosenProgram != "" {
			break
		}
	}

	if chosenProgram == "" {
		return "", fmt.Errorf("no op-plat supports opcode: %s", opcode)
	}

	// 2. 选择该程序下负载最低的进程实例
	instances, err := rdb.Keys(ctx, "/sys/op-plat/*").Result()
	if err != nil {
		return "", fmt.Errorf("list op-plat instances: %w", err)
	}

	type instInfo struct {
		Program string  `json:"program"`
		Status  string  `json:"status"`
		Load    float64 `json:"load"`
	}

	bestLoad := math.MaxFloat64
	bestInstance := ""

	for _, instKey := range instances {
		if !strings.Contains(instKey, chosenProgram) {
			continue
		}

		val, err := rdb.Get(ctx, instKey).Result()
		if err != nil {
			continue
		}
		var info instInfo
		if err := json.Unmarshal([]byte(val), &info); err != nil {
			log.Printf("route.Select: unmarshal instance info %s: %v", instKey, err)
			continue
		}

		if info.Status != "running" {
			continue
		}
		if info.Load < bestLoad {
			bestLoad = info.Load
			// "/sys/op-plat/op-metal:0" → "metal:0"
			parts := strings.Split(instKey, "/")
			lastPart := parts[len(parts)-1]
			bestInstance = strings.TrimPrefix(lastPart, "op-")
		}
	}

	if bestInstance == "" {
		return "", fmt.Errorf("no running op-plat instance for %s (program %s)", opcode, chosenProgram)
	}

	return bestInstance, nil
}

// DetermineBackend 判断 func 的编译后端 (按优先级)
func DetermineBackend(ctx context.Context, rdb *redis.Client, funcName string) string {
	for _, b := range []string{"op-metal", "op-cuda", "op-cpu"} {
		exists, err := rdb.Exists(ctx, fmt.Sprintf("/op/%s/func/%s", b, funcName)).Result()
		if err != nil {
			log.Printf("route.DetermineBackend: EXISTS error for %s: %v", b, err)
			continue
		}
		if exists > 0 {
			return b
		}
	}
	return "op-metal"
}
