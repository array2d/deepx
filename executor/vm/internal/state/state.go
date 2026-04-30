// Package state 提供 vthread 状态管理与 Redis 持久化。
package state

import (
	"context"
	"encoding/json"
	"fmt"
		"deepx/executor/vm/internal/logx"
	"time"

	"github.com/redis/go-redis/v9"
)

// VThreadState 存储在 /vthread/<vtid> 中，表示运行时状态。
type VThreadState struct {
	PC     string            `json:"pc"`
	Status string            `json:"status"`
	Mode   string            `json:"mode,omitempty"` // "single" | "batch", 默认 "single"
	Error  map[string]string `json:"error,omitempty"`
}

// Get 读取 vthread 当前状态。
func Get(ctx context.Context, rdb *redis.Client, vtid string) VThreadState {
	val, err := rdb.Get(ctx, "/vthread/"+vtid).Result()
	if err != nil {
		return VThreadState{Status: "error"}
	}
	var s VThreadState
	if err := json.Unmarshal([]byte(val), &s); err != nil {
		logx.Warn("state.Get: unmarshal vthread %s: %v", vtid, err)
		return VThreadState{Status: "error"}
	}
	return s
}

// Set 更新 vthread 的 PC 和 status。
func Set(ctx context.Context, rdb *redis.Client, vtid string, pc, status string) {
	s := VThreadState{PC: pc, Status: status}
	data, err := json.Marshal(s)
	if err != nil {
		logx.Warn("state.Set: marshal vthread %s: %v", vtid, err)
		return
	}
	rdb.Set(ctx, "/vthread/"+vtid, data, 0)
}

// SetError 标记 vthread 为 error 状态。
func SetError(ctx context.Context, rdb *redis.Client, vtid string, pc string, errMsg string) {
	s := map[string]interface{}{
		"pc":     pc,
		"status": "error",
		"error":  map[string]string{"code": "VM_ERROR", "message": errMsg},
	}
	data, err := json.Marshal(s)
	if err != nil {
		logx.Warn("state.SetError: marshal vthread %s: %v", vtid, err)
		return
	}
	rdb.Set(ctx, "/vthread/"+vtid, data, 0)
}

// WaitDone 阻塞等待 op-plat / heap-plat 完成通知。
func WaitDone(ctx context.Context, rdb *redis.Client, vtid string, timeout time.Duration) (map[string]interface{}, error) {
	doneKey := "done:" + vtid
	result, err := rdb.BLPop(ctx, timeout, doneKey).Result()
	if err != nil {
		return nil, fmt.Errorf("waitDone timeout for %s: %w", doneKey, err)
	}
	var done map[string]interface{}
	if len(result) > 1 {
		if err := json.Unmarshal([]byte(result[1]), &done); err != nil {
			logx.Warn("state.WaitDone: unmarshal done result for %s: %v", vtid, err)
			return nil, fmt.Errorf("unmarshal done result: %w", err)
		}
	}
	return done, nil
}
