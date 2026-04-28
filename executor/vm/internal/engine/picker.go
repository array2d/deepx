package engine

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/redis/go-redis/v9"
)

// VThreadState 存储在 /vthread/<vtid> 中
type VThreadState struct {
	PC     string `json:"pc"`
	Status string `json:"status"`
	Mode   string `json:"mode,omitempty"` // "single" | "batch", 默认 "single"
	Error  map[string]string `json:"error,omitempty"`
}

// PickVthread 扫描 /vthread/*, 原子抢占 status=init 的 vthread
// 返回 vtid，无可用 vthread 时返回空字符串
func PickVthread(ctx context.Context, rdb *redis.Client) string {
	keys, err := rdb.Keys(ctx, "/vthread/*").Result()
	if err != nil || len(keys) == 0 {
		return ""
	}

	for _, key := range keys {
		vtid := extractVtid(key) // "/vthread/42" → "42"
		if vtid == "" {
			continue
		}
		// 跳过子 key: /vthread/1/a, /vthread/1/[0,0], ...
		// 子 key 的 vtid 会包含 "/"
		if containsAny(vtid, "/") {
			continue
		}

		if tryPick(ctx, rdb, vtid) {
			return vtid
		}
	}
	return ""
}

func containsAny(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

var errSkip = fmt.Errorf("skip") // 内部哨兵: 非 init 状态，跳过

func tryPick(ctx context.Context, rdb *redis.Client, vtid string) bool {
	key := "/vthread/" + vtid

	err := rdb.Watch(ctx, func(tx *redis.Tx) error {
		val, err := tx.Get(ctx, key).Result()
		if err == redis.Nil {
			return errSkip
		}
		if err != nil {
			return err
		}

		var state VThreadState
		if err := json.Unmarshal([]byte(val), &state); err != nil {
			return err
		}
		if state.Status != "init" {
			return errSkip // 已被抢走 / 已 done / 已 error → 跳过
		}

		state.Status = "running"
		data, err := json.Marshal(state)
		if err != nil {
			return err
		}

		_, err = tx.TxPipelined(ctx, func(pipe redis.Pipeliner) error {
			pipe.Set(ctx, key, data, 0)
			return nil
		})
		return err
	}, key)

	if err != nil && err != errSkip {
		log.Printf("tryPick %s error: %v", vtid, err)
	}
	return err == nil
}

func extractVtid(key string) string {
	const prefix = "/vthread/"
	if len(key) > len(prefix) {
		return key[len(prefix):]
	}
	return ""
}

// WaitForVthread 阻塞等待新 vthread 创建通知
func WaitForVthread(ctx context.Context, rdb *redis.Client) {
	rdb.BLPop(ctx, 5*time.Second, "notify:vm")
}

// getState 读取 vthread 当前状态
func getState(ctx context.Context, rdb *redis.Client, vtid string) VThreadState {
	val, err := rdb.Get(ctx, "/vthread/" + vtid).Result()
	if err != nil {
		return VThreadState{Status: "error"}
	}
	var state VThreadState
	json.Unmarshal([]byte(val), &state)
	return state
}

// setState 更新 vthread 的 PC 和 status
func setState(ctx context.Context, rdb *redis.Client, vtid string, pc, status string) {
	state := VThreadState{PC: pc, Status: status}
	data, _ := json.Marshal(state)
	rdb.Set(ctx, "/vthread/"+vtid, data, 0)
}

// setError 标记 vthread 为 error 状态
func setError(ctx context.Context, rdb *redis.Client, vtid string, pc string, errMsg string) {
	state := map[string]interface{}{
		"pc":     pc,
		"status": "error",
		"error":  map[string]string{"code": "VM_ERROR", "message": errMsg},
	}
	data, _ := json.Marshal(state)
	rdb.Set(ctx, "/vthread/"+vtid, data, 0)
}

// waitDone 阻塞等待 op-plat / heap-plat 完成通知
func waitDone(ctx context.Context, rdb *redis.Client, vtid string, timeout time.Duration) (map[string]interface{}, error) {
	doneKey := "done:" + vtid
	result, err := rdb.BLPop(ctx, timeout, doneKey).Result()
	if err != nil {
		return nil, err
	}
	var done map[string]interface{}
	if len(result) > 1 {
		json.Unmarshal([]byte(result[1]), &done)
	}
	return done, nil
}
