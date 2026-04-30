// Package picker 负责原子拾取 status=init 的 vthread。
package picker

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"deepx/executor/vm/internal/state"
	"github.com/redis/go-redis/v9"
)

var errSkip = fmt.Errorf("skip") // 内部哨兵: 非 init 状态，跳过

// PickVthread 扫描 /vthread/*, 原子抢占 status=init 的 vthread。
// 返回 vtid，无可用 vthread 时返回空字符串。
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

		var s state.VThreadState
		if err := json.Unmarshal([]byte(val), &s); err != nil {
			return err
		}
		if s.Status != "init" {
			return errSkip
		}

		s.Status = "running"
		data, err := json.Marshal(s)
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
		log.Printf("tryPick %s error (vthread will be marked error): %v", vtid, err)
		// 标记 vthread 为 error 状态，避免被反复尝试
		errData, _ := json.Marshal(state.VThreadState{PC: "[0,0]", Status: "error"})
		rdb.Set(ctx, key, errData, 0)
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

// WaitForVthread 阻塞等待新 vthread 创建通知。
func WaitForVthread(ctx context.Context, rdb *redis.Client) {
	rdb.BLPop(ctx, 5*time.Second, "notify:vm")
}
