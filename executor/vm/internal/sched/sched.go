// Package sched 负责原子拾取和等待虚线程。
package sched

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"deepx/executor/vm/internal/logx"
	"deepx/executor/vm/internal/state"
	"github.com/redis/go-redis/v9"
)

// Pick 扫描 /vthread/*，原子抢占 status=init 的 vthread。返回 vtid 或空串。
func Pick(ctx context.Context, rdb *redis.Client) string {
	keys, err := rdb.Keys(ctx, "/vthread/*").Result()
	if err != nil {
		logx.Debug("picker KEYS error: %v", err)
		return ""
	}
	for _, key := range keys {
		vtid := key[len("/vthread/"):]
		// Skip non-numeric vtid (e.g., system keys nested under /vthread/)
		// 实际上 key pattern `/vthread/*` 不会匹配 `/vthread/1/sub`，但还是做一次 GET 检查
		val, err := rdb.Get(ctx, key).Result()
		if err != nil {
			continue
		}
		var s state.VThreadState
		if json.Unmarshal([]byte(val), &s) != nil {
			continue
		}
		if s.Status != "init" {
			continue
		}
		// 原子 CAS: set status to "running" if it's still "init"
		updated := state.VThreadState{PC: s.PC, Status: "running", Mode: s.Mode}
		data, _ := json.Marshal(updated)
		// Lua script for atomic compare-and-set
		script := `
			local key = KEYS[1]
			local val = redis.call('GET', key)
			if not val then return 0 end
			local decoded = cjson.decode(val)
			if decoded.status ~= 'init' then return 0 end
			redis.call('SET', key, ARGV[1])
			return 1
		`
		result, err := rdb.Eval(ctx, script, []string{key}, string(data)).Int64()
		if err != nil || result == 0 {
			continue
		}
		return vtid
	}
	return ""
}

// Wait 阻塞等待新的 vthread 通知 (BLPOP notify:vm)。
func Wait(ctx context.Context, rdb *redis.Client) {
	vals, err := rdb.BLPop(ctx, 5*time.Second, "notify:vm").Result()
	if err != nil {
		if err.Error() != "redis: nil" {
			logx.Debug("picker BLPOP: %v", err)
		}
		return
	}
	if len(vals) >= 2 {
		var notify struct {
			Event string `json:"event"`
			Vtid  string `json:"vtid"`
		}
		if json.Unmarshal([]byte(vals[1]), &notify) == nil {
			logx.Debug("notify: %s vtid=%s", notify.Event, notify.Vtid)
		}
	}
}

var _ = fmt.Println // keep fmt import
