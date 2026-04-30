// Package redis — WRITE operations.
//
// WARNING: All functions in this file MUTATE Redis state.
// They are ONLY called during explicit user "run" requests (POST /api/run).
// The status-monitoring code path NEVER touches these functions.
//
// Each write operation opens its own short-lived Redis connection,
// performs the write, and closes immediately. This ensures the
// dashboard's primary read connection remains read-only in practice.

package redis

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	goredis "github.com/redis/go-redis/v9"
)

// RunVThread loads functions via loader, then creates a vthread and wakes the VM.
// It opens a temporary write connection for INCR + SET + LPUSH.
//
// Returns the allocated vtid.
func RunVThread(redisAddr, entryFunc string) (int64, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	rdb := goredis.NewClient(&goredis.Options{
		Addr:         redisAddr,
		PoolSize:     1,
		MinIdleConns: 0,
	})
	defer rdb.Close()

	if err := rdb.Ping(ctx).Err(); err != nil {
		return 0, fmt.Errorf("write connection failed: %w", err)
	}

	// 1. Allocate vtid
	vtid, err := rdb.Incr(ctx, "/sys/vtid_counter").Result()
	if err != nil {
		return 0, fmt.Errorf("INCR /sys/vtid_counter: %w", err)
	}

	// 2. Create vthread
	base := fmt.Sprintf("/vthread/%d", vtid)
	status := fmt.Sprintf(`{"pc":"[0,0]","status":"init"}`)

	pipe := rdb.Pipeline()
	pipe.Set(ctx, base, status, 0)
	pipe.Set(ctx, base+"/[0,0]", entryFunc, 0)
	pipe.Set(ctx, base+"/[0,1]", "./ret", 0)
	if _, err := pipe.Exec(ctx); err != nil {
		return 0, fmt.Errorf("create vthread %d: %w", vtid, err)
	}

	// 3. Wake VM
	notify := map[string]interface{}{
		"event": "new_vthread",
		"vtid":  fmt.Sprintf("%d", vtid),
	}
	data, _ := json.Marshal(notify)
	if err := rdb.LPush(ctx, "notify:vm", data).Err(); err != nil {
		return 0, fmt.Errorf("LPUSH notify:vm: %w", err)
	}

	return vtid, nil
}
