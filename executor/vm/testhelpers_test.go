package vm_test

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
)

// connectRedisIntegration connects to Redis for integration tests.
// Uses REDIS_ADDR env or defaults to 127.0.0.1:6379.
func connectRedisIntegration(t *testing.T) (*redis.Client, context.Context) {
	t.Helper()
	addr := os.Getenv("REDIS_ADDR")
	if addr == "" {
		addr = "127.0.0.1:16379"
	}
	ctx := context.Background()
	rdb := redis.NewClient(&redis.Options{Addr: addr, PoolSize: 10, MinIdleConns: 2})
	if err := rdb.Ping(ctx).Err(); err != nil {
		t.Fatalf("Redis not available at %s: %v (set REDIS_ADDR or start Redis)", addr, err)
	}
	rdb.FlushDB(ctx)
	return rdb, ctx
}

// waitVthreadDone polls the vthread state until it reaches "done" or "error".
// Returns named slot values on success.
func waitVthreadDone(t *testing.T, rdb *redis.Client, vtid string, timeout time.Duration) (map[string]string, bool) {
	t.Helper()
	ctx := context.Background()
	ticker := time.NewTicker(30 * time.Millisecond)
	defer ticker.Stop()
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		<-ticker.C
		val, err := rdb.Get(ctx, "/vthread/"+vtid).Result()
		if err == redis.Nil {
			continue
		}
		if err != nil {
			continue
		}
		var s struct {
			Status string            `json:"status"`
			PC     string            `json:"pc"`
			Error  map[string]string `json:"error,omitempty"`
		}
		json.Unmarshal([]byte(val), &s)

		switch s.Status {
		case "done":
			// read named slots
			keys, _ := rdb.Keys(ctx, "/vthread/"+vtid+"/*").Result()
			outputs := make(map[string]string)
			prefix := "/vthread/" + vtid + "/"
			for _, k := range keys {
				if v, err := rdb.Get(ctx, k).Result(); err == nil {
					slot := k[len(prefix):]
					if len(slot) > 0 && slot[0] != '[' {
						outputs[slot] = v
					}
				}
			}
			return outputs, true
		case "error":
			t.Logf("vtid=%s error: %v", vtid, s.Error)
			return nil, false
		}
	}
	t.Logf("vtid=%s timeout after %v", vtid, timeout)
	return nil, false
}