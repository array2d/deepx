// Package redis provides Redis connection, FLUSHDB, and system key status checks
// for deepxctl process orchestration.
//
// Allowed operations (per doc/deepxctl/CLAUDE.md):
//
//	PING, FLUSHDB, DBSIZE
//	GET /sys/op-plat/*, /sys/heap-plat/*, /sys/vm/*
//	GET /vthread/<vtid> (status polling)
//	SET /vthread/<vtid> (vthread creation)
//	SET /vthread/<vtid>/[0,0], /vthread/<vtid>/[0,1] (entry CALL)
//	INCR /sys/vtid_counter
//	LPUSH notify:vm
//	GET /src/func/<name> (verification)
//	KEYS /src/func/* (function listing)
package redis

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	goredis "github.com/redis/go-redis/v9"
)

// DefaultAddr is the default Redis address for development.
const DefaultAddr = "127.0.0.1:16379"

// Connect dials Redis with a short timeout and verifies with PING.
func Connect(addr string) (*goredis.Client, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	rdb := goredis.NewClient(&goredis.Options{
		Addr:        addr,
		PoolSize:    4,
		MinIdleConns: 1,
	})
	if err := rdb.Ping(ctx).Err(); err != nil {
		rdb.Close()
		return nil, fmt.Errorf("redis PING failed [%s]: %w", addr, err)
	}
	log.Printf("[redis] connected to %s", addr)
	return rdb, nil
}

// FlushDB resets the current Redis database.
// Only call this in development (port 16379).
func FlushDB(rdb *goredis.Client) error {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	if err := rdb.FlushDB(ctx).Err(); err != nil {
		return fmt.Errorf("FLUSHDB failed: %w", err)
	}
	// Verify
	size, err := rdb.DBSize(ctx).Result()
	if err != nil {
		return fmt.Errorf("DBSIZE after FLUSHDB failed: %w", err)
	}
	log.Printf("[redis] FLUSHDB done, dbsize=%d", size)
	return nil
}

// WaitForInstance polls a /sys/ key until it contains status="running" or timeout.
func WaitForInstance(rdb *goredis.Client, key string, timeout time.Duration) error {
	ctx := context.Background()
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		val, err := rdb.Get(ctx, key).Result()
		if err != nil {
			time.Sleep(200 * time.Millisecond)
			continue
		}
		var m map[string]interface{}
		if err := json.Unmarshal([]byte(val), &m); err != nil {
			time.Sleep(200 * time.Millisecond)
			continue
		}
		if s, ok := m["status"].(string); ok && s == "running" {
			log.Printf("[redis] %s is running", key)
			return nil
		}
		time.Sleep(200 * time.Millisecond)
	}
	return fmt.Errorf("timeout waiting for %s to be running (%.0fs)", key, timeout.Seconds())
}

// AllocVtid atomically increments the vthread counter and returns the new ID.
func AllocVtid(rdb *goredis.Client) (int64, error) {
	ctx := context.Background()
	id, err := rdb.Incr(ctx, "/sys/vtid_counter").Result()
	if err != nil {
		return 0, fmt.Errorf("INCR /sys/vtid_counter: %w", err)
	}
	return id, nil
}

// VThreadStatus represents the status of a vthread from Redis.
type VThreadStatus struct {
	PC     string `json:"pc"`
	Status string `json:"status"`
	Error  *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// GetVThreadStatus reads the vthread JSON from Redis.
func GetVThreadStatus(rdb *goredis.Client, vtid int64) (*VThreadStatus, error) {
	ctx := context.Background()
	key := fmt.Sprintf("/vthread/%d", vtid)
	val, err := rdb.Get(ctx, key).Result()
	if err != nil {
		return nil, fmt.Errorf("GET %s: %w", key, err)
	}
	var s VThreadStatus
	if err := json.Unmarshal([]byte(val), &s); err != nil {
		return nil, fmt.Errorf("parse %s: %w", key, err)
	}
	return &s, nil
}

// CreateVThread writes a vthread with a single top-level CALL instruction.
//
// Writes:
//
//	/vthread/<vtid>          = {"pc":"[0,0]","status":"init"}
//	/vthread/<vtid>/[0,0]    = "<entryFunc>"
//	/vthread/<vtid>/[0,1]    = "./ret"
func CreateVThread(rdb *goredis.Client, vtid int64, entryFunc string) error {
	ctx := context.Background()
	base := fmt.Sprintf("/vthread/%d", vtid)

	status := fmt.Sprintf(`{"pc":"[0,0]","status":"init"}`)

	pipe := rdb.Pipeline()
	pipe.Set(ctx, base, status, 0)
	pipe.Set(ctx, base+"/[0,0]", entryFunc, 0)
	pipe.Set(ctx, base+"/[0,1]", "./ret", 0)
	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("create vthread %d: %w", vtid, err)
	}
	log.Printf("[redis] created vthread %d entry=%s", vtid, entryFunc)
	return nil
}

// WakeVM pushes a new_vthread notification to the VM wake queue.
func WakeVM(rdb *goredis.Client, vtid int64) error {
	ctx := context.Background()
	notify := map[string]interface{}{
		"event": "new_vthread",
		"vtid":  fmt.Sprintf("%d", vtid),
	}
	data, _ := json.Marshal(notify)
	if err := rdb.LPush(ctx, "notify:vm", data).Err(); err != nil {
		return fmt.Errorf("LPUSH notify:vm: %w", err)
	}
	log.Printf("[redis] notified VM: vtid=%d", vtid)
	return nil
}

// SrcFuncKeys returns all registered function names under /src/func/.
func SrcFuncKeys(rdb *goredis.Client) ([]string, error) {
	ctx := context.Background()
	keys, err := rdb.Keys(ctx, "/src/func/*").Result()
	if err != nil {
		return nil, err
	}
	// Filter out sub-keys like /src/func/name/0, return unique names
	seen := make(map[string]bool)
	var names []string
	for _, k := range keys {
		// /src/func/name    → name
		// /src/func/name/0  → name
		name := k
		if len(k) > 11 { // len("/src/func/")
			rest := k[10:] // after "/src/func/"
			// find first /
			for i, c := range rest {
				if c == '/' {
					name = "/src/func/" + rest[:i]
					break
				}
			}
		}
		if !seen[name] {
			seen[name] = true
			names = append(names, name[len("/src/func/"):])
		}
	}
	return names, nil
}

// SrcFuncExists returns true if /src/func/<name> exists (non-empty).
func SrcFuncExists(rdb *goredis.Client, name string) bool {
	ctx := context.Background()
	val, err := rdb.Get(ctx, "/src/func/"+name).Result()
	if err != nil {
		return false
	}
	return val != ""
}
