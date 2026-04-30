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
	"time"

	goredis "github.com/redis/go-redis/v9"

	"deepx/tool/deepxctl/internal/logx"
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
	logx.Debug("redis connected", "addr", addr)
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
	logx.Debug("FLUSHDB done", "dbsize", size)
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
			logx.Debug("redis instance running", "key", key)
			return nil
		}
		time.Sleep(200 * time.Millisecond)
	}
	return fmt.Errorf("timeout waiting for %s to be running (%.0fs)", key, timeout.Seconds())
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

