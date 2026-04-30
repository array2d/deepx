// Package redis — READ-ONLY operations.
package redis

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	goredis "github.com/redis/go-redis/v9"
)

const DefaultAddr = "127.0.0.1:16379"

// ── Connection (read-only by contract) ──

func Connect(addr string) (*goredis.Client, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	rdb := goredis.NewClient(&goredis.Options{
		Addr:         addr,
		PoolSize:     8,
		MinIdleConns: 2,
	})
	if err := rdb.Ping(ctx).Err(); err != nil {
		rdb.Close()
		return nil, fmt.Errorf("redis PING failed [%s]: %w", addr, err)
	}
	return rdb, nil
}

// ── System Status (read-only) ──

type ComponentStatus struct {
	Id             string `json:"id"`
	Status         string `json:"status"`
	Pid            int    `json:"pid"`
	HeartbeatAgeMs int64  `json:"heartbeat_age_ms,omitempty"`
	Load           string `json:"load,omitempty"`
	Device         string `json:"device,omitempty"`
	Program        string `json:"program,omitempty"`
	StartedAt      int64  `json:"started_at,omitempty"`
}

type SystemStatus struct {
	OpPlat   []ComponentStatus `json:"op_plat"`
	HeapPlat []ComponentStatus `json:"heap_plat"`
	VM       *ComponentStatus  `json:"vm"`
	Term     *ComponentStatus  `json:"term"`
	Functions struct {
		Count int      `json:"count"`
		Names []string `json:"names"`
	} `json:"functions"`
	VThreads struct {
		Total    int            `json:"total"`
		ByStatus map[string]int `json:"by_status"`
	} `json:"vthreads"`
	OpRegistry map[string]int `json:"op_registry"`
	DBSize     int64           `json:"dbsize"`
}

func GetSystemStatus(rdb *goredis.Client) (*SystemStatus, error) {
	ctx := context.Background()
	now := time.Now()
	s := &SystemStatus{
		OpPlat:   []ComponentStatus{},
		HeapPlat: []ComponentStatus{},
	}

	for _, key := range ScanOpPlatInstances(ctx, rdb) {
		cs := readComponent(ctx, rdb, key, now)
		if cs == nil {
			continue
		}
		cs.Id = instanceId(key)
		if age := heartbeatAge(ctx, rdb, HeartbeatKeyFromInstance(key), now); age >= 0 {
			cs.HeartbeatAgeMs = age
		}
		s.OpPlat = append(s.OpPlat, *cs)
	}

	for _, key := range ScanHeapPlatInstances(ctx, rdb) {
		cs := readComponent(ctx, rdb, key, now)
		if cs == nil {
			continue
		}
		cs.Id = instanceId(key)
		if age := heartbeatAge(ctx, rdb, HeartbeatKeyFromInstance(key), now); age >= 0 {
			cs.HeartbeatAgeMs = age
		}
		s.HeapPlat = append(s.HeapPlat, *cs)
	}

	vmKey := ScanVMInstance(ctx, rdb)
	if vmKey != "" {
		s.VM = readComponent(ctx, rdb, vmKey, now)
		if s.VM != nil {
			s.VM.Id = instanceId(vmKey)
			if age := heartbeatAge(ctx, rdb, HeartbeatKeyFromInstance(vmKey), now); age >= 0 {
				s.VM.HeartbeatAgeMs = age
			}
		}
	}

	// Terminal: /sys/term/*
	termKey := ScanTermInstance(ctx, rdb)
	if termKey != "" {
		s.Term = readComponent(ctx, rdb, termKey, now)
		if s.Term != nil {
			s.Term.Id = instanceId(termKey)
			if age := heartbeatAge(ctx, rdb, "/sys/heartbeat/term:dashboard", now); age >= 0 {
				s.Term.HeartbeatAgeMs = age
			}
		}
	}

	names, _ := SrcFuncNames(rdb)
	s.Functions.Count = len(names)
	s.Functions.Names = names
	if s.Functions.Names == nil {
		s.Functions.Names = []string{}
	}

	s.VThreads.ByStatus = make(map[string]int)
	keys, _ := rdb.Keys(ctx, "/vthread/*").Result()
	for _, k := range keys {
		if strings.Count(k, "/") == 2 && !strings.Contains(k, "[") {
			s.VThreads.Total++
			val, err := rdb.Get(ctx, k).Result()
			if err != nil {
				continue
			}
			var vt struct {
				Status string `json:"status"`
			}
			if json.Unmarshal([]byte(val), &vt) == nil {
				s.VThreads.ByStatus[vt.Status]++
			}
		}
	}

	s.OpRegistry = make(map[string]int)
	opKeys, _ := rdb.Keys(ctx, "/op/*/list").Result()
	for _, key := range opKeys {
		parts := strings.Split(key, "/")
		if len(parts) < 3 {
			continue
		}
		backend := parts[2]
		count := rdb.LLen(ctx, key).Val()
		if count > 0 || rdb.Exists(ctx, key).Val() > 0 {
			s.OpRegistry[backend] = int(count)
		}
	}

	s.DBSize, _ = rdb.DBSize(ctx).Result()
	return s, nil
}

func instanceId(key string) string {
	parts := strings.Split(key, "/")
	if len(parts) == 0 {
		return ""
	}
	return parts[len(parts)-1]
}

func readComponent(ctx context.Context, rdb *goredis.Client, key string, now time.Time) *ComponentStatus {
	val, err := rdb.Get(ctx, key).Result()
	if err != nil {
		return nil
	}
	var m map[string]interface{}
	if err := json.Unmarshal([]byte(val), &m); err != nil {
		return nil
	}
	cs := &ComponentStatus{}
	if s, ok := m["status"].(string); ok {
		cs.Status = s
	}
	if pid, ok := m["pid"].(float64); ok {
		cs.Pid = int(pid)
	}
	if l, ok := m["load"].(float64); ok {
		cs.Load = fmt.Sprintf("%.2f", l)
	}
	if d, ok := m["device"].(string); ok {
		cs.Device = d
	}
	if p, ok := m["program"].(string); ok {
		cs.Program = p
	}
	if sa, ok := m["started_at"].(float64); ok {
		cs.StartedAt = int64(sa)
	}
	return cs
}

func heartbeatAge(ctx context.Context, rdb *goredis.Client, key string, now time.Time) int64 {
	val, err := rdb.Get(ctx, key).Result()
	if err != nil {
		return -1
	}
	var hb struct {
		Ts int64 `json:"ts"`
	}
	if json.Unmarshal([]byte(val), &hb) != nil {
		return -1
	}
	if hb.Ts == 0 {
		return -1
	}
	return now.UnixMilli() - hb.Ts*1000
}

// ── Functions (read-only) ──

func SrcFuncNames(rdb *goredis.Client) ([]string, error) {
	ctx := context.Background()
	keys, err := rdb.Keys(ctx, "/src/func/*").Result()
	if err != nil {
		return nil, err
	}
	seen := make(map[string]bool)
	var names []string
	for _, k := range keys {
		rest := strings.TrimPrefix(k, "/src/func/")
		if idx := strings.Index(rest, "/"); idx >= 0 {
			rest = rest[:idx]
		}
		if rest != "" && !seen[rest] {
			seen[rest] = true
			names = append(names, rest)
		}
	}
	return names, nil
}

// ── VThread (read-only) ──

type VThreadInfo struct {
	Vtid      int64  `json:"vtid"`
	Status    string `json:"status"`
	PC        string `json:"pc"`
	CreatedAt int64  `json:"created_at"`
	Error     *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
	DurationMs int64 `json:"duration_ms,omitempty"`
}

func GetVThread(rdb *goredis.Client, vtid int64) (*VThreadInfo, error) {
	ctx := context.Background()
	key := fmt.Sprintf("/vthread/%d", vtid)
	val, err := rdb.Get(ctx, key).Result()
	if err != nil {
		return nil, fmt.Errorf("vthread %d not found", vtid)
	}
	var raw struct {
		PC     string `json:"pc"`
		Status string `json:"status"`
		Error  *struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error,omitempty"`
	}
	if err := json.Unmarshal([]byte(val), &raw); err != nil {
		return nil, fmt.Errorf("parse vthread %d: %w", vtid, err)
	}
	return &VThreadInfo{
		Vtid:   vtid,
		Status: raw.Status,
		PC:     raw.PC,
		Error:  raw.Error,
	}, nil
}

func ListVThreads(rdb *goredis.Client) ([]VThreadInfo, error) {
	ctx := context.Background()
	keys, err := rdb.Keys(ctx, "/vthread/*").Result()
	if err != nil {
		return nil, err
	}
	var vts []VThreadInfo
	for _, k := range keys {
		if strings.Count(k, "/") != 2 || strings.Contains(k, "[") {
			continue
		}
		var vtid int64
		fmt.Sscanf(k, "/vthread/%d", &vtid)
		val, err := rdb.Get(ctx, k).Result()
		if err != nil {
			continue
		}
		var raw struct {
			PC     string `json:"pc"`
			Status string `json:"status"`
		}
		if json.Unmarshal([]byte(val), &raw) != nil {
			continue
		}
		vts = append(vts, VThreadInfo{
			Vtid:   vtid,
			Status: raw.Status,
			PC:     raw.PC,
		})
	}
	return vts, nil
}
