package redis

import (
	"context"
	"strings"

	goredis "github.com/redis/go-redis/v9"
)

// ScanOpPlatInstances discovers all op-plat instance keys from /sys/op-plat/*.
func ScanOpPlatInstances(ctx context.Context, rdb *goredis.Client) []string {
	keys, _ := rdb.Keys(ctx, "/sys/op-plat/*").Result()
	return keys
}

// ScanHeapPlatInstances discovers all heap-plat instance keys from /sys/heap-plat/*.
func ScanHeapPlatInstances(ctx context.Context, rdb *goredis.Client) []string {
	keys, _ := rdb.Keys(ctx, "/sys/heap-plat/*").Result()
	return keys
}

// ScanVMInstance discovers a VM instance key from /sys/vm/* (excluding heartbeat keys).
func ScanVMInstance(ctx context.Context, rdb *goredis.Client) string {
	keys, err := rdb.Keys(ctx, "/sys/vm/*").Result()
	if err != nil {
		return ""
	}
	for _, k := range keys {
		if strings.Contains(k, "/heartbeat/") {
			continue
		}
		return k
	}
	return ""
}

// ScanTermInstance discovers a terminal instance key from /sys/term/* (excluding heartbeat keys).
func ScanTermInstance(ctx context.Context, rdb *goredis.Client) string {
	keys, err := rdb.Keys(ctx, "/sys/term/*").Result()
	if err != nil {
		return ""
	}
	for _, k := range keys {
		if strings.Contains(k, "/heartbeat/") {
			continue
		}
		return k
	}
	return ""
}

// HeartbeatKeyFromInstance converts an instance key to its heartbeat key.
// e.g., "/sys/op-plat/exop-metal:0" → "/sys/heartbeat/exop-metal:0"
//       "/sys/vm/0"             → "/sys/heartbeat/vm:0"
func HeartbeatKeyFromInstance(instKey string) string {
	parts := strings.Split(instKey, "/")
	if len(parts) < 3 {
		return ""
	}
	// parts[1] = "sys", parts[2] = "op-plat" | "heap-plat" | "vm"
	sysType := parts[2]
	// "/sys/op-plat/exop-metal:0" → "/sys/heartbeat/exop-metal:0"
	// "/sys/vm/0"              → "/sys/heartbeat/vm:0"
	if sysType == "op-plat" || sysType == "heap-plat" {
		id := parts[len(parts)-1] // e.g., "exop-metal:0"
		return "/sys/heartbeat/" + id
	}
	if sysType == "vm" {
		id := parts[len(parts)-1] // e.g., "0"
		return "/sys/heartbeat/vm:" + id
	}
	return ""
}
