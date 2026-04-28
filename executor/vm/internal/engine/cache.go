package engine

import (
	"context"
	"fmt"
	"strings"

	"github.com/redis/go-redis/v9"
)

// SubStackCache 子栈指令本地缓存
// CALL 翻译后加载，RETURN 时释放
type SubStackCache struct {
	Root  string            // "/vthread/1/[2,0]/"
	Cells map[string]string // 相对 key → value, e.g., "[0,0]"→"matmul"
}

// LoadSubStack 从 Redis MGET 加载整个子栈到本地
func LoadSubStack(ctx context.Context, rdb *redis.Client, vtid string, pc string) *SubStackCache {
	root := fmt.Sprintf("/vthread/%s/%s/", vtid, pc)

	keys, err := rdb.Keys(ctx, root+"*").Result()
	if err != nil || len(keys) == 0 {
		return nil
	}

	vals, err := rdb.MGet(ctx, keys...).Result()
	if err != nil {
		return nil
	}

	cache := &SubStackCache{
		Root:  root,
		Cells: make(map[string]string, len(keys)),
	}

	for i, key := range keys {
		localKey := strings.TrimPrefix(key, root)
		if s, ok := vals[i].(string); ok {
			cache.Cells[localKey] = s
		}
	}

	return cache
}

// Get 从本地缓存读取指令坐标的值
func (c *SubStackCache) Get(addr0, addr1 int) string {
	return c.Cells[fmt.Sprintf("[%d,%d]", addr0, addr1)]
}
