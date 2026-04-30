// loader — 加载 dxlang 源码到 Redis KV 空间
//
// 用法:
//
//	./loader <path> [redis_addr]
//	    加载 .dx 文件到 /src/func/<name>
//	    <path> 可以是 .dx 文件或包含 .dx 文件的目录
//	    默认 redis_addr: 127.0.0.1:16379
//
// 示例:
//
//	# 加载单个文件
//	./loader example/dxlang/lifecycle/full.dx
//
//	# 加载整个目录 (递归)
//	./loader example/dxlang/nn/
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"deepx/executor/vm/internal/ast"
	"deepx/executor/vm/internal/parser"
	"deepx/executor/vm/internal/logx"
	"github.com/redis/go-redis/v9"
)

func main() {
	args := os.Args[1:]
	if len(args) < 1 {
		printUsage()
		os.Exit(1)
	}

	path := args[0]
	redisAddr := "127.0.0.1:16379"
	if v := os.Getenv("REDIS_ADDR"); v != "" {
		redisAddr = v
	}
	if len(args) >= 2 {
		redisAddr = args[1]
	}

	rdb, ctx := connectRedis(redisAddr)
	defer rdb.Close()

	files, err := collectDxFiles(path)
	if err != nil {
		logx.Fatal("collect .dx files: %v", err)
	}
	if len(files) == 0 {
		logx.Fatal("no .dx files found in: %s", path)
	}

	logx.Info("found %d .dx file(s)", len(files))
	loaded := 0
	hasMain := false
	var allPreamble []string
	for _, f := range files {
		df, err := parser.ParseFile(f)
		if err != nil {
			logx.Warn("SKIP %s: %v", f, err)
			continue
		}

		// Register all function definitions
		for i := range df.Funcs {
			fn := &df.Funcs[i]
			if err := fn.Register(ctx, rdb); err != nil {
				logx.Error("FAIL %s: %v", f, err)
				continue
			}
			loaded++
			logx.Info("OK   %-50s → /src/func/%-30s (%d body lines)", f, fn.Name, len(fn.Body))
			if fn.Name == "main" {
				hasMain = true
			}
		}

		// Collect preamble lines from this file
		allPreamble = append(allPreamble, df.PreambleLines...)
	}

	// ── Build pre_main from all preamble lines ──
	if len(allPreamble) > 0 {
		body := make([]string, len(allPreamble))
		copy(body, allPreamble)
		if hasMain {
			body = append(body, "main() -> './pre_main_ret'")
		}
		preMain := ast.Func{
			Name:      "pre_main",
			Signature: "def pre_main() -> ()",
			Body:      body,
		}
		if err := preMain.Register(ctx, rdb); err != nil {
			logx.Error("FAIL register pre_main: %v", err)
		} else {
			entryMap := map[string]interface{}{
				"entry":  "pre_main",
				"reads":  []string{},
				"writes": []string{},
			}
			entryData, _ := json.Marshal(entryMap)
			if err := rdb.Set(ctx, "/func/main", entryData, 0).Err(); err != nil {
				logx.Error("FAIL write /func/main: %v", err)
			} else {
				logx.Info("PREMAIN %d preamble lines → /src/func/pre_main (hasMain=%v)", len(body), hasMain)
				logx.Info("ENTRY /func/main → pre_main")
			}
		}
	}
	logx.Info("loaded %d/%d functions into Redis", loaded, len(files))
}

func printUsage() {
	fmt.Fprint(os.Stderr, `loader — dxlang source loader for deepx KV space

USAGE:
  loader <path> [redis_addr]
      Load .dx file(s) into /src/func/
      <path> can be a .dx file or a directory containing .dx files

EXAMPLES:
  loader example/dxlang/lifecycle/full.dx
  loader example/dxlang/nn/
  REDIS_ADDR=127.0.0.1:6379 loader example/dxlang/
`)
}

func connectRedis(addr string) (*redis.Client, context.Context) {
	ctx := context.Background()
	rdb := redis.NewClient(&redis.Options{Addr: addr, PoolSize: 4, MinIdleConns: 1})
	if err := rdb.Ping(ctx).Err(); err != nil {
		logx.Fatal("Redis connect failed [%s]: %v", addr, err)
	}
	logx.Info("connected to Redis %s", addr)
	return rdb, ctx
}

// collectDxFiles returns all .dx files under path (single file or directory).
func collectDxFiles(path string) ([]string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}

	if !info.IsDir() {
		if strings.HasSuffix(path, ".dx") {
			return []string{path}, nil
		}
		return nil, fmt.Errorf("not a .dx file: %s", path)
	}

	var files []string
	err = filepath.Walk(path, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(p, ".dx") {
			files = append(files, p)
		}
		return nil
		})
	return files, err
}
