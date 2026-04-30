// Package cmd provides shared utilities used by boot, run, and shutdown subcommands.
package cmd

import (
	"fmt"
	"os"
	"strings"

	"deepx/tool/deepxctl/internal/logx"
)

// ── Output helpers ──

func printHeader(redisAddr string) {
	logx.Debug("deepxctl run", "redis", redisAddr)
}

func printSeparator() {}

func step(n, total int, label string) {
	logx.Debug("step", "n", n, "total", total, "label", label)
}

func ok() {}

func okInline() {}

func greenCheck() {}

func errorX(format string, args ...interface{}) {
	fmt.Println("✗")
	fmt.Fprintf(os.Stderr, "\n─────────────────────────────────────────\n")
	fmt.Fprintf(os.Stderr, "ERROR  "+format+"\n", args...)
	fmt.Fprintf(os.Stderr, "─────────────────────────────────────────\n")
}

// splitRedisAddr splits "host:port" into host, port.
func splitRedisAddr(addr string) (host, port string) {
	host = "127.0.0.1"
	port = "16379"
	if idx := strings.LastIndex(addr, ":"); idx > 0 {
		host = addr[:idx]
		port = addr[idx+1:]
	}
	return
}
