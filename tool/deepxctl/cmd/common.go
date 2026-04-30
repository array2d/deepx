// Package cmd provides shared utilities used by boot, run, and shutdown subcommands.
package cmd

import (
	"fmt"
	"os"
	"strings"
)

// ── Output helpers ──

func printHeader(redisAddr string) {
	fmt.Println()
	fmt.Printf(" deepxctl  |  redis: %s\n", redisAddr)
	printSeparator()
	fmt.Println()
}

func printSeparator() {
	fmt.Println("─────────────────────────────────────────")
}

func step(n, total int, label string) {
	fmt.Printf("[%d/%d] %-28s", n, total, label)
}

func ok() {
	fmt.Println("✓")
}

func okInline() {
	fmt.Println("✓")
}

func greenCheck() {
	fmt.Print("  ✓  ")
}

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
