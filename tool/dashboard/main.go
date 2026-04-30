package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"deepx/tool/dashboard/internal/redis"
	"deepx/tool/dashboard/internal/server"
)

func main() {
	addr := flag.String("addr", ":8080", "HTTP listen address")
	redisAddr := flag.String("redis", redis.DefaultAddr, "Redis address")
	loaderBin := flag.String("loader", "/tmp/deepx/vm/loader", "Path to loader binary")
	flag.Parse()

	rdb, err := redis.Connect(*redisAddr)
	if err != nil {
		log.Fatalf("Redis connection failed: %v", err)
	}
	defer rdb.Close()
	log.Printf("Connected to Redis: %s", *redisAddr)

	srv := server.New(rdb, *loaderBin, *redisAddr, *addr)
	httpServer := &http.Server{Addr: *addr, Handler: srv.Handler()}

	// Graceful shutdown
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Println("Shutting down...")
		httpServer.Close()
	}()

	fmt.Printf("\n  dashboard  |  http://localhost%s  |  redis: %s\n", *addr, *redisAddr)
	fmt.Println("  ─────────────────────────────────────────")
	fmt.Println()
	if err := httpServer.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("HTTP server: %v", err)
	}
}
