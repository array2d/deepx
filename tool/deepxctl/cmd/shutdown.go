// Package cmd implements the "shutdown" subcommand for deepxctl.
//
//	deepxctl shutdown
//
// Ordered shutdown via Redis system commands:
//  1. plats (op-metal, heap-metal) — send sys:shutdown, wait for stopped heartbeat
//  2. VM — send sys:shutdown, wait for stopped heartbeat
//  3. Verify all heartbeats, log final values
//  4. Clean up PID file
//
// OS SIGKILL is only used as last-resort fallback if Redis is unreachable.
package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"syscall"
	"time"

	goredis "github.com/redis/go-redis/v9"

	"deepx/tool/deepxctl/internal/redis"
)

// Shutdown is the entry point for the "shutdown" subcommand.
func Shutdown() {
	if err := ExecShutdown(); err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %v\n", err)
		os.Exit(1)
	}

	fmt.Println()
	printSeparator()
	fmt.Println("Shutdown complete. All services stopped.")
	printSeparator()
}

// ExecShutdown performs the ordered shutdown of all booted services.
// It sends sys:shutdown commands via Redis (with OS signal fallback) and
// removes the PID file. Exported so the run command can reuse it with --rm.
func ExecShutdown() error {
	return shutdown()
}

// heartbeatVal represents a heartbeat entry from Redis.
type heartbeatVal struct {
	Ts     int64  `json:"ts"`
	Status string `json:"status"`
	Pid    int    `json:"pid"`
}

type platInfo struct {
	name     string
	sysQueue string
	hbKey    string
	pid      int
}

func shutdown() error {
	state, err := ReadBootState()
	if err != nil {
		return fmt.Errorf("read boot state: %w", err)
	}
	if state == nil {
		fmt.Println("No boot state found. Nothing to shut down.")
		fmt.Printf("(expected %s — was 'deepxctl boot' run?)\n", BootPIDFile)
		return nil
	}

	fmt.Printf("Ordered shutdown via Redis sys commands (redis: %s)\n", state.RedisAddr)

	// Connect to Redis
	rdb, err := redis.Connect(state.RedisAddr)
	if err != nil {
		fmt.Printf("  Redis not reachable (%v), falling back to OS signals...\n", err)
		return forceKill(state)
	}
	defer rdb.Close()

	ctx := context.Background()
	shutdownCmd, _ := json.Marshal(map[string]string{"cmd": "shutdown"})

	// ═══════════════════════════════════════════════════════════════
	// Phase 1: Shutdown plats (op-metal → heap-metal)
	// ═══════════════════════════════════════════════════════════════
	fmt.Println("\n── Phase 1: Stopping plats (op-metal, heap-metal) ──")

	plats := []platInfo{
		{"op-metal", "sys:cmd:op-metal:0", "/sys/heartbeat/op-metal:0", state.OpMetal},
		{"heap-metal", "sys:cmd:heap-metal:0", "/sys/heartbeat/heap-metal:0", state.HeapMetal},
	}

	for _, p := range plats {
		if !pidAlive(p.pid) {
			fmt.Printf("  %-15s pid=%-6d already stopped\n", p.name, p.pid)
			continue
		}
		fmt.Printf("  %-15s pid=%-6d sending sys:shutdown → %s...", p.name, p.pid, p.sysQueue)
		if err := rdb.LPush(ctx, p.sysQueue, shutdownCmd).Err(); err != nil {
			fmt.Printf(" LPUSH failed: %v\n", err)
		} else {
			fmt.Println(" sent")
		}
	}

	// Wait for plats heartbeats to show "stopped"
	fmt.Print("  waiting for plats to stop...")
	if !waitHeartbeats(rdb, plats, 10*time.Second) {
		fmt.Println(" timeout")
	} else {
		fmt.Println(" done")
	}

	// ═══════════════════════════════════════════════════════════════
	// Phase 2: Shutdown VM
	// ═══════════════════════════════════════════════════════════════
	fmt.Println("\n── Phase 2: Stopping VM ──")

	vmPlats := []platInfo{
		{"vm", "sys:cmd:vm:0", "/sys/heartbeat/vm:0", state.VM},
	}

	if !pidAlive(state.VM) {
		fmt.Printf("  VM              pid=%-6d already stopped\n", state.VM)
	} else {
		fmt.Printf("  VM              pid=%-6d sending sys:shutdown → sys:cmd:vm:0...", state.VM)
		if err := rdb.LPush(ctx, "sys:cmd:vm:0", shutdownCmd).Err(); err != nil {
			fmt.Printf(" LPUSH failed: %v\n", err)
		} else {
			fmt.Println(" sent")
		}

		fmt.Print("  waiting for VM to stop...")
		if !waitHeartbeats(rdb, vmPlats, 10*time.Second) {
			fmt.Println(" timeout")
		} else {
			fmt.Println(" done")
		}
	}

	// ═══════════════════════════════════════════════════════════════
	// Phase 3: Verify all final heartbeats
	// ═══════════════════════════════════════════════════════════════
	fmt.Println("\n── Phase 3: Final heartbeat verification ──")

	allHbKeys := []string{
		"/sys/heartbeat/op-metal:0",
		"/sys/heartbeat/heap-metal:0",
		"/sys/heartbeat/vm:0",
	}

	for _, key := range allHbKeys {
		val, err := rdb.Get(ctx, key).Result()
		if err != nil {
			fmt.Printf("  %-40s ── (cleaned)\n", key)
			continue
		}
		var hb heartbeatVal
		if err := json.Unmarshal([]byte(val), &hb); err != nil {
			fmt.Printf("  %-40s parse error: %v\n", key, err)
			continue
		}
		ts := time.Unix(hb.Ts, 0).Format("15:04:05")
		icon := "✓"
		if hb.Status != "stopped" {
			icon = "✗"
		}
		fmt.Printf("  %s %-40s status=%-8s pid=%-6d ts=%s\n", icon, key, hb.Status, hb.Pid, ts)
	}

	// ═══════════════════════════════════════════════════════════════
	// Grace period — lets processes finish exiting after heartbeat stop
	// ═══════════════════════════════════════════════════════════════
	time.Sleep(500 * time.Millisecond)

	// ═══════════════════════════════════════════════════════════════
	// Phase 4: Force kill any remaining processes (fallback)
	// ═══════════════════════════════════════════════════════════════
	needForce := false
	for _, r := range []struct {
		name string
		pid  int
	}{
		{"op-metal", state.OpMetal},
		{"heap-metal", state.HeapMetal},
		{"vm", state.VM},
	} {
		if pidAlive(r.pid) {
			needForce = true
			break
		}
	}

	if needForce {
		fmt.Println("\n── Force killing remaining processes (fallback) ──")
		for _, r := range []struct {
			name string
			pid  int
		}{
			{"op-metal", state.OpMetal},
			{"heap-metal", state.HeapMetal},
			{"vm", state.VM},
		} {
			if pidAlive(r.pid) {
				fmt.Printf("  %-15s pid=%-6d SIGKILL...", r.name, r.pid)
				syscall.Kill(r.pid, syscall.SIGKILL)
				time.Sleep(100 * time.Millisecond)
				if pidAlive(r.pid) {
					fmt.Println(" still alive!")
				} else {
					fmt.Println(" killed")
				}
			}
		}
	}

	// Remove PID file
	if err := os.Remove(BootPIDFile); err != nil && !os.IsNotExist(err) {
		log.Printf("[shutdown] could not remove %s: %v", BootPIDFile, err)
	} else {
		log.Printf("[shutdown] removed %s", BootPIDFile)
	}

	return nil
}

// waitHeartbeats polls heartbeat keys until all show "stopped" or PID dies, or timeout.
func waitHeartbeats(rdb *goredis.Client, plats []platInfo, timeout time.Duration) bool {
	ctx := context.Background()
	deadline := time.Now().Add(timeout)

	remaining := make(map[string]bool)
	for _, p := range plats {
		if pidAlive(p.pid) {
			remaining[p.name] = true
		}
	}
	if len(remaining) == 0 {
		return true
	}

	for len(remaining) > 0 && time.Now().Before(deadline) {
		for _, p := range plats {
			if !remaining[p.name] {
				continue
			}
			// Check 1: PID dead = component exited
			if !pidAlive(p.pid) {
				delete(remaining, p.name)
				continue
			}
			// Check 2: Heartbeat shows "stopped"
			val, err := rdb.Get(ctx, p.hbKey).Result()
			if err != nil {
				continue
			}
			var hb heartbeatVal
			if json.Unmarshal([]byte(val), &hb) == nil && hb.Status == "stopped" {
				delete(remaining, p.name)
			}
		}
		if len(remaining) > 0 {
			time.Sleep(300 * time.Millisecond)
		}
	}
	return len(remaining) == 0
}

// forceKill sends SIGTERM → wait → SIGKILL to all booted processes.
// Used as fallback when Redis is unreachable.
func forceKill(state *BootState) error {
	pids := map[string]int{
		"op-metal":   state.OpMetal,
		"heap-metal": state.HeapMetal,
		"vm":         state.VM,
	}

	for name, pid := range pids {
		if !pidAlive(pid) {
			fmt.Printf("  %-15s pid=%-6d already stopped\n", name, pid)
			continue
		}
		fmt.Printf("  %-15s pid=%-6d SIGTERM...", name, pid)
		syscall.Kill(pid, syscall.SIGTERM)
		if waitPID(pid, 5*time.Second) {
			fmt.Println(" stopped")
			continue
		}
		fmt.Print(" SIGKILL...")
		syscall.Kill(pid, syscall.SIGKILL)
		time.Sleep(200 * time.Millisecond)
		if pidAlive(pid) {
			fmt.Println(" still alive!")
		} else {
			fmt.Println(" killed")
		}
	}

	os.Remove(BootPIDFile)
	return nil
}

// pidFromState returns the PID for a named component from boot state.
func pidFromState(state *BootState, name string) int {
	switch name {
	case "op-metal":
		return state.OpMetal
	case "heap-metal":
		return state.HeapMetal
	case "vm":
		return state.VM
	}
	return -1
}

// waitPID polls until the process exits or timeout elapses.
func waitPID(pid int, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if !pidAlive(pid) {
			return true
		}
		time.Sleep(200 * time.Millisecond)
	}
	return false
}
