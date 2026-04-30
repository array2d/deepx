// Package cmd implements the "boot" subcommand for deepxctl.
//
//	deepxctl boot [flags]
//
// Boots the full deepx runtime: Redis reset → build → launch op-metal + heap-metal + VM.
// Writes PID state to /tmp/deepx-boot.json for later shutdown.
package cmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"syscall"
	"time"

	"deepx/tool/deepxctl/internal/builder"
	"deepx/tool/deepxctl/internal/logx"
	"deepx/tool/deepxctl/internal/process"
	"deepx/tool/deepxctl/internal/redis"
)

// BootPIDFile is the path where boot writes process PIDs.
const BootPIDFile = "/tmp/deepx-boot.json"

// BootState holds the PIDs of booted services.
type BootState struct {
	OpMetal   int    `json:"op-metal"`
	HeapMetal int    `json:"heap-metal"`
	VM        int    `json:"vm"`
	RedisAddr string `json:"redis_addr"`
}

// BootFlags holds the parsed flags for the boot command.
type BootFlags struct {
	RedisAddr  string
	ForceBuild bool
	NoReset    bool
	Verbose    bool
}

// Boot is the entry point for the "boot" subcommand.
func Boot(args []string) {
	flags := parseBootFlags(args)

	if err := boot(flags); err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %v\n", err)
		os.Exit(1)
	}

	fmt.Println()
	printSeparator()
	fmt.Println("Boot complete. Services are running.")
	fmt.Printf("PID file: %s\n", BootPIDFile)
	fmt.Println("Run 'deepxctl run <file.dx>' to execute, 'deepxctl shutdown' to stop.")
	printSeparator()
}

func parseBootFlags(args []string) BootFlags {
	fs := flag.NewFlagSet("boot", flag.ExitOnError)

	var flags BootFlags
	fs.StringVar(&flags.RedisAddr, "r", redis.DefaultAddr, "Redis address")
	fs.StringVar(&flags.RedisAddr, "redis", redis.DefaultAddr, "Redis address")
	fs.BoolVar(&flags.ForceBuild, "b", false, "Force rebuild all binaries")
	fs.BoolVar(&flags.ForceBuild, "build", false, "Force rebuild all binaries")
	fs.BoolVar(&flags.NoReset, "no-reset", false, "Skip Redis FLUSHDB")
	fs.BoolVar(&flags.Verbose, "v", false, "Verbose output")
	fs.BoolVar(&flags.Verbose, "verbose", false, "Verbose output")

	fs.Parse(args)
	return flags
}

func boot(flags BootFlags) error {
	printHeader(flags.RedisAddr)

	repoRoot, err := builder.RepoRoot()
	if err != nil {
		return fmt.Errorf("find repo root: %w", err)
	}

	// ── [1/3] Redis ──
	step(1, 3, "Redis")
	rdb, err := redis.Connect(flags.RedisAddr)
	if err != nil {
		errorX("Redis connection failed: %v", err)
		return err
	}
	defer rdb.Close()

	if !flags.NoReset {
		if err := redis.FlushDB(rdb); err != nil {
			errorX("FLUSHDB: %v", err)
			return err
		}
	}
	ok()

	// ── [2/3] Build ──
	step(2, 3, "Build")
	if err := builder.All(repoRoot, flags.ForceBuild); err != nil {
		errorX("Build failed: %v", err)
		return err
	}
	ok()

	// ── [3/3] Start services ──
	step(3, 3, "Start services")
	fmt.Println()

	mgr := process.NewManager(flags.Verbose)
	mgr.SetWorkDir(repoRoot)
	mgr.SetLogDir("/tmp/deepx/logs")

	redisHost, redisPort := splitRedisAddr(flags.RedisAddr)

	// ① op-plat
	if _, err := mgr.Start("op-metal", builder.OpMetal, redisHost, redisPort); err != nil {
		errorX("op-plat: %v", err)
		mgr.StopAll(5 * time.Second)
		return err
	}
	fmt.Print("  op-plat .....................")
	if err := redis.WaitForInstance(rdb, "/sys/op-plat/op-metal:0", 30*time.Second); err != nil {
		errorX("op-plat not ready: %v", err)
		mgr.StopAll(5 * time.Second)
		return err
	}
	okInline()

	// ② heap-plat
	if _, err := mgr.Start("heap-metal", builder.HeapMetal, redisHost, redisPort); err != nil {
		errorX("heap-plat: %v", err)
		mgr.StopAll(5 * time.Second)
		return err
	}
	fmt.Print("  heap-plat ...................")
	if err := redis.WaitForInstance(rdb, "/sys/heap-plat/heap-metal:0", 30*time.Second); err != nil {
		errorX("heap-plat not ready: %v", err)
		mgr.StopAll(5 * time.Second)
		return err
	}
	okInline()

	// ③ VM
	if _, err := mgr.Start("vm", builder.VM, flags.RedisAddr); err != nil {
		errorX("VM: %v", err)
		mgr.StopAll(5 * time.Second)
		return err
	}
	fmt.Print("  VM ..........................")
	if err := redis.WaitForInstance(rdb, "/sys/vm/0", 30*time.Second); err != nil {
		errorX("VM not ready: %v", err)
		mgr.StopAll(5 * time.Second)
		return err
	}
	okInline()

	// ── Write PID file ──
	state := BootState{
		OpMetal:   mgr.PID("op-metal"),
		HeapMetal: mgr.PID("heap-metal"),
		VM:        mgr.PID("vm"),
		RedisAddr: flags.RedisAddr,
	}
	if err := writeBootState(state); err != nil {
		errorX("write PID file: %v", err)
		mgr.StopAll(5 * time.Second)
		return err
	}

	fmt.Printf("\n  PID file written: %s\n", BootPIDFile)
	ok()

	// Detach manager — processes stay running after boot exits.
	// The PID file is the authoritative record for shutdown.
	mgr.Detach()
	logx.Debug("boot complete, services running")
	return nil
}

// writeBootState writes the boot state to BootPIDFile.
func writeBootState(state BootState) error {
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal boot state: %w", err)
	}
	if err := os.WriteFile(BootPIDFile, data, 0644); err != nil {
		return fmt.Errorf("write %s: %w", BootPIDFile, err)
	}
	return nil
}

// ReadBootState reads the boot state from BootPIDFile.
// Returns nil if the file does not exist.
func ReadBootState() (*BootState, error) {
	data, err := os.ReadFile(BootPIDFile)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read %s: %w", BootPIDFile, err)
	}
	var state BootState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("parse %s: %w", BootPIDFile, err)
	}
	return &state, nil
}

// IsBooted checks whether the booted services are still running.
// Returns true if BootPIDFile exists and all PIDs are alive.
func IsBooted() bool {
	state, err := ReadBootState()
	if err != nil || state == nil {
		return false
	}
	return pidAlive(state.OpMetal) && pidAlive(state.HeapMetal) && pidAlive(state.VM)
}

// pidAlive checks if a process with the given PID is running (Unix).
func pidAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	// Signal 0 is the null signal — used to check process existence.
	return syscall.Kill(pid, 0) == nil
}
