// Package cmd implements the "run" subcommand for deepxctl.
//
//	deepxctl run <file.dx> [flags]
//
// Orchestrates: Redis reset → build → launch plats + VM → loader → execute → cleanup.
package cmd

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"git.array2d.com/ai/deepx/tool/deepxctl/internal/build"
	"git.array2d.com/ai/deepx/tool/deepxctl/internal/executor"
	"git.array2d.com/ai/deepx/tool/deepxctl/internal/loader"
	"git.array2d.com/ai/deepx/tool/deepxctl/internal/process"
	"git.array2d.com/ai/deepx/tool/deepxctl/internal/redis"
)

// RunFlags holds the parsed flags for the run command.
type RunFlags struct {
	RedisAddr string
	ForceBuild bool
	NoReset    bool
	KeepAlive  bool
	Verbose    bool
	Entry      string
	Timeout    int
	FilePath   string
}

// Execute is the entry point for the "run" subcommand.
// os.Args should already have the subcommand name stripped.
func Execute(args []string) {
	flags := parseFlags(args)

	if flags.FilePath == "" {
		fmt.Fprintln(os.Stderr, "Usage: deepxctl run <file.dx> [flags]")
		flag.PrintDefaults()
		os.Exit(1)
	}

	if err := run(flags); err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %v\n", err)
		os.Exit(1)
	}
}

func parseFlags(args []string) RunFlags {
	fs := flag.NewFlagSet("run", flag.ExitOnError)

	var flags RunFlags
	fs.StringVar(&flags.RedisAddr, "r", redis.DefaultAddr, "Redis address")
	fs.StringVar(&flags.RedisAddr, "redis", redis.DefaultAddr, "Redis address")
	fs.BoolVar(&flags.ForceBuild, "b", false, "Force rebuild")
	fs.BoolVar(&flags.ForceBuild, "build", false, "Force rebuild")
	fs.BoolVar(&flags.NoReset, "no-reset", false, "Skip Redis FLUSHDB")
	fs.BoolVar(&flags.KeepAlive, "keep-alive", false, "Keep processes running after execution")
	fs.BoolVar(&flags.Verbose, "v", false, "Verbose output")
	fs.BoolVar(&flags.Verbose, "verbose", false, "Verbose output")
	fs.StringVar(&flags.Entry, "entry", "", "Entry function name")
	fs.IntVar(&flags.Timeout, "timeout", 60, "Execution timeout in seconds (0=no limit)")

	fs.Parse(args)

	if fs.NArg() > 0 {
		flags.FilePath = fs.Arg(0)
	}

	return flags
}

func run(flags RunFlags) error {
	printHeader(flags)

	repoRoot, err := build.RepoRoot()
	if err != nil {
		return fmt.Errorf("find repo root: %w", err)
	}

	// ── [1/6] Redis ──
	step(1, 6, "Redis")
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

	// ── [2/6] Build ──
	step(2, 6, "Build")

	if flags.ForceBuild {
		log.Printf("[run] force rebuild requested")
	}
	missing := build.Missing()
	if len(missing) > 0 && !flags.ForceBuild {
		log.Printf("[run] missing binaries: %v", missing)
	}

	if err := build.All(repoRoot, flags.ForceBuild); err != nil {
		errorX("Build failed: %v", err)
		return err
	}
	ok()

	// ── [3/6] Start processes ──
	step(3, 6, "Start plats + VM")

	mgr := process.NewManager(flags.Verbose)

	// Split redis addr for components that take host+port separately
	redisHost := "127.0.0.1"
	redisPort := "16379"
	if flags.RedisAddr != "" {
		if idx := lastIndexByte(flags.RedisAddr, ':'); idx > 0 {
			redisHost = flags.RedisAddr[:idx]
			redisPort = flags.RedisAddr[idx+1:]
		}
	}

	// ① op-plat
	if _, err := mgr.Start("op-metal", build.OpMetal, redisHost, redisPort); err != nil {
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
	if _, err := mgr.Start("heap-metal", build.HeapMetal, redisHost, redisPort); err != nil {
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
	vmArgs := []string{flags.RedisAddr}
	if _, err := mgr.Start("vm", build.VM, vmArgs...); err != nil {
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

	// ── [4/6] Load dx ──
	step(4, 6, "Load dx")
	dxPath, _ := loader.NormalizePath(repoRoot, flags.FilePath)
	funcs, err := loader.Load(build.Loader, dxPath, flags.RedisAddr)
	if err != nil {
		errorX("Load: %v", err)
		mgr.StopAll(5 * time.Second)
		return err
	}
	if len(funcs) == 0 {
		errorX("No functions loaded from %s", flags.FilePath)
		mgr.StopAll(5 * time.Second)
		return fmt.Errorf("no functions loaded from %s", flags.FilePath)
	}
	ok()

	// ── [5/6] Execute ──
	step(5, 6, "Execute")

	// Find entry function
	entryFunc, err := loader.FindEntryFunc(funcs, flags.Entry)
	if err != nil {
		errorX("%v", err)
		mgr.StopAll(5 * time.Second)
		return err
	}
	fmt.Printf("  entry: %s\n", entryFunc)

	// Run
	timeout := time.Duration(flags.Timeout) * time.Second
	if flags.Timeout == 0 {
		timeout = 0 // no timeout — will use default in executor
	}
	if timeout == 0 {
		timeout = 5 * time.Minute // safety default when user passes 0
	}

	result, err := executor.Run(rdb, entryFunc, timeout)
	if err != nil {
		errorX("Execute: %v", err)
		mgr.StopAll(5 * time.Second)
		return err
	}

	if result.Success {
		greenCheck()
		fmt.Printf("  vtid=%d  status=%s  pc=%s  %v\n", result.Vtid, result.Status, result.PC, result.Duration)
	} else {
		errorX("vtid=%d status=%s pc=%s", result.Vtid, result.Status, result.PC)
		if result.ErrCode != "" {
			fmt.Fprintf(os.Stderr, "  code:    %s\n", result.ErrCode)
			fmt.Fprintf(os.Stderr, "  message: %s\n", result.ErrMsg)
		}
		mgr.StopAll(5 * time.Second)
		return fmt.Errorf("execution failed")
	}

	// ── [6/6] Cleanup ──
	step(6, 6, "Cleanup")
	if flags.KeepAlive {
		fmt.Println("  (--keep-alive: processes left running)")
	} else {
		mgr.StopAll(5 * time.Second)
	}
	ok()

	// ── Final summary ──
	fmt.Println()
	printSeparator()
	fmt.Printf("SUCCESS  vtid=%d  status=%s  %v\n", result.Vtid, result.Status, result.Duration)
	printSeparator()

	return nil
}

// ── Output helpers ──

func printHeader(flags RunFlags) {
	fmt.Println()
	fmt.Printf(" deepxctl  |  redis: %s\n", flags.RedisAddr)
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

func lastIndexByte(s string, c byte) int {
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == c {
			return i
		}
	}
	return -1
}
