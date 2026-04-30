// Package cmd implements the "run" subcommand for deepxctl.
//
//	deepxctl run [flags] <file.dx>
//
// Flags (standard Go convention: must appear before the file path):
//
//	--rm        After execution, shutdown all services and flush Redis (guaranteed cleanup)
//	--entry     Manual entry function (overrides top-level call detection)
//	--timeout   Execution timeout in seconds (0 = no limit, default 60)
//	--boot      Auto-boot if not already booted (default true)
//	-r, --redis Redis address (default 127.0.0.1:16379)
//
// Execution semantics:
//   - If the .dx file has top-level call expressions (outside any def block),
//     the loader writes /func/main and the VM auto-executes. deepxctl polls for the result.
//   - If the .dx file only has function definitions (no top-level call),
//     the loader only registers them. deepxctl reports the loaded functions and exits.
//   - Use --entry to manually specify an entry function (writes /func/main even
//     when the file has no top-level call).
//
// Services are left running after completion (unless --rm).
package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	goredis "github.com/redis/go-redis/v9"

	"deepx/tool/deepxctl/internal/builder"
	"deepx/tool/deepxctl/internal/redis"
)

// RunFlags holds the parsed flags for the run command.
type RunFlags struct {
	RedisAddr string
	Entry     string
	Timeout   int
	FilePath  string
	Rm        bool
}

// Run is the entry point for the "run" subcommand.
func Run(args []string) {
	flags := parseRunFlags(args)

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

func parseRunFlags(args []string) RunFlags {
	fs := flag.NewFlagSet("run", flag.ExitOnError)

	var flags RunFlags
	fs.StringVar(&flags.RedisAddr, "r", redis.DefaultAddr, "Redis address")
	fs.StringVar(&flags.RedisAddr, "redis", redis.DefaultAddr, "Redis address")
	fs.StringVar(&flags.Entry, "entry", "", "Manual entry function (overrides top-level call detection)")
	fs.IntVar(&flags.Timeout, "timeout", 60, "Execution timeout in seconds (0=no limit)")
	fs.BoolVar(&flags.Rm, "rm", false, "After execution, flush Redis and shutdown all services")

	fs.Parse(args)

	if fs.NArg() > 0 {
		flags.FilePath = fs.Arg(0)
	}

	return flags
}

func run(flags RunFlags) error {
	printHeader(flags.RedisAddr)

	// rdb is declared here so the --rm defer can access it for FLUSHDB
	var rdb *goredis.Client

	// --rm: guaranteed cleanup on exit (success, error, or early return)
	if flags.Rm {
		defer func() {
			fmt.Println()
			printSeparator()
			fmt.Println("── Cleanup (--rm) ──")
			// ① Shutdown services first (ExecShutdown uses its own Redis connection
			//    and relies on intact heartbeat/command keys, so shutdown before FLUSHDB)
			if err := ExecShutdown(); err != nil {
				fmt.Fprintf(os.Stderr, "  shutdown: %v\n", err)
			}
			// ② Flush remaining Redis state (only if we have a connection)
			if rdb != nil {
				if err := redis.FlushDB(rdb); err != nil {
					fmt.Fprintf(os.Stderr, "  FLUSHDB: %v\n", err)
				} else {
					fmt.Println("  Redis FLUSHDB ✓")
				}
			}
			fmt.Println("── All components stopped ──")
			printSeparator()
		}()
	}

	// ── [1/3] Verify boot ──
	step(1, 3, "Check services")
	if !IsBooted() {
		errorX("Services not booted. Run 'deepxctl boot' first.")
		fmt.Fprintf(os.Stderr, "\n  Expected boot state at: %s\n", BootPIDFile)
		fmt.Fprintf(os.Stderr, "  If you believe services are running, check with 'make status'.\n")
		return fmt.Errorf("services not booted")
	}
	ok()

	// Verify each service is registered in Redis
	var err error
	rdb, err = redis.Connect(flags.RedisAddr)
	if err != nil {
		errorX("Redis connection failed: %v", err)
		return err
	}
	defer rdb.Close()

	services := map[string]string{
		"op-plat":   "/sys/op-plat/op-metal:0",
		"heap-plat": "/sys/heap-plat/heap-metal:0",
		"vm":        "/sys/vm/0",
	}
	for name, key := range services {
		if err := redis.WaitForInstance(rdb, key, 5*time.Second); err != nil {
			errorX("%s not ready (%s): %v", name, key, err)
			return fmt.Errorf("service %s not ready — re-run 'deepxctl boot'", name)
		}
	}
	log.Printf("[run] all services verified")

	// ── [2/3] Load dx ──
	step(2, 3, "Load dx")
	dxPath, _ := normalizePath(flags.FilePath)
	funcs, entryCreated, err := loadDx(builder.Loader, dxPath, flags.RedisAddr)
	if err != nil {
		errorX("Load: %v", err)
		return err
	}
	if len(funcs) == 0 {
		errorX("No functions loaded from %s", flags.FilePath)
		return fmt.Errorf("no functions loaded from %s", flags.FilePath)
	}
	ok()

	// ── Manual entry override ──
	// If --entry is specified, write /func/main directly
	if flags.Entry != "" {
		log.Printf("[run] --entry=%s → writing /func/main", flags.Entry)
		entryData, _ := json.Marshal(map[string]interface{}{
			"entry":  flags.Entry,
			"reads":  []string{},
			"writes": []string{},
		})
		if err := rdb.Set(context.Background(), "/func/main", entryData, 0).Err(); err != nil {
			errorX("write /func/main: %v", err)
			return err
		}
		entryCreated = true
		fmt.Printf("  entry: %s (manual override)\n", flags.Entry)
	}

	// ── [3/3] Execute (only if /func/main was created) ──
	if !entryCreated {
		// No entry point — just loaded definitions
		fmt.Println()
		printSeparator()
		fmt.Printf("Loaded %d function(s) into KV Space.\n", len(funcs))
		fmt.Println("(no top-level call found — VM is waiting for /func/main)")
		fmt.Println("Use --entry <funcName> to execute a loaded function.")
		printSeparator()
		return nil
	}

	step(3, 3, "Execute")
	timeout := time.Duration(flags.Timeout) * time.Second
	if flags.Timeout == 0 {
		timeout = 5 * time.Minute
	}

	result, err := pollFuncMain(rdb, timeout)
	if err != nil {
		errorX("Execute: %v", err)
		return err
	}

	if result.Success {
		greenCheck()
		fmt.Printf("  vtid=%s  status=%s  %v\n", result.Vtid, result.Status, result.Duration)
	} else {
		errorX("vtid=%s status=%s", result.Vtid, result.Status)
		if result.ErrCode != "" {
			fmt.Fprintf(os.Stderr, "  code:    %s\n", result.ErrCode)
			fmt.Fprintf(os.Stderr, "  message: %s\n", result.ErrMsg)
		}
		return fmt.Errorf("execution failed")
	}

	// ── Final summary ──
	fmt.Println()
	printSeparator()
	fmt.Printf("SUCCESS  vtid=%s  status=%s  %v\n", result.Vtid, result.Status, result.Duration)
	if !flags.Rm {
		fmt.Println("(services left running — use 'deepxctl shutdown' to stop)")
	}
	printSeparator()

	return nil
}

// ── Loader helpers ──

// normalizePath resolves the .dx file path. Relative paths are resolved against CWD.
func normalizePath(path string) (string, error) {
	if filepath.IsAbs(path) {
		return path, nil
	}
	return filepath.Abs(path)
}

// loadDx exec's the loader binary to load .dx files into /src/func/.
// Returns the set of function names loaded, and whether an entry point (/func/main) was created.
func loadDx(loaderBin, path, redisAddr string) (funcs []string, entryCreated bool, err error) {
	log.Printf("[loader] loading %s ...", path)

	cmd := exec.Command(loaderBin, path, redisAddr)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, false, fmt.Errorf("loader failed: %w\noutput: %s", err, stderr.String())
	}

	// Parse function names and entry info from loader output
	output := stdout.String() + stderr.String()
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)

		// Parse: "OK   /path/file.dx → /src/func/compute (N body lines)"
		if idx := strings.Index(line, "/src/func/"); idx >= 0 && strings.Contains(line, "OK") {
			namePart := line[idx+len("/src/func/"):]
			name := strings.SplitN(namePart, " ", 2)[0]
			funcs = append(funcs, name)
		}

		// Parse: "ENTRY /func/main → funcName"
		if strings.Contains(line, "ENTRY /func/main →") {
			entryCreated = true
		}
	}

	log.Printf("[loader] loaded %d functions: %v, entryCreated=%v", len(funcs), funcs, entryCreated)
	return funcs, entryCreated, nil
}

// ── /func/main execution polling ──

// funcMainResult holds the execution result from the /func/main protocol.
type funcMainResult struct {
	Success  bool
	Vtid     string
	Status   string
	ErrCode  string
	ErrMsg   string
	Duration time.Duration
}

// pollFuncMain waits for the VM to pick up /func/main, execute, and report completion.
func pollFuncMain(rdb *goredis.Client, timeout time.Duration) (*funcMainResult, error) {
	startTime := time.Now()
	deadline := time.Now().Add(timeout)
	ctx := context.Background()
	const key = "/func/main"

	// Phase 1: Wait for VM to claim /func/main and write vtid
	var vtid string
	for time.Now().Before(deadline) {
		val, err := rdb.Get(ctx, key).Result()
		if err != nil {
			// Key may not exist yet, or VM already processed it
			time.Sleep(200 * time.Millisecond)
			continue
		}

		var entry struct {
			Vtid   string `json:"vtid"`
			Status string `json:"status"`
		}
		if err := json.Unmarshal([]byte(val), &entry); err != nil {
			time.Sleep(200 * time.Millisecond)
			continue
		}

		if entry.Vtid != "" {
			vtid = entry.Vtid
			log.Printf("[run] VM picked up /func/main, vtid=%s", vtid)
			break
		}

		// Still waiting for VM to pick up (value has "entry" but not yet "vtid")
		time.Sleep(200 * time.Millisecond)
	}

	if vtid == "" {
		return nil, fmt.Errorf("timeout waiting for VM to pick up /func/main")
	}

	// Phase 2: Poll vthread status
	pollInterval := 100 * time.Millisecond
	for time.Now().Before(deadline) {
		status, err := redis.GetVThreadStatus(rdb, parseVtid(vtid))
		if err != nil {
			time.Sleep(pollInterval)
			continue
		}

		switch status.Status {
		case "done":
			// Clean up /func/main
			rdb.Del(ctx, key)
			return &funcMainResult{
				Success:  true,
				Vtid:     vtid,
				Status:   status.Status,
				Duration: time.Since(startTime),
			}, nil

		case "error":
			r := &funcMainResult{
				Success:  false,
				Vtid:     vtid,
				Status:   status.Status,
				Duration: time.Since(startTime),
			}
			if status.Error != nil {
				r.ErrCode = status.Error.Code
				r.ErrMsg = status.Error.Message
			}
			rdb.Del(ctx, key)
			return r, nil

		case "init", "running", "wait":
			time.Sleep(pollInterval)

		default:
			time.Sleep(pollInterval)
		}
	}

	return nil, fmt.Errorf("vthread %s execution timeout after %v", vtid, timeout)
}

// parseVtid converts a string vtid to int64 for compatibility with redis helpers.
func parseVtid(s string) int64 {
	n, _ := strconv.ParseInt(s, 10, 64)
	return n
}
