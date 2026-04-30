// Package loader handles loading .dx source files into Redis KV space
// by exec'ing the loader binary.
//
// Allowed operations (per doc/deepxctl/CLAUDE.md):
//
//	exec loader binary
//	verify GET /src/func/<name>
//
// Prohibited:
//
//	parsing dxlang syntax (that's the loader binary's job)
//	writing to /src/func/ directly
package loader

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"os/exec"
	"path/filepath"
	"strings"

	goredis "github.com/redis/go-redis/v9"
)

// Load runs the loader binary to load .dx files into /src/func/.
//
// loaderBin: path to compiled loader binary (/tmp/deepx-vm/loader)
// path: .dx file or directory containing .dx files
// redisAddr: Redis address (host:port)
//
// Returns the set of function names loaded, parsed from loader's stdout.
func Load(loaderBin, path, redisAddr string) ([]string, error) {
	log.Printf("[loader] loading %s ...", path)

	cmd := exec.Command(loaderBin, path, redisAddr)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("loader failed: %w\noutput: %s", err, stderr.String())
	}

	// Parse function names from loader stderr lines like:
	//   "2026/04/30 11:27:24 OK   /path/file.dx → /src/func/compute  (5 body lines)"
	// The loader writes log output to stderr.
	output := stdout.String() + stderr.String()
	var funcs []string
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		idx := strings.Index(line, "/src/func/")
		if idx < 0 {
			continue
		}
		if !strings.Contains(line, "OK") {
			continue
		}
		namePart := line[idx+len("/src/func/"):]
		name := strings.SplitN(namePart, " ", 2)[0]
		funcs = append(funcs, name)
	}

	log.Printf("[loader] loaded %d functions: %v", len(funcs), funcs)
	return funcs, nil
}

// FindEntryFunc determines the entry function from a set of loaded function names
// and an optional user-specified entry.
//
// Rules (per doc/deepxctl/CLAUDE.md):
//  1. --entry specified → use that
//  2. "main" exists → use "main"
//  3. single function → use that
//  4. multiple functions, no main → error
func FindEntryFunc(funcs []string, userEntry string) (string, error) {
	if userEntry != "" {
		return userEntry, nil
	}

	if len(funcs) == 0 {
		return "", fmt.Errorf("no functions loaded")
	}

	// Check for "main"
	for _, f := range funcs {
		if f == "main" {
			return "main", nil
		}
	}

	// Only one function
	if len(funcs) == 1 {
		return funcs[0], nil
	}

	// Multiple, no main — suggest using --entry
	return "", fmt.Errorf(
		"multiple functions found (%s) but no 'main'.\nUse --entry to specify which one to run",
		strings.Join(funcs, ", "),
	)
}

// Verify checks that all loaded functions exist in Redis /src/func/.
func Verify(rdb *goredis.Client, funcs []string) error {
	for _, f := range funcs {
		if !SrcFuncExists(rdb, f) {
			return fmt.Errorf("function %s not found in /src/func/ after loading", f)
		}
	}
	return nil
}

// SrcFuncExists checks if a function exists in Redis /src/func/<name>.
func SrcFuncExists(rdb *goredis.Client, name string) bool {
	ctx := context.Background()
	val, err := rdb.Get(ctx, "/src/func/"+name).Result()
	if err != nil {
		return false
	}
	return val != ""
}

// NormalizePath resolves relative .dx file paths against repoRoot.
// If the path is already absolute, returns it as-is.
func NormalizePath(repoRoot, path string) (string, error) {
	if filepath.IsAbs(path) {
		return path, nil
	}
	return filepath.Join(repoRoot, path), nil
}
