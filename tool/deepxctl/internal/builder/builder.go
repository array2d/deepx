// Package builder handles building deepx components by exec'ing existing build.sh scripts.
//
// Allowed operations (per doc/deepxctl/CLAUDE.md):
//
//	exec executor/*/build.sh
//	detect existing binary
//
// Prohibited:
//
//	modifying build scripts
//	modifying CMakeLists.txt
package builder

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"deepx/tool/deepxctl/internal/logx"
)

// Binary paths for metal platform binaries.
var (
	OpMetal   = "/tmp/deepx/op-metal/build/deepx-op-metal"
	HeapMetal = "/tmp/deepx/heap-metal/build/deepx-heap-metal"
	VM        = "/tmp/deepx-vm/vm"
	Loader    = "/tmp/deepx-vm/loader"
)

// Script paths relative to repo root.
type Scripts struct {
	OpMetal   string
	HeapMetal string
	VM        string
}

// DefaultScripts returns the standard build script locations.
func DefaultScripts(repoRoot string) Scripts {
	return Scripts{
		OpMetal:   filepath.Join(repoRoot, "executor/op-metal/build.sh"),
		HeapMetal: filepath.Join(repoRoot, "executor/heap-metal/build.sh"),
		VM:        filepath.Join(repoRoot, "executor/vm/build.sh"),
	}
}

// binaryExists checks if a binary exists on disk.
func binaryExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// Missing returns the list of missing binaries.
func Missing() []string {
	var missing []string
	if !binaryExists(OpMetal) {
		missing = append(missing, "op-metal")
	}
	if !binaryExists(HeapMetal) {
		missing = append(missing, "heap-metal")
	}
	if !binaryExists(VM) {
		missing = append(missing, "vm")
	}
	if !binaryExists(Loader) {
		missing = append(missing, "loader")
	}
	return missing
}

// All builds all components by exec'ing their build.sh scripts.
// repoRoot is the path to the deepx repository root.
func All(repoRoot string, force bool) error {
	scripts := DefaultScripts(repoRoot)

	components := []struct {
		name   string
		script string
		bin    string
	}{
		{"op-metal", scripts.OpMetal, OpMetal},
		{"heap-metal", scripts.HeapMetal, HeapMetal},
		{"vm (+loader)", scripts.VM, VM},
	}

	for _, c := range components {
		if !force && binaryExists(c.bin) {
			logx.Debug("binary exists, skipping build", "component", c.name)
			continue
		}
		logx.Debug("building component", "name", c.name)
		cmd := exec.Command("bash", c.script)
		cmd.Dir = repoRoot
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("build %s failed: %w", c.name, err)
		}
		if !binaryExists(c.bin) {
			return fmt.Errorf("build %s succeeded but binary not found at %s", c.name, c.bin)
		}
		logx.Debug("build complete", "name", c.name, "binary", c.bin)
	}

	// loader is built as part of vm build.sh
	if !binaryExists(Loader) {
		return fmt.Errorf("loader binary not found at %s (should be built by vm/build.sh)", Loader)
	}

	return nil
}

// RepoRoot attempts to find the repository root by walking up from the
// executable's directory, looking for go.mod or executor/ directory.
func RepoRoot() (string, error) {
	// Start from executable path, or current working directory.
	exe, err := os.Executable()
	if err != nil {
		exe, _ = os.Getwd()
	}
	dir := filepath.Dir(exe)

	// Walk up to find repo root (look for executor/ or go.mod at top level)
	for {
		if _, err := os.Stat(filepath.Join(dir, "executor")); err == nil {
			if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
				return dir, nil
			}
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}

	// Fallback: use cwd
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("cannot determine repo root")
	}
	// Walk up from cwd too
	dir = cwd
	for {
		if _, err := os.Stat(filepath.Join(dir, "executor")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return "", fmt.Errorf("cannot find repo root (no executor/ directory found)")
}
