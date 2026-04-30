// Package process manages the lifecycle of deepx subprocesses:
// op-plat, heap-plat, and VM.
//
// Allowed operations (per doc/deepxctl/CLAUDE.md):
//
//	exec.Command start
//	pass args (redis addr)
//	capture stdout/stderr
//	SIGTERM / SIGKILL
//	detect exit status
package process

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"time"
)

// Proc represents a managed subprocess.
type Proc struct {
	Name    string
	cmd     *exec.Cmd
	stdout  bytes.Buffer
	stderr  bytes.Buffer
	logFile *os.File // if set, stdout+stderr are also written to this file
	done    chan error
}

// Manager tracks all subprocesses started by deepxctl.
type Manager struct {
	mu      sync.Mutex
	procs   []*Proc
	verbose bool
	workDir string // if set, all subprocesses run with this CWD
	logDir  string // if set, each subprocess logs to <logDir>/<name>.log
}

// NewManager creates a process manager.
// If verbose is true, subprocess stdout/stderr are also streamed to os.Stdout/os.Stderr.
func NewManager(verbose bool) *Manager {
	return &Manager{verbose: verbose}
}

// SetWorkDir sets the working directory for all subprocesses started by this manager.
func (m *Manager) SetWorkDir(dir string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.workDir = dir
}

// SetLogDir sets a directory for per-process log files.
// Each subprocess gets <logDir>/<name>.log with combined stdout+stderr.
// Files persist after the manager exits (safe for boot → detach → run workflows).
func (m *Manager) SetLogDir(dir string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.logDir = dir
}

// Start launches a subprocess.
//
// binPath: path to the compiled binary
// args: arguments passed to the binary
//
// If logDir is set, stdout+stderr are written directly to <logDir>/<name>.log.
// This ensures logs survive after the manager exits (safe for boot → detach workflow).
// If logDir is not set, stdout+stderr are captured in memory via pipes.
func (m *Manager) Start(name, binPath string, args ...string) (*Proc, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	cmd := exec.Command(binPath, args...)
	if m.workDir != "" {
		cmd.Dir = m.workDir
	}

	p := &Proc{
		Name: name,
		cmd:  cmd,
		done: make(chan error, 1),
	}

	// Case 1: log file redirection (survives parent exit).
	if m.logDir != "" {
		if err := os.MkdirAll(m.logDir, 0755); err != nil {
			return nil, fmt.Errorf("create log dir %s: %w", m.logDir, err)
		}
		logPath := filepath.Join(m.logDir, name+".log")
		f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
		if err != nil {
			return nil, fmt.Errorf("open log file %s: %w", logPath, err)
		}
		p.logFile = f
		// Write directly to file — no pipes, so survives parent exit.
		cmd.Stdout = f
		cmd.Stderr = f
		log.Printf("[process] %s logging to %s", name, logPath)
	} else {
		// Case 2: in-memory capture via pipes (for run command where parent stays alive).
		if m.verbose {
			cmd.Stdout = io.MultiWriter(&p.stdout, os.Stdout)
			cmd.Stderr = io.MultiWriter(&p.stderr, os.Stderr)
		} else {
			cmd.Stdout = &p.stdout
			cmd.Stderr = &p.stderr
		}
	}

	if err := cmd.Start(); err != nil {
		if p.logFile != nil {
			p.logFile.Close()
		}
		return nil, fmt.Errorf("start %s: %w", name, err)
	}

	log.Printf("[process] %s started pid=%d", name, cmd.Process.Pid)

	// Monitor exit in background
	go func() {
		p.done <- cmd.Wait()
		if p.logFile != nil {
			p.logFile.Close()
		}
	}()

	m.procs = append(m.procs, p)
	return p, nil
}

// PID returns the PID of a named process, or -1 if not found.
func (m *Manager) PID(name string) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, p := range m.procs {
		if p.Name == name && p.cmd.Process != nil {
			return p.cmd.Process.Pid
		}
	}
	return -1
}

// StopAll sends SIGTERM to all processes, waits up to shutdownTimeout, then SIGKILL.
func (m *Manager) StopAll(shutdownTimeout time.Duration) {
	m.mu.Lock()
	procs := make([]*Proc, len(m.procs))
	copy(procs, m.procs)
	m.mu.Unlock()

	if len(procs) == 0 {
		return
	}

	log.Printf("[process] stopping %d subprocesses...", len(procs))

	// Phase 1: SIGTERM
	for _, p := range procs {
		if p.cmd.Process != nil {
			p.cmd.Process.Signal(syscall.SIGTERM)
		}
	}

	// Phase 2: Wait for graceful shutdown, then collect exit statuses.
	shutdownDeadline := time.After(shutdownTimeout)
	allExited := true
	for _, p := range procs {
		select {
		case err := <-p.done:
			if err != nil {
				log.Printf("[process] %s exited: %v", p.Name, err)
			} else {
				log.Printf("[process] %s exited ok", p.Name)
			}
		case <-shutdownDeadline:
			allExited = false
		}
	}

	if !allExited {
		log.Printf("[process] timeout, sending SIGKILL...")
		for _, p := range procs {
			if p.cmd.Process != nil {
				p.cmd.Process.Signal(syscall.SIGKILL)
			}
		}
		time.Sleep(500 * time.Millisecond)
		// Drain remaining done channels after kill
		for _, p := range procs {
			select {
			case err := <-p.done:
				if err != nil {
					log.Printf("[process] %s killed: %v", p.Name, err)
				}
			default:
			}
		}
	}
}

// Stdout returns captured stdout for a named process.
func (m *Manager) Stdout(name string) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, p := range m.procs {
		if p.Name == name {
			return p.stdout.String()
		}
	}
	return ""
}

// Detach clears the internal process list without stopping any processes.
// Use this when the manager should exit but processes must keep running
// (e.g., after boot, managed by PID file instead).
func (m *Manager) Detach() {
	m.mu.Lock()
	defer m.mu.Unlock()
	log.Printf("[process] detaching %d subprocesses", len(m.procs))
	m.procs = nil
}

// Stderr returns captured stderr for a named process.
func (m *Manager) Stderr(name string) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, p := range m.procs {
		if p.Name == name {
			return p.stderr.String()
		}
	}
	return ""
}
