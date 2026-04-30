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
	"sync"
	"syscall"
	"time"
)

// Proc represents a managed subprocess.
type Proc struct {
	Name   string
	cmd    *exec.Cmd
	stdout bytes.Buffer
	stderr bytes.Buffer
	done   chan error
}

// Manager tracks all subprocesses started by deepxctl.
type Manager struct {
	mu       sync.Mutex
	procs    []*Proc
	verbose  bool
	workDir  string // if set, all subprocesses run with this CWD
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

// Start launches a subprocess.
//
// binPath: path to the compiled binary
// args: arguments passed to the binary
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

	if m.verbose {
		cmd.Stdout = io.MultiWriter(&p.stdout, os.Stdout)
		cmd.Stderr = io.MultiWriter(&p.stderr, os.Stderr)
	} else {
		cmd.Stdout = &p.stdout
		cmd.Stderr = &p.stderr
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start %s: %w", name, err)
	}

	log.Printf("[process] %s started pid=%d", name, cmd.Process.Pid)

	// Monitor exit in background
	go func() {
		p.done <- cmd.Wait()
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
