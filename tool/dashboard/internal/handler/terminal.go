// Package handler — Terminal WebSocket + PTY handler.
//
// Web terminal (xterm.js):
//   - /api/terminal     → browser ←WebSocket→ dashboard → PTY → /bin/bash
//
// VM I/O relay (3 independent endpoints, raw data, no JSON):
//   - /api/term/stdout  → VM writes stdout → dashboard → broadcast → browser terminals
//   - /api/term/stderr  → VM writes stderr → dashboard → broadcast (ANSI red) → browser terminals
//   - /api/term/stdin   → VM reads stdin  ← browser input ← dashboard
//
// KV space registration:
//   - /sys/term/dashboard        → instance status
//   - /sys/term/dashboard/stdout → HASH {type:"websocket", detail:"ws://addr/api/term/stdout"}
//   - /sys/term/dashboard/stderr → HASH {type:"websocket", detail:"ws://addr/api/term/stderr"}
//   - /sys/term/dashboard/stdin  → HASH {type:"websocket", detail:"ws://addr/api/term/stdin"}
//   - /sys/heartbeat/term:dashboard → heartbeat
package handler

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/creack/pty"
	goredis "github.com/redis/go-redis/v9"
	"github.com/gorilla/websocket"
)

const (
	DefaultTermRows = 24
	DefaultTermCols = 80

	TermWriterType = "websocket"
)

type TerminalStatus struct {
	Status    string `json:"status"`
	Pid       int    `json:"pid"`
	StartedAt int64  `json:"started_at"`
}

type TermIOChannel struct {
	Type   string `json:"type"`
	Detail string `json:"detail"`
}

// ── Broadcast hub ──

// termHub fans out VM stdout/stderr to all connected browser terminals,
// and fans in browser input to the VM stdin connection.
type termHub struct {
	mu      sync.RWMutex
	clients map[*websocket.Conn]chan []byte // browser display channels

	stdinConn *websocket.Conn // VM stdin connection (reverse: browser → VM)
	stdinMu   sync.Mutex
}

var hub = &termHub{
	clients: make(map[*websocket.Conn]chan []byte),
}

func (h *termHub) addBrowser(conn *websocket.Conn) chan []byte {
	h.mu.Lock()
	defer h.mu.Unlock()
	ch := make(chan []byte, 256)
	h.clients[conn] = ch
	return ch
}

func (h *termHub) removeBrowser(conn *websocket.Conn) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if ch, ok := h.clients[conn]; ok {
		close(ch)
		delete(h.clients, conn)
	}
}

// broadcast sends raw data to all browser terminals.
func (h *termHub) broadcast(data []byte) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	for _, ch := range h.clients {
		select {
		case ch <- data:
		default:
		}
	}
}

// setStdin registers a new VM stdin connection (closes previous if any).
func (h *termHub) setStdin(conn *websocket.Conn) {
	h.stdinMu.Lock()
	defer h.stdinMu.Unlock()
	if h.stdinConn != nil {
		h.stdinConn.Close()
	}
	h.stdinConn = conn
}

// clearStdin removes the VM stdin connection.
func (h *termHub) clearStdin() {
	h.stdinMu.Lock()
	defer h.stdinMu.Unlock()
	h.stdinConn = nil
}

// sendStdin forwards browser input to the VM stdin connection.
func (h *termHub) sendStdin(data []byte) {
	h.stdinMu.Lock()
	defer h.stdinMu.Unlock()
	if h.stdinConn != nil {
		h.stdinConn.WriteMessage(websocket.BinaryMessage, data)
	}
}

// ── Registration ──

func RegisterTerminal(rdb *goredis.Client, listenAddr string) {
	ctx := context.Background()

	// Instance status
	status := TerminalStatus{
		Status:    "running",
		Pid:       os.Getpid(),
		StartedAt: time.Now().UnixMilli(),
	}
	data, _ := json.Marshal(status)
	rdb.Set(ctx, "/sys/term/dashboard", string(data), 0)
	log.Printf("[term] registered instance at /sys/term/dashboard")

	// I/O channels: stdout, stderr, stdin — each with its own WebSocket URL
	host := listenAddr
	if host[0] == ':' {
		host = "127.0.0.1" + host
	}
	base := "ws://" + host

	streams := map[string]string{
		"stdout": base + "/api/term/stdout",
		"stderr": base + "/api/term/stderr",
		"stdin":  base + "/api/term/stdin",
	}
	for stream, wsURL := range streams {
		key := "/sys/term/dashboard/" + stream
		ch := TermIOChannel{Type: TermWriterType, Detail: wsURL}
		rdb.HSet(ctx, key, "type", ch.Type, "detail", ch.Detail)
		log.Printf("[term] registered I/O %s → %s", key, wsURL)
	}

	// Heartbeat
	go func() {
		ticker := time.NewTicker(3 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			hb := map[string]interface{}{"ts": time.Now().Unix(), "status": "running"}
			hbData, _ := json.Marshal(hb)
			rdb.Set(ctx, "/sys/heartbeat/term:dashboard", string(hbData), 10*time.Second)
		}
	}()
}

// ── WebSocket handlers ──

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin:     func(r *http.Request) bool { return true },
}

type ResizeMsg struct {
	Cols int `json:"cols"`
	Rows int `json:"rows"`
}

// ServeTerminal handles browser xterm.js → PTY shell.
// GET /api/terminal
func ServeTerminal(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("[term] ws upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	log.Printf("[term] browser connected from %s", r.RemoteAddr)

	// Register for VM output broadcast (stdout/stderr)
	browserCh := hub.addBrowser(conn)
	defer hub.removeBrowser(conn)

	shell := os.Getenv("SHELL")
	if shell == "" {
		shell = "/bin/bash"
	}

	cmd := exec.Command(shell)
	cmd.Env = append(os.Environ(), "TERM=xterm-256color", "COLORTERM=truecolor")

	tty, err := pty.StartWithSize(cmd, &pty.Winsize{
		Rows: DefaultTermRows, Cols: DefaultTermCols,
	})
	if err != nil {
		conn.WriteMessage(websocket.TextMessage, []byte("Terminal failed: "+err.Error()+"\r\n"))
		return
	}
	defer func() { cmd.Process.Kill(); tty.Close() }()

	log.Printf("[term] PTY started, shell=%s, pid=%d", shell, cmd.Process.Pid)

	var wg sync.WaitGroup
	wg.Add(2)

	// PTY stdout → browser WebSocket
	go func() {
		defer wg.Done()
		buf := make([]byte, 4096)
		for {
			n, err := tty.Read(buf)
			if err != nil {
				return
			}
			if err := conn.WriteMessage(websocket.BinaryMessage, buf[:n]); err != nil {
				return
			}
		}
	}()

	// Browser input → PTY stdin  +  VM stdout/stderr display  +  forward to VM stdin
	go func() {
		defer wg.Done()
		for {
			select {
			case vmData := <-browserCh:
				// Display VM stdout/stderr in terminal (stderr already wrapped in ANSI red)
				conn.WriteMessage(websocket.TextMessage, vmData)
				continue
			default:
			}

			msgType, msg, err := conn.ReadMessage()
			if err != nil {
				return
			}

			switch msgType {
			case websocket.TextMessage:
				var resize ResizeMsg
				if json.Unmarshal(msg, &resize) == nil && resize.Cols > 0 && resize.Rows > 0 {
					pty.Setsize(tty, &pty.Winsize{Rows: uint16(resize.Rows), Cols: uint16(resize.Cols)})
					continue
				}
				fallthrough
			case websocket.BinaryMessage:
				tty.Write(msg)
				// Forward to VM stdin if connected
				hub.sendStdin(msg)
			}
		}
	}()

	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	go func() { cmd.Wait(); conn.Close() }()

	<-done
	log.Printf("[term] browser disconnected from %s", r.RemoteAddr)
}

// ── VM I/O endpoints (3 independent WebSocket paths, raw data, no JSON) ──

// ServeTermStdout — VM stdout → broadcast to all browser terminals.
// GET /api/term/stdout
func ServeTermStdout(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("[term-stdout] ws upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	log.Printf("[term-stdout] VM connected from %s", r.RemoteAddr)

	for {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			log.Printf("[term-stdout] VM disconnected: %v", err)
			return
		}
		hub.broadcast(msg)
	}
}

// ServeTermStderr — VM stderr → broadcast with ANSI red prefix.
// GET /api/term/stderr
func ServeTermStderr(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("[term-stderr] ws upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	log.Printf("[term-stderr] VM connected from %s", r.RemoteAddr)

	redPrefix := []byte("\x1b[31m")
	resetSuffix := []byte("\x1b[0m")

	for {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			log.Printf("[term-stderr] VM disconnected: %v", err)
			return
		}
		output := make([]byte, 0, len(redPrefix)+len(msg)+len(resetSuffix))
		output = append(output, redPrefix...)
		output = append(output, msg...)
		output = append(output, resetSuffix...)
		hub.broadcast(output)
	}
}

// ServeTermStdin — VM stdin ← browser input (reverse direction).
// VM connects and reads; dashboard forwards browser keystrokes to this connection.
// GET /api/term/stdin
func ServeTermStdin(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("[term-stdin] ws upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	hub.setStdin(conn)
	defer hub.clearStdin()

	log.Printf("[term-stdin] VM connected from %s", r.RemoteAddr)

	// Keep connection alive — VM reads, browsers write via hub.sendStdin()
	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			log.Printf("[term-stdin] VM disconnected: %v", err)
			return
		}
	}
}
