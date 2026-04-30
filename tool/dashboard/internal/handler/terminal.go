// Package handler — Terminal WebSocket + PTY handler.
//
// Web terminal (xterm.js):
//   - /api/terminal     → browser ←WebSocket→ dashboard → PTY → /bin/bash
//
// VM I/O relay:
//   - /api/term/io      → VM ←WebSocket→ dashboard → broadcast → browser terminals
//
// KV space registration:
//   - /sys/term/dashboard        → instance status
//   - /sys/term/dashboard/stdout → HASH {type:"websocket", detail:"ws://addr/api/term/io"}
//   - /sys/term/dashboard/stderr → HASH {type:"websocket", detail:"ws://addr/api/term/io"}
//   - /sys/term/dashboard/stdin  → HASH {type:"websocket", detail:"ws://addr/api/term/io"}
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

// ── VM ↔ Browser relay hub ──

// VM sends JSON messages over WebSocket:
//   {"stream":"stdout","data":"line\n"}
//   {"stream":"stderr","data":"error\n"}
type VMMsg struct {
	Stream string `json:"stream"`
	Data   string `json:"data"`
}

type termHub struct {
	mu       sync.RWMutex
	clients  map[*websocket.Conn]chan []byte // browser connections
	vmConn   *websocket.Conn                  // single VM connection
	vmMu     sync.Mutex
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

	// I/O channels: stdout, stderr, stdin — all point to same WebSocket endpoint
	host := listenAddr
	if host[0] == ':' {
		host = "127.0.0.1" + host
	}
	wsURL := "ws://" + host + "/api/term/io"
	ch := TermIOChannel{Type: TermWriterType, Detail: wsURL}
	for _, stream := range []string{"stdout", "stderr", "stdin"} {
		key := "/sys/term/dashboard/" + stream
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

	// Register for VM output broadcast
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

	// Browser input + VM broadcast → PTY stdin
	go func() {
		defer wg.Done()
		for {
			select {
			case vmData := <-browserCh:
				tty.Write(vmData)
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
			}
		}
	}()

	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	go func() { cmd.Wait(); conn.Close() }()

	<-done
	log.Printf("[term] browser disconnected from %s", r.RemoteAddr)
}

// ServeTermIO handles VM stdout/stderr → broadcast to browser terminals.
// GET /api/term/io
//
// VM connects via WebSocket and sends JSON:
//   {"stream":"stdout","data":"hello\n"}
//   {"stream":"stderr","data":"error\n"}
//
// Dashboard broadcasts to all connected browser terminals.
func ServeTermIO(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("[term-io] ws upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	hub.vmMu.Lock()
	if hub.vmConn != nil {
		hub.vmConn.Close()
	}
	hub.vmConn = conn
	hub.vmMu.Unlock()

	defer func() {
		hub.vmMu.Lock()
		hub.vmConn = nil
		hub.vmMu.Unlock()
	}()

	log.Printf("[term-io] VM connected from %s", r.RemoteAddr)

	for {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			log.Printf("[term-io] VM disconnected: %v", err)
			return
		}

		var vmMsg VMMsg
		if json.Unmarshal(msg, &vmMsg) != nil {
			continue
		}

		// Format for terminal display
		prefix := ""
		if vmMsg.Stream == "stderr" {
			prefix = "\x1b[31m" // red
		}
		output := prefix + vmMsg.Data + "\x1b[0m"

		// Broadcast to all browser terminals
		hub.broadcast([]byte(output))
	}
}
