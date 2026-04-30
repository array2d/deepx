// Package handler — Terminal WebSocket + PTY handler.
//
// Provides a web-based terminal via xterm.js:
//   - WebSocket upgrade at /api/terminal
//   - Spawns /bin/bash via PTY (pseudo-terminal)
//   - Bridges stdin/stdout between browser and shell
//   - Handles terminal resize from xterm.js fit addon
//
// Registration:
//   - /sys/term/dashboard        → instance status (string, JSON)
//   - /sys/term/dashboard/stdout → HASH {type, detail}  VM output target
//   - /sys/term/dashboard/stderr → HASH {type, detail}  VM error target
//   - /sys/term/dashboard/stdin  → HASH {type, detail}  VM input source
//   - /sys/heartbeat/term:dashboard → heartbeat
//
// I/O flow:
//   VM 输出 → PUBLISH term:dashboard:stdout → dashboard SUBSCRIBE → WS → xterm.js
//   VM 错误 → PUBLISH term:dashboard:stderr → dashboard SUBSCRIBE → WS → xterm.js
package handler

import (
	"context"
	"encoding/json"
	"io"
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

	// Redis channels for VM → terminal I/O relay
	ChannelTermStdout = "term:dashboard:stdout"
	ChannelTermStderr = "term:dashboard:stderr"
	ChannelTermStdin  = "term:dashboard:stdin"

	// KV space registration keys
	KeyTermStatus = "/sys/term/dashboard"
	KeyTermStdout = "/sys/term/dashboard/stdout"
	KeyTermStderr = "/sys/term/dashboard/stderr"
	KeyTermStdin  = "/sys/term/dashboard/stdin"
)

// TermWriterType is the writer type for the web pseudo-terminal.
// "websocket" means the writer is a WebSocket endpoint.
const TermWriterType = "websocket"

type TerminalStatus struct {
	Status    string `json:"status"`
	Pid       int    `json:"pid"`
	StartedAt int64  `json:"started_at"`
}

// Terminal I/O channel descriptor (HASH stored in Redis KV).
type TermIOChannel struct {
	Type   string `json:"type"`
	Detail string `json:"detail"`
}

// termHub relays Redis pub/sub messages to connected terminal WebSocket clients.
type termHub struct {
	mu      sync.RWMutex
	clients map[*websocket.Conn]chan []byte
	rdb     *goredis.Client
}

var globalTermHub *termHub

// RegisterTerminal writes the terminal's instance status and I/O channels to Redis KV space.
//
// Keys written:
//   /sys/term/dashboard         → {"status":"running","pid":...,"started_at":...}
//   /sys/term/dashboard/stdout  → {"type":"websocket","detail":"term:dashboard:stdout"}
//   /sys/term/dashboard/stderr  → {"type":"websocket","detail":"term:dashboard:stderr"}
//   /sys/term/dashboard/stdin   → {"type":"websocket","detail":"term:dashboard:stdin"}
//
// Also starts:
//   - Heartbeat goroutine to /sys/heartbeat/term:dashboard
//   - Redis pub/sub relay for VM output → terminal WebSocket
func RegisterTerminal(rdb *goredis.Client) {
	ctx := context.Background()

	// Instance status
	status := TerminalStatus{
		Status:    "running",
		Pid:       os.Getpid(),
		StartedAt: time.Now().UnixMilli(),
	}
	data, _ := json.Marshal(status)
	rdb.Set(ctx, KeyTermStatus, string(data), 0)
	log.Printf("[term] registered instance at %s", KeyTermStatus)

	// I/O channels: stdout, stderr, stdin — each is a HASH
	writeIOChannel := func(key, channel string) {
		ch := TermIOChannel{
			Type:   TermWriterType,
			Detail: channel,
		}
		chData, _ := json.Marshal(ch)
		// HSet with JSON string values — Redis HASH
		rdb.HSet(ctx, key,
			"type", ch.Type,
			"detail", ch.Detail,
		)
		log.Printf("[term] registered I/O channel %s → %s", key, string(chData))
	}
	writeIOChannel(KeyTermStdout, ChannelTermStdout)
	writeIOChannel(KeyTermStderr, ChannelTermStderr)
	writeIOChannel(KeyTermStdin, ChannelTermStdin)

	// Heartbeat
	go func() {
		ticker := time.NewTicker(3 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			hb := map[string]interface{}{
				"ts":     time.Now().Unix(),
				"status": "running",
			}
			hbData, _ := json.Marshal(hb)
			rdb.Set(ctx, "/sys/heartbeat/term:dashboard", string(hbData), 10*time.Second)
		}
	}()

	// Pub/sub relay: VM publishes to channels, dashboard subscribes and forwards to WebSocket clients
	hub := &termHub{
		clients: make(map[*websocket.Conn]chan []byte),
		rdb:     rdb,
	}
	globalTermHub = hub
	go hub.runRelay(ChannelTermStdout)
	go hub.runRelay(ChannelTermStderr)
}

// subscribe adds a WebSocket client to receive VM output.
func (h *termHub) subscribe(conn *websocket.Conn) chan []byte {
	h.mu.Lock()
	defer h.mu.Unlock()
	ch := make(chan []byte, 256)
	h.clients[conn] = ch
	return ch
}

// unsubscribe removes a WebSocket client.
func (h *termHub) unsubscribe(conn *websocket.Conn) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if ch, ok := h.clients[conn]; ok {
		close(ch)
		delete(h.clients, conn)
	}
}

// runRelay subscribes to a Redis pub/sub channel and forwards messages to all WebSocket clients.
func (h *termHub) runRelay(channel string) {
	ctx := context.Background()
	pubsub := h.rdb.Subscribe(ctx, channel)
	defer pubsub.Close()

	log.Printf("[term] relay subscribed to %s", channel)

	for {
		msg, err := pubsub.ReceiveMessage(ctx)
		if err != nil {
			log.Printf("[term] relay %s receive error: %v", channel, err)
			// Reconnect after 1s
			time.Sleep(time.Second)
			pubsub.Close()
			pubsub = h.rdb.Subscribe(ctx, channel)
			continue
		}

		// Prefix with channel info so terminal knows it's VM output
		prefix := ""
		if channel == ChannelTermStderr {
			prefix = "\x1b[31m" // red
		}
		payload := prefix + msg.Payload + "\x1b[0m\r\n"

		h.mu.RLock()
		for _, ch := range h.clients {
			select {
			case ch <- []byte(payload):
			default:
				// skip slow client
			}
		}
		h.mu.RUnlock()
	}
}

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

type ResizeMsg struct {
	Cols int `json:"cols"`
	Rows int `json:"rows"`
}

// ServeTerminal handles WebSocket upgrade and PTY session.
func ServeTerminal(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("[term] ws upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	log.Printf("[term] client connected from %s", r.RemoteAddr)

	// Subscribe to VM output relay (if hub is running)
	var relayCh chan []byte
	if globalTermHub != nil {
		relayCh = globalTermHub.subscribe(conn)
		defer globalTermHub.unsubscribe(conn)
	}

	shell := os.Getenv("SHELL")
	if shell == "" {
		shell = "/bin/bash"
	}

	cmd := exec.Command(shell)
	cmd.Env = append(os.Environ(),
		"TERM=xterm-256color",
		"COLORTERM=truecolor",
	)

	tty, err := pty.StartWithSize(cmd, &pty.Winsize{
		Rows: DefaultTermRows,
		Cols: DefaultTermCols,
	})
	if err != nil {
		log.Printf("[term] pty start failed: %v", err)
		conn.WriteMessage(websocket.TextMessage,
			[]byte("Terminal failed: "+err.Error()+"\r\n"))
		return
	}
	defer func() {
		cmd.Process.Kill()
		tty.Close()
	}()

	log.Printf("[term] PTY started, shell=%s, pid=%d", shell, cmd.Process.Pid)

	var wg sync.WaitGroup
	wg.Add(2)

	// PTY stdout → WebSocket
	go func() {
		defer wg.Done()
		buf := make([]byte, 4096)
		for {
			n, readErr := tty.Read(buf)
			if readErr != nil {
				if readErr != io.EOF {
					log.Printf("[term] pty read error: %v", readErr)
				}
				return
			}
			if writeErr := conn.WriteMessage(websocket.BinaryMessage, buf[:n]); writeErr != nil {
				log.Printf("[term] ws write error: %v", writeErr)
				return
			}
		}
	}()

	// WebSocket + relay → PTY stdin
	go func() {
		defer wg.Done()
		for {
			select {
			case relayData, ok := <-relayCh:
				if !ok {
					return
				}
				// VM output → write to PTY (appears in terminal)
				tty.Write(relayData)
				// Also send directly to WebSocket so it renders immediately
				conn.WriteMessage(websocket.TextMessage, relayData)
				continue
			default:
			}

			msgType, msg, readErr := conn.ReadMessage()
			if readErr != nil {
				if websocket.IsUnexpectedCloseError(readErr,
					websocket.CloseGoingAway, websocket.CloseNormalClosure) {
					log.Printf("[term] ws read error: %v", readErr)
				}
				return
			}

			switch msgType {
			case websocket.TextMessage:
				var resize ResizeMsg
				if json.Unmarshal(msg, &resize) == nil && resize.Cols > 0 && resize.Rows > 0 {
					ws := &pty.Winsize{
						Rows: uint16(resize.Rows),
						Cols: uint16(resize.Cols),
					}
					if szErr := pty.Setsize(tty, ws); szErr != nil {
						log.Printf("[term] resize failed: %v", szErr)
					} else {
						log.Printf("[term] resized to %dx%d", resize.Cols, resize.Rows)
					}
					continue
				}
				fallthrough
			case websocket.BinaryMessage:
				if _, writeErr := tty.Write(msg); writeErr != nil {
					log.Printf("[term] pty write error: %v", writeErr)
					return
				}
			}

			// After processing WS message, check relay again
			select {
			case relayData, ok := <-relayCh:
				if !ok {
					return
				}
				tty.Write(relayData)
				conn.WriteMessage(websocket.TextMessage, relayData)
			default:
			}
		}
	}()

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	go func() {
		cmd.Wait()
		conn.Close()
	}()

	<-done
	log.Printf("[term] client disconnected from %s", r.RemoteAddr)
}
