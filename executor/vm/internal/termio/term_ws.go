// Package termio — 终端 I/O 传输层（WebSocket + 文件）。
//
// 终端发现流程：
//   /vthread/<vtid>/term → 终端名称 $name（默认空字符串，空则无终端）
//   /sys/term/${name}/stdout  → HASH {type, detail}
//   /sys/term/${name}/stderr  → HASH {type, detail}
//   /sys/term/${name}/stdin   → HASH {type, detail}
//
// type 取值: "websocket" | "file"
// detail: ws://url 或文件路径
//
// 不做任何序列化，直接传原始字节流。
package termio

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// TermStream 表示一个已解析的终端流配置。
type TermStream struct {
	Type   string // "websocket" | "file" | ""
	Detail string // ws://url 或文件路径
}

// IsZero 终端未配置时返回 true。
func (s TermStream) IsZero() bool { return s.Type == "" }

type wsConn struct {
	conn  *websocket.Conn
	mu    sync.Mutex
	wsURL string
}

var (
	conns   = map[string]*wsConn{}
	connsMu sync.Mutex
)

// getConn 获取或创建到 wsURL 的 WebSocket 连接（按 URL 缓存复用）。
func getConn(ctx context.Context, wsURL string) (*wsConn, error) {
	connsMu.Lock()
	defer connsMu.Unlock()

	if c, ok := conns[wsURL]; ok {
		if err := c.conn.WriteControl(websocket.PingMessage, []byte{}, time.Now().Add(3*time.Second)); err == nil {
			return c, nil
		}
		c.conn.Close()
		delete(conns, wsURL)
	}

	conn, _, err := websocket.DefaultDialer.DialContext(ctx, wsURL, nil)
	if err != nil {
		return nil, fmt.Errorf("dial %s: %w", wsURL, err)
	}

	c := &wsConn{conn: conn, wsURL: wsURL}
	conns[wsURL] = c
	return c, nil
}

// writeWS 发送原始文本到 WebSocket。
func writeWS(ctx context.Context, wsURL, text string) error {
	c, err := getConn(ctx, wsURL)
	if err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.conn.WriteMessage(websocket.TextMessage, []byte(text))
}

// readWS 从 WebSocket 读取一行原始文本（阻塞，超时 30s）。
func readWS(ctx context.Context, wsURL string) (string, error) {
	c, err := getConn(ctx, wsURL)
	if err != nil {
		return "", err
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	// 设置读超时
	c.conn.SetReadDeadline(time.Now().Add(30 * time.Second))
	defer c.conn.SetReadDeadline(time.Time{})

	_, data, err := c.conn.ReadMessage()
	if err != nil {
		return "", fmt.Errorf("read: %w", err)
	}
	return string(data), nil
}

// ── 文件 I/O ──

var (
	fileWriters   = map[string]*os.File{}
	fileWritersMu sync.Mutex
)

// writeFile 追加一行文本到文件。
func writeFile(path, text string) error {
	fileWritersMu.Lock()
	defer fileWritersMu.Unlock()

	f, ok := fileWriters[path]
	if !ok {
		var err error
		f, err = os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			return fmt.Errorf("open %s: %w", path, err)
		}
		fileWriters[path] = f
	}
	if _, err := f.WriteString(text + "\n"); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	if err := f.Sync(); err != nil {
		return fmt.Errorf("sync %s: %w", path, err)
	}
	return nil
}

// readFile 从文件读取一行文本（去除尾部换行）。
func readFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", path, err)
	}
	s := string(data)
	// strip trailing newline
	if len(s) > 0 && s[len(s)-1] == '\n' {
		s = s[:len(s)-1]
	}
	if len(s) > 0 && s[len(s)-1] == '\r' {
		s = s[:len(s)-1]
	}
	return s, nil
}

// WriteTerm 根据 TermStream 类型将文本写入终端。
func WriteTerm(ctx context.Context, s TermStream, text string) error {
	switch s.Type {
	case "websocket":
		return writeWS(ctx, s.Detail, text)
	case "file":
		return writeFile(s.Detail, text)
	default:
		return nil // 无终端，静默丢弃
	}
}

// ReadTerm 根据 TermStream 类型从终端读取一行文本。
func ReadTerm(ctx context.Context, s TermStream) (string, error) {
	switch s.Type {
	case "websocket":
		return readWS(ctx, s.Detail)
	case "file":
		return readFile(s.Detail)
	default:
		return "", nil // 无终端，返回空
	}
}
