// Package dispatch — 原始字节流 WebSocket 终端 I/O。
//
// 三路独立 WebSocket，各自互不影响：
//   /vthread/<vtid>/stdout → ws://...   VM 写入
//   /vthread/<vtid>/stderr → ws://...   VM 写入
//   /vthread/<vtid>/stdin  → ws://...   VM 读取
//
// 不做任何序列化，直接传原始字节流（TextMessage）。
package dispatch

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

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
