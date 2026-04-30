// Package dispatch — WebSocket 终端客户端。
//
// VM 通过 WebSocket 直接连接 dashboard 终端，不使用 Redis Pub/Sub。
// Redis 仅用于发现 WebSocket URL（/sys/term/*/<stream> HASH）。
//
// 消息格式 (JSON):
//
//	VM → Dashboard:
//	  {"stream":"stdout","vtid":"1","text":"hello"}
//	  {"stream":"stderr","vtid":"1","text":"error"}
//	  {"stream":"stdin-req","vtid":"1"}
//
//	Dashboard → VM:
//	  {"stream":"stdin-resp","vtid":"1","text":"user input"}
package dispatch

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"sync"
	"time"

	"deepx/executor/vm/internal/logx"
	"github.com/gorilla/websocket"
	"github.com/redis/go-redis/v9"
)

// TermWSMsg VM 与 dashboard 终端之间的 WebSocket 消息。
type TermWSMsg struct {
	Stream string `json:"stream"`          // "stdout" | "stderr" | "stdin-req" | "stdin-resp"
	Vtid   string `json:"vtid,omitempty"`  // 虚线程 ID
	Text   string `json:"text,omitempty"`  // 文本内容
}

// termWSClient 维护与 dashboard 终端的 WebSocket 长连接。
type termWSClient struct {
	conn       *websocket.Conn
	mu         sync.Mutex
	wsURL      string
	inputChans map[string]chan string // vtid → stdin 响应通道
	ichanMu    sync.Mutex
}

var (
	wsClient   *termWSClient
	wsClientMu sync.Mutex
)

// getOrCreateWSClient 获取或创建与 dashboard 的 WebSocket 连接。
// 从 Redis 发现 WebSocket URL：扫描 /sys/term/*/stdout 查找 type=websocket 的 HASH。
func getOrCreateWSClient(ctx context.Context, rdb *redis.Client) (*termWSClient, error) {
	wsClientMu.Lock()
	defer wsClientMu.Unlock()

	if wsClient != nil {
		// 探活：发送 ping
		err := wsClient.conn.WriteControl(websocket.PingMessage, []byte{}, time.Now().Add(3*time.Second))
		if err == nil {
			return wsClient, nil
		}
		// 连接断开 → 重连
		logx.Debug("ws term ping failed, reconnecting: %v", err)
		wsClient.conn.Close()
		wsClient = nil
	}

	wsURL, err := discoverWSURL(ctx, rdb)
	if err != nil {
		return nil, fmt.Errorf("discover ws url: %w", err)
	}

	conn, _, err := websocket.DefaultDialer.DialContext(ctx, wsURL, nil)
	if err != nil {
		return nil, fmt.Errorf("dial %s: %w", wsURL, err)
	}

	c := &termWSClient{
		conn:       conn,
		wsURL:      wsURL,
		inputChans: make(map[string]chan string),
	}

	// 启动读协程：接收 stdin 响应
	go c.readLoop()

	wsClient = c
	logx.Debug("ws term connected: %s", wsURL)
	return c, nil
}

// discoverWSURL 从 Redis 发现 dashboard 终端的 WebSocket URL。
// 扫描 /sys/term/*/stdout，找到第一个 type=websocket 的 HASH，读取 url 字段。
func discoverWSURL(ctx context.Context, rdb *redis.Client) (string, error) {
	keys, err := rdb.Keys(ctx, "/sys/term/*/stdout").Result()
	if err != nil {
		return "", fmt.Errorf("KEYS: %w", err)
	}
	for _, key := range keys {
		keyType, tErr := rdb.Type(ctx, key).Result()
		if tErr != nil || keyType != "hash" {
			continue
		}
		termType, hErr := rdb.HGet(ctx, key, "type").Result()
		if hErr != nil || termType != "websocket" {
			continue
		}
		wsURL, uErr := rdb.HGet(ctx, key, "url").Result()
		if uErr != nil || wsURL == "" {
			continue
		}
		// 验证 URL 格式
		parsed, pErr := url.Parse(wsURL)
		if pErr != nil || (parsed.Scheme != "ws" && parsed.Scheme != "wss") {
			logx.Warn("invalid ws url: %s, skipping", wsURL)
			continue
		}
		logx.Debug("discovered ws term url: %s from %s", wsURL, key)
		return wsURL, nil
	}
	return "", fmt.Errorf("no websocket terminal found (check /sys/term/*/stdout)")
}

// send 发送 JSON 消息到 dashboard。
func (c *termWSClient) send(msg TermWSMsg) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	if err := c.conn.WriteMessage(websocket.TextMessage, data); err != nil {
		return fmt.Errorf("write: %w", err)
	}
	return nil
}

// sendToTermWS 发送 stdout / stderr 输出到 dashboard。
func sendToTermWS(ctx context.Context, rdb *redis.Client, vtid, stream, text string) error {
	c, err := getOrCreateWSClient(ctx, rdb)
	if err != nil {
		logx.Debug("[%s] ws term not available: %v", vtid, err)
		return err
	}
	return c.send(TermWSMsg{Stream: stream, Vtid: vtid, Text: text})
}

// requestStdinWS 通过 WebSocket 向 dashboard 请求 stdin 输入，阻塞等待响应。
// 超时 30 秒。
func requestStdinWS(ctx context.Context, rdb *redis.Client, vtid string) (string, error) {
	c, err := getOrCreateWSClient(ctx, rdb)
	if err != nil {
		return "", fmt.Errorf("ws connect: %w", err)
	}

	// 创建响应通道
	respCh := make(chan string, 1)
	c.ichanMu.Lock()
	c.inputChans[vtid] = respCh
	c.ichanMu.Unlock()
	defer func() {
		c.ichanMu.Lock()
		delete(c.inputChans, vtid)
		c.ichanMu.Unlock()
	}()

	// 发送 stdin 请求
	if err := c.send(TermWSMsg{Stream: "stdin-req", Vtid: vtid}); err != nil {
		return "", fmt.Errorf("send stdin-req: %w", err)
	}

	// 等待响应
	select {
	case resp := <-respCh:
		return resp, nil
	case <-time.After(30 * time.Second):
		return "", fmt.Errorf("stdin timeout for vtid %s", vtid)
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

// readLoop 持续读取 dashboard 发来的消息，分发 stdin 响应。
func (c *termWSClient) readLoop() {
	for {
		_, data, err := c.conn.ReadMessage()
		if err != nil {
			logx.Debug("ws term read error: %v", err)
			// 连接断开 → 清理，下次请求时重连
			wsClientMu.Lock()
			if wsClient == c {
				wsClient = nil
				c.conn.Close()
			}
			wsClientMu.Unlock()
			return
		}

		var msg TermWSMsg
		if err := json.Unmarshal(data, &msg); err != nil {
			logx.Debug("ws term bad message: %v", err)
			continue
		}

		if msg.Stream == "stdin-resp" && msg.Vtid != "" {
			c.ichanMu.Lock()
			if ch, ok := c.inputChans[msg.Vtid]; ok {
				ch <- msg.Text
			}
			c.ichanMu.Unlock()
		}
	}
}
