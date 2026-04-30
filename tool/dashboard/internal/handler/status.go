package handler

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	goredis "github.com/redis/go-redis/v9"

	"deepx/tool/dashboard/internal/redis"
)

// ── REST handlers ──

func GetStatus(rdb *goredis.Client) (interface{}, error) {
	return redis.GetSystemStatus(rdb)
}

func ListVThreads(rdb *goredis.Client) (interface{}, error) {
	vts, err := redis.ListVThreads(rdb)
	if err != nil {
		return nil, err
	}
	return map[string]interface{}{"vthreads": vts}, nil
}

func GetVThread(rdb *goredis.Client, w http.ResponseWriter, r *http.Request) {
	idStr := r.URL.Path[len("/api/vthread/"):]
	var vtid int64
	if _, err := fmt.Sscanf(idStr, "%d", &vtid); err != nil || vtid <= 0 {
		http.Error(w, `{"error":"invalid vtid"}`, http.StatusBadRequest)
		return
	}
	vt, err := redis.GetVThread(rdb, vtid)
	if err != nil {
		http.Error(w, `{"error":"`+err.Error()+`"}`, http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(vt)
}

func ListFunctions(rdb *goredis.Client) (interface{}, error) {
	names, err := redis.SrcFuncNames(rdb)
	if err != nil {
		return nil, err
	}
	return map[string]interface{}{"functions": names}, nil
}

// ── SSE (Server-Sent Events) Hub ──

type sseClient struct {
	ch     chan []byte
	done   <-chan struct{}
}

type StatusHub struct {
	rdb     *goredis.Client
	clients map[*sseClient]bool
	mu      sync.Mutex
}

func NewStatusHub(rdb *goredis.Client) *StatusHub {
	return &StatusHub{
		rdb:     rdb,
		clients: make(map[*sseClient]bool),
	}
}

func (h *StatusHub) ServeSSE(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	ch := make(chan []byte, 8)
	client := &sseClient{ch: ch, done: r.Context().Done()}

	h.mu.Lock()
	h.clients[client] = true
	h.mu.Unlock()

	// Send an initial status immediately
	if data := h.buildStatusMsg(); data != nil {
		ch <- data
	}

	defer func() {
		h.mu.Lock()
		delete(h.clients, client)
		h.mu.Unlock()
		close(ch)
	}()

	for {
		select {
		case <-r.Context().Done():
			return
		case data, ok := <-ch:
			if !ok {
				return
			}
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
	}
}

func (h *StatusHub) buildStatusMsg() []byte {
	status, err := redis.GetSystemStatus(h.rdb)
	if err != nil {
		log.Printf("[sse] status fetch error: %v", err)
		return nil
	}
	msg := map[string]interface{}{
		"type": "status",
		"ts":   time.Now().Unix(),
		"data": status,
	}
	data, _ := json.Marshal(msg)
	return data
}

func (h *StatusHub) Run() {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		data := h.buildStatusMsg()
		if data == nil {
			continue
		}

		h.mu.Lock()
		for client := range h.clients {
			select {
			case <-client.done:
				delete(h.clients, client)
			case client.ch <- data:
			default:
				// Client too slow, skip
			}
		}
		h.mu.Unlock()
	}
}
