package handler

import (
	"encoding/json"
	"net/http"
	"strings"

	goredis "github.com/redis/go-redis/v9"
)

// GetOps handles GET /api/ops/<backend> — returns the full operator list for any backend.
//
// All backends use the same RPUSH format at /op/<backend>/list (key auto-discovered).
// No hardcoded backend names — any component registering to /op/<name>/list is supported.
//
// Examples: /api/ops/exop-metal, /api/ops/buildin, /api/ops/exop-cpu
func GetOps(rdb *goredis.Client, w http.ResponseWriter, r *http.Request) {
	backend := strings.TrimPrefix(r.URL.Path, "/api/ops/")
	if backend == "" {
		http.Error(w, `{"error":"missing backend"}`, http.StatusBadRequest)
		return
	}

	// 统一从 /op/<backend>/list 读取 (RPUSH 格式)
	key := "/op/" + backend + "/list"
	ops, err := rdb.LRange(r.Context(), key, 0, -1).Result()
	if err != nil {
		http.Error(w, `{"error":"`+err.Error()+`"}`, http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"backend": backend,
		"count":   len(ops),
		"ops":     ops,
	})
}
