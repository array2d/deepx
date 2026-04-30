package server

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	goredis "github.com/redis/go-redis/v9"

	"deepx/tool/dashboard/internal/handler"
)

type Server struct {
	rdb       *goredis.Client
	loaderBin string
	redisAddr string
	listenAddr string
	statusHub *handler.StatusHub
}

func New(rdb *goredis.Client, loaderBin, redisAddr, listenAddr string) *Server {
	s := &Server{
		rdb:        rdb,
		loaderBin:  loaderBin,
		redisAddr:  redisAddr,
		listenAddr: listenAddr,
		statusHub:  handler.NewStatusHub(rdb),
	}
	go s.statusHub.Run()
	handler.RegisterTerminal(rdb, listenAddr)
	return s
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// React dev proxy: check if dist exists, otherwise serve nothing for /api routes
	distDir := filepath.Join("frontend", "dist")

	// API routes
	mux.HandleFunc("/api/status", func(w http.ResponseWriter, r *http.Request) {
		data, err := handler.GetStatus(s.rdb)
		writeJSON(w, data, err)
	})

	mux.HandleFunc("/api/status/stream", func(w http.ResponseWriter, r *http.Request) {
		s.statusHub.ServeSSE(w, r)
	})

	mux.HandleFunc("/api/vthreads", func(w http.ResponseWriter, r *http.Request) {
		data, err := handler.ListVThreads(s.rdb)
		writeJSON(w, data, err)
	})

	mux.HandleFunc("/api/vthread/", func(w http.ResponseWriter, r *http.Request) {
		handler.GetVThread(s.rdb, w, r)
	})

	mux.HandleFunc("/api/functions", func(w http.ResponseWriter, r *http.Request) {
		data, err := handler.ListFunctions(s.rdb)
		writeJSON(w, data, err)
	})

	mux.HandleFunc("/api/terminal", func(w http.ResponseWriter, r *http.Request) {
		handler.ServeTerminal(w, r)
	})

	mux.HandleFunc("/api/term/stdout", func(w http.ResponseWriter, r *http.Request) {
		handler.ServeTermStdout(w, r)
	})

	mux.HandleFunc("/api/term/stderr", func(w http.ResponseWriter, r *http.Request) {
		handler.ServeTermStderr(w, r)
	})

	mux.HandleFunc("/api/term/stdin", func(w http.ResponseWriter, r *http.Request) {
		handler.ServeTermStdin(w, r)
	})

	mux.HandleFunc("/api/ops/", func(w http.ResponseWriter, r *http.Request) {
		handler.GetOps(s.rdb, w, r)
	})

	mux.HandleFunc("/api/run", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		handler.Run(s.loaderBin, s.redisAddr, w, r)
	})

	// Static files (React build) or SPA fallback
	fs := http.FileServer(http.Dir(distDir))
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// API and SSE paths handled above
		if strings.HasPrefix(r.URL.Path, "/api/") {
			http.NotFound(w, r)
			return
		}

		// Try to serve static file
		path := filepath.Join(distDir, r.URL.Path)
		if r.URL.Path == "/" {
			path = filepath.Join(distDir, "index.html")
		}

		if info, err := os.Stat(path); err == nil && !info.IsDir() {
			fs.ServeHTTP(w, r)
			return
		}

		// SPA fallback: serve index.html for all non-file routes
		http.ServeFile(w, r, filepath.Join(distDir, "index.html"))
	})

	return withCORS(mux)
}

func writeJSON(w http.ResponseWriter, data interface{}, err error) {
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	json.NewEncoder(w).Encode(data)
}

func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		next.ServeHTTP(w, r)
	})
}
