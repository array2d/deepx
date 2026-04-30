package handler

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"

	"deepx/tool/dashboard/internal/redis"
)

// RunRequest is the JSON body for POST /api/run.
type RunRequest struct {
	Source  string `json:"source"`
	Entry   string `json:"entry,omitempty"`
	Timeout int    `json:"timeout,omitempty"`
}

// RunResponse is returned after vthread creation.
type RunResponse struct {
	Vtid   int64  `json:"vtid"`
	Status string `json:"status"`
	Entry  string `json:"entry"`
}

// Run handles POST /api/run.
//
// Write path: temp .dx → exec loader (writes /src/func/*) → redis.RunVThread (creates vthread).
// The dashboard's own Redis connection is NEVER used for writes — RunVThread opens a
// separate short-lived connection.
func Run(loaderBin, redisAddr string, w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	var req RunRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid JSON"}`, http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.Source) == "" {
		http.Error(w, `{"error":"source is required"}`, http.StatusBadRequest)
		return
	}
	if req.Timeout <= 0 {
		req.Timeout = 60
	}

	// 1. Write source to temp file
	tmpFile, err := os.CreateTemp("", "dashboard-*.dx")
	if err != nil {
		writeError(w, fmt.Sprintf("create temp file: %v", err))
		return
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString(req.Source); err != nil {
		tmpFile.Close()
		writeError(w, fmt.Sprintf("write temp file: %v", err))
		return
	}
	tmpFile.Close()
	log.Printf("[run] wrote source to %s", tmpFile.Name())

	// 2. Exec loader — loader writes /src/func/* to Redis (using its own connection)
	funcs, entryCreated, err := execLoader(loaderBin, tmpFile.Name(), redisAddr)
	if err != nil {
		writeError(w, fmt.Sprintf("loader: %v", err))
		return
	}
	if len(funcs) == 0 {
		writeError(w, "no functions loaded from source")
		return
	}
	log.Printf("[run] loaded %d functions: %v, entryCreated=%v", len(funcs), funcs, entryCreated)

	// 3. Determine entry function
	entryFunc := req.Entry
	if entryFunc == "" {
		entryFunc = detectEntry(funcs)
	}
	if entryFunc == "" && !entryCreated {
		writeError(w, "multiple functions found, please specify entry (e.g., --entry main)")
		return
	}
	if entryFunc == "" {
		entryFunc = funcs[0] // loader created /func/main, use first func as best guess
	}

	// If loader didn't create /func/main, no top-level call found → error
	if !entryCreated {
		writeError(w, "no entry point (no top-level call and no --entry specified)")
		return
	}

	// 4. Create vthread + wake VM — write via isolated short-lived connection
	vtid, err := redis.RunVThread(redisAddr, entryFunc)
	if err != nil {
		writeError(w, fmt.Sprintf("create vthread: %v", err))
		return
	}

	log.Printf("[run] vtid=%d entry=%s → VM notified (via isolated write connection)", vtid, entryFunc)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(RunResponse{
		Vtid:   vtid,
		Status: "init",
		Entry:  entryFunc,
	})
}

// execLoader runs the loader binary with the given .dx file.
func execLoader(loaderBin, path, redisAddr string) (funcs []string, entryCreated bool, err error) {
	log.Printf("[loader] exec %s %s %s", loaderBin, path, redisAddr)

	cmd := exec.Command(loaderBin, path, redisAddr)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	// Set a timeout for the loader
	timer := time.AfterFunc(10*time.Second, func() {
		cmd.Process.Kill()
	})
	defer timer.Stop()

	if err := cmd.Run(); err != nil {
		return nil, false, fmt.Errorf("loader failed: %w\nstderr: %s", err, stderr.String())
	}

	output := stdout.String() + stderr.String()
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)

		if idx := strings.Index(line, "/src/func/"); idx >= 0 && strings.Contains(line, "OK") {
			namePart := line[idx+len("/src/func/"):]
			name := strings.SplitN(namePart, " ", 2)[0]
			funcs = append(funcs, name)
		}

		if strings.Contains(line, "ENTRY /func/main →") {
			entryCreated = true
		}
	}
	return funcs, entryCreated, nil
}

// detectEntry picks an entry function from loaded functions.
func detectEntry(funcs []string) string {
	for _, f := range funcs {
		if f == "main" {
			return "main"
		}
	}
	if len(funcs) == 1 {
		return funcs[0]
	}
	return ""
}

func writeError(w http.ResponseWriter, msg string) {
	log.Printf("[run] ERROR: %s", msg)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusBadRequest)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}
