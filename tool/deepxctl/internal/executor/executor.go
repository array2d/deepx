// Package executor handles vthread creation, VM wake-up, and status polling.
//
// Allowed operations (per doc/deepxctl/CLAUDE.md):
//
//	INCR /sys/vtid_counter
//	SET /vthread/<vtid> (init state)
//	SET /vthread/<vtid>/[0,0] (entry CALL)
//	SET /vthread/<vtid>/[0,1] (return slot)
//	LPUSH notify:vm
//	GET /vthread/<vtid> (status polling)
//
// Prohibited:
//
//	writing to /vthread/<vtid>/[*,*] beyond the initial entry CALL
//	consuming done:<vtid> queue
//	writing to cmd:* queues
package executor

import (
	"fmt"
	"log"
	"time"

	goredis "github.com/redis/go-redis/v9"

	"deepx/tool/deepxctl/internal/redis"
)

// Result holds the execution result.
type Result struct {
	Success  bool
	Vtid     int64
	Status   string
	PC       string
	ErrCode  string
	ErrMsg   string
	Duration time.Duration
}

// Run creates a vthread for entryFunc, wakes the VM, and polls until done/error/timeout.
func Run(rdb *goredis.Client, entryFunc string, timeout time.Duration) (*Result, error) {
	startTime := time.Now()

	// 1. Allocate vtid
	vtid, err := redis.AllocVtid(rdb)
	if err != nil {
		return nil, fmt.Errorf("alloc vtid: %w", err)
	}

	// 2. Create vthread with single CALL instruction
	if err := redis.CreateVThread(rdb, vtid, entryFunc); err != nil {
		return nil, fmt.Errorf("create vthread: %w", err)
	}

	// 3. Wake VM
	if err := redis.WakeVM(rdb, vtid); err != nil {
		return nil, fmt.Errorf("wake vm: %w", err)
	}

	log.Printf("[executor] vthread %d started, entry=%s, waiting...", vtid, entryFunc)

	// 4. Poll until done/error/timeout
	deadline := time.Now().Add(timeout)
	pollInterval := 100 * time.Millisecond

	for time.Now().Before(deadline) {
		status, err := redis.GetVThreadStatus(rdb, vtid)
		if err != nil {
			// VThread key might not exist yet (VM hasn't picked it up)
			time.Sleep(pollInterval)
			continue
		}

		switch status.Status {
		case "done":
			return &Result{
				Success:  true,
				Vtid:     vtid,
				Status:   status.Status,
				PC:       status.PC,
				Duration: time.Since(startTime),
			}, nil

		case "error":
			r := &Result{
				Success:  false,
				Vtid:     vtid,
				Status:   status.Status,
				PC:       status.PC,
				Duration: time.Since(startTime),
			}
			if status.Error != nil {
				r.ErrCode = status.Error.Code
				r.ErrMsg = status.Error.Message
			}
			return r, nil

		case "init", "running", "wait":
			// Still executing, continue polling
			time.Sleep(pollInterval)

		default:
			log.Printf("[executor] unexpected vthread status: %s", status.Status)
			time.Sleep(pollInterval)
		}
	}

	return nil, fmt.Errorf("vthread %d execution timeout after %v", vtid, timeout)
}
