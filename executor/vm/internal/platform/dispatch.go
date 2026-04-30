// Package platform 负责算子分发到 op-plat 和 heap-plat。
package platform

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"deepx/executor/vm/internal/ir"
	"deepx/executor/vm/internal/logx"
	"deepx/executor/vm/internal/parser"
	"deepx/executor/vm/internal/state"
	"github.com/redis/go-redis/v9"
)

type OpTask struct {
	Vtid    string                 `json:"vtid"`
	PC      string                 `json:"pc"`
	Opcode  string                 `json:"opcode"`
	Inputs  []ParamRef             `json:"inputs"`
	Outputs []ParamRef             `json:"outputs"`
	Params  map[string]interface{} `json:"params,omitempty"`
}

type ParamRef struct {
	Key     string                 `json:"key"`
	Dtype   string                 `json:"dtype,omitempty"`
	Shape   []int                  `json:"shape,omitempty"`
	Address map[string]interface{} `json:"address,omitempty"`
}

type HeapTask struct {
	Vtid   string `json:"vtid"`
	PC     string `json:"pc"`
	Op     string `json:"op"`
	Key    string `json:"key"`
	Device string `json:"device,omitempty"`
	Dtype  string `json:"dtype,omitempty"`
	Shape  []int  `json:"shape,omitempty"`
	Src    string `json:"src,omitempty"`
	Dst    string `json:"dst,omitempty"`
}

func IsRelative(param string) bool {
	return len(param) >= 2 && param[:2] == "./"
}

func resolveParam(ctx context.Context, rdb *redis.Client, vtid, param string) ParamRef {
	ref := ParamRef{Key: param}
	resolvedKey := param
	if IsRelative(param) {
		resolvedKey = "/vthread/" + vtid + "/" + param[2:]
	}
	ref.Key = resolvedKey
	val, err := rdb.Get(ctx, resolvedKey).Result()
	if err != nil {
		return ref
	}
	var meta map[string]interface{}
	if json.Unmarshal([]byte(val), &meta) != nil {
		return ref
	}
	if dtype, ok := meta["dtype"].(string); ok {
		ref.Dtype = dtype
	}
	if shapeRaw, ok := meta["shape"].([]interface{}); ok {
		for _, s := range shapeRaw {
			if n, ok := s.(float64); ok {
				ref.Shape = append(ref.Shape, int(n))
			}
		}
	}
	if addr, ok := meta["address"].(map[string]interface{}); ok {
		ref.Address = addr
	}
	return ref
}

func isLiteral(s string) bool {
	if IsRelative(s) {
		return false
	}
	if len(s) > 0 && s[0] == '/' {
		return false
	}
	return true
}

func buildOpTask(ctx context.Context, rdb *redis.Client, vtid, pc string, inst *ir.Instruction) *OpTask {
	task := &OpTask{Vtid: vtid, PC: pc, Opcode: inst.Opcode, Params: make(map[string]interface{})}
	switch inst.Opcode {
	case "save":
		for i, r := range inst.Reads {
			if i == 0 {
				task.Inputs = append(task.Inputs, resolveParam(ctx, rdb, vtid, r))
			} else {
				task.Params[fmt.Sprintf("arg%d", len(task.Params))] = r
			}
		}
	case "load":
		for _, r := range inst.Reads {
			task.Params[fmt.Sprintf("arg%d", len(task.Params))] = r
		}
		for _, w := range inst.Writes {
			task.Outputs = append(task.Outputs, resolveParam(ctx, rdb, vtid, w))
		}
	case "print":
		for _, r := range inst.Reads {
			task.Inputs = append(task.Inputs, resolveParam(ctx, rdb, vtid, r))
		}
	default:
		for _, r := range inst.Reads {
			if isLiteral(r) {
				task.Params[fmt.Sprintf("arg%d", len(task.Params))] = r
			} else {
				task.Inputs = append(task.Inputs, resolveParam(ctx, rdb, vtid, r))
			}
		}
		for _, w := range inst.Writes {
			task.Outputs = append(task.Outputs, resolveParam(ctx, rdb, vtid, w))
		}
	}
	return task
}

func buildHeapTask(vtid, pc string, inst *ir.Instruction) *HeapTask {
	task := &HeapTask{Vtid: vtid, PC: pc, Op: inst.Opcode}
	switch inst.Opcode {
	case "newtensor":
		if len(inst.Writes) > 0 {
			task.Key = inst.Writes[0]
		}
		if len(inst.Reads) > 0 {
			task.Dtype = inst.Reads[0]
		}
		if len(inst.Reads) > 1 {
			task.Shape = parseShapeParam(inst.Reads[1])
		}
	case "deltensor":
		if len(inst.Reads) > 0 {
			task.Key = inst.Reads[0]
		}
	case "clonetensor":
		if len(inst.Reads) > 0 {
			task.Src = inst.Reads[0]
		}
		if len(inst.Writes) > 0 {
			task.Dst = inst.Writes[0]
		}
	}
	return task
}

func parseShapeParam(raw string) []int {
	raw = strings.Trim(raw, "[] ")
	if raw == "" {
		return nil
	}
	var shape []int
	for _, s := range strings.Split(raw, ",") {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		var n int
		fmt.Sscanf(s, "%d", &n)
		shape = append(shape, n)
	}
	return shape
}

// Compute 分发张量计算指令到 op-plat。
func Compute(ctx context.Context, rdb *redis.Client, vtid, pc string, inst *ir.Instruction) error {
	instance, err := Select(ctx, rdb, inst.Opcode)
	if err != nil {
		return fmt.Errorf("route: %w", err)
	}
	task := buildOpTask(ctx, rdb, vtid, pc, inst)
	cmdQueue := fmt.Sprintf("cmd:op-%s", instance)
	taskJSON, _ := json.Marshal(task)
	if err := rdb.RPush(ctx, cmdQueue, taskJSON).Err(); err != nil {
		return fmt.Errorf("push task: %w", err)
	}
	logx.Debug("[%s] PUSH %s → %s", vtid, inst.Opcode, cmdQueue)
	state.Set(ctx, rdb, vtid, pc, "wait")
	done, err := state.WaitDone(ctx, rdb, vtid, 30*time.Second)
	if err != nil {
		state.SetError(ctx, rdb, vtid, pc, fmt.Sprintf("BLPOP timeout: %v", err))
		return err
	}
	if status, ok := done["status"].(string); ok && status == "error" {
		errInfo := fmt.Sprintf("%v", done["error"])
		state.SetError(ctx, rdb, vtid, pc, errInfo)
		return fmt.Errorf("op error: %s", errInfo)
	}
	logx.Debug("[%s] DONE %s", vtid, inst.Opcode)
	state.Set(ctx, rdb, vtid, ir.NextPC(pc), "running")
	return nil
}

// Lifecycle 分发生命周期指令到 heap-plat。
func Lifecycle(ctx context.Context, rdb *redis.Client, vtid, pc string, inst *ir.Instruction) error {
	task := buildHeapTask(vtid, pc, inst)
	taskJSON, _ := json.Marshal(task)
	if err := rdb.RPush(ctx, "cmd:heap-metal:0", taskJSON).Err(); err != nil {
		return fmt.Errorf("push heap task: %w", err)
	}
	logx.Debug("[%s] PUSH %s → cmd:heap-metal:0", vtid, inst.Opcode)
	done, err := state.WaitDone(ctx, rdb, vtid, 5*time.Second)
	if err != nil {
		state.SetError(ctx, rdb, vtid, pc, fmt.Sprintf("heap op timeout: %v", err))
		return err
	}
	if status, ok := done["status"].(string); ok && status == "error" {
		errInfo := fmt.Sprintf("%v", done["error"])
		state.SetError(ctx, rdb, vtid, pc, errInfo)
		return fmt.Errorf("heap error: %s", errInfo)
	}
	logx.Debug("[%s] HEAP %s done", vtid, inst.Opcode)
	state.Set(ctx, rdb, vtid, ir.NextPC(pc), "running")
	return nil
}

var _ = parser.ParseLine
