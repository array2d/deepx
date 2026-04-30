// Package dispatch 负责指令分发到 op-plat / heap-plat 或本地执行。
package dispatch

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"

	"deepx/executor/vm/internal/ir"
	"deepx/executor/vm/internal/route"
	"deepx/executor/vm/internal/state"
	"github.com/redis/go-redis/v9"
)

// OpTask 发送给 op-plat 的计算任务。
type OpTask struct {
	Vtid    string                   `json:"vtid"`
	PC      string                   `json:"pc"`
	Opcode  string                   `json:"opcode"`
	Inputs  []ParamRef               `json:"inputs"`
	Outputs []ParamRef               `json:"outputs"`
	Params  map[string]interface{}   `json:"params,omitempty"`
}

// ParamRef 参数引用 (tensor 元信息)。
type ParamRef struct {
	Key     string                 `json:"key"`
	Dtype   string                 `json:"dtype,omitempty"`
	Shape   []int                  `json:"shape,omitempty"`
	Address map[string]interface{} `json:"address,omitempty"`
}

// HeapTask 发送给 heap-plat 的生命周期任务。
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

// IsRelative 判断是否为相对路径引用 (./ 前缀)。
func IsRelative(param string) bool {
	return len(param) >= 2 && param[:2] == "./"
}

// ResolveWriteKey 将相对路径解析为绝对 Redis key。
func ResolveWriteKey(vtid, param string) string {
	if IsRelative(param) {
		return "/vthread/" + vtid + "/" + param[2:]
	}
	return param
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

// resolveParam 解析参数引用，返回完整的 tensor 元信息。
func resolveParam(ctx context.Context, rdb *redis.Client, vtid string, param string) ParamRef {
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
	if err := json.Unmarshal([]byte(val), &meta); err != nil {
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

func buildOpTask(ctx context.Context, rdb *redis.Client, vtid string, pc string, inst *ir.Instruction) *OpTask {
	task := &OpTask{
		Vtid:   vtid,
		PC:     pc,
		Opcode: inst.Opcode,
		Params: make(map[string]interface{}),
	}

	switch inst.Opcode {
	case "save":
		// save(tensor, filepath)
		// Reads[0] = tensor → input
		// Reads[1] = filepath → param
		for i, r := range inst.Reads {
			if i == 0 {
				task.Inputs = append(task.Inputs, resolveParam(ctx, rdb, vtid, r))
			} else {
				task.Params[fmt.Sprintf("arg%d", len(task.Params))] = r
			}
		}

	case "load":
		// load(filepath) → output_tensor
		// Reads[0] = filepath → param
		// Writes[0] = output tensor → output
		for _, r := range inst.Reads {
			task.Params[fmt.Sprintf("arg%d", len(task.Params))] = r
		}
		for _, w := range inst.Writes {
			task.Outputs = append(task.Outputs, resolveParam(ctx, rdb, vtid, w))
		}

	case "print":
		// print(tensor) — all reads are tensor inputs, no outputs
		for _, r := range inst.Reads {
			task.Inputs = append(task.Inputs, resolveParam(ctx, rdb, vtid, r))
		}
		// no outputs

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

func buildHeapTask(vtid string, pc string, inst *ir.Instruction) *HeapTask {
	task := &HeapTask{
		Vtid: vtid,
		PC:   pc,
		Op:   inst.Opcode,
	}

	switch inst.Opcode {
	case "newtensor":
		// Writes[0] = tensor key (e.g., "/data/x")
		// Reads[0] = dtype (e.g., "f32")
		// Reads[1] = shape (e.g., "[10,10]" or "[100]")
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

// parseShapeParam converts "[10,10]" or "[100]" to []int.
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
func Compute(ctx context.Context, rdb *redis.Client, vtid string, pc string, inst *ir.Instruction) error {
	instance, err := route.Select(ctx, rdb, inst.Opcode)
	if err != nil {
		return fmt.Errorf("route: %w", err)
	}

	task := buildOpTask(ctx, rdb, vtid, pc, inst)
	cmdQueue := fmt.Sprintf("cmd:op-%s", instance)

	taskJSON, err := json.Marshal(task)
	if err != nil {
		return fmt.Errorf("marshal task: %w", err)
	}

	if err := rdb.RPush(ctx, cmdQueue, taskJSON).Err(); err != nil {
		return fmt.Errorf("push task: %w", err)
	}

	log.Printf("[%s] PUSH %s → %s", vtid, inst.Opcode, cmdQueue)

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

	log.Printf("[%s] DONE %s", vtid, inst.Opcode)
	state.Set(ctx, rdb, vtid, ir.NextPC(pc), "running")
	return nil
}

// Lifecycle 分发生命周期指令到 heap-plat。
func Lifecycle(ctx context.Context, rdb *redis.Client, vtid string, pc string, inst *ir.Instruction) error {
	task := buildHeapTask(vtid, pc, inst)
	taskJSON, err := json.Marshal(task)
	if err != nil {
		return fmt.Errorf("marshal heap task: %w", err)
	}

	if err := rdb.RPush(ctx, "cmd:heap-metal:0", taskJSON).Err(); err != nil {
		return fmt.Errorf("push heap task: %w", err)
	}

	log.Printf("[%s] PUSH %s → cmd:heap-metal:0", vtid, inst.Opcode)

	done, err := state.WaitDone(ctx, rdb, vtid, 5*time.Second)
	if err != nil {
		state.SetError(ctx, rdb, vtid, pc, fmt.Sprintf("heap op timeout: %v", err))
		return err
	}

	if status, ok := done["status"].(string); ok && status == "error" {
		errInfo := fmt.Sprintf("%v", done["error"])
		state.SetError(ctx, rdb, vtid, pc, errInfo)
		return fmt.Errorf("heap op error: %s", errInfo)
	}

	state.Set(ctx, rdb, vtid, ir.NextPC(pc), "running")
	return nil
}

// If 处理 IF 条件分支。
func If(ctx context.Context, rdb *redis.Client, vtid string, pc string, inst *ir.Instruction) error {
	if len(inst.Reads) == 0 {
		return fmt.Errorf("if without condition")
	}

	condVal := inst.Reads[0]
	if IsRelative(condVal) {
		resolvedKey := "/vthread/" + vtid + "/" + condVal[2:]
		val, err := rdb.Get(ctx, resolvedKey).Result()
		if err == nil {
			condVal = val
		}
	}

	cond := isTruthy(condVal)
	var branchPC string
	if cond {
		branchPC = pc + "/true/0"
	} else {
		branchPC = pc + "/false/0"
	}

	log.Printf("[%s] IF %v → %s", vtid, cond, branchPC)
	state.Set(ctx, rdb, vtid, branchPC, "running")
	return nil
}

func isTruthy(val string) bool {
	switch strings.ToLower(strings.TrimSpace(val)) {
	case "true", "1", "yes":
		return true
	default:
		return false
	}
}
