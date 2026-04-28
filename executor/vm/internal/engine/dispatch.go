package engine

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"

	"git.array2d.com/ai/deepx/executor/vm/internal/ir"
	"git.array2d.com/ai/deepx/executor/vm/internal/route"
	"github.com/redis/go-redis/v9"
)

// OpTask 发送给 op-plat 的计算任务
type OpTask struct {
	Vtid   string     `json:"vtid"`
	PC     string     `json:"pc"`
	Opcode string     `json:"opcode"`
	Inputs []ParamRef `json:"inputs"`
	Outputs []ParamRef `json:"outputs"`
	Params map[string]interface{} `json:"params,omitempty"`
}

// ParamRef 参数引用（tensor 元信息）
type ParamRef struct {
	Key     string `json:"key"`
	Dtype   string `json:"dtype,omitempty"`
	Shape   []int  `json:"shape,omitempty"`
	Address map[string]interface{} `json:"address,omitempty"`
}

// HeapTask 发送给 heap-plat 的生命周期任务
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

// resolveParam 解析参数引用，返回完整的 tensor 元信息
func resolveParam(ctx context.Context, rdb *redis.Client, vtid string, param string) ParamRef {
	ref := ParamRef{Key: param}

	// 解析路径
	resolvedKey := param
	if isRelative(param) {
		resolvedKey = "/vthread/" + vtid + "/" + param[2:]
	}
	ref.Key = resolvedKey

	// 尝试读取 tensor 元信息
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

func isRelative(param string) bool {
	return len(param) >= 2 && param[:2] == "./"
}

// buildOpTask 构建发送给 op-plat 的任务
func buildOpTask(ctx context.Context, rdb *redis.Client, vtid string, pc string, inst *ir.Instruction) *OpTask {
	task := &OpTask{
		Vtid:   vtid,
		PC:     pc,
		Opcode: inst.Opcode,
		Params: make(map[string]interface{}),
	}

	for _, r := range inst.Reads {
		// 立即数判断
		if isLiteral(r) {
			task.Params[fmt.Sprintf("arg%d", len(task.Params))] = r
		} else {
			task.Inputs = append(task.Inputs, resolveParam(ctx, rdb, vtid, r))
		}
	}

	for _, w := range inst.Writes {
		task.Outputs = append(task.Outputs, resolveParam(ctx, rdb, vtid, w))
	}

	return task
}

func isLiteral(s string) bool {
	// 简单判断: 不是路径格式的就是立即数
	if isRelative(s) {
		return false
	}
	if len(s) > 0 && s[0] == '/' {
		return false
	}
	return true
}

// buildHeapTask 构建发送给 heap-plat 的任务
func buildHeapTask(vtid string, pc string, inst *ir.Instruction) *HeapTask {
	task := &HeapTask{
		Vtid: vtid,
		PC:   pc,
		Op:   inst.Opcode,
	}

	switch inst.Opcode {
	case "newtensor":
		if len(inst.Writes) > 0 {
			task.Key = inst.Writes[0]
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

// dispatchCompute 分发张量计算指令到 op-plat
func dispatchCompute(ctx context.Context, rdb *redis.Client, vtid string, pc string, inst *ir.Instruction) error {
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

	log.Printf("[%s] PUSH %s → cmd:op-%s", vtid, inst.Opcode, instance)

	// 等待完成
	setState(ctx, rdb, vtid, pc, "wait")
	done, err := waitDone(ctx, rdb, vtid, 30*time.Second)
	if err != nil {
		setError(ctx, rdb, vtid, pc, fmt.Sprintf("BLPOP timeout: %v", err))
		return err
	}

	if status, _ := done["status"].(string); status == "error" {
		errInfo := fmt.Sprintf("%v", done["error"])
		setError(ctx, rdb, vtid, pc, errInfo)
		return fmt.Errorf("op error: %s", errInfo)
	}

	log.Printf("[%s] DONE %s", vtid, inst.Opcode)

	// PC++
	setState(ctx, rdb, vtid, ir.NextPC(pc), "running")
	return nil
}

// dispatchLifecycle 分发生命周期指令到 heap-plat
func dispatchLifecycle(ctx context.Context, rdb *redis.Client, vtid string, pc string, inst *ir.Instruction) error {
	task := buildHeapTask(vtid, pc, inst)
	taskJSON, _ := json.Marshal(task)

	// 默认发到 metal heap (TODO: 根据 device 选择)
	if err := rdb.RPush(ctx, "cmd:heapmem-metal:0", taskJSON).Err(); err != nil {
		return fmt.Errorf("push heap task: %w", err)
	}

	log.Printf("[%s] PUSH %s → cmd:heapmem-metal:0", vtid, inst.Opcode)

	// heap 操作通常同步快速完成
	done, err := waitDone(ctx, rdb, vtid, 5*time.Second)
	if err != nil {
		setError(ctx, rdb, vtid, pc, fmt.Sprintf("heap op timeout: %v", err))
		return err
	}
	_ = done

	setState(ctx, rdb, vtid, ir.NextPC(pc), "running")
	return nil
}

// dispatchIf 处理 IF 条件分支
func dispatchIf(ctx context.Context, rdb *redis.Client, vtid string, pc string, inst *ir.Instruction) error {
	// 解析条件值
	if len(inst.Reads) == 0 {
		return fmt.Errorf("if without condition")
	}

	condVal := inst.Reads[0]
	// 如果是命名槽位引用，读取实际值
	if isRelative(condVal) {
		resolvedKey := "/vthread/" + vtid + "/" + condVal[2:]
		val, err := rdb.Get(ctx, resolvedKey).Result()
		if err == nil {
			condVal = val
		}
	}

	// 评估条件
	cond := isTruthy(condVal)
	var branchPC string
	if cond {
		branchPC = pc + "/true/0"
	} else {
		branchPC = pc + "/false/0"
	}

	log.Printf("[%s] IF %v → %s", vtid, cond, branchPC)
	setState(ctx, rdb, vtid, branchPC, "running")
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
