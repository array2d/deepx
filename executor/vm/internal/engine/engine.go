// Package engine 提供 VM 核心执行循环与指令分发。
//
// engine 是 VM 的编排者，协调 picker/dispatch/translate/state 等子包。
package engine

import (
	"context"
	"fmt"

	"deepx/executor/vm/internal/dispatch"
	"deepx/executor/vm/internal/logx"
	"deepx/executor/vm/internal/ir"
	"deepx/executor/vm/internal/picker"
	"deepx/executor/vm/internal/state"
	"deepx/executor/vm/internal/translate"
	"github.com/redis/go-redis/v9"
)

// RunWorker 单个 worker 的主循环。
func RunWorker(ctx context.Context, rdb *redis.Client, id int) {
	logx.Debug("worker-%d started", id)
	for {
		select {
		case <-ctx.Done():
			logx.Debug("worker-%d stopped", id)
			return
		default:
		}

		vtid := picker.PickVthread(ctx, rdb)
		if vtid == "" {
			picker.WaitForVthread(ctx, rdb)
			continue
		}

		logx.Debug("worker-%d picked vthread %s", id, vtid)
		Execute(ctx, rdb, vtid)
		// 注意: VM 不负责清理 vthread key，由调用方在读取结果后自行清理
	}
}

// Execute 执行一个 vthread 直到完成或出错。
func Execute(ctx context.Context, rdb *redis.Client, vtid string) {
	for {
		s := state.Get(ctx, rdb, vtid)
		if s.Status == "done" || s.Status == "error" {
			return
		}

		pc := s.PC
		inst, err := ir.Decode(ctx, rdb, vtid, pc)
		if err != nil {
			logx.Debug("[%s] decode error at %s: %v", vtid, pc, err)
			state.SetError(ctx, rdb, vtid, pc, fmt.Sprintf("decode: %v", err))
			return
		}

		if inst.Opcode == "" {
			logx.Debug("[%s] done (no more instructions at %s)", vtid, pc)
			state.Set(ctx, rdb, vtid, pc, "done")
			return
		}

		logx.Debug("[%s] PC=%s OP=%s READS=%v WRITES=%v", vtid, pc, inst.Opcode, inst.Reads, inst.Writes)

		var execErr error

		switch {
		case ir.IsControlOp(inst.Opcode):
			execErr = dispatchControl(ctx, rdb, vtid, pc, inst)

		case ir.IsNativeOp(inst.Opcode):
			execErr = dispatch.Native(ctx, rdb, vtid, pc, inst)

		case ir.IsLifecycleOp(inst.Opcode):
			execErr = dispatch.Lifecycle(ctx, rdb, vtid, pc, inst)

		case isFunctionCall(ctx, rdb, inst.Opcode):
			// 非内置关键字的标识符 → 函数调用
			// 将 funcName(A, B) -> ./C 转换为内部 call(funcName, A, B) -> ./C
			inst.Reads = append([]string{inst.Opcode}, inst.Reads...)
			inst.Opcode = "call"
			execErr = dispatchControl(ctx, rdb, vtid, pc, inst)

		case ir.IsComputeOp(inst.Opcode):
			execErr = dispatch.Compute(ctx, rdb, vtid, pc, inst)

		default:
			state.Set(ctx, rdb, vtid, ir.NextPC(pc), "running")
		}

		if execErr != nil {
			logx.Debug("[%s] error: %v", vtid, execErr)
			return
		}
	}
}

// dispatchControl 处理控制流指令 (call / return / if)。
func dispatchControl(ctx context.Context, rdb *redis.Client, vtid string, pc string, inst *ir.Instruction) error {
	switch inst.Opcode {
	case "call":
		substackPC := translate.HandleCall(ctx, rdb, vtid, pc, inst)
		state.Set(ctx, rdb, vtid, substackPC, "running")
		logx.Debug("[%s] CALL → substack %s", vtid, substackPC)
		return nil

	case "return":
		parentPC := translate.HandleReturn(ctx, rdb, vtid, pc)
		logx.Debug("[%s] RETURN → parent %s", vtid, parentPC)

		if parentPC == pc {
			state.Set(ctx, rdb, vtid, pc, "done")
			return nil
		}
		state.Set(ctx, rdb, vtid, parentPC, "running")
		return nil

	case "if":
		return dispatch.If(ctx, rdb, vtid, pc, inst)

	default:
		return fmt.Errorf("unknown control opcode: %s", inst.Opcode)
	}
}

// isFunctionCall 判断 opcode 是否是一个已注册的函数名 (而非算子)。
// 检查 /src/func/<name> 或 /op/*/func/<name> 是否存在。
func isFunctionCall(ctx context.Context, rdb *redis.Client, opcode string) bool {
	exists, err := rdb.Exists(ctx, "/src/func/"+opcode).Result()
	if err == nil && exists > 0 {
		return true
	}
	for _, backend := range []string{"op-metal", "op-cuda", "op-cpu"} {
		exists, err := rdb.Exists(ctx, fmt.Sprintf("/op/%s/func/%s", backend, opcode)).Result()
		if err == nil && exists > 0 {
			return true
		}
	}
	return false
}
