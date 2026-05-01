// Package vm 提供 VM 核心执行循环与控制流指令。
package vm

import (
	"context"
	"fmt"

	"deepx/executor/vm/internal/codegen"
	"deepx/executor/vm/internal/platform"
	"deepx/executor/vm/internal/ir"
	"deepx/executor/vm/internal/logx"
	"deepx/executor/vm/internal/sched"
	"deepx/executor/vm/internal/state"
	"deepx/executor/vm/internal/termio"
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
		vtid := sched.Pick(ctx, rdb)
		if vtid == "" {
			sched.Wait(ctx, rdb)
			continue
		}
		logx.Debug("worker-%d picked vthread %s", id, vtid)
		Execute(ctx, rdb, vtid)
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
			logx.Debug("[%s] done at %s", vtid, pc)
			state.Set(ctx, rdb, vtid, pc, "done")
			return
		}
		logx.Debug("[%s] PC=%s OP=%s READS=%v WRITES=%v", vtid, pc, inst.Opcode, inst.Reads, inst.Writes)

		var execErr error
		switch {
		case ir.IsControlOp(inst.Opcode):
			execErr = handleControl(ctx, rdb, vtid, pc, inst)
		case ir.IsNativeOp(inst.Opcode):
			execErr = termio.Native(ctx, rdb, vtid, pc, inst)
		case ir.IsLifecycleOp(inst.Opcode):
			execErr = platform.Lifecycle(ctx, rdb, vtid, pc, inst)
		case isFunctionCall(ctx, rdb, inst.Opcode):
			inst.Reads = append([]string{inst.Opcode}, inst.Reads...)
			inst.Opcode = "call"
			execErr = handleControl(ctx, rdb, vtid, pc, inst)
		case ir.IsComputeOp(inst.Opcode):
			execErr = platform.Compute(ctx, rdb, vtid, pc, inst)
		default:
			state.Set(ctx, rdb, vtid, ir.NextPC(pc), "running")
		}
		if execErr != nil {
			logx.Debug("[%s] error: %v", vtid, execErr)
			return
		}
	}
}

func handleControl(ctx context.Context, rdb *redis.Client, vtid, pc string, inst *ir.Instruction) error {
	switch inst.Opcode {
	case "call":
		substackPC := codegen.HandleCall(ctx, rdb, vtid, pc, inst)
		if substackPC == pc {
			// HandleCall already set error state (func not found, parse failure, etc.)
			return fmt.Errorf("call %s failed", inst.Reads[0])
		}
		state.Set(ctx, rdb, vtid, substackPC, "running")
		logx.Debug("[%s] CALL → %s", vtid, substackPC)
		return nil
	case "return":
		parentPC := codegen.HandleReturn(ctx, rdb, vtid, pc)
		logx.Debug("[%s] RETURN → %s", vtid, parentPC)
		if parentPC == pc {
			state.Set(ctx, rdb, vtid, pc, "done")
			return nil
		}
		state.Set(ctx, rdb, vtid, parentPC, "running")
		return nil
	case "if":
		return If(ctx, rdb, vtid, pc, inst)
	default:
		return fmt.Errorf("unknown control op: %s", inst.Opcode)
	}
}

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
