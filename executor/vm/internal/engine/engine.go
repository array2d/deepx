package engine

import (
	"context"
	"fmt"
	"log"

	"git.array2d.com/ai/deepx/executor/vm/internal/ir"
	"github.com/redis/go-redis/v9"
)

// mgetAll 批量读取指定 base 路径下的所有编译层/源码层指令
// base 格式: "/op/op-metal/func/gemm" 或 "/src/func/gemm"
func mgetAll(ctx context.Context, rdb *redis.Client, base string) []string {
	keys, err := rdb.Keys(ctx, base+"/*").Result()
	if err != nil || len(keys) == 0 {
		return nil
	}

	vals, err := rdb.MGet(ctx, keys...).Result()
	if err != nil {
		return nil
	}

	result := make([]string, 0, len(vals))
	for _, v := range vals {
		if s, ok := v.(string); ok {
			result = append(result, s)
		}
	}
	return result
}

// replaceParams 将形参替换为实参 (原地修改)
func replaceParams(params []string, bindings map[string]string) {
	for i, p := range params {
		if v, ok := bindings[p]; ok {
			params[i] = v
		}
	}
}

// RunWorker 单个 worker 的主循环
func RunWorker(ctx context.Context, rdb *redis.Client, id int) {
	log.Printf("worker-%d started", id)
	for {
		select {
		case <-ctx.Done():
			log.Printf("worker-%d stopped", id)
			return
		default:
		}

		vtid := PickVthread(ctx, rdb)
		if vtid == "" {
			WaitForVthread(ctx, rdb)
			continue
		}

		log.Printf("worker-%d picked vthread %s", id, vtid)
		Execute(ctx, rdb, vtid)
		// 注意: VM 不负责清理 vthread key，由调用方在读取结果后自行清理
	}
}

// Execute 执行一个 vthread 直到完成或出错
func Execute(ctx context.Context, rdb *redis.Client, vtid string) {
	for {
		state := getState(ctx, rdb, vtid)
		if state.Status == "done" || state.Status == "error" {
			return
		}

		pc := state.PC
		inst, err := ir.Decode(ctx, rdb, vtid, pc)
		if err != nil {
			log.Printf("[%s] decode error at %s: %v", vtid, pc, err)
			setError(ctx, rdb, vtid, pc, fmt.Sprintf("decode: %v", err))
			return
		}

		if inst.Opcode == "" {
			// 无更多指令 → vthread done
			log.Printf("[%s] done (no more instructions at %s)", vtid, pc)
			setState(ctx, rdb, vtid, pc, "done")
			return
		}

		log.Printf("[%s] PC=%s OP=%s READS=%v WRITES=%v", vtid, pc, inst.Opcode, inst.Reads, inst.Writes)

		var execErr error

		switch {
		case ir.IsControlOp(inst.Opcode):
			// 控制流: CALL / RETURN / IF
			execErr = dispatchControl(ctx, rdb, vtid, pc, inst)

		case ir.IsNativeOp(inst.Opcode):
			// 原生求值: + - * / == != < > && || ! & | ^ << >>
			execErr = dispatchNative(ctx, rdb, vtid, pc, inst)

		case ir.IsLifecycleOp(inst.Opcode):
			// 生命周期: newtensor / deltensor / clonetensor
			execErr = dispatchLifecycle(ctx, rdb, vtid, pc, inst)

		case ir.IsComputeOp(inst.Opcode):
			// 张量计算 → op-plat
			execErr = dispatchCompute(ctx, rdb, vtid, pc, inst)

		default:
			setState(ctx, rdb, vtid, ir.NextPC(pc), "running")
		}

		if execErr != nil {
			log.Printf("[%s] error: %v", vtid, execErr)
			return
		}
	}
}

// dispatchControl 处理控制流指令
func dispatchControl(ctx context.Context, rdb *redis.Client, vtid string, pc string, inst *ir.Instruction) error {
	switch inst.Opcode {
	case "call":
		substackPC := handleCall(ctx, rdb, vtid, pc, inst)
		setState(ctx, rdb, vtid, substackPC, "running")
		log.Printf("[%s] CALL → substack %s", vtid, substackPC)
		return nil

	case "return":
		parentPC := handleReturn(ctx, rdb, vtid, pc)
		log.Printf("[%s] RETURN → parent %s", vtid, parentPC)

		// 检查是否回到根栈末尾
		if parentPC == pc {
			// 根栈 return
			setState(ctx, rdb, vtid, pc, "done")
			return nil
		}
		setState(ctx, rdb, vtid, parentPC, "running")
		return nil

	case "if":
		return dispatchIf(ctx, rdb, vtid, pc, inst)

	default:
		return fmt.Errorf("unknown control opcode: %s", inst.Opcode)
	}
}
