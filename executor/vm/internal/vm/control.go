package vm

import (
	"context"
	"fmt"

	"deepx/executor/vm/internal/ir"
	"deepx/executor/vm/internal/logx"
	"deepx/executor/vm/internal/state"
	"github.com/redis/go-redis/v9"
)

// If 处理 if 控制流指令。
// if(cond) -> branch_true, branch_false
// cond 为 true 时跳转到第一个 reads，否则第二个。
func If(ctx context.Context, rdb *redis.Client, vtid, pc string, inst *ir.Instruction) error {
	if len(inst.Reads) < 1 {
		return fmt.Errorf("if requires condition")
	}

	// Evaluate condition
	condKey := inst.Reads[0]
	var condVal string
	var err error
	if len(condKey) >= 2 && condKey[:2] == "./" {
		condVal, err = rdb.Get(ctx, "/vthread/"+vtid+"/"+condKey[2:]).Result()
	} else {
		condVal = condKey
	}
	if err != nil {
		condVal = condKey
	}

	isTrue := condVal != "" && condVal != "0" && condVal != "false"
	var target string
	if isTrue {
		if len(inst.Reads) > 1 {
			target = inst.Reads[1]
		}
	} else {
		if len(inst.Reads) > 2 {
			target = inst.Reads[2]
		}
	}

	if target == "" {
		// No branch → skip
		logx.Debug("[%s] IF %q → no branch, skip", vtid, condVal)
		state.Set(ctx, rdb, vtid, ir.NextPC(pc), "running")
		return nil
	}

	logx.Debug("[%s] IF %q → %s", vtid, condVal, target)
	state.Set(ctx, rdb, vtid, target, "running")
	return nil
}
