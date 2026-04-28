package engine

import (
	"context"
	"fmt"
	"strings"

	"git.array2d.com/ai/deepx/executor/vm/internal/ir"
	"git.array2d.com/ai/deepx/executor/vm/internal/route"
	"github.com/redis/go-redis/v9"
)

// handleCall 执行 CALL 指令的 eager 翻译
// 返回子栈第一条指令的 PC
func handleCall(ctx context.Context, rdb *redis.Client, vtid string, pc string, inst *ir.Instruction) string {
	funcName := inst.Reads[0]

	// 1. 确定 backend
	backend := route.DetermineBackend(ctx, rdb, funcName)

	// 2. 读取编译层函数签名
	sig, err := rdb.Get(ctx, fmt.Sprintf("/op/%s/func/%s", backend, funcName)).Result()
	if err != nil {
		sig, err = rdb.Get(ctx, "/src/func/"+funcName).Result()
		if err != nil {
			panic(fmt.Sprintf("func %s not found in /op/%s/func/ or /src/func/", funcName, backend))
		}
	}

	// 3. 解析签名 → 形参列表
	formalParams := parseSignature(sig)

	// 4. 建立形参→实参映射
	bindings := make(map[string]string)
	for i, param := range formalParams.Reads {
		if i+1 < len(inst.Reads) {
			bindings[param] = inst.Reads[i+1]
		}
	}
	for i, param := range formalParams.Writes {
		if i < len(inst.Writes) {
			bindings[param] = inst.Writes[i]
		}
	}

	// 5. 批量 MGET 编译层所有指令
	compiled := mgetAll(ctx, rdb, fmt.Sprintf("/op/%s/func/%s", backend, funcName))
	if len(compiled) == 0 {
		compiled = mgetAll(ctx, rdb, "/src/func/"+funcName)
	}

	// 6. 逐条翻译 → Pipeline 批量写入子栈
	substackRoot := fmt.Sprintf("/vthread/%s/%s/", vtid, pc)
	pipe := rdb.Pipeline()

	bodyCount := len(compiled)
	for i, dxlangLine := range compiled {
		parsed, err := ir.ParseDxlang(dxlangLine)
		if err != nil {
			continue
		}

		replaceParams(parsed.Reads, bindings)
		replaceParams(parsed.Writes, bindings)

		pipe.Set(ctx, fmt.Sprintf("%s[%d,0]", substackRoot, i), parsed.Opcode, 0)
		for j, r := range parsed.Reads {
			pipe.Set(ctx, fmt.Sprintf("%s[%d,-%d]", substackRoot, i, j+1), r, 0)
		}
		for j, w := range parsed.Writes {
			pipe.Set(ctx, fmt.Sprintf("%s[%d,%d]", substackRoot, i, j+1), w, 0)
		}
	}

	// 7. 追加隐式 return 指令 (将最后一个输出形参的值回传父栈)
	if len(formalParams.Writes) > 0 {
		retIdx := bodyCount
		retSlot := formalParams.Writes[0] // e.g., "C"
		pipe.Set(ctx, fmt.Sprintf("%s[%d,0]", substackRoot, retIdx), "return", 0)
		pipe.Set(ctx, fmt.Sprintf("%s[%d,-1]", substackRoot, retIdx), "./"+retSlot, 0)
	}

	_, err = pipe.Exec(ctx)
	if err != nil {
		panic(fmt.Sprintf("CALL translate pipeline failed: %v", err))
	}

	// 8. 返回子栈第一条指令的 PC
	return pc + "/[0,0]"
}

// handleReturn 处理 RETURN 指令
// 返回父栈 CALL 指令的下一条 PC
func handleReturn(ctx context.Context, rdb *redis.Client, vtid string, pc string) string {
	lastSlash := strings.LastIndex(pc, "/")
	if lastSlash < 0 {
		return pc // 根栈 return → vthread 即将 done
	}

	parentPC := pc[:lastSlash]

	// 1. 读取返回值写入父 CALL 的返回值槽位
	inst, err := ir.Decode(ctx, rdb, vtid, pc)
	if err == nil {
		parentInst, err := ir.Decode(ctx, rdb, vtid, parentPC)
		if err == nil && len(parentInst.Writes) > 0 && len(inst.Reads) > 0 {
			retSlot := parentInst.Writes[0] // e.g., "./c"
			retRef := inst.Reads[0]         // e.g., "./C" (slot reference)
			// 解析 retRef: 如果是相对路径则读取其实际值
			retVal := retRef
			if strings.HasPrefix(retRef, "./") {
				srcKey := "/vthread/" + vtid + "/" + retRef[2:]
				if v, e := rdb.Get(ctx, srcKey).Result(); e == nil {
					retVal = v
				}
			}
			if strings.HasPrefix(retSlot, "./") {
				slotKey := "/vthread/" + vtid + "/" + retSlot[2:]
				rdb.Set(ctx, slotKey, retVal, 0)
			}
		}
	}

	// 2. 删除当前子栈
	keys, _ := rdb.Keys(ctx, "/vthread/"+vtid+"/"+parentPC+"/*").Result()
	if len(keys) > 0 {
		rdb.Del(ctx, keys...)
	}

	// 3. PC 恢复到父栈 CALL 的下一条
	return ir.NextPC(parentPC)
}

// parseSignature 解析函数签名
//
//	"def add_test(A:int, B:int) -> (C:int)"   → Reads:["A","B"], Writes:["C"]
//	"(add_test(A, B) -> (C))"                 → Reads:["A","B"], Writes:["C"] (legacy)
func parseSignature(sig string) FormalParams {
	var fp FormalParams
	sig = strings.TrimSpace(sig)

	// strip "def " prefix (new format)
	if strings.HasPrefix(sig, "def ") {
		sig = strings.TrimSpace(sig[4:])
	}

	// strip outer parens (legacy format)
	if len(sig) >= 2 && sig[0] == '(' && sig[len(sig)-1] == ')' {
		sig = sig[1 : len(sig)-1]
	}

	arrow := strings.Index(sig, "->")
	if arrow < 0 {
		return fp
	}

	left := strings.TrimSpace(sig[:arrow])
	right := strings.TrimSpace(sig[arrow+2:])

	if lp := strings.Index(left, "("); lp >= 0 {
		rp := strings.LastIndex(left, ")")
		if rp > lp {
			fp.Reads = extractParamNames(left[lp+1 : rp])
		}
	}

	right = strings.TrimSpace(right)
	if len(right) >= 2 && right[0] == '(' && right[len(right)-1] == ')' {
		fp.Writes = extractParamNames(right[1 : len(right)-1])
	} else {
		fp.Writes = extractParamNames(right)
	}

	return fp
}

// FormalParams 函数形参列表
type FormalParams struct {
	Reads  []string
	Writes []string
}

// extractParamNames 从 "A:tensor, B:tensor, alpha:f32" 提取 ["A", "B", "alpha"]
func extractParamNames(s string) []string {
	var names []string
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if colon := strings.Index(p, ":"); colon >= 0 {
			p = p[:colon]
		}
		p = strings.TrimSpace(p)
		if p != "" {
			names = append(names, p)
		}
	}
	return names
}
