// Package ast 定义 dxlang 抽象语法树节点类型。
package ast

import (
	"context"
	"fmt"

	"github.com/redis/go-redis/v9"
)

// File 表示一个完整的 .dx 源文件。
type File struct {
	Funcs         []Func
	TopLevelCalls []TopLevelCall
}

// Func 表示一个函数定义。
type Func struct {
	Name      string   // 函数名
	Signature string   // def funcName(A:int, B:int) -> (C:int)
	Body      []string // dxlang 指令行
}

// TopLevelCall 表示 def 块外部的顶层调用表达式。
type TopLevelCall struct {
	FuncName string   // 函数名
	Args     []string // 实参
	Outputs  []string // 输出槽位
}

// Instruction 表示解析后的单条 dxlang 指令。
type Instruction struct {
	Opcode string   // 操作码
	Reads  []string // 输入参数
	Writes []string // 输出槽位
}

// FormalParams 表示函数签名的形参列表。
type FormalParams struct {
	Reads  []string
	Writes []string
}

// Register 将函数注册到 Redis KV 空间：/src/func/<name> = 签名，/src/func/<name>/<i> = 指令行。
func (f *Func) Register(ctx context.Context, rdb *redis.Client) error {
	if err := rdb.Set(ctx, "/src/func/"+f.Name, f.Signature, 0).Err(); err != nil {
		return fmt.Errorf("register sig: %w", err)
	}
	for i, line := range f.Body {
		key := fmt.Sprintf("/src/func/%s/%d", f.Name, i)
		if err := rdb.Set(ctx, key, line, 0).Err(); err != nil {
			return fmt.Errorf("register body[%d]: %w", i, err)
		}
	}
	return nil
}
