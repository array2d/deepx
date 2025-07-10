package funcs

import "aispace"

type Func struct {
	// Name 函数名称
	Name    string              `json:"name"`
	Args    []aispace.CellValue `json:"args"`    // 函数参数列表
	Returns []aispace.CellValue `json:"returns"` // 函数返回值列表
}

type FuncSpace struct {
	// Name 函数名称
	root    Func
	Pointer *FuncSpace // 指向父函数空间
}
