package funcs

type Var struct {
	funcspace *FuncSpace // 函数空间
	Name      string     // 变量名称
}

type Bool struct {
	Var
	Value bool
}

type Byte struct {
	Var
	Value byte
}

type Int8 struct {
	Var
	Value int8
}

type Int16 struct {
	Var
	Value int16
}

type IntPair struct {
	Var
	Value [2]int
}
