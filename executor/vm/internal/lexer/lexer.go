// Package lexer 提供 dxlang 词法分析：源代码行 → Token 流。
package lexer

import (
	"fmt"
	"strings"
)

// Kind 标记 Token 类型。
type Kind int

const (
	Ident     Kind = iota // 标识符 (opcode, 变量名)
	Literal                // 字面量 (数字, 字符串, 路径 "./a", "/data/x")
	Arrow                  // -> 或 <-
	LParen                 // (
	RParen                 // )
	Comma                  // ,
	LBrace                 // {
	RBrace                 // }
	Return                 // return
	If                     // if
	For                    // for
)

func (k Kind) String() string {
	switch k {
	case Ident:
		return "IDENT"
	case Literal:
		return "LITERAL"
	case Arrow:
		return "ARROW"
	case LParen:
		return "LPAREN"
	case RParen:
		return "RPAREN"
	case Comma:
		return "COMMA"
	case LBrace:
		return "LBRACE"
	case RBrace:
		return "RBRACE"
	case Return:
		return "RETURN"
	case If:
		return "IF"
	case For:
		return "FOR"
	default:
		return "UNKNOWN"
	}
}

// Token 表示一个词法单元。
type Token struct {
	Kind  Kind
	Value string
}

func (t Token) String() string { return fmt.Sprintf("%s(%q)", t.Kind, t.Value) }

// Tokenize 将一行 dxlang 代码分割为 Token 列表。
func Tokenize(line string) []Token {
	line = strings.TrimSpace(line)
	if line == "" {
		return nil
	}

	var tokens []Token
	i := 0
	for i < len(line) {
		c := line[i]

		// 空白
		if c == ' ' || c == '\t' {
			i++
			continue
		}

		// 单行注释
		if c == '#' {
			break
		}

		// 双引号字符串 → 完整读取
		if c == '"' {
			end := strings.IndexByte(line[i+1:], '"')
			if end >= 0 {
				val := line[i+1 : i+1+end]
				tokens = append(tokens, Token{Kind: Literal, Value: `"` + val + `"`})
				i += end + 2
			} else {
				// 未闭合引号 → 读到行尾
				tokens = append(tokens, Token{Kind: Literal, Value: line[i:]})
				i = len(line)
			}
			continue
		}

		// 左箭头 <-
		if c == '<' && i+1 < len(line) && line[i+1] == '-' {
			tokens = append(tokens, Token{Kind: Arrow, Value: "<-"})
			i += 2
			continue
		}

		// 右箭头 ->   (注意排除 ->arrow 后的连续字符)
		if c == '-' && i+1 < len(line) && line[i+1] == '>' {
			tokens = append(tokens, Token{Kind: Arrow, Value: "->"})
			i += 2
			continue
		}

		// 多字符算子 (在标识符之前匹配)
		if i+1 < len(line) {
			two := line[i : i+2]
			switch two {
			case "==", "!=", "<=", ">=", "&&", "||", "<<", ">>":
				tokens = append(tokens, Token{Kind: Ident, Value: two})
				i += 2
				continue
			}
		}

		// 单字符控制/括号
		switch c {
		case '(':
			tokens = append(tokens, Token{Kind: LParen, Value: "("})
			i++
			continue
		case ')':
			tokens = append(tokens, Token{Kind: RParen, Value: ")"})
			i++
			continue
		case ',':
			tokens = append(tokens, Token{Kind: Comma, Value: ","})
			i++
			continue
		case '{':
			tokens = append(tokens, Token{Kind: LBrace, Value: "{"})
			i++
			continue
		case '}':
			tokens = append(tokens, Token{Kind: RBrace, Value: "}"})
			i++
			continue
		}

		// 单字符符号算子
		switch c {
		case '+', '-', '*', '/', '%', '!', '<', '>', '&', '|', '^':
			tokens = append(tokens, Token{Kind: Ident, Value: string(c)})
			i++
			continue
		}

		// 数字字面量
		if c >= '0' && c <= '9' || c == '.' {
			start := i
			for i < len(line) && ((line[i] >= '0' && line[i] <= '9') || line[i] == '.' || line[i] == 'e' || line[i] == 'E' || line[i] == '-' || line[i] == '+') {
				i++
			}
			// 回退: 嵌套表达式中的减号/加号不应归入数字
			tokens = append(tokens, Token{Kind: Literal, Value: line[start:i]})
			continue
		}

		// 路径字面量 (./a, /data/x)
		if c == '.' && i+1 < len(line) && line[i+1] == '/' {
			start := i
			i += 2
			for i < len(line) && !isDelim(line[i]) {
				i++
			}
			tokens = append(tokens, Token{Kind: Literal, Value: line[start:i]})
			continue
		}
		if c == '/' {
			start := i
			i++
			for i < len(line) && !isDelim(line[i]) {
				i++
			}
			tokens = append(tokens, Token{Kind: Literal, Value: line[start:i]})
			continue
		}

		// 关键字
		start := i
		for i < len(line) && !isDelim(line[i]) {
			i++
		}
		word := line[start:i]
		switch word {
		case "return":
			tokens = append(tokens, Token{Kind: Return, Value: word})
		case "if":
			tokens = append(tokens, Token{Kind: If, Value: word})
		case "for":
			tokens = append(tokens, Token{Kind: For, Value: word})
		default:
			tokens = append(tokens, Token{Kind: Ident, Value: word})
		}
	}
	return tokens
}

func isDelim(c byte) bool {
	switch c {
	case ' ', '\t', ',', ')', '(', '{', '}', '+', '-', '*', '/', '%', '!', '=', '<', '>', '&', '|', '^':
		return true
	}
	return false
}
