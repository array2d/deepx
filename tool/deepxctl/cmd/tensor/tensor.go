package tensor

import (
	"fmt"
	"os"
)

func PrintUsage() {
	fmt.Println("使用方法:")
	fmt.Println("  tensor print <文件路径>")
	fmt.Println("  tensor help")
}

func Execute() {
	if len(os.Args) < 1 {
		PrintUsage()
		os.Exit(1)
	}

	subCmd := "help"
	if len(os.Args) > 0 {
		subCmd = os.Args[0]
	}

	switch subCmd {
	case "print":
		os.Args = os.Args[1:]
		PrintCmd()
	case "help":
		PrintUsage()
	default:
		fmt.Printf("未知的张量命令: %s\n", subCmd)
		PrintUsage()
		os.Exit(1)
	}
}
