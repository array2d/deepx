package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/array2d/deepx/tool/deepxctl/cmd/tensor"
)

var version = "0.1.0"

func printUsage() {
	execName := filepath.Base(os.Args[0])
	fmt.Printf("用法: %s [命令] [参数]\n\n", execName)
	fmt.Println("可用命令:")
	fmt.Println("  tensor    张量操作相关命令")
	fmt.Println("  version   显示版本信息")
	fmt.Println("  help      显示帮助信息")
	fmt.Println("\n使用 '%s help [命令]' 获取命令的详细信息", execName)
}

func main() {
	flag.Usage = printUsage

	if len(os.Args) < 2 {
		printUsage()
		os.Exit(0)
	}

	// 获取子命令
	cmd := os.Args[1]

	// 根据子命令执行相应操作
	switch cmd {
	case "tensor":
		// 移除子命令，让子命令处理剩余的参数
		os.Args = os.Args[2:]
		tensor.Execute()

	case "version":
		fmt.Printf("deepxctl 版本 %s\n", version)

	case "help":
		if len(os.Args) > 2 {
			helpCmd := os.Args[2]
			switch helpCmd {
			case "tensor":
				tensor.PrintUsage()
			default:
				fmt.Printf("未知命令: %s\n", helpCmd)
				printUsage()
			}
		} else {
			printUsage()
		}

	default:
		fmt.Printf("未知命令: %s\n", cmd)
		printUsage()
		os.Exit(1)
	}
}
