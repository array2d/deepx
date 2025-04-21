package tensor

import (
	"flag"
	"fmt"
	"os"

	coretensor "github.com/array2d/deepx/deepxctl/tensor"
)

func PrintCmd() {
	printCmd := flag.NewFlagSet("print", flag.ExitOnError)
	tensorPath := os.Args[0]
	if tensorPath == "" {
		fmt.Println("请指定文件路径")
		printCmd.Usage()
		return
	}
	var err error
	var t coretensor.Tensor
	t, err = coretensor.LoadTensor(tensorPath)
	if err != nil {
		fmt.Println("读取文件失败:", err)
	}
	t.Print()
}
