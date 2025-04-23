package tensor

import (
	"flag"
	"fmt"
	"os"

	coretensor "github.com/array2d/deepx/tool/deepxctl/tensor"
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
	var shape coretensor.Shape
	shape, err = coretensor.LoadShape(tensorPath)
	if err != nil {
		fmt.Println("读取文件失败:", err)
	}
	switch shape.Dtype {
	case "bool":
		var t coretensor.Tensor[bool]
		t, err = coretensor.LoadTensor[bool](tensorPath)
		if err != nil {
			fmt.Println("读取文件失败:", err)
		}
		t.Print()
	case "int8":
		var t coretensor.Tensor[int8]
		t, err = coretensor.LoadTensor[int8](tensorPath)
		if err != nil {
			fmt.Println("读取文件失败:", err)
		}
		t.Print()
	case "int16":
		var t coretensor.Tensor[int16]
		t, err = coretensor.LoadTensor[int16](tensorPath)
		if err != nil {
			fmt.Println("读取文件失败:", err)
		}
		t.Print()
	case "int32":
		var t coretensor.Tensor[int32]
		t, err = coretensor.LoadTensor[int32](tensorPath)
		if err != nil {
			fmt.Println("读取文件失败:", err)
		}
		t.Print()
	case "int64":
		var t coretensor.Tensor[int64]
		t, err = coretensor.LoadTensor[int64](tensorPath)
		if err != nil {
			fmt.Println("读取文件失败:", err)
		}
		t.Print()
	case "float16":
		// var t coretensor.Tensor[float16]
		// t, err = coretensor.LoadTensor[float16](tensorPath)
		// if err != nil {
		// 	fmt.Println("读取文件失败:", err)
		// }
		// t.Print()
	case "float32":
		var t coretensor.Tensor[float32]
		t, err = coretensor.LoadTensor[float32](tensorPath)
		if err != nil {
			fmt.Println("读取文件失败:", err)
		}
		t.Print()
	case "float64":
		var t coretensor.Tensor[float64]
		t, err = coretensor.LoadTensor[float64](tensorPath)
		if err != nil {
			fmt.Println("读取文件失败:", err)
		}
		t.Print()
	}
}
