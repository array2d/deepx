package main

import (
	"fmt"
	"os"
	"path/filepath"

	"deepx/tool/deepxctl/cmd"
	"deepx/tool/deepxctl/cmd/tensor"
)

var version = "0.2.0"

func printUsage() {
	execName := filepath.Base(os.Args[0])
	fmt.Printf("Usage: %s [command] [arguments]\n\n", execName)
	fmt.Println("Commands:")
	fmt.Println("  run        Run a .dx file (one-shot: Redis → build → launch → load → exec)")
	fmt.Println("  tensor     Tensor file operations (print)")
	fmt.Println("  version    Show version")
	fmt.Println("  help       Show this help")
	fmt.Println()
	fmt.Printf("Run '%s run --help' for run flags.\n", execName)
}

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(0)
	}

	subcmd := os.Args[1]

	switch subcmd {
	case "run":
		os.Args = os.Args[1:] // strip "deepxctl", keep "run ..."
		cmd.Execute(os.Args[1:])

	case "tensor":
		os.Args = os.Args[1:]
		tensor.Execute()

	case "version":
		fmt.Printf("deepxctl version %s\n", version)

	case "help", "-h", "--help":
		printUsage()

	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", subcmd)
		printUsage()
		os.Exit(1)
	}
}
