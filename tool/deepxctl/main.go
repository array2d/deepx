package main

import (
	"fmt"
	"os"
	"path/filepath"

	"deepx/tool/deepxctl/cmd"
	"deepx/tool/deepxctl/cmd/tensor"
	"deepx/tool/deepxctl/internal/logx"
)

var version = "0.2.0"

func printUsage() {
	execName := filepath.Base(os.Args[0])
	fmt.Printf("Usage: %s [command] [arguments]\n\n", execName)
	fmt.Println("Commands:")
	fmt.Println("  boot       Start services (Redis → build → launch op-metal + heap-metal + VM)")
	fmt.Println("  run        Run a .dx file (requires prior boot)")
	fmt.Println("  shutdown   Stop all booted services")
	fmt.Println("  tensor     Tensor file operations (print)")
	fmt.Println("  version    Show version")
	fmt.Println("  help       Show this help")
	fmt.Println()
	fmt.Println("Typical workflow:")
	fmt.Printf("  %s boot                     # start services once\n", execName)
	fmt.Printf("  %s run file.dx              # execute .dx (repeatable)\n", execName)
	fmt.Printf("  %s shutdown                 # stop services when done\n", execName)
	fmt.Println()
	fmt.Printf("Run '%s [command] --help' for per-command flags.\n", execName)
}

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(0)
	}

	subcmd := os.Args[1]

	switch subcmd {
	case "boot":
		cmd.Boot(os.Args[2:])

	case "run":
		cmd.Run(os.Args[2:])

	case "shutdown":
		cmd.Shutdown()

	case "tensor":
		os.Args = os.Args[1:]
		tensor.Execute()

	case "version":
		fmt.Printf("deepxctl version %s\n", version)

	case "help", "-h", "--help":
		printUsage()

	default:
		logx.Error("unknown command", "cmd", subcmd)
		fmt.Fprintln(os.Stderr)
		printUsage()
		os.Exit(1)
	}
}
