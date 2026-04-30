// Package logx 提供基于 log/slog 的级别日志，通过 LOG_LEVEL 环境变量控制。
//
//	LOG_LEVEL=debug  输出所有级别
//	LOG_LEVEL=info   输出 info/warn/error (默认)
//	LOG_LEVEL=warn   输出 warn/error
//	LOG_LEVEL=error  仅输出 error
//
// 用法: logx.Debug("worker-%d picked vthread %s", id, vtid)
package logx

import (
	"fmt"
	"log/slog"
	"os"
	"strings"
)

var level = new(slog.LevelVar)

func init() {
	switch strings.ToLower(os.Getenv("LOG_LEVEL")) {
	case "debug":
		level.Set(slog.LevelDebug)
	case "info", "":
		level.Set(slog.LevelInfo)
	case "warn":
		level.Set(slog.LevelWarn)
	case "error":
		level.Set(slog.LevelError)
	default:
		level.Set(slog.LevelInfo)
	}

	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: level,
	})
	slog.SetDefault(slog.New(handler))
}

// Debug 调试信息，仅在 LOG_LEVEL=debug 时输出。
func Debug(format string, args ...any) {
	slog.Debug(fmt.Sprintf(format, args...))
}

// Info 常规运行信息。
func Info(format string, args ...any) {
	slog.Info(fmt.Sprintf(format, args...))
}

// Warn 警告信息（可恢复的问题）。
func Warn(format string, args ...any) {
	slog.Warn(fmt.Sprintf(format, args...))
}

// Error 错误信息。
func Error(format string, args ...any) {
	slog.Error(fmt.Sprintf(format, args...))
}

// Fatal 输出错误日志后退出。
func Fatal(format string, args ...any) {
	slog.Error(fmt.Sprintf(format, args...))
	os.Exit(1)
}
