// Package logx provides a slog-based structured logger for deepxctl.
//
// Log level is controlled via the DEEPXCTL_LOG_LEVEL environment variable:
//
//	debug  — show all messages (Debug, Info, Warn, Error)
//	info   — show Info, Warn, Error (default)
//	warn   — show Warn, Error only
//	error  — show Error only
//
// Internal debug/diagnostic messages should use Debug level so they are
// suppressed by default and only visible when explicitly requested.
package logx

import (
	"log/slog"
	"os"
	"strings"
)

// Logger is the package-level structured logger.
// All deepxctl packages should use this for logging.
var Logger *slog.Logger

func init() {
	level := parseLevel(os.Getenv("DEEPXCTL_LOG_LEVEL"))
	Logger = slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: level,
	}))
	slog.SetDefault(Logger)
}

// parseLevel converts a string to an slog.Level, defaulting to Info.
func parseLevel(s string) slog.Level {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "debug":
		return slog.LevelDebug
	case "info":
		return slog.LevelInfo
	case "warn", "warning":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}

// Debug logs a debug message (suppressed at default info level).
func Debug(msg string, args ...any) {
	Logger.Debug(msg, args...)
}

// Info logs an informational message.
func Info(msg string, args ...any) {
	Logger.Info(msg, args...)
}

// Warn logs a warning message.
func Warn(msg string, args ...any) {
	Logger.Warn(msg, args...)
}

// Error logs an error message.
func Error(msg string, args ...any) {
	Logger.Error(msg, args...)
}
