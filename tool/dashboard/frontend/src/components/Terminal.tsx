import { useEffect, useRef, useCallback } from 'react';
import { Terminal as XTerm } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import { WebLinksAddon } from '@xterm/addon-web-links';
import { termWsUrl } from '../api/client';
import '@xterm/xterm/css/xterm.css';

interface Props {
  active: boolean;
}

export function TerminalPanel({ active }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const xtermRef = useRef<XTerm | null>(null);
  const fitRef = useRef<FitAddon | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const activeRef = useRef(active);
  activeRef.current = active;

  const connectWs = useCallback((term: XTerm, fit: FitAddon) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(termWsUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      // Send initial terminal size
      if (fitRef.current) {
        const dims = fitRef.current.proposeDimensions();
        if (dims) {
          ws.send(JSON.stringify({ cols: dims.cols, rows: dims.rows }));
        }
      }
    };

    ws.onmessage = (event) => {
      if (event.data instanceof Blob) {
        event.data.arrayBuffer().then((buf) => {
          if (activeRef.current) {
            term.write(new Uint8Array(buf));
          }
        });
      } else {
        if (activeRef.current) {
          term.write(event.data);
        }
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
      // Reconnect after 2s
      if (activeRef.current) {
        setTimeout(() => {
          if (activeRef.current && xtermRef.current && fitRef.current) {
            connectWs(xtermRef.current, fitRef.current);
          }
        }, 2000);
      }
    };

    ws.onerror = () => {
      ws.close();
    };

    term.onData((data) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(data);
      }
    });

    // Resize handler
    const onResize = () => {
      if (fitRef.current) {
        fitRef.current.fit();
        if (ws.readyState === WebSocket.OPEN) {
          const dims = fitRef.current.proposeDimensions();
          if (dims) {
            ws.send(JSON.stringify({ cols: dims.cols, rows: dims.rows }));
          }
        }
      }
    };

    // Debounced resize observer
    let resizeTimer: ReturnType<typeof setTimeout>;
    const ro = new ResizeObserver(() => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(onResize, 100);
    });

    if (containerRef.current) {
      ro.observe(containerRef.current);
    }

    // Cleanup function stored for disconnection
    const cleanup = () => ro.disconnect();
    (term as any).__cleanup = cleanup;
  }, []);

  useEffect(() => {
    if (!active || !containerRef.current) return;

    // Destroy existing terminal if needed
    if (xtermRef.current) {
      // Tab switched away — we can reconnect if needed
      if (fitRef.current) {
        setTimeout(() => fitRef.current?.fit(), 50);
      }
      return;
    }

    const term = new XTerm({
      cursorBlink: true,
      cursorStyle: 'bar',
      fontSize: 13,
      fontFamily: "'SF Mono', 'Fira Code', 'Cascadia Code', Menlo, Monaco, monospace",
      theme: {
        background: '#0d1117',
        foreground: '#e6edf3',
        cursor: '#58a6ff',
        cursorAccent: '#0d1117',
        selectionBackground: '#3b5998',
        black: '#484f58',
        red: '#f85149',
        green: '#3fb950',
        yellow: '#d29922',
        blue: '#58a6ff',
        magenta: '#a371f7',
        cyan: '#39c5cf',
        white: '#b1bac4',
        brightBlack: '#6e7681',
        brightRed: '#ff7b72',
        brightGreen: '#56d364',
        brightYellow: '#e3b341',
        brightBlue: '#79c0ff',
        brightMagenta: '#d2a8ff',
        brightCyan: '#56d4dd',
        brightWhite: '#f0f6fc',
      },
      allowProposedApi: true,
      allowTransparency: false,
      scrollback: 5000,
      tabStopWidth: 4,
    });

    const fit = new FitAddon();
    const webLinks = new WebLinksAddon();

    term.loadAddon(fit);
    term.loadAddon(webLinks);
    term.open(containerRef.current);
    fit.fit();

    xtermRef.current = term;
    fitRef.current = fit;

    connectWs(term, fit);

    return () => {
      // Cleanup
      const t = term as any;
      if (t.__cleanup) t.__cleanup();
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      term.dispose();
      xtermRef.current = null;
      fitRef.current = null;
    };
  }, [active, connectWs]);

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        overflow: 'hidden',
      }}
    />
  );
}
