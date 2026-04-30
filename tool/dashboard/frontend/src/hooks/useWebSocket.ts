import { useEffect, useRef, useState, useCallback } from 'react';

interface SseMessage {
  type: string;
  ts?: number;
  data?: unknown;
}

/**
 * useSSE — Server-Sent Events hook.
 * Auto-reconnects on disconnect (browser EventSource handles this natively,
 * but we add a manual fallback with reconnect timer).
 */
export function useSSE(url: string, onMessage: (msg: SseMessage) => void) {
  const [connected, setConnected] = useState(false);
  const esRef = useRef<EventSource | null>(null);
  const reconnectTimer = useRef<number>(0);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  const connect = useCallback(() => {
    if (esRef.current) {
      esRef.current.close();
    }

    const es = new EventSource(url);
    esRef.current = es;

    es.onopen = () => {
      setConnected(true);
      console.log('[sse] connected');
    };

    es.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as SseMessage;
        onMessageRef.current(msg);
      } catch (e) {
        console.error('[sse] parse error:', e);
      }
    };

    es.onerror = () => {
      setConnected(false);
      es.close();
      esRef.current = null;
      console.log('[sse] disconnected, reconnecting in 3s...');
      reconnectTimer.current = window.setTimeout(connect, 3000);
    };
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      esRef.current?.close();
    };
  }, [connect]);

  return { connected };
}
