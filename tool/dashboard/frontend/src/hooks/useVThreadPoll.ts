import { useState, useEffect, useRef, useCallback } from 'react';
import { VThreadInfo, fetchVThread } from '../api/client';

interface VThreadPollResult {
  status: VThreadInfo | null;
  loading: boolean;
  error: string | null;
  startPolling: (vtid: number) => void;
  stopPolling: () => void;
}

export function useVThreadPoll(): VThreadPollResult {
  const [status, setStatus] = useState<VThreadInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const vtidRef = useRef<number | null>(null);
  const timerRef = useRef<number>(0);

  const stopPolling = useCallback(() => {
    clearTimeout(timerRef.current);
    vtidRef.current = null;
  }, []);

  const startPolling = useCallback((vtid: number) => {
    stopPolling();
    vtidRef.current = vtid;
    setLoading(true);
    setError(null);
    setStatus(null);

    const poll = async () => {
      if (vtidRef.current !== vtid) return;
      try {
        const vt = await fetchVThread(vtid);
        setStatus(vt);
        setLoading(false);
        if (vt.status === 'done' || vt.status === 'error') {
          vtidRef.current = null;
          return;
        }
        timerRef.current = window.setTimeout(poll, 500);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'poll failed');
        setLoading(false);
        timerRef.current = window.setTimeout(poll, 2000);
      }
    };
    poll();
  }, [stopPolling]);

  useEffect(() => {
    return () => clearTimeout(timerRef.current);
  }, []);

  return { status, loading, error, startPolling, stopPolling };
}
