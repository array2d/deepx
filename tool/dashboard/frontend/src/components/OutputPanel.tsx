import { useState, useEffect } from 'react';
import { VThreadInfo, fetchVThreads } from '../api/client';
import { TerminalPanel } from './Terminal';

interface Props {
  runResult: VThreadInfo | null;
  runError: string | null;
  vthreadStatus: VThreadInfo | null;
  vthreadLoading: boolean;
  vthreadError: string | null;
  isRunning: boolean;
}

type Tab = 'output' | 'vthreads' | 'history' | 'terminal';

interface HistoryEntry {
  vtid: number;
  entry: string;
  status: string;
  duration_ms?: number;
  error?: { code: string; message: string };
  timestamp: number;
}

export function OutputPanel({
  runResult, runError, vthreadStatus, vthreadLoading, vthreadError, isRunning,
}: Props) {
  const [tab, setTab] = useState<Tab>('output');
  const [vthreads, setVThreads] = useState<VThreadInfo[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>(() => {
    try {
      return JSON.parse(localStorage.getItem('dx-run-history') || '[]');
    } catch { return []; }
  });

  // Poll vthread list periodically
  useEffect(() => {
    const load = async () => {
      try {
        const data = await fetchVThreads();
        setVThreads(data.vthreads);
      } catch { /* ignore */ }
    };
    load();
    const timer = setInterval(load, 3000);
    return () => clearInterval(timer);
  }, []);

  // Save to history on completion
  useEffect(() => {
    if (runResult && (runResult.status === 'done' || runResult.status === 'error')) {
      const entry: HistoryEntry = {
        vtid: runResult.vtid,
        entry: 'main',
        status: runResult.status,
        duration_ms: runResult.duration_ms,
        error: runResult.error,
        timestamp: Date.now(),
      };
      const updated = [entry, ...history].slice(0, 50);
      setHistory(updated);
      localStorage.setItem('dx-run-history', JSON.stringify(updated));
    }
  }, [runResult]);

  const vtStatus = vthreadStatus || runResult;

  return (
    <aside className="panel output-panel">
      <div className="tabs">
        <button
          className={`tab ${tab === 'output' ? 'active' : ''}`}
          onClick={() => setTab('output')}
        >
          Output
        </button>
        <button
          className={`tab ${tab === 'vthreads' ? 'active' : ''}`}
          onClick={() => setTab('vthreads')}
        >
          VThreads
        </button>
        <button
          className={`tab ${tab === 'history' ? 'active' : ''}`}
          onClick={() => setTab('history')}
        >
          History
        </button>
        <button
          className={`tab ${tab === 'terminal' ? 'active' : ''}`}
          onClick={() => setTab('terminal')}
        >
          Terminal
        </button>
      </div>

      <div className="tab-content">
        {tab === 'output' && (
          <OutputTab
            status={vtStatus}
            error={runError || vthreadError}
            loading={vthreadLoading || isRunning}
          />
        )}
        {tab === 'vthreads' && <VThreadsTab vthreads={vthreads} />}
        {tab === 'history' && <HistoryTab history={history} />}
        {tab === 'terminal' && <TerminalPanel active={tab === 'terminal'} />}
      </div>
    </aside>
  );
}

// ── Output Tab ──

function OutputTab({ status, error, loading }: {
  status: VThreadInfo | null;
  error: string | null;
  loading: boolean;
}) {
  if (loading && !status) {
    return (
      <div className="output-placeholder">
        <span className="spinner" />
        <p>Waiting for execution...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="output-error">
        <div className="result-header error">
          <span>✗ Error</span>
        </div>
        <pre className="error-msg">{error}</pre>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="output-placeholder">
        <p>No execution yet.</p>
        <p className="hint">Enter dxlang code and press ▶ Run</p>
      </div>
    );
  }

  if (status.status === 'done') {
    return (
      <div className="output-success">
        <div className="result-header success">
          <span>✓ Execution completed</span>
        </div>
        <div className="result-details">
          <div className="detail-row">
            <span className="detail-label">vtid</span>
            <span className="detail-value">{status.vtid}</span>
          </div>
          <div className="detail-row">
            <span className="detail-label">status</span>
            <span className="detail-value success-text">{status.status}</span>
          </div>
          <div className="detail-row">
            <span className="detail-label">pc</span>
            <span className="detail-value mono">{status.pc}</span>
          </div>
          {status.duration_ms && (
            <div className="detail-row">
              <span className="detail-label">duration</span>
              <span className="detail-value">{formatDuration(status.duration_ms)}</span>
            </div>
          )}
        </div>
      </div>
    );
  }

  if (status.status === 'error') {
    return (
      <div className="output-failure">
        <div className="result-header error">
          <span>✗ Execution failed</span>
        </div>
        <div className="result-details">
          <div className="detail-row">
            <span className="detail-label">vtid</span>
            <span className="detail-value">{status.vtid}</span>
          </div>
          <div className="detail-row">
            <span className="detail-label">status</span>
            <span className="detail-value error-text">{status.status}</span>
          </div>
          <div className="detail-row">
            <span className="detail-label">pc</span>
            <span className="detail-value mono">{status.pc}</span>
          </div>
          {status.error && (
            <>
              <div className="detail-row">
                <span className="detail-label">code</span>
                <span className="detail-value error-text">{status.error.code}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">message</span>
                <span className="detail-value">{status.error.message}</span>
              </div>
            </>
          )}
        </div>
      </div>
    );
  }

  // init, running, wait
  return (
    <div className="output-progress">
      <span className="spinner" />
      <div className="result-details">
        <div className="detail-row">
          <span className="detail-label">vtid</span>
          <span className="detail-value">{status.vtid}</span>
        </div>
        <div className="detail-row">
          <span className="detail-label">status</span>
          <span className="detail-value status-running">{status.status}</span>
        </div>
        <div className="detail-row">
          <span className="detail-label">pc</span>
          <span className="detail-value mono">{status.pc}</span>
        </div>
      </div>
    </div>
  );
}

// ── VThreads Tab ──

const STATUS_ORDER: Record<string, number> = {
  init: 0, running: 1, wait: 2, done: 3, error: 4,
};

function VThreadsTab({ vthreads }: { vthreads: VThreadInfo[] }) {
  const sorted = [...vthreads].sort((a, b) =>
    (STATUS_ORDER[a.status] ?? 99) - (STATUS_ORDER[b.status] ?? 99)
  );

  if (sorted.length === 0) {
    return <div className="output-placeholder"><p>No active vthreads</p></div>;
  }

  return (
    <table className="vt-table">
      <thead>
        <tr>
          <th>Vtid</th>
          <th>Status</th>
          <th>PC</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((vt) => (
          <tr key={vt.vtid} className={`vt-row ${vt.status}`}>
            <td className="mono">{vt.vtid}</td>
            <td>
              <span className={`vt-status-badge ${vt.status}`}>{vt.status}</span>
            </td>
            <td className="mono pc-cell">{vt.pc || '-'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ── History Tab ──

function HistoryTab({ history }: { history: HistoryEntry[] }) {
  if (history.length === 0) {
    return <div className="output-placeholder"><p>No execution history</p></div>;
  }

  return (
    <div className="history-list">
      {history.map((h, i) => (
        <div key={i} className={`history-item ${h.status}`}>
          <div className="history-top">
            <span className="mono">vtid:{h.vtid}</span>
            <span className={`vt-status-badge ${h.status}`}>{h.status}</span>
            {h.duration_ms && (
              <span className="history-time">{formatDuration(h.duration_ms)}</span>
            )}
          </div>
          {h.error && (
            <div className="history-error">{h.error.code}: {h.error.message}</div>
          )}
        </div>
      ))}
    </div>
  );
}

// ── Helpers ──

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}
