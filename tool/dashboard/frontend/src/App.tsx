import { useState, useCallback } from 'react';
import { SystemStatus, VThreadInfo, RunRequest, runCode } from './api/client';
import { useSSE } from './hooks/useWebSocket';
import { useVThreadPoll } from './hooks/useVThreadPoll';
import { useResizable } from './hooks/useResizable';
import { StatusPanel } from './components/StatusPanel';
import { CodeEditor } from './components/CodeEditor';
import { OutputPanel } from './components/OutputPanel';
import { ResizeHandle } from './components/ResizeHandle';

export default function App() {
  const [sysStatus, setSysStatus] = useState<SystemStatus | null>(null);
  const [sseConnected, setSseConnected] = useState(false);
  const [runResult, setRunResult] = useState<VThreadInfo | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  const vthread = useVThreadPoll();

  // Resizable panels
  const leftPanel = useResizable({ initialSize: 260, minSize: 200, maxSize: 400, direction: 'left' });
  const rightPanel = useResizable({ initialSize: 320, minSize: 240, maxSize: 500, direction: 'right' });

  // SSE: receive real-time system status
  const { connected } = useSSE(
    '/api/status/stream',
    useCallback((msg) => {
      if (msg.type === 'status' && msg.data) {
        setSysStatus(msg.data as SystemStatus);
        setSseConnected(true);
      }
    }, [])
  );

  const isConnected = sseConnected || connected;

  // Handle code execution
  const handleRun = useCallback(async (req: RunRequest) => {
    setIsRunning(true);
    setRunError(null);
    setRunResult(null);
    try {
      const resp = await runCode(req);
      vthread.startPolling(resp.vtid);
    } catch (e) {
      setRunError(e instanceof Error ? e.message : 'Unknown error');
      setIsRunning(false);
    }
  }, [vthread]);

  // When vthread poll completes
  const vthreadDone = vthread.status &&
    (vthread.status.status === 'done' || vthread.status.status === 'error');

  if (vthreadDone && isRunning) {
    setIsRunning(false);
    setRunResult(vthread.status);
  }

  return (
    <div className="dashboard">
      <header className="header">
        <span className="logo">◈ deepx dashboard</span>
        <span className="connection">
          <span className={`dot ${isConnected ? 'green' : 'red'}`} />
          {isConnected ? 'connected' : 'disconnected'}
        </span>
      </header>
      <div className="panels">
        <div style={{ width: leftPanel.size, minWidth: leftPanel.size, maxWidth: leftPanel.size, flexShrink: 0 }}>
          <StatusPanel status={sysStatus} connected={isConnected} />
        </div>
        <ResizeHandle onMouseDown={leftPanel.onMouseDown} active={leftPanel.dragging} />
        <CodeEditor onRun={handleRun} isRunning={isRunning} />
        <ResizeHandle onMouseDown={rightPanel.onMouseDown} active={rightPanel.dragging} />
        <div style={{ width: rightPanel.size, minWidth: rightPanel.size, maxWidth: rightPanel.size, flexShrink: 0 }}>
          <OutputPanel
            runResult={runResult}
            runError={runError}
            vthreadStatus={vthread.status}
            vthreadLoading={vthread.loading}
            vthreadError={vthread.error}
            isRunning={isRunning}
          />
        </div>
      </div>
    </div>
  );
}
