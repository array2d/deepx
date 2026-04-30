import { SystemStatus, ComponentInfo } from '../api/client';

interface Props {
  status: SystemStatus | null;
  connected: boolean;
}

function CompCard({ name, info }: { name: string; info: ComponentInfo | null }) {
  if (!info) {
    return (
      <div className="comp-card offline">
        <span className="comp-dot grey" />
        <span className="comp-name">{name}</span>
        <span className="comp-status">offline</span>
      </div>
    );
  }

  const statusClass = info.status === 'running' ? 'green' :
    info.status === 'error' ? 'red' : 'yellow';

  return (
    <div className={`comp-card ${info.status}`}>
      <span className={`comp-dot ${statusClass}`} />
      <span className="comp-name">{name}</span>
      <span className="comp-status">{info.status}</span>
      {info.pid > 0 && <span className="comp-pid">pid:{info.pid}</span>}
      {info.load && <span className="comp-load">load:{info.load}</span>}
      {info.heartbeat_age_ms > 0 && (
        <span className="comp-hb">hb:{formatMs(info.heartbeat_age_ms)}</span>
      )}
    </div>
  );
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function StatusPanel({ status, connected }: Props) {
  return (
    <aside className="panel status-panel">
      <div className="panel-header">
        <h3>System Status</h3>
        <span className={`indicator ${connected ? 'online' : 'offline'}`}>
          {connected ? '● live' : '○ offline'}
        </span>
      </div>

      <div className="components">
        <CompCard name="op-plat" info={status?.op_plat ?? null} />
        <CompCard name="heap-plat" info={status?.heap_plat ?? null} />
        <CompCard name="vm" info={status?.vm ?? null} />
      </div>

      {status && (
        <div className="stats">
          <div className="stat-row">
            <span>Functions</span>
            <span className="stat-val">{status.functions.count}</span>
          </div>
          <div className="stat-row">
            <span>VThreads</span>
            <span className="stat-val">{status.vthreads.total}</span>
          </div>
          <div className="stat-row">
            <span>DB Keys</span>
            <span className="stat-val">{status.dbsize}</span>
          </div>
          {Object.entries(status.op_registry).map(([k, v]) => (
            <div className="stat-row" key={k}>
              <span>Op({k})</span>
              <span className="stat-val">{v}</span>
            </div>
          ))}
        </div>
      )}

      {status && status.vthreads.total > 0 && (
        <div className="vthread-summary">
          <div className="subtitle">vthread status</div>
          {Object.entries(status.vthreads.by_status).map(([s, c]) => (
            <div className="stat-row" key={s}>
              <span className={`vt-status-badge ${s}`}>{s}</span>
              <span className="stat-val">{c}</span>
            </div>
          ))}
        </div>
      )}

      {status && status.functions.names.length > 0 && (
        <div className="func-list">
          <div className="subtitle">functions</div>
          {status.functions.names.map((name) => (
            <code key={name} className="func-name">{name}</code>
          ))}
        </div>
      )}
    </aside>
  );
}
