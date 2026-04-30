import { useState } from 'react';
import { SystemStatus, ComponentInfo, fetchOps, OpList } from '../api/client';

interface Props {
  status: SystemStatus | null;
  connected: boolean;
}

function CompCard({ name, info }: { name: string; info: ComponentInfo }) {
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

function InstanceSection({ title, instances }: { title: string; instances: ComponentInfo[] }) {
  return (
    <div className="comp-section">
      <div className="subtitle">{title}</div>
      {instances.length === 0 ? (
        <div className="comp-card offline">
          <span className="comp-dot grey" />
          <span className="comp-name">—</span>
          <span className="comp-status">no instances</span>
        </div>
      ) : (
        instances.map((inst) => (
          <CompCard key={inst.id} name={inst.id} info={inst} />
        ))
      )}
    </div>
  );
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function OpRow({ backend, count }: { backend: string; count: number }) {
  const [expanded, setExpanded] = useState(false);
  const [ops, setOps] = useState<OpList | null>(null);
  const [loading, setLoading] = useState(false);

  const handleToggle = async () => {
    if (expanded) { setExpanded(false); return; }
    setExpanded(true);
    if (!ops) {
      setLoading(true);
      try { const data = await fetchOps(backend); setOps(data); } catch {}
      setLoading(false);
    }
  };

  return (
    <div className="op-row">
      <div className="stat-row clickable" onClick={handleToggle}>
        <span className="op-label">{expanded ? '▾' : '▸'} op/{backend}</span>
        <span className="stat-val">{count}</span>
      </div>
      {expanded && (
        <div className="op-list">
          {loading ? (
            <span className="op-loading">loading...</span>
          ) : ops && ops.ops.length > 0 ? (
            <div className="op-tags">
              {ops.ops.map((op) => (
                <code key={op} className="op-tag">{op}</code>
              ))}
            </div>
          ) : (
            <span className="op-empty">(empty)</span>
          )}
        </div>
      )}
    </div>
  );
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
        {status ? (
          <>
            <InstanceSection title="op-plat" instances={status.op_plat} />
            <InstanceSection title="heap-plat" instances={status.heap_plat} />
            {status.vm && <CompCard name={status.vm.id} info={status.vm} />}
            {!status.vm && (
              <div className="comp-card offline">
                <span className="comp-dot grey" /><span className="comp-name">vm</span><span className="comp-status">offline</span>
              </div>
            )}
            <div className="comp-section">
              <div className="subtitle">terminal</div>
              {status.term ? (
                <CompCard name={status.term.id} info={status.term} />
              ) : (
                <div className="comp-card offline">
                  <span className="comp-dot grey" /><span className="comp-name">—</span><span className="comp-status">offline</span>
                </div>
              )}
            </div>
          </>
        ) : (
          <>
            <div className="comp-card offline"><span className="comp-dot grey" /><span className="comp-name">op-plat</span><span className="comp-status">offline</span></div>
            <div className="comp-card offline"><span className="comp-dot grey" /><span className="comp-name">heap-plat</span><span className="comp-status">offline</span></div>
            <div className="comp-card offline"><span className="comp-dot grey" /><span className="comp-name">vm</span><span className="comp-status">offline</span></div>
            <div className="comp-card offline"><span className="comp-dot grey" /><span className="comp-name">term</span><span className="comp-status">offline</span></div>
          </>
        )}
      </div>

      {status && (
        <div className="stats">
          <div className="stat-row"><span>Functions</span><span className="stat-val">{status.functions.count}</span></div>
          <div className="stat-row"><span>VThreads</span><span className="stat-val">{status.vthreads.total}</span></div>
          <div className="stat-row"><span>DB Keys</span><span className="stat-val">{status.dbsize}</span></div>
          {Object.entries(status.op_registry).map(([k, v]) => (
            <OpRow key={k} backend={k} count={v} />
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
