const BASE = '';

export interface SystemStatus {
  op_plat: ComponentInfo[];
  heap_plat: ComponentInfo[];
  vm: ComponentInfo | null;
  term: ComponentInfo | null;
  functions: { count: number; names: string[] };
  vthreads: { total: number; by_status: Record<string, number> };
  op_registry: Record<string, number>;
  dbsize: number;
}

export interface ComponentInfo {
  id: string;
  status: string;
  pid: number;
  heartbeat_age_ms: number;
  load: string;
  device: string;
  program: string;
  started_at: number;
}

export interface VThreadInfo {
  vtid: number;
  status: string;
  pc: string;
  error?: { code: string; message: string };
  duration_ms?: number;
}

export interface RunRequest {
  source: string;
  entry?: string;
  timeout?: number;
}

export interface RunResponse {
  vtid: number;
  status: string;
  entry: string;
}

export async function fetchStatus(): Promise<SystemStatus> {
  const res = await fetch(`${BASE}/api/status`);
  if (!res.ok) throw new Error(`status fetch failed: ${res.status}`);
  return res.json();
}

export async function fetchVThreads(): Promise<{ vthreads: VThreadInfo[] }> {
  const res = await fetch(`${BASE}/api/vthreads`);
  if (!res.ok) throw new Error(`vthreads fetch failed: ${res.status}`);
  return res.json();
}

export async function fetchVThread(vtid: number): Promise<VThreadInfo> {
  const res = await fetch(`${BASE}/api/vthread/${vtid}`);
  if (!res.ok) throw new Error(`vthread ${vtid} fetch failed: ${res.status}`);
  return res.json();
}

export async function fetchFunctions(): Promise<{ functions: string[] }> {
  const res = await fetch(`${BASE}/api/functions`);
  if (!res.ok) throw new Error(`functions fetch failed: ${res.status}`);
  return res.json();
}

export interface OpList {
  backend: string;
  count: number;
  ops: string[];
}

export async function fetchOps(backend: string): Promise<OpList> {
  const res = await fetch(`${BASE}/api/ops/${backend}`);
  if (!res.ok) throw new Error(`ops ${backend} fetch failed: ${res.status}`);
  return res.json();
}

export async function runCode(req: RunRequest): Promise<RunResponse> {
  const res = await fetch(`${BASE}/api/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || 'run failed');
  }
  return res.json();
}

export function sseUrl(): string {
  return '/api/status/stream';
}

export function termWsUrl(): string {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${location.host}/api/terminal`;
}
