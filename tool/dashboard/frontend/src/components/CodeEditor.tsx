import { useState, useCallback, useRef, useEffect, KeyboardEvent } from 'react';
import { RunRequest } from '../api/client';
import { highlightToHtml } from '../utils/highlight';

interface Props {
  onRun: (req: RunRequest) => void;
  isRunning: boolean;
}

const DEFAULT_CODE = `def add(A:int, B:int) -> (C:int) {
    A + B -> ./C
}

def main() -> (Z:int) {
    add(3, 5) -> ./Z
}`;

export function CodeEditor({ onRun, isRunning }: Props) {
  const [source, setSource] = useState(() => {
    return localStorage.getItem('dx-editor-source') || DEFAULT_CODE;
  });
  const [entry, setEntry] = useState('main');
  const [timeout, setTimeout_] = useState(60);
  const [lineCount, setLineCount] = useState(0);
  const [scrollTop, setScrollTop] = useState(0);
  const [scrollLeft, setScrollLeft] = useState(0);
  const editorRef = useRef<HTMLTextAreaElement>(null);
  const highlightRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    localStorage.setItem('dx-editor-source', source);
  }, [source]);

  useEffect(() => {
    setLineCount(source.split('\n').length);
  }, [source]);

  useEffect(() => {
    const defs = source.match(/def\s+(\w+)/g);
    if (defs) {
      const names = defs.map(d => d.replace('def ', ''));
      if (names.includes('main')) {
        setEntry('main');
      } else if (names.length === 1) {
        setEntry(names[0]);
      }
    }
  }, [source]);

  const handleScroll = useCallback(() => {
    const ta = editorRef.current;
    if (ta) {
      setScrollTop(ta.scrollTop);
      setScrollLeft(ta.scrollLeft);
    }
  }, []);

  const handleRun = useCallback(() => {
    if (isRunning) return;
    onRun({ source, entry: entry || undefined, timeout });
  }, [source, entry, timeout, isRunning, onRun]);

  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      handleRun();
    }
    if (e.key === 'Tab') {
      e.preventDefault();
      const ta = e.currentTarget;
      const start = ta.selectionStart;
      const end = ta.selectionEnd;
      setSource(source.substring(0, start) + '    ' + source.substring(end));
      requestAnimationFrame(() => {
        ta.selectionStart = ta.selectionEnd = start + 4;
      });
    }
  }, [source, handleRun]);

  const handleClear = () => {
    setSource('');
    editorRef.current?.focus();
  };

  const lineNumbers = Array.from({ length: Math.max(lineCount, 1) }, (_, i) => i + 1);
  const highlighted = highlightToHtml(source + '\n');

  return (
    <main className="panel editor-panel">
      <div className="panel-header">
        <h3>Code Editor</h3>
        <span className="hint">dxlang</span>
      </div>

      <div className="editor-toolbar">
        <div className="toolbar-group">
          <label>Entry:</label>
          <input
            type="text"
            value={entry}
            onChange={(e) => setEntry(e.target.value)}
            className="input-sm"
            placeholder="main"
          />
        </div>
        <div className="toolbar-group">
          <label>Timeout:</label>
          <select
            value={timeout}
            onChange={(e) => setTimeout_(Number(e.target.value))}
            className="input-sm"
          >
            <option value={10}>10s</option>
            <option value={30}>30s</option>
            <option value={60}>60s</option>
            <option value={120}>120s</option>
            <option value={300}>300s</option>
          </select>
        </div>
        <div className="toolbar-actions">
          <button
            className="btn btn-run"
            onClick={handleRun}
            disabled={isRunning}
          >
            {isRunning ? '⟳ Running...' : '▶ Run'}
          </button>
          <button className="btn btn-secondary" onClick={handleClear}>
            Clear
          </button>
        </div>
      </div>

      <div className="editor-body">
        <div className="line-numbers">
          {lineNumbers.map((n) => (
            <span key={n} className="line-num">{n}</span>
          ))}
        </div>
        <div className="editor-code-container">
          <pre
            ref={highlightRef}
            className="editor-highlight"
            aria-hidden="true"
            dangerouslySetInnerHTML={{ __html: highlighted }}
            style={{
              transform: `translate(-${scrollLeft}px, -${scrollTop}px)`,
            }}
          />
          <textarea
            ref={editorRef}
            className="editor-textarea"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            onKeyDown={handleKeyDown}
            onScroll={handleScroll}
            placeholder="Enter dxlang code here..."
            spellCheck={false}
          />
        </div>
      </div>

      <div className="editor-statusbar">
        <span>{lineCount} lines</span>
        <span className="statusbar-right">
          <kbd>Cmd+Enter</kbd> run &nbsp; <kbd>Tab</kbd> indent
        </span>
      </div>
    </main>
  );
}
