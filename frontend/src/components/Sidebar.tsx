import { useState } from "react";
import type { HealthResponse, TestRunResponse } from "../types";
import { formatTimestamp } from "../utils";

interface SidebarProps {
  health: HealthResponse | null;
  onGenerate: (n: number) => void;
  generating: boolean;
  onRun: (seed: number) => void;
  running: boolean;
  onRunTests: () => void;
  testing: boolean;
  testResult: TestRunResponse | null;
}

export function Sidebar({
  health,
  onGenerate,
  generating,
  onRun,
  running,
  onRunTests,
  testing,
  testResult,
}: SidebarProps) {
  const [nParticipants, setNParticipants] = useState(8);
  const [seed, setSeed] = useState(0);

  return (
    <aside className="sidebar">
      <div className="panel-section">
        <div className="panel-title">Dataset</div>
        <div className="field">
          <label htmlFor="n_participants">Synthetic participants</label>
          <input
            id="n_participants"
            type="number"
            min={1}
            max={200}
            value={nParticipants}
            onChange={(e) => setNParticipants(Number(e.target.value))}
          />
        </div>
        <button
          className="btn"
          disabled={generating}
          onClick={() => onGenerate(nParticipants)}
        >
          {generating && <span className="spinner" />}
          {generating ? "Generating…" : "Generate dataset"}
        </button>
        {health?.dataset.exists && (
          <div style={{ marginTop: 2 }}>
            <div className="status-row">
              <span>On disk</span>
              <span className="v">{health.dataset.n_participants} participants</span>
            </div>
            {Object.entries(health.dataset.splits).map(([split, count]) => (
              <div className="status-row" key={split}>
                <span className="faint">{split}</span>
                <span className="v">{count}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="panel-section">
        <div className="panel-title">Pipeline</div>
        <div className="field">
          <label htmlFor="seed">Fusion head seed</label>
          <input
            id="seed"
            type="number"
            min={0}
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
          />
        </div>
        <button
          className="btn btn-primary"
          disabled={running || !health?.dataset.exists}
          onClick={() => onRun(seed)}
        >
          {running && <span className="spinner" />}
          {running ? "Running…" : "Run pipeline"}
        </button>
        {!health?.dataset.exists && (
          <p className="faint" style={{ fontSize: 11, lineHeight: 1.5 }}>
            Generate a dataset first — the pipeline reads participant sessions from disk.
          </p>
        )}
      </div>

      <div className="panel-section">
        <div className="panel-title">Status</div>
        <div className="status-row">
          <span>Encoders</span>
          <span className="v">mock (stub)</span>
        </div>
        <div className="status-row">
          <span>Model seed</span>
          <span className="v">{health?.model_seed ?? "—"}</span>
        </div>
        <div className="status-row">
          <span>Last run</span>
          <span className="v">{formatTimestamp(health?.last_run_at ?? null)}</span>
        </div>
      </div>

      <div className="panel-section">
        <div className="panel-title">Test suite</div>
        <button className="btn btn-ghost" disabled={testing} onClick={onRunTests}>
          {testing && <span className="spinner" />}
          {testing ? "Running pytest…" : "Run tests/"}
        </button>
        {testResult && (
          <>
            <div className="status-row">
              <span>Result</span>
              <span className="v" style={{ color: testResult.success ? "var(--sev-0)" : "var(--error)" }}>
                {testResult.passed} passed · {testResult.failed} failed
              </span>
            </div>
            <details>
              <summary style={{ fontSize: 11, color: "var(--text-faint)", cursor: "pointer" }}>
                pytest output
              </summary>
              <div className="test-output">{testResult.output}</div>
            </details>
          </>
        )}
      </div>
    </aside>
  );
}
