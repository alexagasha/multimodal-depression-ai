import { useEffect, useState } from "react";
import { api, ApiError } from "./api";
import { EmptyState } from "./components/EmptyState";
import { ParticipantDetail } from "./components/ParticipantDetail";
import { ParticipantTable } from "./components/ParticipantTable";
import { PredictionChart } from "./components/PredictionChart";
import { Sidebar } from "./components/Sidebar";
import { StageRail, type RunStatus } from "./components/StageRail";
import { SummaryStrip } from "./components/SummaryStrip";
import type {
  HealthResponse,
  Metrics,
  ParticipantDetail as ParticipantDetailT,
  ParticipantResult,
  TestRunResponse,
} from "./types";

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [results, setResults] = useState<ParticipantResult[] | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [attributionsMap, setAttributionsMap] = useState<Record<number, Record<string, number>>>({});

  const [selectedPid, setSelectedPid] = useState<number | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<ParticipantDetailT | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const [generating, setGenerating] = useState(false);
  const [running, setRunning] = useState(false);
  const [runStatus, setRunStatus] = useState<RunStatus>("idle");
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<TestRunResponse | null>(null);

  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const h = await api.health();
        setHealth(h);
        if (h.predictions_available) {
          await loadResults();
        }
      } catch (e) {
        setError(describeError(e));
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadResults() {
    try {
      const [preds, m, attrs] = await Promise.all([
        api.getPredictions(),
        api.getMetrics(),
        api.getAllAttributions(),
      ]);
      setResults(preds);
      setMetrics(m);
      const map: Record<number, Record<string, number>> = {};
      attrs.forEach((a) => {
        map[a.participant_id] = a.modality_attributions;
      });
      setAttributionsMap(map);
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) return; // nothing to show yet
      setError(describeError(e));
    }
  }

  async function handleGenerate(n: number) {
    setGenerating(true);
    setError(null);
    try {
      await api.generateDataset(n);
      const h = await api.health();
      setHealth(h);
      setResults(null);
      setMetrics(null);
      setAttributionsMap({});
      setSelectedPid(null);
      setSelectedDetail(null);
    } catch (e) {
      setError(describeError(e));
    } finally {
      setGenerating(false);
    }
  }

  async function handleRun(seed: number) {
    setRunning(true);
    setRunStatus("running");
    setError(null);
    try {
      const res = await api.runPipeline(seed);
      setResults(res.results);
      setMetrics(res.metrics);
      const h = await api.health();
      setHealth(h);
      const attrs = await api.getAllAttributions();
      const map: Record<number, Record<string, number>> = {};
      attrs.forEach((a) => {
        map[a.participant_id] = a.modality_attributions;
      });
      setAttributionsMap(map);
      setRunStatus("done");
    } catch (e) {
      setError(describeError(e));
      setRunStatus("error");
    } finally {
      setRunning(false);
      setTimeout(() => setRunStatus("idle"), 1600);
    }
  }

  async function handleSelect(pid: number) {
    setSelectedPid(pid);
    setDetailLoading(true);
    try {
      const detail = await api.getParticipant(pid);
      setSelectedDetail(detail);
    } catch (e) {
      setError(describeError(e));
      setSelectedPid(null);
    } finally {
      setDetailLoading(false);
    }
  }

  function handleCloseDetail() {
    setSelectedPid(null);
    setSelectedDetail(null);
  }

  async function handleRunTests() {
    setTesting(true);
    try {
      const r = await api.runTests();
      setTestResult(r);
    } catch (e) {
      setError(describeError(e));
    } finally {
      setTesting(false);
    }
  }

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand">
          <span className="brand-mark" />
          <span className="brand-title">Depression Severity Pipeline</span>
          <span className="brand-sub">multimodal · PHQ-8</span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <span className="badge mock">
            <span className="badge-dot" />
            mock encoders · synthetic data
          </span>
          {health?.dataset.exists && (
            <span className="badge">{health.dataset.n_participants} participants on disk</span>
          )}
        </div>
      </div>

      <StageRail status={runStatus} />

      <div className="app-body">
        <Sidebar
          health={health}
          onGenerate={handleGenerate}
          generating={generating}
          onRun={handleRun}
          running={running}
          onRunTests={handleRunTests}
          testing={testing}
          testResult={testResult}
        />

        <main className="main">
          {error && (
            <div className="banner error">
              <span>⚠</span>
              <span>{error}</span>
            </div>
          )}

          {!results && !error && (
            <EmptyState title="No predictions yet">
              Generate a synthetic dataset and run the pipeline from the panel on the left. Everything
              here runs against deterministic mock encoders — see README.md — so this proves the
              plumbing end-to-end while E-DAIC-WOZ access is pending.
            </EmptyState>
          )}

          {results && metrics && (
            <>
              <SummaryStrip metrics={metrics} />

              <div className="section-header">
                <h2>Participants</h2>
                <span className="count">{results.length} total</span>
              </div>
              <ParticipantTable
                results={results}
                attributions={attributionsMap}
                selectedPid={selectedPid}
                onSelect={handleSelect}
              />

              <PredictionChart results={results} threshold={metrics.depression_threshold} />
            </>
          )}
        </main>
      </div>

      <ParticipantDetail detail={selectedDetail} loading={detailLoading} onClose={handleCloseDetail} />
    </div>
  );
}

function describeError(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return "Something went wrong.";
}
