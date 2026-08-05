"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { api, ApiError, ScoreResult, VisitSummary } from "@/lib/api";
import SessionStatusStepper from "@/components/SessionStatusStepper";
import RiskBanner from "@/components/RiskBanner";
import ScaleForm from "@/components/ScaleForm";
import AudioRecorder from "@/components/AudioRecorder";
import ClinicalNotes from "@/components/ClinicalNotes";
import NoteDraftPanel from "@/components/NoteDraftPanel";
import ScoreModal from "@/components/ScoreModal";
import { Card } from "@/components/FormField";

export default function VisitWorkspacePage() {
  const { id } = useParams<{ id: string }>();
  const [visit, setVisit] = useState<VisitSummary | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [scoring, setScoring] = useState(false);
  const [scoreError, setScoreError] = useState<string | null>(null);
  const [result, setResult] = useState<ScoreResult | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [notesRefreshKey, setNotesRefreshKey] = useState(0);

  function refreshVisit() {
    api
      .getVisit(id)
      .then(setVisit)
      .catch((e) => {
        if (e instanceof ApiError && e.status === 404) setNotFound(true);
      });
  }

  useEffect(refreshVisit, [id]);

  async function runScoring() {
    setScoring(true);
    setScoreError(null);
    try {
      const r = await api.scoreVisit(id);
      setResult(r);
      setModalOpen(true);
      refreshVisit();
    } catch (e) {
      setScoreError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setScoring(false);
    }
  }

  async function viewLastResult() {
    const r = await api.getVisitResults(id);
    setResult(r);
    setModalOpen(true);
  }

  if (notFound) {
    return <p className="text-sm text-sage-700">Visit {id} not found.</p>;
  }

  if (!visit) {
    return <p className="text-sm text-sage-700">Loading visit…</p>;
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="font-display text-2xl font-semibold text-sage-900">
            Visit {visit.session_id}
          </h1>
          <p className="text-sm text-sage-700">
            Patient {visit.participant_id} ·{" "}
            <Link href={`/patients/${visit.participant_id}`} className="underline decoration-sage-300">
              view record
            </Link>
          </p>
        </div>
        {visit.status === "scored" && (
          <button
            onClick={viewLastResult}
            className="rounded-full border border-sage-300 px-4 py-2 text-sm font-medium text-sage-700 hover:bg-sage-100"
          >
            View last result
          </button>
        )}
      </div>

      <Card>
        <SessionStatusStepper status={visit.status} />
      </Card>

      {visit.risk_flag && <RiskBanner />}

      <ScaleForm visitId={id} onDone={() => refreshVisit()} />

      <AudioRecorder visitId={id} onDone={() => refreshVisit()} />

      <Card className="space-y-3">
        <h2 className="font-display text-lg font-semibold text-sage-800">Score</h2>
        <button
          onClick={runScoring}
          disabled={scoring}
          className="rounded-full bg-clay-500 px-4 py-2 text-sm font-semibold text-white hover:bg-clay-600 disabled:opacity-50"
        >
          {scoring ? "Scoring…" : "Run scoring"}
        </button>
        {scoreError && (
          <p className="rounded-lg bg-[var(--color-danger-bg)] px-3 py-2 text-sm text-[var(--color-danger)]">
            {scoreError}
          </p>
        )}
      </Card>

      <NoteDraftPanel visitId={id} onSaved={() => setNotesRefreshKey((k) => k + 1)} />

      <ClinicalNotes visitId={id} refreshKey={notesRefreshKey} />

      {modalOpen && result && (
        <ScoreModal result={result} visitId={id} onClose={() => setModalOpen(false)} />
      )}
    </div>
  );
}
