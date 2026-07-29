"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { api, ApiError, ScoreResult, SessionSummary } from "@/lib/api";
import SessionStatusStepper from "@/components/SessionStatusStepper";
import RiskBanner from "@/components/RiskBanner";
import ScaleForm from "@/components/ScaleForm";
import AudioRecorder from "@/components/AudioRecorder";
import ClinicalNotes from "@/components/ClinicalNotes";
import ScoreModal from "@/components/ScoreModal";
import { Card } from "@/components/FormField";

export default function SessionWorkspacePage() {
  const { id } = useParams<{ id: string }>();
  const [session, setSession] = useState<SessionSummary | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [scoring, setScoring] = useState(false);
  const [scoreError, setScoreError] = useState<string | null>(null);
  const [result, setResult] = useState<ScoreResult | null>(null);
  const [modalOpen, setModalOpen] = useState(false);

  function refreshSession() {
    api
      .getSession(id)
      .then(setSession)
      .catch((e) => {
        if (e instanceof ApiError && e.status === 404) setNotFound(true);
      });
  }

  useEffect(refreshSession, [id]);

  async function runScoring() {
    setScoring(true);
    setScoreError(null);
    try {
      const r = await api.scoreSession(id);
      setResult(r);
      setModalOpen(true);
      refreshSession();
    } catch (e) {
      setScoreError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setScoring(false);
    }
  }

  async function viewLastResult() {
    const r = await api.getResults(id);
    setResult(r);
    setModalOpen(true);
  }

  if (notFound) {
    return <p className="text-sm text-sage-700">Session {id} not found.</p>;
  }

  if (!session) {
    return <p className="text-sm text-sage-700">Loading session…</p>;
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="font-display text-2xl font-semibold text-sage-900">
            Session {session.session_id}
          </h1>
          <p className="text-sm text-sage-700">
            Participant {session.participant_id} ·{" "}
            <Link href={`/participants/${session.participant_id}`} className="underline decoration-sage-300">
              view history
            </Link>
          </p>
        </div>
        {session.status === "scored" && (
          <button
            onClick={viewLastResult}
            className="rounded-full border border-sage-300 px-4 py-2 text-sm font-medium text-sage-700 hover:bg-sage-100"
          >
            View last result
          </button>
        )}
      </div>

      <Card>
        <SessionStatusStepper status={session.status} />
      </Card>

      {session.risk_flag && <RiskBanner />}

      <ScaleForm sessionId={id} onDone={() => refreshSession()} />

      <AudioRecorder sessionId={id} onDone={() => refreshSession()} />

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

      <ClinicalNotes sessionId={id} />

      {modalOpen && result && (
        <ScoreModal result={result} sessionId={id} onClose={() => setModalOpen(false)} />
      )}
    </div>
  );
}
