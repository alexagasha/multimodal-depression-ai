"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { api, ApiError, NoteDraft, ScoreResult, TranscriptRow, VisitSummary } from "@/lib/api";
import SessionStatusStepper from "@/components/SessionStatusStepper";
import RiskBanner from "@/components/RiskBanner";
import RiskAssessmentForm from "@/components/RiskAssessmentForm";
import ScaleForm from "@/components/ScaleForm";
import AudioRecorder from "@/components/AudioRecorder";
import LiveTranscript from "@/components/LiveTranscript";
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
  const [recording, setRecording] = useState(false);
  const [transcriptRows, setTranscriptRows] = useState<TranscriptRow[]>([]);
  const [liveNote, setLiveNote] = useState<NoteDraft | null>(null);
  const [riskAssessed, setRiskAssessed] = useState(false);
  const [closeError, setCloseError] = useState<string | null>(null);

  function refreshVisit() {
    api
      .getVisit(id)
      .then(setVisit)
      .catch((e) => {
        if (e instanceof ApiError && e.status === 404) setNotFound(true);
      });
  }

  useEffect(refreshVisit, [id]);

  // Drives the stepper's risk gate. Refetched whenever an assessment is saved.
  useEffect(() => {
    api
      .listRiskAssessments(id)
      .then((a) => setRiskAssessed(a.length > 0))
      .catch(() => {});
  }, [id, notesRefreshKey]);

  async function closeVisit() {
    setCloseError(null);
    try {
      await api.closeVisit(id);
      refreshVisit();
    } catch (e) {
      // A 409 here is the risk gate, and its message says so.
      setCloseError(e instanceof ApiError ? e.message : String(e));
    }
  }

  // Catch up on any transcript/note already streamed for this visit, so a
  // reload mid-interview doesn't come back to an empty panel.
  useEffect(() => {
    api
      .getLiveState(id)
      .then((live) => {
        setTranscriptRows(live.rows);
        if (live.note) setLiveNote(live.note);
      })
      .catch(() => {});
  }, [id]);

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

      <Card className="space-y-3">
        <SessionStatusStepper
          status={visit.status}
          riskFlag={!!visit.risk_flag}
          riskAssessed={riskAssessed}
        />
        {visit.status !== "closed" && (
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={closeVisit}
              className="rounded-full border border-sage-300 px-4 py-1.5 text-sm font-medium text-sage-700 hover:bg-sage-100"
            >
              Close visit
            </button>
            {closeError && (
              <p className="text-sm text-[var(--color-danger)]">{closeError}</p>
            )}
          </div>
        )}
      </Card>

      {visit.risk_flag && <RiskBanner />}

      {/* The flag alone recorded nothing about what the clinician found or
          decided. Shown immediately under the banner it raises, and only when
          it is raised. */}
      {visit.risk_flag && (
        <RiskAssessmentForm
          visitId={id}
          onSaved={() => setNotesRefreshKey((k) => k + 1)}
        />
      )}

      <ScaleForm visitId={id} onDone={() => refreshVisit()} />

      <AudioRecorder
        visitId={id}
        onRecordingChange={setRecording}
        onTranscriptRows={(rows) => setTranscriptRows((prev) => [...prev, ...rows])}
        onNote={setLiveNote}
        onDone={() => {
          refreshVisit();
          // Recording finished and the transcript is finalised — score without
          // making the clinician ask for it.
          void runScoring();
        }}
      />

      <LiveTranscript rows={transcriptRows} recording={recording} />

      {scoring && (
        <p className="text-sm text-sage-700">Scoring this visit…</p>
      )}
      {scoreError && (
        <p className="rounded-lg bg-[var(--color-danger-bg)] px-3 py-2 text-sm text-[var(--color-danger)]">
          {scoreError}
        </p>
      )}

      <NoteDraftPanel
        visitId={id}
        liveDraft={liveNote}
        recording={recording}
        onSaved={() => setNotesRefreshKey((k) => k + 1)}
      />

      <ClinicalNotes visitId={id} refreshKey={notesRefreshKey} />

      {modalOpen && result && (
        <ScoreModal result={result} visitId={id} onClose={() => setModalOpen(false)} />
      )}
    </div>
  );
}
