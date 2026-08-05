"use client";

import { useEffect, useState } from "react";
import { api, ApiError, Review, ScoreResult } from "@/lib/api";
import Gauge from "@/components/Gauge";
import RiskBanner from "@/components/RiskBanner";
import SubtypeDifferential from "@/components/SubtypeDifferential";
import EvidenceQuotes from "@/components/EvidenceQuotes";
import PatientSummaryCard from "@/components/PatientSummaryCard";
import { Field, TextInput, TextArea } from "@/components/FormField";

const MODALITY_LABEL: Record<string, string> = {
  text: "Text (transcript)",
  audio: "Audio (prosody)",
  metadata: "Metadata",
};

function ReviewSection({ visitId }: { visitId: string }) {
  const [review, setReview] = useState<Review | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [reviewer, setReviewer] = useState("");
  const [agrees, setAgrees] = useState(true);
  const [adjustedPhq9, setAdjustedPhq9] = useState("");
  const [adjustedHamd, setAdjustedHamd] = useState("");
  const [comment, setComment] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getReview(visitId).then((r) => {
      setReview(r);
      setLoaded(true);
    });
  }, [visitId]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!reviewer.trim()) return;
    setSubmitting(true);
    setError(null);
    try {
      const saved = await api.submitReview(visitId, {
        reviewer: reviewer.trim(),
        agrees,
        adjusted_phq9: adjustedPhq9 ? Number(adjustedPhq9) : null,
        adjusted_hamd: adjustedHamd ? Number(adjustedHamd) : null,
        comment: comment.trim() || null,
      });
      setReview(saved);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  if (!loaded) return null;

  if (review) {
    return (
      <div className="rounded-xl border border-sage-200 bg-sage-50 px-4 py-3" data-no-print>
        <p className="text-sm font-medium text-sage-800">
          Reviewed by {review.reviewer} — {review.agrees ? "agrees with model" : "adjusted"}
        </p>
        {!review.agrees && (
          <p className="text-sm text-sage-700">
            Adjusted: PHQ-9 {review.adjusted_phq9 ?? "–"} / HAM-D {review.adjusted_hamd ?? "–"}
          </p>
        )}
        {review.comment && <p className="mt-1 text-sm text-ink-700">&ldquo;{review.comment}&rdquo;</p>}
      </div>
    );
  }

  return (
    <form onSubmit={onSubmit} className="space-y-3 rounded-xl border border-sage-200 p-4" data-no-print>
      <h3 className="font-display text-sm font-semibold text-sage-800">Clinician review</h3>
      <Field label="Reviewer">
        <TextInput value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="Dr. …" required />
      </Field>
      <div className="flex gap-4 text-sm">
        <label className="flex items-center gap-1.5">
          <input type="radio" checked={agrees} onChange={() => setAgrees(true)} /> Agrees with model
        </label>
        <label className="flex items-center gap-1.5">
          <input type="radio" checked={!agrees} onChange={() => setAgrees(false)} /> Adjusting
        </label>
      </div>
      {!agrees && (
        <div className="grid grid-cols-2 gap-3">
          <Field label="Adjusted PHQ-9">
            <TextInput type="number" value={adjustedPhq9} onChange={(e) => setAdjustedPhq9(e.target.value)} />
          </Field>
          <Field label="Adjusted HAM-D">
            <TextInput type="number" value={adjustedHamd} onChange={(e) => setAdjustedHamd(e.target.value)} />
          </Field>
        </div>
      )}
      <Field label="Comment (optional)">
        <TextArea rows={2} value={comment} onChange={(e) => setComment(e.target.value)} />
      </Field>
      {error && <p className="text-sm text-[var(--color-danger)]">{error}</p>}
      <button
        type="submit"
        disabled={submitting}
        className="rounded-full bg-sage-500 px-4 py-2 text-sm font-semibold text-white hover:bg-sage-600 disabled:opacity-50"
      >
        {submitting ? "Saving…" : "Submit review"}
      </button>
    </form>
  );
}

export default function ScoreModal({
  result,
  visitId,
  onClose,
}: {
  result: ScoreResult;
  visitId: string;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-30 flex items-center justify-center bg-black/40 p-4" data-no-print>
      <div className="max-h-[90vh] w-full max-w-xl overflow-y-auto rounded-3xl bg-cream-50 shadow-xl">
        <div data-print-area className="space-y-5 p-6">
          <h2 className="font-display text-xl font-semibold text-sage-900">Screening result</h2>

          {result.risk_flag && <RiskBanner />}

          <div className="flex items-center justify-around">
            <Gauge label="PHQ-9" value={result.phq9_pred} clinicianValue={result.phq9_clinician} max={27} />
            <Gauge label="HAM-D" value={result.hamd_pred} clinicianValue={result.hamd_clinician} max={44} />
            <div className="flex flex-col items-center">
              <span
                className={`rounded-full px-3 py-1.5 text-sm font-semibold ${
                  result.binary_pred
                    ? "bg-clay-100 text-clay-600"
                    : "bg-sage-100 text-sage-700"
                }`}
              >
                {result.binary_pred ? "Case" : "Non-case"}
              </span>
              <span className="mt-2 text-sm font-medium text-sage-700">Caseness</span>
            </div>
          </div>

          <div>
            <h3 className="font-display text-sm font-semibold text-sage-800">Modality attribution</h3>
            <div className="mt-2 space-y-2">
              {Object.entries(result.modality_attributions).map(([modality, score]) => (
                <div key={modality} className="flex items-center gap-2 text-sm">
                  <span className="w-36 text-sage-700">{MODALITY_LABEL[modality] ?? modality}</span>
                  <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-sage-100">
                    <div
                      className="h-full rounded-full bg-sage-500"
                      style={{ width: `${Math.min(100, Math.abs(score) * 100)}%` }}
                    />
                  </div>
                  <span className="w-12 text-right text-sage-600">{score.toFixed(2)}</span>
                </div>
              ))}
            </div>
            <div className="mt-2">
              <EvidenceQuotes data={result.evidence} />
            </div>
          </div>

          <div className="rounded-2xl bg-sage-50 p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-sage-500">
              AI-generated commentary — not a diagnosis
            </p>
            <p className="mt-1 text-sm leading-relaxed text-ink-900">{result.narrative}</p>
          </div>

          <SubtypeDifferential data={result.subtype_differential} />

          <div className="rounded-2xl bg-sage-50 p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-sage-500">
              Next-step considerations — AI-suggested, always advisory
            </p>
            {!result.treatment_suggestions || result.treatment_suggestions.length === 0 ? (
              <p className="mt-1 text-sm text-sage-600">
                Not available — requires an LLM connection (ANTHROPIC_API_KEY).
              </p>
            ) : (
              <ul className="mt-1 list-disc space-y-1 pl-4 text-sm text-ink-900">
                {result.treatment_suggestions.map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
            )}
          </div>

          <PatientSummaryCard summary={result.patient_summary} />

          <ReviewSection visitId={visitId} />
        </div>

        <div className="flex items-center justify-end gap-2 border-t border-sage-200 px-6 py-4" data-no-print>
          <button
            onClick={() => window.print()}
            className="rounded-full border border-sage-300 px-4 py-2 text-sm font-medium text-sage-700 hover:bg-sage-100"
          >
            Print / Save PDF
          </button>
          <button
            onClick={onClose}
            className="rounded-full bg-sage-500 px-4 py-2 text-sm font-semibold text-white hover:bg-sage-600"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
