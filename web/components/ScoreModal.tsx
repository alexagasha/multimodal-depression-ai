"use client";

import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { api, ApiError, Review, ScoreResult } from "@/lib/api";
import Gauge from "@/components/Gauge";
import RiskBanner from "@/components/RiskBanner";
import SubtypeDifferential from "@/components/SubtypeDifferential";
import EvidenceQuotes from "@/components/EvidenceQuotes";
import PatientSummaryCard from "@/components/PatientSummaryCard";
import { useTypewriter } from "@/lib/useTypewriter";
import { Field, TextInput, TextArea } from "@/components/FormField";

const MODALITY_LABEL: Record<string, string> = {
  text: "Text (transcript)",
  audio: "Audio (prosody)",
  metadata: "Metadata",
};

/** Plain-language names for the prosodic measures. The model reports feature
 *  keys; a clinician should read "how much pitch varies", not "f0_sd_mean".
 *  Suffixes: _mean is the average across the interview's answers, _sd is how
 *  much it varied between them. */
const PROSODY_LABEL: Record<string, string> = {
  f0_mean_mean: "Average pitch",
  f0_mean_sd: "Pitch varies between answers",
  f0_sd_mean: "Pitch variation within answers",
  f0_sd_sd: "Consistency of pitch variation",
  f0_range_st_mean: "Pitch range (semitones)",
  f0_range_st_sd: "Pitch range varies between answers",
  rms_mean_mean: "Loudness",
  rms_mean_sd: "Loudness varies between answers",
  rms_sd_mean: "Loudness variation within answers",
  rms_sd_sd: "Consistency of loudness variation",
  db_range_mean: "Dynamic range",
  db_range_sd: "Dynamic range varies between answers",
  voiced_frac_mean: "Proportion of speech that is voiced",
  voiced_frac_sd: "Voicing varies between answers",
  speech_frac_mean: "Proportion of answer spent speaking",
  speech_frac_sd: "Speaking proportion varies between answers",
  n_pauses_per_s_mean: "Pause frequency",
  n_pauses_per_s_sd: "Pause frequency varies between answers",
  pause_frac_mean: "Proportion of answer spent pausing",
  pause_frac_sd: "Pausing varies between answers",
  mean_pause_mean: "Average pause length",
  mean_pause_sd: "Pause length varies between answers",
  longest_pause_mean: "Longest pause",
  longest_pause_sd: "Longest pause varies between answers",
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

function AttributionBar({
  modality,
  score,
  index,
}: {
  modality: string;
  score: number;
  index: number;
}) {
  const width = Math.min(100, Math.abs(score) * 100);
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="w-36 text-sage-700">{MODALITY_LABEL[modality] ?? modality}</span>
      <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-sage-100">
        <motion.div
          className="h-full rounded-full bg-sage-500"
          initial={{ width: 0 }}
          animate={{ width: `${width}%` }}
          transition={{ duration: 0.7, delay: 0.2 + index * 0.12, ease: "easeOut" }}
        />
      </div>
      <span className="w-12 text-right text-sage-600">{score.toFixed(2)}</span>
    </div>
  );
}

function NarrativeCard({ narrative }: { narrative: string }) {
  const typed = useTypewriter(narrative, 10);
  return (
    <div className="rounded-2xl bg-sage-50 p-4">
      <p className="text-xs font-semibold uppercase tracking-wide text-sage-500">
        AI-generated commentary — not a diagnosis
      </p>
      <p className="mt-1 min-h-[3em] text-sm leading-relaxed text-ink-900">
        {typed}
        {typed.length < narrative.length && (
          <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-sage-400 align-middle" />
        )}
      </p>
    </div>
  );
}

function TreatmentSuggestionsCard({ suggestions }: { suggestions: string[] | null }) {
  return (
    <div className="rounded-2xl bg-sage-50 p-4">
      <p className="text-xs font-semibold uppercase tracking-wide text-sage-500">
        Next-step considerations — AI-suggested, always advisory
      </p>
      {!suggestions || suggestions.length === 0 ? (
        <p className="mt-1 text-sm text-sage-600">
          Not available — requires an LLM connection (ANTHROPIC_API_KEY).
        </p>
      ) : (
        <ul className="mt-1 list-disc space-y-1 pl-4 text-sm text-ink-900">
          {suggestions.map((s, i) => (
            <motion.li
              key={i}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.15 * i, duration: 0.35 }}
            >
              {s}
            </motion.li>
          ))}
        </ul>
      )}
    </div>
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
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.2 }}
      className="fixed inset-0 z-30 flex items-center justify-center bg-black/40 p-4"
      data-no-print
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.94, y: 12 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        className="max-h-[90vh] w-full max-w-xl overflow-y-auto rounded-3xl bg-cream-50 shadow-xl"
      >
        <div data-print-area className="space-y-5 p-6">
          <h2 className="font-display text-xl font-semibold text-sage-900">Screening result</h2>

          {result.risk_flag && <RiskBanner />}

          <div className="flex items-center justify-around">
            <Gauge label="PHQ-9" value={result.phq9_pred} clinicianValue={result.phq9_clinician} max={27} />
            <Gauge label="HAM-D" value={result.hamd_pred} clinicianValue={result.hamd_clinician} max={44} />
            <motion.div
              initial={{ opacity: 0, scale: 0.7 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.5, duration: 0.3 }}
              className="flex flex-col items-center"
            >
              {/* Deliberately NOT "Case"/"Non-case". At the calibrated operating
                  point PPV is 0.93 but NPV is 0.46, so a negative is weak
                  evidence — labelling it "Non-case" invites a clinician to read
                  a rule-out that the model cannot support. */}
              <span
                className={`rounded-full px-3 py-1.5 text-sm font-semibold ${
                  result.binary_pred
                    ? "bg-clay-100 text-clay-600"
                    : "bg-sage-100 text-sage-700"
                }`}
              >
                {result.binary_pred ? "Higher priority" : "Not prioritised"}
              </span>
              <span className="mt-2 text-sm font-medium text-sage-700">
                Screening priority
              </span>
            </motion.div>
          </div>

          <p className="rounded-xl border border-sage-200 bg-sage-50 px-4 py-3 text-xs leading-relaxed text-sage-700">
            <span className="font-semibold">Decision aid, not a diagnosis.</span>{" "}
            Estimated from 135 interviews at two Ugandan sites and not yet validated
            elsewhere. It supports deciding who to assess first and{" "}
            <span className="font-semibold">cannot rule depression out</span> — when
            it does not prioritise someone it is right fewer than half the time.
            The clinician-administered scores and clinical judgement take precedence.
          </p>

          <div>
            <h3 className="font-display text-sm font-semibold text-sage-800">Modality attribution</h3>
            <div className="mt-2 space-y-2">
              {Object.entries(result.modality_attributions).map(([modality, score], i) => (
                <AttributionBar key={modality} modality={modality} score={score} index={i} />
              ))}
            </div>
            {result.acoustic_drivers && result.acoustic_drivers.length > 0 && (
              <div className="mt-3">
                <h4 className="text-xs font-semibold uppercase tracking-wide text-sage-600">
                  Voice features driving this estimate
                </h4>
                <ul className="mt-1.5 space-y-1">
                  {result.acoustic_drivers.map((d) => (
                    <li key={d.feature} className="flex items-baseline justify-between gap-3 text-xs">
                      <span className="text-sage-700">{PROSODY_LABEL[d.feature] ?? d.feature}</span>
                      <span
                        className={
                          d.contribution > 0 ? "shrink-0 text-clay-600" : "shrink-0 text-sage-600"
                        }
                      >
                        {d.contribution > 0 ? "↑" : "↓"} {Math.abs(d.contribution).toFixed(2)}
                      </span>
                    </li>
                  ))}
                </ul>
                <p className="mt-1.5 text-[11px] leading-snug text-sage-500">
                  Measured from the recording, not inferred from the words. These are
                  associations in this cohort, not causes.
                </p>
              </div>
            )}

            <div className="mt-2">
              <EvidenceQuotes data={result.evidence} />
            </div>
          </div>

          <NarrativeCard narrative={result.narrative} />

          <SubtypeDifferential data={result.subtype_differential} />

          <TreatmentSuggestionsCard suggestions={result.treatment_suggestions} />

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
            className="rounded-full bg-sage-500 px-4 py-2 text-sm font-semibold text-white transition-transform hover:bg-sage-600 active:scale-95"
          >
            Close
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}
