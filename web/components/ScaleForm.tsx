"use client";

import { useState } from "react";
import { api, ApiError, ScaleResponsesIn } from "@/lib/api";
import { Field, TextInput, Card } from "@/components/FormField";
import RiskBanner from "@/components/RiskBanner";
import { useDraftAutosave } from "@/lib/useDraftAutosave";

const ITEM_SPEECH: Record<string, string> = {
  phq9_total: "PHQ-9 total score, from zero to twenty seven.",
  hamd_total: "HAM-D total score, from zero to forty four.",
  phq9_item9:
    "Over the last two weeks, how often have you been bothered by thoughts that you " +
    "would be better off dead, or of hurting yourself in some way? Zero means not at all. " +
    "One means several days. Two means more than half the days. Three means nearly every day.",
  hamd_suicide_item:
    "Clinician-rated suicide domain, zero to four: zero means absent, increasing severity " +
    "up to four for suicide attempt.",
};

function speak(text: string) {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

function ListenButton({ itemKey }: { itemKey: keyof typeof ITEM_SPEECH }) {
  return (
    <button
      type="button"
      onClick={() => speak(ITEM_SPEECH[itemKey])}
      aria-label={`Listen: ${ITEM_SPEECH[itemKey]}`}
      className="rounded-full border border-sage-300 px-2 py-1 text-xs text-sage-700 hover:bg-sage-100"
    >
      🔊 Listen
    </button>
  );
}

export default function ScaleForm({
  sessionId,
  onDone,
}: {
  sessionId: string;
  onDone: (riskFlag: boolean) => void;
}) {
  const [form, setForm, clearDraft] = useDraftAutosave<ScaleResponsesIn>(`scales:${sessionId}`, {
    phq9_total: 0,
    hamd_total: 0,
    phq9_item9: 0,
    hamd_suicide_item: 0,
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [riskFlag, setRiskFlag] = useState<boolean | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const { risk_flag } = await api.submitScaleResponses(sessionId, form);
      setRiskFlag(risk_flag);
      clearDraft();
      onDone(risk_flag);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Card className="space-y-4">
      <h2 className="font-display text-lg font-semibold text-sage-800">
        PHQ-9 / HAM-D scale responses
      </h2>

      <form onSubmit={onSubmit} className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="PHQ-9 total (0–27)">
            <div className="flex items-center gap-2">
              <TextInput
                type="number"
                min={0}
                max={27}
                required
                value={form.phq9_total}
                onChange={(e) => setForm({ ...form, phq9_total: Number(e.target.value) })}
              />
              <ListenButton itemKey="phq9_total" />
            </div>
          </Field>
          <Field label="HAM-D total (0–44)">
            <div className="flex items-center gap-2">
              <TextInput
                type="number"
                min={0}
                max={44}
                required
                value={form.hamd_total}
                onChange={(e) => setForm({ ...form, hamd_total: Number(e.target.value) })}
              />
              <ListenButton itemKey="hamd_total" />
            </div>
          </Field>
          <Field
            label="PHQ-9 item 9 (0–3)"
            hint="Thoughts of self-harm — carries a mandatory referral protocol."
          >
            <div className="flex items-center gap-2">
              <TextInput
                type="number"
                min={0}
                max={3}
                required
                value={form.phq9_item9}
                onChange={(e) => setForm({ ...form, phq9_item9: Number(e.target.value) })}
              />
              <ListenButton itemKey="phq9_item9" />
            </div>
          </Field>
          <Field label="HAM-D suicide domain (0–4)" hint="Same referral protocol.">
            <div className="flex items-center gap-2">
              <TextInput
                type="number"
                min={0}
                max={4}
                required
                value={form.hamd_suicide_item}
                onChange={(e) => setForm({ ...form, hamd_suicide_item: Number(e.target.value) })}
              />
              <ListenButton itemKey="hamd_suicide_item" />
            </div>
          </Field>
        </div>

        {error && (
          <p className="rounded-lg bg-[var(--color-danger-bg)] px-3 py-2 text-sm text-[var(--color-danger)]">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={submitting}
          className="rounded-full bg-sage-500 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-sage-600 disabled:opacity-50"
        >
          {submitting ? "Submitting…" : "Submit scale responses"}
        </button>
      </form>

      {riskFlag !== null && (riskFlag ? <RiskBanner /> : (
        <p className="text-sm text-sage-700">Recorded — no referral flag.</p>
      ))}
    </Card>
  );
}
