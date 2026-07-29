"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { api, ApiError, ParticipantIn } from "@/lib/api";
import { Field, Select, Card } from "@/components/FormField";
import { useDraftAutosave } from "@/lib/useDraftAutosave";

const DEFAULT: ParticipantIn = {
  age_band: "26-35",
  sex: "female",
  marital_status: "married",
  ethnicity: "bantu",
  residence: "urban",
  education_level: "secondary",
  employment_status: "employed",
  smartphone: "yes",
  site: "butabika",
};

const OPTIONS: Record<keyof ParticipantIn, { value: string; label: string }[]> = {
  age_band: ["18-25", "26-35", "36-45", "46-55", "56-65"].map((v) => ({ value: v, label: v })),
  sex: [
    { value: "female", label: "Female" },
    { value: "male", label: "Male" },
  ],
  marital_status: [
    { value: "married", label: "Married" },
    { value: "divorced", label: "Divorced" },
    { value: "widowed", label: "Widowed" },
    { value: "single", label: "Single" },
  ],
  ethnicity: [
    { value: "bantu", label: "Bantu" },
    { value: "luo", label: "Luo" },
    { value: "other", label: "Other" },
  ],
  residence: [
    { value: "urban", label: "Urban" },
    { value: "semi-urban", label: "Semi-urban" },
    { value: "rural", label: "Rural" },
  ],
  education_level: [
    { value: "none", label: "None" },
    { value: "primary", label: "Primary" },
    { value: "secondary", label: "Secondary" },
    { value: "tertiary_university", label: "Tertiary / University" },
  ],
  employment_status: [
    { value: "employed", label: "Employed" },
    { value: "unemployed", label: "Unemployed" },
    { value: "self-employed", label: "Self-employed" },
    { value: "student", label: "Student" },
  ],
  smartphone: [
    { value: "yes", label: "Yes" },
    { value: "no", label: "No" },
  ],
  site: [
    { value: "butabika", label: "Butabika" },
    { value: "mulago", label: "Mulago" },
    { value: "other", label: "Other" },
  ],
};

const LABELS: Record<keyof ParticipantIn, string> = {
  age_band: "Age band",
  sex: "Sex",
  marital_status: "Marital status",
  ethnicity: "Ethnicity / tribe",
  residence: "Residence",
  education_level: "Education level",
  employment_status: "Employment status",
  smartphone: "Owns a smartphone?",
  site: "Site",
};

export default function IntakePage() {
  const router = useRouter();
  const [form, setForm, clearDraft] = useDraftAutosave<ParticipantIn>("intake", DEFAULT);
  const [consented, setConsented] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const { participant_id } = await api.createParticipant(form);
      const { session_id } = await api.createSession(participant_id);
      clearDraft();
      router.push(`/sessions/${session_id}`);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <div>
        <h1 className="font-display text-2xl font-semibold text-sage-900">
          New participant intake
        </h1>
        <p className="text-sm text-sage-700">Section A demographics, then a new session begins.</p>
      </div>

      <Card>
        <h2 className="font-display text-base font-semibold text-sage-800">Informed consent</h2>
        <p className="mt-2 text-sm leading-relaxed text-ink-700">
          This session records interview audio and demographic details to produce an AI-assisted
          PHQ-9/HAM-D severity estimate for clinician review — it does not replace a clinical
          diagnosis. Data is de-identified for the model (site, tribe, and other provenance
          fields are kept separate from the scoring pipeline). Participation is voluntary and can
          be withdrawn at any time.
        </p>
        <label className="mt-3 flex items-start gap-2 text-sm text-ink-900">
          <input
            type="checkbox"
            checked={consented}
            onChange={(e) => setConsented(e.target.checked)}
            className="mt-0.5 h-4 w-4 rounded border-sage-300 text-sage-600 focus:ring-sage-400"
          />
          The participant has been informed and consents to proceed.
        </label>
      </Card>

      <form onSubmit={onSubmit}>
        <Card className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            {(Object.keys(DEFAULT) as (keyof ParticipantIn)[]).map((key) => (
              <Field key={key} label={LABELS[key]}>
                <Select
                  value={form[key]}
                  onChange={(e) => setForm({ ...form, [key]: e.target.value })}
                  required
                >
                  {OPTIONS[key].map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </Select>
              </Field>
            ))}
          </div>

          {error && (
            <p className="rounded-lg bg-[var(--color-danger-bg)] px-3 py-2 text-sm text-[var(--color-danger)]">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={!consented || submitting}
            className="w-full rounded-full bg-sage-500 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-sage-600 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {submitting ? "Creating session…" : "Create participant & session"}
          </button>
        </Card>
      </form>
    </div>
  );
}
