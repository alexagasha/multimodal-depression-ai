"use client";

import { useEffect, useState } from "react";
import {
  api,
  ApiError,
  Disposition,
  Ideation,
  RiskAssessment,
  RiskAssessmentIn,
} from "@/lib/api";
import { AUTHOR_ROLES, ROLE_LABELS } from "@/lib/clinical";
import { Field, TextInput, TextArea, Select, Card } from "@/components/FormField";

/**
 * The record of a risk assessment actually being done.
 *
 * Until this existed the referral banner fired and nothing captured what the
 * clinician found or decided. Deliberately NOT themed with the sage/clay
 * palette, for the same reason RiskBanner isn't: this is the safety path, and
 * it should not look like the rest of the dashboard.
 *
 * Nothing here is prefilled by the model. The one AI affordance is the list of
 * candidate verbatim quotes, which the clinician chooses to insert or ignore —
 * a patient's own words are more defensible in the record than a paraphrase.
 */
const IDEATION_OPTIONS: { value: Ideation; label: string }[] = [
  { value: "none", label: "None reported" },
  { value: "passive", label: "Passive — better off dead, not wanting to wake up" },
  { value: "active", label: "Active — thoughts of killing themselves" },
];

const DISPOSITION_OPTIONS: { value: Disposition; label: string }[] = [
  { value: "routine_follow_up", label: "Routine follow-up" },
  { value: "urgent_follow_up", label: "Urgent follow-up" },
  { value: "referral", label: "Referral" },
  { value: "same_day_referral", label: "Same-day referral" },
  { value: "admission", label: "Admission" },
];

const EMPTY: RiskAssessmentIn = {
  assessor: "",
  assessor_role: "psychiatrist",
  ideation: "none",
  intent: false,
  plan: false,
  plan_description: "",
  means_access: false,
  means_description: "",
  prior_attempts: false,
  prior_attempts_description: "",
  protective_factors: [],
  verbatim_quotes: [],
  safety_plan: "",
  disposition: "routine_follow_up",
  clinical_reasoning: "",
};

export default function RiskAssessmentForm({
  visitId,
  onSaved,
}: {
  visitId: string;
  onSaved?: () => void;
}) {
  const [existing, setExisting] = useState<RiskAssessment[]>([]);
  const [form, setForm] = useState<RiskAssessmentIn>(EMPTY);
  const [protectiveText, setProtectiveText] = useState("");
  const [quotes, setQuotes] = useState<string[]>([]);
  const [open, setOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function refresh() {
    api.listRiskAssessments(visitId).then(setExisting).catch(() => {});
  }

  useEffect(refresh, [visitId]);

  useEffect(() => {
    if (!open) return;
    // Suggestions are best-effort: no key, no transcript, no quotes — the
    // form stays fully usable either way.
    api
      .getRiskQuotes(visitId)
      .then((r) => setQuotes(r.quotes))
      .catch(() => setQuotes([]));
  }, [open, visitId]);

  function set<K extends keyof RiskAssessmentIn>(key: K, value: RiskAssessmentIn[K]) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      await api.addRiskAssessment(visitId, {
        ...form,
        assessor: form.assessor.trim(),
        protective_factors: protectiveText
          .split("\n")
          .map((s) => s.trim())
          .filter(Boolean),
      });
      setForm(EMPTY);
      setProtectiveText("");
      setOpen(false);
      refresh();
      onSaved?.();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  const ready = form.assessor.trim() && form.clinical_reasoning.trim();

  return (
    <Card className="space-y-4 border-2 border-[var(--color-danger-border)]">
      <div>
        <h2 className="font-display text-lg font-semibold text-[var(--color-danger)]">
          Risk assessment
        </h2>
        <p className="text-xs text-sage-600">
          Completed by the clinician. Nothing on this form is filled in by the model — quote
          suggestions are drawn verbatim from the transcript and are yours to use or ignore.
        </p>
      </div>

      {existing.length > 0 && (
        <ul className="space-y-2">
          {existing.map((a) => (
            <li key={a.assessment_id} className="rounded-xl bg-sage-50 px-3 py-2 text-sm">
              <div className="flex items-baseline justify-between text-xs text-sage-600">
                <span className="font-medium text-sage-800">
                  {a.assessor}
                  {a.assessor_role && ` · ${ROLE_LABELS[a.assessor_role] ?? a.assessor_role}`}
                </span>
                <span>{new Date(a.assessed_at).toLocaleString()}</span>
              </div>
              <p className="mt-1 text-ink-900">
                <strong>{IDEATION_OPTIONS.find((o) => o.value === a.ideation)?.label}</strong>
                {a.intent && " · intent"}
                {a.plan && " · plan"}
                {a.means_access && " · access to means"}
                {a.prior_attempts && " · prior attempts"}
              </p>
              {a.protective_factors.length > 0 && (
                <p className="mt-1 text-xs text-sage-700">
                  Protective: {a.protective_factors.join("; ")}
                </p>
              )}
              <p className="mt-1 text-xs text-sage-700">
                Disposition:{" "}
                {DISPOSITION_OPTIONS.find((o) => o.value === a.disposition)?.label}
              </p>
              <p className="mt-1 text-ink-900">{a.clinical_reasoning}</p>
            </li>
          ))}
        </ul>
      )}

      {!open ? (
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="rounded-full bg-[var(--color-danger)] px-4 py-2 text-sm font-semibold text-white"
        >
          {existing.length ? "Record a reassessment" : "Record risk assessment"}
        </button>
      ) : (
        <form onSubmit={onSubmit} className="space-y-3 border-t border-sage-200 pt-4">
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Assessor">
              <TextInput
                value={form.assessor}
                onChange={(e) => set("assessor", e.target.value)}
                placeholder="Dr. …"
                required
              />
            </Field>
            <Field label="Role">
              <Select
                value={form.assessor_role ?? ""}
                onChange={(e) => set("assessor_role", e.target.value)}
              >
                {AUTHOR_ROLES.map((r) => (
                  <option key={r.value} value={r.value}>
                    {r.label}
                  </option>
                ))}
              </Select>
            </Field>
          </div>

          <Field label="Ideation">
            <Select
              value={form.ideation}
              onChange={(e) => set("ideation", e.target.value as Ideation)}
            >
              {IDEATION_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </Select>
          </Field>

          <fieldset className="grid gap-2 sm:grid-cols-2">
            {(
              [
                ["intent", "Intent to act"],
                ["plan", "Has a plan"],
                ["means_access", "Access to means"],
                ["prior_attempts", "Prior attempts"],
              ] as const
            ).map(([key, label]) => (
              <label key={key} className="flex items-center gap-2 text-sm text-ink-900">
                <input
                  type="checkbox"
                  checked={form[key] as boolean}
                  onChange={(e) => set(key, e.target.checked)}
                  className="h-4 w-4 accent-[var(--color-danger)]"
                />
                {label}
              </label>
            ))}
          </fieldset>

          {form.plan && (
            <Field label="Plan — describe">
              <TextArea
                rows={2}
                value={form.plan_description ?? ""}
                onChange={(e) => set("plan_description", e.target.value)}
              />
            </Field>
          )}
          {form.means_access && (
            <Field label="Means — describe">
              <TextArea
                rows={2}
                value={form.means_description ?? ""}
                onChange={(e) => set("means_description", e.target.value)}
              />
            </Field>
          )}
          {form.prior_attempts && (
            <Field label="Prior attempts — describe">
              <TextArea
                rows={2}
                value={form.prior_attempts_description ?? ""}
                onChange={(e) => set("prior_attempts_description", e.target.value)}
              />
            </Field>
          )}

          <Field
            label="Protective factors"
            hint="One per line. What the disposition is weighed against."
          >
            <TextArea
              rows={3}
              value={protectiveText}
              onChange={(e) => setProtectiveText(e.target.value)}
              placeholder={"Two young children\nAttends church weekly"}
            />
          </Field>

          {quotes.length > 0 && (
            <div className="rounded-xl bg-sage-50 px-3 py-2">
              <p className="text-xs font-medium text-sage-800">
                From the transcript — click to quote in the record
              </p>
              <ul className="mt-1 space-y-1">
                {quotes.map((q) => {
                  const chosen = form.verbatim_quotes.includes(q);
                  return (
                    <li key={q}>
                      <button
                        type="button"
                        onClick={() =>
                          set(
                            "verbatim_quotes",
                            chosen
                              ? form.verbatim_quotes.filter((x) => x !== q)
                              : [...form.verbatim_quotes, q],
                          )
                        }
                        className={`w-full rounded-lg px-2 py-1 text-left text-xs ${
                          chosen
                            ? "bg-sage-500 text-white"
                            : "bg-white text-ink-900 hover:bg-sage-100"
                        }`}
                      >
                        &ldquo;{q}&rdquo;
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}

          <Field label="Safety plan" hint="Collaborative coping steps, not a no-harm contract.">
            <TextArea
              rows={2}
              value={form.safety_plan ?? ""}
              onChange={(e) => set("safety_plan", e.target.value)}
            />
          </Field>

          <Field label="Disposition">
            <Select
              value={form.disposition}
              onChange={(e) => set("disposition", e.target.value as Disposition)}
            >
              {DISPOSITION_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </Select>
          </Field>

          <Field
            label="Clinical reasoning"
            hint="Why this disposition — name the specific factors. Required."
          >
            <TextArea
              rows={3}
              value={form.clinical_reasoning}
              onChange={(e) => set("clinical_reasoning", e.target.value)}
              required
            />
          </Field>

          {error && <p className="text-sm text-[var(--color-danger)]">{error}</p>}

          <div className="flex gap-2">
            <button
              type="submit"
              disabled={saving || !ready}
              className="rounded-full bg-[var(--color-danger)] px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
            >
              {saving ? "Saving…" : "Save risk assessment"}
            </button>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="rounded-full border border-sage-300 px-4 py-2 text-sm font-medium text-sage-700 hover:bg-sage-100"
            >
              Cancel
            </button>
          </div>
        </form>
      )}
    </Card>
  );
}
