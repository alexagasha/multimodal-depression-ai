"use client";

import { useState } from "react";
import { api, ApiError, NoteDraft } from "@/lib/api";
import { Field, TextArea, TextInput, Card } from "@/components/FormField";

const SOAP_FIELDS = ["subjective", "objective", "assessment", "plan"] as const;

export default function NoteDraftPanel({
  visitId,
  onSaved,
}: {
  visitId: string;
  onSaved: () => void;
}) {
  const [draft, setDraft] = useState<NoteDraft | null>(null);
  const [author, setAuthor] = useState("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function requestDraft() {
    setLoading(true);
    setError(null);
    try {
      setDraft(await api.getNoteDraft(visitId));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function saveAsNote() {
    if (!draft || !author.trim()) return;
    setSaving(true);
    setError(null);
    try {
      const note_text =
        `S: ${draft.subjective}\nO: ${draft.objective}\n` +
        `A: ${draft.assessment}\nP: ${draft.plan}`;
      await api.addNote(visitId, author.trim(), note_text);
      setDraft(null);
      onSaved();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  return (
    <Card className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="font-display text-lg font-semibold text-sage-800">AI-drafted note</h2>
        {!draft && (
          <button
            type="button"
            onClick={requestDraft}
            disabled={loading}
            className="rounded-full bg-sage-500 px-4 py-2 text-sm font-semibold text-white hover:bg-sage-600 disabled:opacity-50"
          >
            {loading ? "Drafting…" : "Draft SOAP note"}
          </button>
        )}
      </div>
      <p className="text-xs text-sage-600">
        A starting point only — review, edit, and correct before saving. You remain responsible
        for the note&apos;s accuracy, same as any note you write yourself.
      </p>

      {error && <p className="text-sm text-[var(--color-danger)]">{error}</p>}

      {draft && (
        <div className="space-y-3">
          {SOAP_FIELDS.map((field) => (
            <Field key={field} label={field[0].toUpperCase() + field.slice(1)}>
              <TextArea
                rows={2}
                value={draft[field]}
                onChange={(e) => setDraft({ ...draft, [field]: e.target.value })}
              />
            </Field>
          ))}
          <Field label="Author">
            <TextInput
              value={author}
              onChange={(e) => setAuthor(e.target.value)}
              placeholder="Dr. …"
            />
          </Field>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={saveAsNote}
              disabled={saving || !author.trim()}
              className="rounded-full bg-sage-500 px-4 py-2 text-sm font-semibold text-white hover:bg-sage-600 disabled:opacity-50"
            >
              {saving ? "Saving…" : "Save as clinical note"}
            </button>
            <button
              type="button"
              onClick={() => setDraft(null)}
              className="rounded-full border border-sage-300 px-4 py-2 text-sm font-medium text-sage-700 hover:bg-sage-100"
            >
              Discard
            </button>
          </div>
        </div>
      )}
    </Card>
  );
}
