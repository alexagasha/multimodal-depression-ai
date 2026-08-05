"use client";

import { useEffect, useState } from "react";
import { api, ApiError, ClinicalNote } from "@/lib/api";
import { Field, TextInput, TextArea, Card } from "@/components/FormField";

export default function ClinicalNotes({ visitId }: { visitId: string }) {
  const [notes, setNotes] = useState<ClinicalNote[]>([]);
  const [author, setAuthor] = useState("");
  const [text, setText] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function refresh() {
    api.listNotes(visitId).then(setNotes).catch(() => {});
  }

  useEffect(refresh, [visitId]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!author.trim() || !text.trim()) return;
    setSubmitting(true);
    setError(null);
    try {
      await api.addNote(visitId, author.trim(), text.trim());
      setText("");
      refresh();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Card className="space-y-4">
      <div>
        <h2 className="font-display text-lg font-semibold text-sage-800">Clinical notes</h2>
        <p className="text-xs text-sage-600">
          Append-only — amend with a new note rather than editing history, matching standard
          clinical documentation practice.
        </p>
      </div>

      <ul className="space-y-2">
        {notes.map((n) => (
          <li key={n.note_id} className="rounded-xl bg-sage-50 px-3 py-2">
            <div className="flex items-baseline justify-between text-xs text-sage-600">
              <span className="font-medium text-sage-800">{n.author}</span>
              <span>{new Date(n.created_at).toLocaleString()}</span>
            </div>
            <p className="mt-1 text-sm text-ink-900">{n.note_text}</p>
          </li>
        ))}
        {notes.length === 0 && <p className="text-sm text-sage-600">No notes yet.</p>}
      </ul>

      <form onSubmit={onSubmit} className="space-y-3">
        <Field label="Author">
          <TextInput
            value={author}
            onChange={(e) => setAuthor(e.target.value)}
            placeholder="Dr. …"
            required
          />
        </Field>
        <Field label="Note">
          <TextArea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={3}
            placeholder="Clinical observation…"
            required
          />
        </Field>
        {error && <p className="text-sm text-[var(--color-danger)]">{error}</p>}
        <button
          type="submit"
          disabled={submitting}
          className="rounded-full bg-sage-500 px-4 py-2 text-sm font-semibold text-white hover:bg-sage-600 disabled:opacity-50"
        >
          {submitting ? "Adding…" : "Add note"}
        </button>
      </form>
    </Card>
  );
}
