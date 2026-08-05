"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { api, ApiError, NoteDraft } from "@/lib/api";
import { Field, TextArea, TextInput, Card } from "@/components/FormField";

const SOAP_FIELDS = ["subjective", "objective", "assessment", "plan"] as const;

/**
 * The AI note writes itself during the interview: `liveDraft` is pushed in
 * from the recorder as the server redrafts it from the running transcript,
 * so there's no "draft this for me" button to click. Once the clinician edits
 * a section, incoming live updates stop overwriting their work — losing typed
 * text to a background refresh would be far worse than a slightly stale draft.
 */
export default function NoteDraftPanel({
  visitId,
  liveDraft,
  recording,
  onSaved,
}: {
  visitId: string;
  liveDraft: NoteDraft | null;
  recording: boolean;
  onSaved: () => void;
}) {
  const [draft, setDraft] = useState<NoteDraft | null>(null);
  const [edited, setEdited] = useState(false);
  const [author, setAuthor] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prevLiveDraft, setPrevLiveDraft] = useState<NoteDraft | null>(liveDraft);

  // Adopt each new server draft during render (React's documented "adjusting
  // state when a prop changes" pattern) rather than from an effect, which
  // would cascade an extra render on every redraft.
  if (liveDraft !== prevLiveDraft) {
    setPrevLiveDraft(liveDraft);
    if (liveDraft && !edited) setDraft(liveDraft);
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
      setEdited(false);
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
        <AnimatePresence>
          {recording && !edited && (
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-1.5 text-xs font-medium text-clay-600"
            >
              <span className="flex gap-0.5">
                {[0, 1, 2].map((i) => (
                  <motion.span
                    key={i}
                    className="h-1 w-1 rounded-full bg-clay-500"
                    animate={{ opacity: [0.2, 1, 0.2] }}
                    transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
                  />
                ))}
              </span>
              Writing as you talk
            </motion.span>
          )}
        </AnimatePresence>
      </div>
      <p className="text-xs text-sage-600">
        A starting point only — review, edit, and correct before saving. You remain responsible
        for the note&apos;s accuracy, same as any note you write yourself.
      </p>

      {error && <p className="text-sm text-[var(--color-danger)]">{error}</p>}

      {!draft ? (
        <p className="text-sm text-sage-600">
          {recording
            ? "The draft appears once there's enough of the interview to summarise."
            : "Not available — requires an LLM connection (ANTHROPIC_API_KEY)."}
        </p>
      ) : (
        <div className="space-y-3">
          {SOAP_FIELDS.map((field) => (
            <Field key={field} label={field[0].toUpperCase() + field.slice(1)}>
              <TextArea
                rows={2}
                value={draft[field]}
                onChange={(e) => {
                  setEdited(true);
                  setDraft({ ...draft, [field]: e.target.value });
                }}
              />
            </Field>
          ))}
          {edited && (
            <p className="text-xs text-sage-600">
              Live updates paused — your edits won&apos;t be overwritten.
            </p>
          )}
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
              onClick={() => {
                setDraft(null);
                setEdited(false);
              }}
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
