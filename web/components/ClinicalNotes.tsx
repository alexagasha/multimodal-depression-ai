"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { api, ApiError, ClinicalNote } from "@/lib/api";
import {
  AUTHOR_ROLES,
  ROLE_LABELS,
  SOAP_SECTIONS,
  isQuickReview,
  sourceLabel,
} from "@/lib/clinical";
import { Field, TextInput, TextArea, Select, Card } from "@/components/FormField";

/** A note written here is SOAP-structured like the AI-drafted one, so the
 * record holds the same shape however it was produced. Free text remains
 * available for addenda, where forcing four headings would be noise. */
type Mode = "soap" | "free";

export default function ClinicalNotes({
  visitId,
  refreshKey,
}: {
  visitId: string;
  /** Bump this (e.g. a counter) to force a re-fetch after a note was added
   * elsewhere — NoteDraftPanel saves via the same POST .../notes endpoint
   * but this component owns its own list state. */
  refreshKey?: number;
}) {
  const [notes, setNotes] = useState<ClinicalNote[]>([]);
  const [author, setAuthor] = useState("");
  const [role, setRole] = useState("psychiatrist");
  const [mode, setMode] = useState<Mode>("soap");
  const [soap, setSoap] = useState({ subjective: "", objective: "", assessment: "", plan: "" });
  const [text, setText] = useState("");
  /** Set when writing a correction to an existing note — append-only means
   * the original stays, and this links the two. */
  const [amends, setAmends] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function refresh() {
    api.listNotes(visitId).then(setNotes).catch(() => {});
  }

  useEffect(refresh, [visitId, refreshKey]);

  const hasContent = mode === "soap" ? Object.values(soap).some((v) => v.trim()) : text.trim();

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!author.trim() || !hasContent) return;
    setSubmitting(true);
    setError(null);
    try {
      await api.addNote(visitId, {
        author: author.trim(),
        author_role: role,
        note_type: amends ? "addendum" : "progress",
        amends,
        source: "clinician",
        ...(mode === "soap" ? soap : { note_text: text.trim() }),
      });
      setSoap({ subjective: "", objective: "", assessment: "", plan: "" });
      setText("");
      setAmends(null);
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
        <AnimatePresence initial={false}>
          {notes.map((n) => (
            <motion.li
              key={n.note_id}
              initial={{ opacity: 0, y: -8, height: 0 }}
              animate={{ opacity: 1, y: 0, height: "auto" }}
              transition={{ duration: 0.3 }}
              className="overflow-hidden rounded-xl bg-sage-50 px-3 py-2"
            >
              <div className="flex items-baseline justify-between gap-2 text-xs text-sage-600">
                <span className="font-medium text-sage-800">
                  {n.author}
                  {n.author_role && (
                    <span className="font-normal text-sage-600">
                      {" "}
                      · {ROLE_LABELS[n.author_role] ?? n.author_role}
                    </span>
                  )}
                </span>
                <span>{new Date(n.signed_at ?? n.created_at).toLocaleString()}</span>
              </div>

              {n.amends && (
                <p className="mt-1 text-xs italic text-clay-600">
                  Amends earlier note {n.amends}
                </p>
              )}

              {SOAP_SECTIONS.some((s) => n[s.key]) ? (
                <dl className="mt-1 space-y-1">
                  {SOAP_SECTIONS.filter((s) => n[s.key]).map((s) => (
                    <div key={s.key} className="text-sm">
                      <dt className="inline font-semibold text-sage-700">{s.label}: </dt>
                      <dd className="inline text-ink-900">{n[s.key]}</dd>
                    </div>
                  ))}
                </dl>
              ) : (
                <p className="mt-1 whitespace-pre-wrap text-sm text-ink-900">{n.note_text}</p>
              )}

              <NoteProvenanceLine note={n} />

              <button
                type="button"
                onClick={() => {
                  setAmends(n.note_id);
                  setMode("free");
                }}
                className="mt-1 text-xs font-medium text-sage-700 underline hover:text-sage-900"
              >
                Amend
              </button>
            </motion.li>
          ))}
        </AnimatePresence>
        {notes.length === 0 && <p className="text-sm text-sage-600">No notes yet.</p>}
      </ul>

      <form onSubmit={onSubmit} className="space-y-3 border-t border-sage-200 pt-4">
        {amends && (
          <div className="flex items-center justify-between rounded-lg bg-clay-50 px-3 py-2 text-xs text-clay-700">
            <span>Writing an amendment to note {amends}.</span>
            <button
              type="button"
              onClick={() => setAmends(null)}
              className="font-medium underline"
            >
              Cancel
            </button>
          </div>
        )}

        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Author">
            <TextInput
              value={author}
              onChange={(e) => setAuthor(e.target.value)}
              placeholder="Dr. …"
              required
            />
          </Field>
          <Field label="Role">
            <Select value={role} onChange={(e) => setRole(e.target.value)}>
              {AUTHOR_ROLES.map((r) => (
                <option key={r.value} value={r.value}>
                  {r.label}
                </option>
              ))}
            </Select>
          </Field>
        </div>

        <div className="flex gap-1 text-xs">
          {(["soap", "free"] as Mode[]).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`rounded-full px-3 py-1 font-medium ${
                mode === m
                  ? "bg-sage-500 text-white"
                  : "border border-sage-300 text-sage-700 hover:bg-sage-100"
              }`}
            >
              {m === "soap" ? "SOAP" : "Free text"}
            </button>
          ))}
        </div>

        {mode === "soap" ? (
          SOAP_SECTIONS.map((s) => (
            <Field key={s.key} label={s.label}>
              <TextArea
                rows={2}
                value={soap[s.key]}
                onChange={(e) => setSoap({ ...soap, [s.key]: e.target.value })}
              />
            </Field>
          ))
        ) : (
          <Field label="Note">
            <TextArea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={3}
              placeholder="Clinical observation…"
            />
          </Field>
        )}

        {error && <p className="text-sm text-[var(--color-danger)]">{error}</p>}
        <button
          type="submit"
          disabled={submitting || !author.trim() || !hasContent}
          className="rounded-full bg-sage-500 px-4 py-2 text-sm font-semibold text-white hover:bg-sage-600 disabled:opacity-50"
        >
          {submitting ? "Adding…" : amends ? "Add amendment" : "Add note"}
        </button>
      </form>
    </Card>
  );
}

/** Who or what wrote this note, and how long it was reviewed for. A saved
 * note must not be indistinguishable from a clinician-written one — that
 * distinction is the whole point of recording provenance at all. */
function NoteProvenanceLine({ note }: { note: ClinicalNote }) {
  const p = note.provenance;
  if (!p) return null;
  const edited = p.edited_fields?.length;
  return (
    <p className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-sage-600">
      <span className="rounded-full bg-sage-200 px-2 py-0.5 font-medium text-sage-800">
        {sourceLabel(p.source)}
      </span>
      {p.ai_model && <span>{p.ai_model}</span>}
      {edited ? <span>edited: {p.edited_fields!.join(", ")}</span> : null}
      {typeof p.review_seconds === "number" && (
        <span className={isQuickReview(note) ? "font-semibold text-[var(--color-danger)]" : ""}>
          reviewed in {p.review_seconds}s
        </span>
      )}
    </p>
  );
}
