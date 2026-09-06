/**
 * Shared clinical-documentation vocabulary for the note surfaces.
 *
 * Kept out of lib/api.ts because none of this is wire format — it is how the
 * stored values are shown to a clinician. See
 * docs/clinical-documentation-plan.md.
 */
import { ClinicalNote, NoteSource } from "@/lib/api";

/** Cadres that actually write mental-health notes at the study sites. Sent as
 * `author_role`; a note is otherwise attributable to a name with no scope. */
export const AUTHOR_ROLES = [
  { value: "psychiatrist", label: "Psychiatrist" },
  { value: "psychiatric_clinical_officer", label: "Psychiatric clinical officer" },
  { value: "psychiatric_nurse", label: "Psychiatric nurse" },
  { value: "medical_officer", label: "Medical officer" },
  { value: "researcher", label: "Researcher" },
];

export const ROLE_LABELS: Record<string, string> = Object.fromEntries(
  AUTHOR_ROLES.map((r) => [r.value, r.label]),
);

const SOURCE_LABELS: Record<NoteSource, string> = {
  clinician: "Clinician-written",
  ai_draft_edited: "AI-drafted, clinician-edited",
  ai_draft_accepted: "AI-drafted, accepted unchanged",
};

export function sourceLabel(source: string): string {
  return SOURCE_LABELS[source as NoteSource] ?? "Clinician-written";
}

/**
 * Whether a note was signed fast enough that the review is worth questioning.
 *
 * Deliberately advisory: it annotates the record, it never blocks a save.
 * A note accepted verbatim seconds after the draft appeared is the specific
 * pattern documentation audits treat as evidence that no review happened
 * (docs/clinical-documentation-plan.md §1.5).
 */
export const QUICK_REVIEW_SECONDS = 20;

export function isQuickReview(note: ClinicalNote): boolean {
  const s = note.provenance?.review_seconds;
  return (
    note.provenance?.source === "ai_draft_accepted" &&
    typeof s === "number" &&
    s < QUICK_REVIEW_SECONDS
  );
}

/** SOAP sections in the order they are read, for rendering a stored note. */
export const SOAP_SECTIONS = [
  { key: "subjective", label: "Subjective" },
  { key: "objective", label: "Objective" },
  { key: "assessment", label: "Assessment" },
  { key: "plan", label: "Plan" },
] as const;
