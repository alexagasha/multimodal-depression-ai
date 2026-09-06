"use client";

import { useState } from "react";
import { api, ApiError, MSEDomain, MSEResult } from "@/lib/api";
import { Card, TextArea } from "@/components/FormField";

/**
 * The mental status examination, as a scaffold rather than an output.
 *
 * Every domain is editable and three of them start deliberately empty:
 * appearance, motor behaviour and attitude cannot be observed from a
 * recording, and the panel says so on the row instead of quietly leaving a
 * gap. Speech is filled from measured prosody, not from the language model,
 * and carries no asserted normal range.
 *
 * The clinician completes it and copies it into the note's Objective section,
 * which is otherwise unstructured prose.
 */
const SOURCE_BADGE: Record<string, { label: string; className: string }> = {
  measured: { label: "measured", className: "bg-sage-200 text-sage-800" },
  transcript: { label: "from transcript", className: "bg-clay-100 text-clay-700" },
  not_observable: {
    label: "you must observe",
    className: "bg-[var(--color-danger-bg)] text-[var(--color-danger)]",
  },
};

export default function MSEPanel({
  visitId,
  onCopyToNote,
}: {
  visitId: string;
  /** Hand the completed examination back for pasting into Objective. */
  onCopyToNote?: (text: string) => void;
}) {
  const [mse, setMse] = useState<MSEResult | null>(null);
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const r = await api.getMSE(visitId);
      setMse(r);
      setEdits(
        Object.fromEntries(r.domains.map((d) => [d.domain, d.finding ?? ""])),
      );
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function asText(): string {
    if (!mse) return "";
    return mse.domains
      .map((d) => {
        const body =
          d.source === "measured" ? [describeMeasures(d), edits[d.domain]].filter(Boolean).join(" ")
          : edits[d.domain];
        return body?.trim() ? `${d.label}: ${body.trim()}` : null;
      })
      .filter(Boolean)
      .join("\n");
  }

  return (
    <Card className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="font-display text-lg font-semibold text-sage-800">
            Mental status examination
          </h2>
          <p className="text-xs text-sage-600">
            The structure behind the note&apos;s Objective section. Appearance, behaviour and
            attitude cannot be observed from a recording — they are yours to complete.
          </p>
        </div>
        {!mse && (
          <button
            type="button"
            onClick={load}
            disabled={loading}
            className="rounded-full bg-sage-500 px-4 py-2 text-sm font-semibold text-white hover:bg-sage-600 disabled:opacity-50"
          >
            {loading ? "Building…" : "Build examination"}
          </button>
        )}
      </div>

      {error && <p className="text-sm text-[var(--color-danger)]">{error}</p>}

      {mse && (
        <>
          <ul className="space-y-3">
            {mse.domains.map((d) => {
              const badge = SOURCE_BADGE[d.source];
              return (
                <li key={d.domain} className="space-y-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-semibold text-sage-800">{d.label}</span>
                    <span
                      className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${badge.className}`}
                    >
                      {badge.label}
                    </span>
                  </div>

                  {d.measures && (
                    <dl className="grid grid-cols-2 gap-x-3 gap-y-0.5 rounded-lg bg-sage-50 px-3 py-2 text-xs sm:grid-cols-4">
                      {d.measures.map((m) => (
                        <div key={m.label}>
                          <dt className="text-sage-600">{m.label}</dt>
                          <dd className="font-medium text-ink-900">
                            {m.value}
                            {m.unit}
                          </dd>
                        </div>
                      ))}
                    </dl>
                  )}

                  {d.note && <p className="text-[11px] italic text-sage-600">{d.note}</p>}

                  <TextArea
                    rows={2}
                    value={edits[d.domain] ?? ""}
                    onChange={(e) => setEdits({ ...edits, [d.domain]: e.target.value })}
                    placeholder={
                      d.source === "not_observable"
                        ? "Your observation…"
                        : d.finding
                          ? undefined
                          : "Not evidenced in the transcript — add your own."
                    }
                  />
                </li>
              );
            })}
          </ul>

          <div className="flex flex-wrap gap-2">
            {onCopyToNote && (
              <button
                type="button"
                onClick={() => onCopyToNote(asText())}
                className="rounded-full bg-sage-500 px-4 py-2 text-sm font-semibold text-white hover:bg-sage-600"
              >
                Use as Objective
              </button>
            )}
            <button
              type="button"
              onClick={load}
              disabled={loading}
              className="rounded-full border border-sage-300 px-4 py-2 text-sm font-medium text-sage-700 hover:bg-sage-100 disabled:opacity-50"
            >
              {loading ? "Rebuilding…" : "Rebuild"}
            </button>
          </div>
        </>
      )}
    </Card>
  );
}

/** Flatten the speech measurements into the one sentence a note would carry. */
function describeMeasures(d: MSEDomain): string {
  if (!d.measures?.length) return "";
  return d.measures.map((m) => `${m.label.toLowerCase()} ${m.value}${m.unit}`).join(", ") + ".";
}
