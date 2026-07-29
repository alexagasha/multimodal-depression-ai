"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, SessionSummary, ApiError } from "@/lib/api";
import EmptyState from "@/components/illustrations/EmptyState";

const STATUS_LABEL: Record<string, string> = {
  created: "Intake complete",
  scale_responses_recorded: "Scales recorded",
  audio_uploaded: "Audio uploaded",
  scored: "Scored",
};

function ScorePill({ label, value }: { label: string; value: number | null }) {
  return (
    <div className="rounded-lg bg-sage-50 px-2.5 py-1 text-xs font-medium text-sage-700">
      {label} {value === null ? "–" : value.toFixed(1)}
    </div>
  );
}

export default function DashboardPage() {
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listSessions()
      .then(setSessions)
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-display text-2xl font-semibold text-sage-900">
            Triage queue
          </h1>
          <p className="text-sm text-sage-700">
            Referral-flagged sessions first, then by severity.
          </p>
        </div>
        <Link
          href="/intake"
          className="rounded-full bg-sage-500 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-sage-600"
        >
          + New session
        </Link>
      </div>

      {error && (
        <div className="rounded-xl border border-[var(--color-danger-border)] bg-[var(--color-danger-bg)] px-4 py-3 text-sm text-[var(--color-danger)]">
          Could not reach the API at {process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000"}: {error}
        </div>
      )}

      {sessions && sessions.length === 0 && (
        <div className="flex flex-col items-center gap-3 rounded-3xl border border-sage-200 bg-white/60 px-6 py-16 text-center">
          <EmptyState />
          <p className="font-display text-lg text-sage-800">No sessions yet</p>
          <p className="max-w-sm text-sm text-sage-600">
            Start a new intake to register a participant and begin a screening session.
          </p>
          <Link
            href="/intake"
            className="mt-2 rounded-full bg-sage-500 px-4 py-2 text-sm font-semibold text-white hover:bg-sage-600"
          >
            + New session
          </Link>
        </div>
      )}

      {sessions && sessions.length > 0 && (
        <ul className="space-y-3">
          {sessions.map((s) => (
            <li key={s.session_id} className="relative">
              {/* "Stretched link" pattern: the session Link covers the whole
                  card via absolute inset-0, while the nested "view history"
                  link sits above it (relative + higher stacking context) so
                  both remain independently clickable without nesting <a>
                  inside <a>, which is invalid HTML and breaks hydration. */}
              <div
                className={`flex flex-wrap items-center justify-between gap-3 rounded-2xl border bg-white/70 px-4 py-3 shadow-sm transition hover:shadow-md ${
                  s.risk_flag ? "border-[var(--color-danger-border)]" : "border-sage-200"
                }`}
              >
                <Link
                  href={`/sessions/${s.session_id}`}
                  className="absolute inset-0 z-0"
                  aria-label={`Open session ${s.session_id}`}
                />
                <div className="relative z-10 flex items-center gap-3">
                  {s.risk_flag && (
                    <span className="rounded-full bg-[var(--color-danger)] px-2 py-0.5 text-xs font-bold text-white">
                      REFERRAL
                    </span>
                  )}
                  <div>
                    <p className="font-medium text-ink-900">Session {s.session_id}</p>
                    <p className="text-xs text-sage-600">
                      Participant {s.participant_id} ·{" "}
                      <Link
                        href={`/participants/${s.participant_id}`}
                        className="relative z-10 underline decoration-sage-300 hover:text-sage-800"
                      >
                        view history
                      </Link>
                    </p>
                  </div>
                </div>
                <div className="relative z-10 flex items-center gap-2">
                  <ScorePill label="PHQ-9" value={s.phq9_pred} />
                  <ScorePill label="HAM-D" value={s.hamd_pred} />
                  {s.reviewed && (
                    <span className="rounded-full bg-clay-100 px-2.5 py-1 text-xs font-medium text-clay-600">
                      Reviewed
                    </span>
                  )}
                  <span className="rounded-full bg-sage-100 px-2.5 py-1 text-xs font-medium text-sage-700">
                    {STATUS_LABEL[s.status] ?? s.status}
                  </span>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
