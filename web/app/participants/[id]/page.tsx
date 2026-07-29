"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { api, ApiError, SessionSummary } from "@/lib/api";
import { Card } from "@/components/FormField";
import TrendSparkline from "@/components/TrendSparkline";

export default function ParticipantTrendPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    api
      .listParticipantSessions(id)
      .then(setSessions)
      .catch((e) => {
        if (e instanceof ApiError && e.status === 404) setNotFound(true);
      });
  }, [id]);

  async function startFollowUpSession() {
    setCreating(true);
    try {
      const { session_id } = await api.createSession(id);
      router.push(`/sessions/${session_id}`);
    } finally {
      setCreating(false);
    }
  }

  if (notFound) return <p className="text-sm text-sage-700">Participant {id} not found.</p>;
  if (!sessions) return <p className="text-sm text-sage-700">Loading…</p>;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-2xl font-semibold text-sage-900">
          Participant {id}
        </h1>
        <p className="text-sm text-sage-700">
          {sessions.length} session{sessions.length === 1 ? "" : "s"} on record.
        </p>
      </div>

      <Card>
        <h2 className="font-display text-base font-semibold text-sage-800">
          Severity over time
        </h2>
        <div className="mt-3">
          <TrendSparkline sessions={sessions} />
        </div>
      </Card>

      <ul className="space-y-2">
        {sessions.map((s) => (
          <li key={s.session_id}>
            <Link
              href={`/sessions/${s.session_id}`}
              className={`flex flex-wrap items-center justify-between gap-2 rounded-2xl border bg-white/70 px-4 py-3 shadow-sm hover:shadow-md ${
                s.risk_flag ? "border-[var(--color-danger-border)]" : "border-sage-200"
              }`}
            >
              <div>
                <p className="font-medium text-ink-900">Session {s.session_id}</p>
                <p className="text-xs text-sage-600">
                  {new Date(s.created_at).toLocaleString()}
                </p>
              </div>
              <div className="flex items-center gap-2 text-sm text-sage-700">
                <span>PHQ-9 {s.phq9_pred?.toFixed(1) ?? "–"}</span>
                <span>HAM-D {s.hamd_pred?.toFixed(1) ?? "–"}</span>
                {s.risk_flag && (
                  <span className="rounded-full bg-[var(--color-danger)] px-2 py-0.5 text-xs font-bold text-white">
                    REFERRAL
                  </span>
                )}
              </div>
            </Link>
          </li>
        ))}
      </ul>

      <div className="flex flex-wrap gap-3">
        <button
          onClick={startFollowUpSession}
          disabled={creating}
          className="rounded-full bg-sage-500 px-4 py-2 text-sm font-semibold text-white hover:bg-sage-600 disabled:opacity-50"
        >
          {creating ? "Starting…" : "+ Follow-up session for this participant"}
        </button>
        <Link
          href="/intake"
          className="rounded-full border border-sage-300 px-4 py-2 text-sm font-medium text-sage-700 hover:bg-sage-100"
        >
          + New participant instead
        </Link>
      </div>
    </div>
  );
}
