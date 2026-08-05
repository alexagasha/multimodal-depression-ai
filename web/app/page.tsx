"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { motion, AnimatePresence } from "motion/react";
import { api, PatientSummary, ApiError, CaseloadQueryResult } from "@/lib/api";
import EmptyState from "@/components/illustrations/EmptyState";

function ScorePill({ label, value }: { label: string; value: number | null }) {
  return (
    <div className="rounded-lg bg-sage-50 px-2.5 py-1 text-xs font-medium text-sage-700">
      {label} {value === null ? "–" : value.toFixed(1)}
    </div>
  );
}

function TrendBadge({ label }: { label: string }) {
  return (
    <span className="rounded-full border border-amber-300 bg-amber-50 px-2 py-0.5 text-xs font-medium text-amber-700">
      {label}
    </span>
  );
}

function CaseloadQueryBox({ onResult }: { onResult: (result: CaseloadQueryResult | null) => void }) {
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setAsking(true);
    setError(null);
    try {
      onResult(await api.queryCaseload(question.trim()));
    } catch (e) {
      onResult(null);
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setAsking(false);
    }
  }

  return (
    <form onSubmit={onSubmit} className="flex flex-wrap items-center gap-2">
      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask your caseload — e.g. “which patients haven't improved in 3 visits?”"
        className="min-w-[280px] flex-1 rounded-xl border border-sage-200 bg-white px-3 py-2 text-sm shadow-sm focus:border-sage-500 focus:outline-none focus:ring-2 focus:ring-sage-200"
      />
      <button
        type="submit"
        disabled={asking || !question.trim()}
        className="rounded-full bg-sage-500 px-4 py-2 text-sm font-semibold text-white hover:bg-sage-600 disabled:opacity-50"
      >
        {asking ? "Asking…" : "Ask"}
      </button>
      {error && <p className="w-full text-sm text-[var(--color-danger)]">{error}</p>}
    </form>
  );
}

export default function PatientRosterPage() {
  const [patients, setPatients] = useState<PatientSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [queryResult, setQueryResult] = useState<CaseloadQueryResult | null>(null);

  useEffect(() => {
    api
      .listPatients()
      .then(setPatients)
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, []);

  const filtered = useMemo(() => {
    if (!patients) return null;
    const q = query.trim().toLowerCase();
    if (!q) return patients;
    return patients.filter((p) => p.participant_id.toLowerCase().includes(q));
  }, [patients, query]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-display text-2xl font-semibold text-sage-900">
            Patients
          </h1>
          <p className="text-sm text-sage-700">
            Referral-flagged patients first, then by severity of their latest visit.
          </p>
        </div>
        <Link
          href="/patients/new"
          className="rounded-full bg-sage-500 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-sage-600"
        >
          + Register new patient
        </Link>
      </div>

      {patients && patients.length > 0 && (
        <CaseloadQueryBox onResult={setQueryResult} />
      )}

      {queryResult && (
        <div className="rounded-xl bg-sage-50 px-4 py-3 text-sm text-ink-900">
          <p className="text-xs font-semibold uppercase tracking-wide text-sage-500">
            AI answer — based only on the roster shown below
          </p>
          <p className="mt-1">{queryResult.answer}</p>
        </div>
      )}

      {patients && patients.length > 0 && (
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by patient ID…"
          className="w-full max-w-sm rounded-xl border border-sage-200 bg-white px-3 py-2 text-sm shadow-sm focus:border-sage-500 focus:outline-none focus:ring-2 focus:ring-sage-200"
        />
      )}

      {error && (
        <div className="rounded-xl border border-[var(--color-danger-border)] bg-[var(--color-danger-bg)] px-4 py-3 text-sm text-[var(--color-danger)]">
          Could not reach the API at {process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000"}: {error}
        </div>
      )}

      {patients && patients.length === 0 && (
        <div className="flex flex-col items-center gap-3 rounded-3xl border border-sage-200 bg-white/60 px-6 py-16 text-center">
          <EmptyState />
          <p className="font-display text-lg text-sage-800">No patients yet</p>
          <p className="max-w-sm text-sm text-sage-600">
            Register a patient to begin their first visit.
          </p>
          <Link
            href="/patients/new"
            className="mt-2 rounded-full bg-sage-500 px-4 py-2 text-sm font-semibold text-white hover:bg-sage-600"
          >
            + Register new patient
          </Link>
        </div>
      )}

      {filtered && filtered.length === 0 && patients && patients.length > 0 && (
        <p className="text-sm text-sage-600">No patients match &ldquo;{query}&rdquo;.</p>
      )}

      {filtered && filtered.length > 0 && (
        <ul className="space-y-3">
          <AnimatePresence initial={true}>
            {filtered.map((p, i) => {
              const isQueryMatch = queryResult?.matching_patient_ids.includes(p.participant_id);
              return (
                <motion.li
                  key={p.participant_id}
                  layout
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ delay: Math.min(i, 8) * 0.05, duration: 0.3 }}
                  className="relative"
                >
                  {/* Stretched-link pattern (see git history) so the card is one
                      big click target without nesting <a> inside <a>. */}
                  <div
                    className={`flex flex-wrap items-center justify-between gap-3 rounded-2xl border bg-white/70 px-4 py-3 shadow-sm transition hover:shadow-md ${
                      p.risk_flag
                        ? "border-[var(--color-danger-border)]"
                        : isQueryMatch
                          ? "border-sage-500 ring-2 ring-sage-200"
                          : "border-sage-200"
                    }`}
                  >
                    <Link
                      href={`/patients/${p.participant_id}`}
                      className="absolute inset-0 z-0"
                      aria-label={`Open patient ${p.participant_id}`}
                    />
                    <div className="relative z-10 flex items-center gap-3">
                      {p.risk_flag && (
                        <span className="relative flex items-center rounded-full bg-[var(--color-danger)] px-2 py-0.5 text-xs font-bold text-white">
                          <span className="absolute inset-0 animate-pulse rounded-full bg-[var(--color-danger)] opacity-60" />
                          <span className="relative">REFERRAL</span>
                        </span>
                      )}
                      <div>
                        <p className="font-medium text-ink-900">Patient {p.participant_id}</p>
                        <p className="text-xs text-sage-600">
                          {p.visit_count} visit{p.visit_count === 1 ? "" : "s"}
                          {p.last_visit_at &&
                            ` · last seen ${new Date(p.last_visit_at).toLocaleDateString()}`}
                        </p>
                      </div>
                    </div>
                    <div className="relative z-10 flex flex-wrap items-center justify-end gap-2">
                      {p.relapse_warning.flag && <TrendBadge label="Worsening trend" />}
                      {p.risk_trajectory.flag && <TrendBadge label="Rising risk pattern" />}
                      <ScorePill label="PHQ-9" value={p.phq9_pred} />
                      <ScorePill label="HAM-D" value={p.hamd_pred} />
                      {p.visit_count === 0 && (
                        <span className="rounded-full bg-sage-100 px-2.5 py-1 text-xs font-medium text-sage-700">
                          No visits yet
                        </span>
                      )}
                    </div>
                  </div>
                </motion.li>
              );
            })}
          </AnimatePresence>
        </ul>
      )}
    </div>
  );
}
