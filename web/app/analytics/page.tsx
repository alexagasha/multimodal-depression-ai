"use client";

import { useEffect, useState } from "react";
import { api, AnalyticsResult, ApiError } from "@/lib/api";
import { Card } from "@/components/FormField";

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-sage-200 bg-white/70 px-4 py-3 text-center shadow-sm">
      <div className="font-display text-2xl font-semibold text-sage-800">{value}</div>
      <div className="mt-1 text-xs text-sage-600">{label}</div>
    </div>
  );
}

export default function AnalyticsPage() {
  const [stats, setStats] = useState<AnalyticsResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getAnalytics()
      .then(setStats)
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, []);

  if (error) {
    return (
      <div className="rounded-xl border border-[var(--color-danger-border)] bg-[var(--color-danger-bg)] px-4 py-3 text-sm text-[var(--color-danger)]">
        Could not reach the API: {error}
      </div>
    );
  }
  if (!stats) return <p className="text-sm text-sage-700">Loading…</p>;

  const maxBand = Math.max(1, ...Object.values(stats.phq9_severity_distribution));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-2xl font-semibold text-sage-900">Practice analytics</h1>
        <p className="text-sm text-sage-700">
          A population view of your caseload — not just one patient&apos;s chart at a time.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        <StatTile label="Patients" value={String(stats.total_patients)} />
        <StatTile label="Total visits" value={String(stats.total_visits)} />
        <StatTile label="Scored visits" value={String(stats.scored_visits)} />
        <StatTile
          label="Referral-flag rate"
          value={stats.referral_flag_rate === null ? "–" : `${Math.round(stats.referral_flag_rate * 100)}%`}
        />
        <StatTile
          label="Caseness rate"
          value={stats.caseness_rate === null ? "–" : `${Math.round(stats.caseness_rate * 100)}%`}
        />
      </div>

      <Card>
        <h2 className="font-display text-base font-semibold text-sage-800">
          PHQ-9 severity distribution
        </h2>
        <div className="mt-3 space-y-2">
          {Object.entries(stats.phq9_severity_distribution).map(([band, count]) => (
            <div key={band} className="flex items-center gap-2 text-sm">
              <span className="w-44 shrink-0 text-sage-700">{band}</span>
              <div className="h-3 flex-1 overflow-hidden rounded-full bg-sage-100">
                <div
                  className="h-full rounded-full bg-sage-500"
                  style={{ width: `${(count / maxBand) * 100}%` }}
                />
              </div>
              <span className="w-6 text-right text-sage-600">{count}</span>
            </div>
          ))}
        </div>
      </Card>

      <div className="rounded-2xl bg-sage-50 p-4">
        <p className="text-xs font-semibold uppercase tracking-wide text-sage-500">
          AI caseload summary
        </p>
        <p className="mt-1 text-sm leading-relaxed text-ink-900">
          {stats.narrative ?? "Not available — requires an LLM connection (ANTHROPIC_API_KEY)."}
        </p>
      </div>
    </div>
  );
}
