"use client";

import { useEffect, useState } from "react";
import { motion, animate, useMotionValue, useTransform, useReducedMotion } from "motion/react";
import { api, AnalyticsResult, ApiError } from "@/lib/api";
import { Card } from "@/components/FormField";
import { useTypewriter } from "@/lib/useTypewriter";

function StatTile({
  label,
  value,
  suffix = "",
  delay = 0,
}: {
  label: string;
  value: number | null;
  suffix?: string;
  delay?: number;
}) {
  const prefersReducedMotion = useReducedMotion();
  // Bound directly into the JSX via useTransform — animate()'s onUpdate ->
  // React setState form proved unreliable under React 19 Strict Mode's
  // double-effect-invoke in dev (silently stalled at the initial value);
  // see Gauge.tsx for the same fix.
  const count = useMotionValue(prefersReducedMotion ? value ?? 0 : 0);
  const display = useTransform(count, (v) => `${Math.round(v)}${suffix}`);

  useEffect(() => {
    if (value === null) return;
    if (prefersReducedMotion) {
      count.jump(value);
      return;
    }
    const controls = animate(count, value, { duration: 0.8, delay, ease: "easeOut" });
    return controls.stop;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, delay, prefersReducedMotion]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.3 }}
      className="rounded-2xl border border-sage-200 bg-white/70 px-4 py-3 text-center shadow-sm"
    >
      <div className="font-display text-2xl font-semibold text-sage-800">
        {value === null ? "–" : <motion.span>{display}</motion.span>}
      </div>
      <div className="mt-1 text-xs text-sage-600">{label}</div>
    </motion.div>
  );
}

function SeverityBar({ band, count, maxBand, delay }: { band: string; count: number; maxBand: number; delay: number }) {
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="w-44 shrink-0 text-sage-700">{band}</span>
      <div className="h-3 flex-1 overflow-hidden rounded-full bg-sage-100">
        <motion.div
          className="h-full rounded-full bg-sage-500"
          initial={{ width: 0 }}
          animate={{ width: `${(count / maxBand) * 100}%` }}
          transition={{ duration: 0.6, delay, ease: "easeOut" }}
        />
      </div>
      <span className="w-6 text-right text-sage-600">{count}</span>
    </div>
  );
}

function CaseloadNarrative({ narrative }: { narrative: string | null }) {
  const typed = useTypewriter(narrative ?? "", 10);
  return (
    <div className="rounded-2xl bg-sage-50 p-4">
      <p className="text-xs font-semibold uppercase tracking-wide text-sage-500">
        AI caseload summary
      </p>
      <p className="mt-1 min-h-[3em] text-sm leading-relaxed text-ink-900">
        {narrative
          ? typed
          : "Not available — requires an LLM connection (ANTHROPIC_API_KEY)."}
        {narrative && typed.length < narrative.length && (
          <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-sage-400 align-middle" />
        )}
      </p>
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
        <StatTile label="Patients" value={stats.total_patients} delay={0} />
        <StatTile label="Total visits" value={stats.total_visits} delay={0.05} />
        <StatTile label="Scored visits" value={stats.scored_visits} delay={0.1} />
        <StatTile
          label="Referral-flag rate"
          value={stats.referral_flag_rate === null ? null : Math.round(stats.referral_flag_rate * 100)}
          suffix="%"
          delay={0.15}
        />
        <StatTile
          label="Caseness rate"
          value={stats.caseness_rate === null ? null : Math.round(stats.caseness_rate * 100)}
          suffix="%"
          delay={0.2}
        />
      </div>

      <Card>
        <h2 className="font-display text-base font-semibold text-sage-800">
          PHQ-9 severity distribution
        </h2>
        <div className="mt-3 space-y-2">
          {Object.entries(stats.phq9_severity_distribution).map(([band, count], i) => (
            <SeverityBar key={band} band={band} count={count} maxBand={maxBand} delay={0.3 + i * 0.08} />
          ))}
        </div>
      </Card>

      <CaseloadNarrative narrative={stats.narrative} />
    </div>
  );
}
