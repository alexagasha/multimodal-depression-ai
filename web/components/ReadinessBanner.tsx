"use client";

import { useEffect, useState } from "react";
import { api, Health } from "@/lib/api";

/**
 * Installation-wide scoring status, shown on every page.
 *
 * The API fails closed: it refuses to score unless it is running the real,
 * released model (api/readiness.py). Without this banner a clinician would only
 * find out when a visit's scoring fails. And on a development override they
 * would never find out at all, because unvalidated scores look exactly like
 * validated ones.
 *
 * Deliberately not themed with the sage/clay palette, like RiskBanner: it is a
 * safety signal, and it must not blend into the rest of the page.
 */
export default function ReadinessBanner() {
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = () =>
      api
        .getHealth()
        .then((h) => !cancelled && setHealth(h))
        .catch(() => !cancelled && setHealth(null));
    load();
    // Picks up a fix (key set, weights restored) without a page reload.
    const id = setInterval(load, 60_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  if (!health || health.scoring_ready) return null;

  const override = health.allow_unvalidated;
  return (
    <div
      role="alert"
      className="border-b-2 border-[var(--color-danger-border)] bg-[var(--color-danger-bg)] px-4 py-2 text-sm text-[var(--color-danger)] sm:px-6"
    >
      <div className="mx-auto max-w-5xl">
        <p className="font-semibold">
          {override
            ? "Development build: scores come from an unvalidated model and must not be used clinically."
            : "Scoring is disabled on this installation: it is not running the validated model release."}
        </p>
        <p className="mt-0.5 text-xs">
          Clinical notes, risk assessment and referral flags still work.
        </p>
        <ul className="mt-1 list-disc pl-5 text-xs">
          {health.blockers.map((b) => (
            <li key={b}>{b}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
