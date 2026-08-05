"use client";

import { useTypewriter } from "@/lib/useTypewriter";

export default function PatientSummaryCard({ summary }: { summary: string | null }) {
  const typed = useTypewriter(summary ?? "", 14);

  function speak() {
    if (typeof window === "undefined" || !("speechSynthesis" in window) || !summary) return;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(summary));
  }

  if (!summary) {
    return (
      <div className="rounded-2xl bg-sage-50 p-4">
        <p className="text-xs font-semibold uppercase tracking-wide text-sage-500">
          Patient handout
        </p>
        <p className="mt-1 text-sm text-sage-600">
          Not available — requires an LLM connection (ANTHROPIC_API_KEY).
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-2xl bg-clay-100/50 p-4">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-clay-600">
          Patient handout — plain language, for the patient to take home
        </p>
        <button
          type="button"
          onClick={speak}
          data-no-print
          className="shrink-0 rounded-full border border-clay-300 px-2 py-1 text-xs text-clay-600 hover:bg-clay-100"
        >
          🔊 Read aloud
        </button>
      </div>
      <p className="mt-1 min-h-[3em] text-sm leading-relaxed text-ink-900">
        {typed}
        {typed.length < summary.length && (
          <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-clay-400 align-middle" />
        )}
      </p>
    </div>
  );
}
