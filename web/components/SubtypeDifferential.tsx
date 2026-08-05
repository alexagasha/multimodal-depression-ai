"use client";

import { motion } from "motion/react";
import { SubtypeDifferentialResult, SubtypeLikelihood } from "@/lib/api";

const SUBTYPE_LABEL: Record<keyof SubtypeDifferentialResult, string> = {
  melancholic: "Melancholic features",
  atypical: "Atypical features",
  anxious_distress: "Anxious distress",
  psychotic_features: "Psychotic features",
};

const LIKELIHOOD_STYLE: Record<SubtypeLikelihood, string> = {
  none: "bg-sage-50 text-sage-500",
  possible: "bg-clay-100 text-clay-600",
  present: "bg-clay-500 text-white",
};

const LIKELIHOOD_LABEL: Record<SubtypeLikelihood, string> = {
  none: "No indication",
  possible: "Possible",
  present: "Present",
};

export default function SubtypeDifferential({
  data,
}: {
  data: SubtypeDifferentialResult | null;
}) {
  if (!data) {
    return (
      <div className="rounded-2xl bg-sage-50 p-4">
        <p className="text-xs font-semibold uppercase tracking-wide text-sage-500">
          Symptom-subtype differential
        </p>
        <p className="mt-1 text-sm text-sage-600">
          Not available — this requires an LLM connection (ANTHROPIC_API_KEY isn&apos;t
          configured). No differential is shown rather than a guess.
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-2xl bg-sage-50 p-4">
      <p className="text-xs font-semibold uppercase tracking-wide text-sage-500">
        Symptom-subtype differential — AI-suggested, not a diagnosis
      </p>
      <div className="mt-2 space-y-2">
        {(Object.keys(SUBTYPE_LABEL) as (keyof SubtypeDifferentialResult)[]).map((key, i) => {
          const entry = data[key];
          const isPsychoticAlert = key === "psychotic_features" && entry.likelihood !== "none";
          return (
            <motion.div
              key={key}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * i, duration: 0.35 }}
              className={`rounded-xl p-2.5 ${
                isPsychoticAlert
                  ? "border border-amber-300 bg-amber-50"
                  : "bg-white/60"
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-medium text-ink-900">
                  {SUBTYPE_LABEL[key]}
                  {isPsychoticAlert && (
                    <span className="ml-1.5 text-xs font-normal text-amber-700">
                      — consider urgent review
                    </span>
                  )}
                </span>
                <motion.span
                  initial={{ scale: 0.5, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: 0.1 * i + 0.2, type: "spring", stiffness: 300, damping: 15 }}
                  className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ${LIKELIHOOD_STYLE[entry.likelihood]}`}
                >
                  {LIKELIHOOD_LABEL[entry.likelihood]}
                </motion.span>
              </div>
              {entry.rationale && (
                <p className="mt-1 text-xs text-sage-700">{entry.rationale}</p>
              )}
            </motion.div>
          );
        })}
      </div>
      <p className="mt-3 text-xs text-sage-500">
        Based only on interview language, read against paraphrased DSM-5 specifier criteria.
        Confirm clinically before acting on any subtype indicated here.
      </p>
    </div>
  );
}
