"use client";

import { useEffect } from "react";
import { motion, useMotionValue, useTransform, animate, useReducedMotion } from "motion/react";

/**
 * Shows the AI-predicted score as the primary ring, with the
 * clinician-administered total as a second inner ring + explicit label —
 * the AI runs alongside manual scoring, not in place of it, so both need to
 * be visible at a glance for comparison. Both rings sweep in and the
 * numbers count up on mount, rather than appearing pre-filled — the score
 * modal's opening "reveal."
 *
 * Uses Framer Motion's useMotionValue/useTransform bound directly into the
 * SVG/text via JSX (motion subscribes and writes the DOM itself) rather
 * than animate()'s onUpdate -> React setState — the latter proved
 * unreliable under React 19 Strict Mode's double-effect-invoke in dev,
 * silently stalling at the initial value.
 */
export default function Gauge({
  label,
  value,
  clinicianValue,
  max,
}: {
  label: string;
  value: number;
  clinicianValue?: number;
  max: number;
}) {
  const prefersReducedMotion = useReducedMotion();
  const rOuter = 34;
  const rInner = 25;
  const cOuter = 2 * Math.PI * rOuter;
  const cInner = 2 * Math.PI * rInner;

  const aiValue = useMotionValue(prefersReducedMotion ? value : 0);
  const clinValue = useMotionValue(prefersReducedMotion ? clinicianValue ?? 0 : 0);

  const outerOffset = useTransform(aiValue, (v) => cOuter * (1 - Math.max(0, Math.min(1, v / max))));
  const innerOffset = useTransform(clinValue, (v) => cInner * (1 - Math.max(0, Math.min(1, v / max))));
  const aiLabel = useTransform(aiValue, (v) => v.toFixed(1));
  const aiLegendLabel = useTransform(aiValue, (v) => `AI ${v.toFixed(1)}`);

  useEffect(() => {
    if (prefersReducedMotion) {
      aiValue.jump(value);
      return;
    }
    const controls = animate(aiValue, value, { duration: 0.9, ease: "easeOut" });
    return controls.stop;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, prefersReducedMotion]);

  useEffect(() => {
    if (clinicianValue === undefined) return;
    if (prefersReducedMotion) {
      clinValue.jump(clinicianValue);
      return;
    }
    const controls = animate(clinValue, clinicianValue, { duration: 0.9, ease: "easeOut" });
    return controls.stop;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clinicianValue, prefersReducedMotion]);

  return (
    <div className="flex flex-col items-center">
      <div className="relative h-24 w-24">
        <svg viewBox="0 0 80 80" className="h-24 w-24 -rotate-90">
          <circle cx="40" cy="40" r={rOuter} stroke="var(--color-sage-100)" strokeWidth="7" fill="none" />
          <motion.circle
            cx="40"
            cy="40"
            r={rOuter}
            stroke="var(--color-sage-500)"
            strokeWidth="7"
            fill="none"
            strokeLinecap="round"
            strokeDasharray={cOuter}
            style={{ strokeDashoffset: outerOffset }}
          />
          {clinicianValue !== undefined && (
            <>
              <circle cx="40" cy="40" r={rInner} stroke="var(--color-clay-100)" strokeWidth="5" fill="none" />
              <motion.circle
                cx="40"
                cy="40"
                r={rInner}
                stroke="var(--color-clay-500)"
                strokeWidth="5"
                fill="none"
                strokeLinecap="round"
                strokeDasharray={cInner}
                style={{ strokeDashoffset: innerOffset }}
              />
            </>
          )}
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.div className="text-xl font-bold text-sage-800">{aiLabel}</motion.div>
          <div className="text-xs text-sage-600">/ {max}</div>
        </div>
      </div>
      <div className="mt-2 text-sm font-medium text-sage-700">{label}</div>
      <div className="mt-1 flex items-center gap-3 text-xs">
        <span className="flex items-center gap-1 text-sage-600">
          <span className="h-2 w-2 rounded-full bg-sage-500" />
          <motion.span>{aiLegendLabel}</motion.span>
        </span>
        {clinicianValue !== undefined && (
          <span className="flex items-center gap-1 text-clay-600">
            <span className="h-2 w-2 rounded-full bg-clay-500" /> Clinician {clinicianValue}
          </span>
        )}
      </div>
    </div>
  );
}
