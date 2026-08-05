"use client";

import { motion } from "motion/react";
import { TreatmentEvent, VisitSummary } from "@/lib/api";

const WIDTH = 420;
const HEIGHT = 150;
const PAD = 24;

const EVENT_LABEL: Record<TreatmentEvent["event_type"], string> = {
  medication_change: "Rx",
  therapy_session: "Tx",
  other: "•",
};

export default function TrendSparkline({
  visits,
  events = [],
}: {
  visits: VisitSummary[];
  /** Treatment-response overlay (feature 5): medication/therapy changes
   * plotted against the same visits, snapped to the nearest visit by date
   * — the chart is visit-index-based, not a true continuous time axis. */
  events?: TreatmentEvent[];
}) {
  const scored = visits.filter((s) => s.phq9_pred !== null && s.hamd_pred !== null);

  if (scored.length < 2) {
    return (
      <p className="text-sm text-sage-600">
        Need at least two scored visits for this patient to show a trend.
      </p>
    );
  }

  const xStep = (WIDTH - PAD * 2) / (scored.length - 1);
  const yFor = (value: number, max: number) => HEIGHT - PAD - (value / max) * (HEIGHT - PAD * 2);
  const xForIndex = (i: number) => PAD + i * xStep;

  const phq9Points = scored
    .map((s, i) => `${xForIndex(i)},${yFor(s.phq9_pred!, 27)}`)
    .join(" ");
  const hamdPoints = scored
    .map((s, i) => `${xForIndex(i)},${yFor(s.hamd_pred!, 44)}`)
    .join(" ");

  // Snap each event to its nearest scored visit by date.
  const eventMarkers = events.map((event) => {
    const eventTime = new Date(event.event_date).getTime();
    let nearestIndex = 0;
    let smallestDelta = Infinity;
    scored.forEach((v, i) => {
      const delta = Math.abs(new Date(v.created_at).getTime() - eventTime);
      if (delta < smallestDelta) {
        smallestDelta = delta;
        nearestIndex = i;
      }
    });
    return { event, x: xForIndex(nearestIndex) };
  });

  return (
    <div>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full">
        <line
          x1={PAD}
          y1={HEIGHT - PAD}
          x2={WIDTH - PAD}
          y2={HEIGHT - PAD}
          stroke="var(--color-sage-200)"
        />
        {eventMarkers.map(({ event, x }, i) => (
          <motion.g
            key={event.event_id}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.9 + i * 0.1, duration: 0.4 }}
          >
            <line
              x1={x} y1={PAD - 6} x2={x} y2={HEIGHT - PAD}
              stroke="var(--color-clay-400)" strokeWidth="1" strokeDasharray="3,2"
            />
            <text x={x} y={PAD - 8} textAnchor="middle" fontSize="9" fill="var(--color-clay-600)">
              {EVENT_LABEL[event.event_type]}
              <title>{`${event.event_type.replace("_", " ")}: ${event.description}`}</title>
            </text>
          </motion.g>
        ))}
        <motion.polyline
          points={phq9Points}
          fill="none"
          stroke="var(--color-sage-500)"
          strokeWidth="2"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        />
        <motion.polyline
          points={hamdPoints}
          fill="none"
          stroke="var(--color-clay-500)"
          strokeWidth="2"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.8, delay: 0.15, ease: "easeOut" }}
        />
        {scored.map((s, i) => (
          <g key={s.session_id}>
            <motion.circle
              cx={xForIndex(i)}
              cy={yFor(s.phq9_pred!, 27)}
              r={3.5}
              fill="var(--color-sage-500)"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.8 + i * 0.05, type: "spring", stiffness: 400, damping: 15 }}
            />
            <motion.circle
              cx={xForIndex(i)}
              cy={yFor(s.hamd_pred!, 44)}
              r={3.5}
              fill="var(--color-clay-500)"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.95 + i * 0.05, type: "spring", stiffness: 400, damping: 15 }}
            />
          </g>
        ))}
      </svg>
      <div className="mt-2 flex flex-wrap gap-4 text-xs text-sage-700">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-sage-500" /> PHQ-9 (/27)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-clay-500" /> HAM-D (/44)
        </span>
        {events.length > 0 && (
          <span className="flex items-center gap-1.5 text-clay-600">
            <span className="h-2 w-0.5 bg-clay-400" /> Rx/Tx = treatment change (hover for detail)
          </span>
        )}
      </div>
    </div>
  );
}
