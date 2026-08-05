"use client";

import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import { TranscriptRow } from "@/lib/api";
import { Card } from "@/components/FormField";

/**
 * The transcript as it arrives, mid-interview — lines slide in the moment the
 * server recognises them, rather than appearing all at once after the
 * recording is uploaded. Auto-scrolls to the newest line so the clinician can
 * glance at it without touching anything.
 */
function formatClock(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function LiveTranscript({
  rows,
  recording,
}: {
  rows: TranscriptRow[];
  recording: boolean;
}) {
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [rows.length]);

  if (rows.length === 0 && !recording) return null;

  return (
    <Card className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <h2 className="font-display text-lg font-semibold text-sage-800">Transcript</h2>
        {recording && (
          <span className="flex items-center gap-1.5 text-xs font-medium text-clay-600">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-clay-500 opacity-75" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-clay-500" />
            </span>
            Live
          </span>
        )}
      </div>

      {rows.length === 0 ? (
        <p className="text-sm text-sage-600">Listening — speech will appear here as it&apos;s recognised.</p>
      ) : (
        <div className="max-h-64 space-y-1.5 overflow-y-auto pr-1">
          <AnimatePresence initial={false}>
            {rows.map((row, i) => (
              <motion.p
                key={`${row.start_time}-${i}`}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className="text-sm leading-relaxed text-ink-900"
              >
                <span className="mr-2 select-none font-mono text-xs text-sage-500">
                  {formatClock(row.start_time)}
                </span>
                {row.value}
              </motion.p>
            ))}
          </AnimatePresence>
          <div ref={endRef} />
        </div>
      )}
    </Card>
  );
}
