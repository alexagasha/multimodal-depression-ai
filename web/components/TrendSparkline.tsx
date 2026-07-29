import { SessionSummary } from "@/lib/api";

const WIDTH = 420;
const HEIGHT = 150;
const PAD = 24;

export default function TrendSparkline({ sessions }: { sessions: SessionSummary[] }) {
  const scored = sessions.filter((s) => s.phq9_pred !== null && s.hamd_pred !== null);

  if (scored.length < 2) {
    return (
      <p className="text-sm text-sage-600">
        Need at least two scored sessions for this participant to show a trend.
      </p>
    );
  }

  const xStep = (WIDTH - PAD * 2) / (scored.length - 1);
  const yFor = (value: number, max: number) => HEIGHT - PAD - (value / max) * (HEIGHT - PAD * 2);

  const phq9Points = scored
    .map((s, i) => `${PAD + i * xStep},${yFor(s.phq9_pred!, 27)}`)
    .join(" ");
  const hamdPoints = scored
    .map((s, i) => `${PAD + i * xStep},${yFor(s.hamd_pred!, 44)}`)
    .join(" ");

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
        <polyline points={phq9Points} fill="none" stroke="var(--color-sage-500)" strokeWidth="2" />
        <polyline points={hamdPoints} fill="none" stroke="var(--color-clay-500)" strokeWidth="2" />
        {scored.map((s, i) => (
          <g key={s.session_id}>
            <circle cx={PAD + i * xStep} cy={yFor(s.phq9_pred!, 27)} r={3.5} fill="var(--color-sage-500)" />
            <circle cx={PAD + i * xStep} cy={yFor(s.hamd_pred!, 44)} r={3.5} fill="var(--color-clay-500)" />
          </g>
        ))}
      </svg>
      <div className="mt-2 flex gap-4 text-xs text-sage-700">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-sage-500" /> PHQ-9 (/27)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-clay-500" /> HAM-D (/44)
        </span>
      </div>
    </div>
  );
}
