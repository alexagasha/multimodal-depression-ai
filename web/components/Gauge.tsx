/**
 * Shows the AI-predicted score as the primary ring, with the
 * clinician-administered total as a second inner ring + explicit label —
 * the AI runs alongside manual scoring, not in place of it, so both need to
 * be visible at a glance for comparison.
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
  const pct = Math.max(0, Math.min(1, value / max));
  const clinicianPct =
    clinicianValue !== undefined ? Math.max(0, Math.min(1, clinicianValue / max)) : null;
  const rOuter = 34;
  const rInner = 25;
  const cOuter = 2 * Math.PI * rOuter;
  const cInner = 2 * Math.PI * rInner;

  return (
    <div className="flex flex-col items-center">
      <div className="relative h-24 w-24">
        <svg viewBox="0 0 80 80" className="h-24 w-24 -rotate-90">
          <circle cx="40" cy="40" r={rOuter} stroke="var(--color-sage-100)" strokeWidth="7" fill="none" />
          <circle
            cx="40"
            cy="40"
            r={rOuter}
            stroke="var(--color-sage-500)"
            strokeWidth="7"
            fill="none"
            strokeLinecap="round"
            strokeDasharray={cOuter}
            strokeDashoffset={cOuter * (1 - pct)}
          />
          {clinicianPct !== null && (
            <>
              <circle cx="40" cy="40" r={rInner} stroke="var(--color-clay-100)" strokeWidth="5" fill="none" />
              <circle
                cx="40"
                cy="40"
                r={rInner}
                stroke="var(--color-clay-500)"
                strokeWidth="5"
                fill="none"
                strokeLinecap="round"
                strokeDasharray={cInner}
                strokeDashoffset={cInner * (1 - clinicianPct)}
              />
            </>
          )}
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div className="text-xl font-bold text-sage-800">{value.toFixed(1)}</div>
          <div className="text-xs text-sage-600">/ {max}</div>
        </div>
      </div>
      <div className="mt-2 text-sm font-medium text-sage-700">{label}</div>
      <div className="mt-1 flex items-center gap-3 text-xs">
        <span className="flex items-center gap-1 text-sage-600">
          <span className="h-2 w-2 rounded-full bg-sage-500" /> AI {value.toFixed(1)}
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
