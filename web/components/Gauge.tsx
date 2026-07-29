export default function Gauge({
  label,
  value,
  max,
}: {
  label: string;
  value: number;
  max: number;
}) {
  const pct = Math.max(0, Math.min(1, value / max));
  const r = 34;
  const circumference = 2 * Math.PI * r;
  return (
    <div className="flex flex-col items-center">
      <div className="relative h-24 w-24">
        <svg viewBox="0 0 80 80" className="h-24 w-24 -rotate-90">
          <circle cx="40" cy="40" r={r} stroke="var(--color-sage-100)" strokeWidth="8" fill="none" />
          <circle
            cx="40"
            cy="40"
            r={r}
            stroke="var(--color-sage-500)"
            strokeWidth="8"
            fill="none"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={circumference * (1 - pct)}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div className="text-xl font-bold text-sage-800">{value.toFixed(1)}</div>
          <div className="text-xs text-sage-600">/ {max}</div>
        </div>
      </div>
      <div className="mt-2 text-sm font-medium text-sage-700">{label}</div>
    </div>
  );
}
