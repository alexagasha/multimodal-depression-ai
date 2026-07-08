import type { Metrics } from "../types";

interface SummaryStripProps {
  metrics: Metrics;
}

export function SummaryStrip({ metrics }: SummaryStripProps) {
  const cm = metrics.confusion_matrix;
  const depressedCount = cm ? cm.matrix[0][1] + cm.matrix[1][1] : null;

  const cards: { label: string; value: string; sub?: string }[] = [
    { label: "Participants", value: String(metrics.n_participants) },
    { label: "F1 (weighted)", value: metrics.f1_weighted.toFixed(3) },
    { label: "RMSE", value: metrics.rmse.toFixed(2), sub: "PHQ-8 points" },
    { label: "MAE", value: metrics.mae.toFixed(2), sub: "PHQ-8 points" },
    {
      label: "Flagged depressed",
      value: depressedCount !== null ? String(depressedCount) : "—",
      sub: `threshold ≥ ${metrics.depression_threshold}`,
    },
  ];

  return (
    <div className="summary-strip">
      {cards.map((c) => (
        <div className="summary-card" key={c.label}>
          <div className="label">{c.label}</div>
          <div className="value">{c.value}</div>
          {c.sub && <div className="sub">{c.sub}</div>}
        </div>
      ))}
    </div>
  );
}
