import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ParticipantResult } from "../types";
import { severityColor } from "../utils";

interface PredictionChartProps {
  results: ParticipantResult[];
  threshold: number;
}

interface ScatterPoint {
  pid: number;
  x: number; // true PHQ-8
  y: number; // predicted PHQ-8
}

// Explicit ticks on both axes — never let the charting library auto-generate
// "nice" tick values. With a fixed [0, 24] domain (the real PHQ-8 range),
// auto-tick algorithms can disagree with the domain as the number of points
// grows, which is what produced the garbled axis at n=200.
const AXIS_TICKS = [0, 4, 8, 12, 16, 20, 24];

export function PredictionChart({ results, threshold }: PredictionChartProps) {
  const data: ScatterPoint[] = results.map((r) => ({
    pid: r.participant_id,
    x: r.phq8_true,
    y: r.phq8_pred,
  }));

  return (
    <div className="chart-panel">
      <div className="section-header">
        <h2>True vs. predicted PHQ-8</h2>
        <span className="count">{results.length} participants</span>
      </div>
      {/* key forces a clean remount when the dataset size changes (e.g. 8 -> 200
          participants), instead of the chart library patching an existing
          render whose internal scale was computed for a different data shape. */}
      <ResponsiveContainer width="100%" height={300} key={`scatter-${data.length}`}>
        <ScatterChart margin={{ top: 8, right: 20, left: 0, bottom: 20 }}>
          <CartesianGrid stroke="#1c2529" />
          <XAxis
            type="number"
            dataKey="x"
            name="True PHQ-8"
            domain={[0, 24]}
            ticks={AXIS_TICKS}
            tick={{ fill: "#8fa3a9", fontSize: 11, fontFamily: "IBM Plex Mono, monospace" }}
            axisLine={{ stroke: "#263338" }}
            tickLine={false}
            label={{ value: "True PHQ-8", position: "insideBottom", offset: -12, fill: "#5c6d72", fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="Predicted PHQ-8"
            domain={[0, 24]}
            ticks={AXIS_TICKS}
            tick={{ fill: "#8fa3a9", fontSize: 11, fontFamily: "IBM Plex Mono, monospace" }}
            axisLine={{ stroke: "#263338" }}
            tickLine={false}
            width={34}
            label={{ value: "Predicted", angle: -90, position: "insideLeft", fill: "#5c6d72", fontSize: 11 }}
          />
          <ReferenceLine
            segment={[{ x: 0, y: 0 }, { x: 24, y: 24 }]}
            stroke="#3a4750"
            strokeDasharray="3 4"
            ifOverflow="extendDomain"
          />
          <ReferenceLine x={threshold} stroke="#e3a548" strokeDasharray="3 4" strokeWidth={1} />
          <ReferenceLine y={threshold} stroke="#e3a548" strokeDasharray="3 4" strokeWidth={1} />
          <Tooltip
            cursor={{ strokeDasharray: "3 3", stroke: "#3a4750" }}
            content={<ScatterTooltip />}
          />
          <Scatter
            data={data}
            fill="#52d6c2"
            shape={(props: unknown) => {
              const p = props as { cx?: number; cy?: number; payload?: ScatterPoint };
              if (p.cx == null || p.cy == null || !p.payload) return <g />;
              const color = severityColor(p.payload.y);
              return <circle cx={p.cx} cy={p.cy} r={4} fill={color} fillOpacity={0.8} stroke={color} strokeWidth={1} />;
            }}
          />
        </ScatterChart>
      </ResponsiveContainer>
      <p className="faint" style={{ fontSize: 11, marginTop: 2 }}>
        Diagonal = perfect prediction. Dashed cross = depression threshold (≥{threshold}) on each axis. Dot
        color follows the same severity scale as the table.
      </p>
    </div>
  );
}

function ScatterTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload: ScatterPoint }[];
}) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0].payload;
  return (
    <div
      style={{
        background: "#182226",
        border: "1px solid #263338",
        borderRadius: 6,
        padding: "6px 10px",
        fontSize: 12,
        fontFamily: "IBM Plex Mono, monospace",
        color: "#e7eef0",
      }}
    >
      <div>P{p.pid}</div>
      <div style={{ color: "#8fa3a9" }}>
        true {p.x} · pred {p.y.toFixed(2)}
      </div>
    </div>
  );
}
