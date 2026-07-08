import { MODALITY_COLORS, MODALITY_LABELS } from "../utils";

const ACTIVE_MODALITIES = ["text", "audio", "metadata"];
const ACTIVE_WIDTH_PCT = 92; // remainder reserved for the "video — pending" sliver

interface FusionStripProps {
  attributions: Record<string, number>;
  size?: "sm" | "lg";
  showLegend?: boolean;
}

/**
 * Renders the fusion vector's modality blocks as a horizontal strip —
 * segment width = that modality's share of the prediction (from the XAI
 * ablation attribution). Same component at table-row scale and detail-view
 * scale, so the visual grammar stays consistent everywhere it appears.
 */
export function FusionStrip({ attributions, size = "sm", showLegend = false }: FusionStripProps) {
  const values = ACTIVE_MODALITIES.map((m) => Math.abs(attributions[m] ?? 0));
  const total = values.reduce((a, b) => a + b, 0);
  const isFlat = total < 1e-9;

  const widths = isFlat
    ? ACTIVE_MODALITIES.map(() => ACTIVE_WIDTH_PCT / ACTIVE_MODALITIES.length)
    : values.map((v) => (v / total) * ACTIVE_WIDTH_PCT);

  return (
    <div>
      <div
        className={`fusion-strip ${size}`}
        title={isFlat ? "No attribution signal yet — mock encoders near baseline" : undefined}
      >
        {ACTIVE_MODALITIES.map((m, i) => (
          <div
            key={m}
            className="fusion-strip-seg"
            style={{
              flexBasis: `${widths[i]}%`,
              background: MODALITY_COLORS[m],
              opacity: isFlat ? 0.3 : 1,
            }}
          />
        ))}
        <div
          className="fusion-strip-seg"
          style={{
            flexBasis: `${100 - ACTIVE_WIDTH_PCT}%`,
            backgroundImage:
              "repeating-linear-gradient(45deg, var(--panel-3), var(--panel-3) 3px, transparent 3px, transparent 6px)",
          }}
          title="Video encoder — Phase 2, not yet wired into fusion"
        />
      </div>

      {showLegend && (
        <div className="fusion-legend">
          {ACTIVE_MODALITIES.map((m) => {
            const v = attributions[m] ?? 0;
            return (
              <div className="fusion-legend-item" key={m}>
                <span className="fusion-legend-swatch" style={{ background: MODALITY_COLORS[m] }} />
                <span>{MODALITY_LABELS[m]}</span>
                <span className="pct">{isFlat ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(0)}%`}</span>
              </div>
            );
          })}
          <div className="fusion-legend-item">
            <span className="fusion-legend-swatch" style={{ background: MODALITY_COLORS.video }} />
            <span className="faint">{MODALITY_LABELS.video}</span>
          </div>
        </div>
      )}
    </div>
  );
}
