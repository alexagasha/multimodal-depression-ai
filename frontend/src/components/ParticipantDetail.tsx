import type { ParticipantDetail as ParticipantDetailT } from "../types";
import { formatSeconds, severityColor, severityLabel } from "../utils";
import { FusionStrip } from "./FusionStrip";

interface ParticipantDetailProps {
  detail: ParticipantDetailT | null;
  loading: boolean;
  onClose: () => void;
}

export function ParticipantDetail({ detail, loading, onClose }: ParticipantDetailProps) {
  if (!detail && !loading) return null;

  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <div className="drawer" role="dialog" aria-label="Participant detail">
        <div className="drawer-header">
          <div>
            <h2 style={{ fontSize: 18 }}>
              Participant <span className="mono">P{detail?.participant_id ?? "…"}</span>
            </h2>
            <p className="faint" style={{ fontSize: 12, marginTop: 4 }}>
              {detail?.n_segments ?? "—"} segments · 30s each · synthetic session
            </p>
          </div>
          <button className="drawer-close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        {loading && <p className="dim">Loading…</p>}

        {detail && (
          <>
            {detail.prediction && (
              <div className="drawer-section">
                <h3>PHQ-8 severity</h3>
                <div className="pred-compare">
                  <div className="item">
                    <div className="k">True</div>
                    <div className="v">{detail.prediction.phq8_true}</div>
                  </div>
                  <div className="item">
                    <div className="k">Predicted</div>
                    <div className="v" style={{ color: severityColor(detail.prediction.phq8_pred) }}>
                      {detail.prediction.phq8_pred.toFixed(2)}
                    </div>
                  </div>
                  <div className="item">
                    <div className="k">Band (pred)</div>
                    <div
                      className="v"
                      style={{ fontSize: 14, color: severityColor(detail.prediction.phq8_pred) }}
                    >
                      {severityLabel(detail.prediction.phq8_pred)}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {detail.attribution && (
              <div className="drawer-section">
                <h3>Modality attribution</h3>
                <FusionStrip
                  attributions={detail.attribution.modality_attributions}
                  size="lg"
                  showLegend
                />
                <p className="faint" style={{ fontSize: 11, marginTop: 10, lineHeight: 1.5 }}>
                  Share of the predicted score attributable to each modality block, from
                  ablating that block in the fusion vector (see src/xai/attribution.py).
                  Negative values pushed the score down.
                </p>
              </div>
            )}

            <div className="drawer-section">
              <h3>Metadata</h3>
              <div className="meta-grid">
                <div className="meta-item">
                  <div className="k">Age</div>
                  <div className="v">{detail.metadata.age}</div>
                </div>
                <div className="meta-item">
                  <div className="k">Gender</div>
                  <div className="v">{detail.metadata.gender}</div>
                </div>
                <div className="meta-item">
                  <div className="k">Education (yrs)</div>
                  <div className="v">{detail.metadata.education_years}</div>
                </div>
                <div className="meta-item">
                  <div className="k">Split</div>
                  <div className="v">{detail.prediction?.split ?? "—"}</div>
                </div>
              </div>
            </div>

            <div className="drawer-section">
              <details className="segments-debug">
                <summary>Segment transcript (synthetic filler text, for pipeline debugging)</summary>
                {detail.segments_preview.map((s) => (
                  <div className="segment-row" key={s.segment_idx}>
                    <span className="t">
                      [{formatSeconds(s.start)}–{formatSeconds(s.end)}]
                    </span>{" "}
                    {s.text || <span className="faint">(no participant speech in window)</span>}
                  </div>
                ))}
              </details>
            </div>
          </>
        )}
      </div>
    </>
  );
}
