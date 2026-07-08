export type RunStatus = "idle" | "running" | "done" | "error";

interface StageRailProps {
  status: RunStatus;
}

const STAGES = [
  { key: "sync", label: "Sync", pending: false },
  { key: "text", label: "Text · BERT", pending: false },
  { key: "audio", label: "Audio · Wav2Vec2", pending: false },
  { key: "video", label: "Video", pending: true },
  { key: "meta", label: "Metadata", pending: false },
  { key: "aggregate", label: "Aggregate", pending: false },
  { key: "fuse", label: "Fuse", pending: false },
  { key: "predict", label: "Predict", pending: false },
] as const;

/**
 * A persistent, honest architecture display — not a fake step-by-step
 * progress fake-out. The pipeline call is a single synchronous request, so
 * this shows the real shape of the system (with Video marked pending, since
 * it isn't wired into fusion yet) and pulses as a whole while a run is
 * in flight.
 */
export function StageRail({ status }: StageRailProps) {
  return (
    <div className={`stage-rail ${status === "running" ? "running" : ""}`}>
      {STAGES.map((stage, i) => {
        const cls = stage.pending ? "pending" : status === "idle" ? "" : status;
        return (
          <div className="stage-node-wrap" key={stage.key} style={{ display: "flex", alignItems: "center" }}>
            <div className={`stage-node ${cls}`}>
              <span className="stage-dot" />
              <span className="stage-label">{stage.label}</span>
            </div>
            {i < STAGES.length - 1 && <span className="stage-connector" />}
          </div>
        );
      })}
    </div>
  );
}
