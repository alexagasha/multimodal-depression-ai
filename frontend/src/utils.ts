// PHQ-8 severity banding follows the standard convention used in the
// clinical literature (Kroenke et al.) — same bands as PHQ-9 minus the
// self-harm item. Scores here are model *outputs* on synthetic data, not
// a diagnosis of anyone.
const BANDS = [
  { max: 4, label: "Minimal" },
  { max: 9, label: "Mild" },
  { max: 14, label: "Moderate" },
  { max: 19, label: "Mod. severe" },
  { max: 24, label: "Severe" },
];

export function severityLabel(score: number): string {
  const s = clamp(score, 0, 24);
  for (const band of BANDS) if (s <= band.max) return band.label;
  return "Severe";
}

// Sequential mint -> teal -> indigo scale over the PHQ-8 range [0, 24].
// Deliberately not red/green: severity is ordered, not binary good/bad,
// and a mental-health score shouldn't be alarm-coded.
const ANCHORS: [number, [number, number, number]][] = [
  [0, [159, 234, 212]],
  [10, [111, 201, 198]],
  [17, [74, 143, 192]],
  [24, [51, 59, 140]],
];

export function severityColor(score: number): string {
  const s = clamp(score, 0, 24);
  for (let i = 0; i < ANCHORS.length - 1; i++) {
    const [stopA, rgbA] = ANCHORS[i];
    const [stopB, rgbB] = ANCHORS[i + 1];
    if (s >= stopA && s <= stopB) {
      const t = stopB === stopA ? 0 : (s - stopA) / (stopB - stopA);
      const r = Math.round(lerp(rgbA[0], rgbB[0], t));
      const g = Math.round(lerp(rgbA[1], rgbB[1], t));
      const b = Math.round(lerp(rgbA[2], rgbB[2], t));
      return `rgb(${r}, ${g}, ${b})`;
    }
  }
  return `rgb(${ANCHORS[ANCHORS.length - 1][1].join(", ")})`;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

// Fixed palette per modality — used everywhere the fusion strip appears so
// the color mapping stays legible across the whole app.
export const MODALITY_COLORS: Record<string, string> = {
  text: "#52d6c2",
  audio: "#e3a548",
  video: "#3a4750",
  metadata: "#8b7fd6",
};

export const MODALITY_LABELS: Record<string, string> = {
  text: "Text (BERT)",
  audio: "Audio (Wav2Vec2)",
  video: "Video — Phase 2",
  metadata: "Metadata",
};

export function formatTimestamp(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatSeconds(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}
