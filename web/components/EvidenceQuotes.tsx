import { EvidenceResult } from "@/lib/api";

const MODALITY_LABEL: Record<string, string> = {
  text: "Text (transcript)",
  audio: "Audio (prosody)",
  metadata: "Metadata",
};

export default function EvidenceQuotes({ data }: { data: EvidenceResult | null }) {
  return (
    <div className="rounded-xl bg-white/60 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-sage-500">
        Supporting evidence
      </p>
      {!data ? (
        <p className="mt-1 text-xs text-sage-600">
          Not available — requires an LLM connection (ANTHROPIC_API_KEY).
        </p>
      ) : data.quotes.length === 0 ? (
        <p className="mt-1 text-xs text-sage-600">
          No specific transcript phrases stood out for the {MODALITY_LABEL[data.modality] ?? data.modality} modality.
        </p>
      ) : (
        <>
          <p className="mt-0.5 text-xs text-sage-600">
            From the {MODALITY_LABEL[data.modality] ?? data.modality} modality:
          </p>
          <ul className="mt-1 space-y-1">
            {data.quotes.map((q, i) => (
              <li key={i} className="text-sm italic text-ink-700">&ldquo;{q}&rdquo;</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
