/**
 * Visit progress. The risk-assessment step is conditional: it appears only for
 * a referral-flagged visit, where it is a real gate — the API refuses to close
 * such a visit until an assessment is on file (POST /sessions/{id}/close).
 * Showing it for every visit would turn a blocking requirement into decoration.
 */
const BASE_STEPS = [
  { key: "created", label: "Intake" },
  { key: "scale_responses_recorded", label: "Scales" },
  { key: "audio_uploaded", label: "Audio" },
  { key: "scored", label: "Scored" },
];

const CLOSED = { key: "closed", label: "Closed" };
const RISK_STEP = { key: "risk_assessed", label: "Risk" };

export default function SessionStatusStepper({
  status,
  riskFlag = false,
  riskAssessed = false,
}: {
  status: string;
  riskFlag?: boolean;
  riskAssessed?: boolean;
}) {
  const steps = riskFlag
    ? [...BASE_STEPS, RISK_STEP, CLOSED]
    : [...BASE_STEPS, CLOSED];

  const statusIndex = steps.findIndex((s) => s.key === status);
  const scoredIndex = steps.findIndex((s) => s.key === "scored");
  const currentIndex = Math.max(0, statusIndex);

  return (
    <ol className="flex items-center gap-2">
      {steps.map((step, i) => {
        // The risk step tracks its own record rather than the session status,
        // which never passes through it.
        const done =
          step.key === RISK_STEP.key
            ? riskAssessed
            : i <= currentIndex;
        // Reached but not satisfied: the visit is scored and this is the step
        // standing between it and being closed.
        const blocking =
          step.key === RISK_STEP.key && !riskAssessed && currentIndex >= scoredIndex;

        return (
          <li key={step.key} className="flex flex-1 items-center gap-2">
            <div className="flex flex-col items-center gap-1">
              <div
                className={`flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold ${
                  blocking
                    ? "bg-[var(--color-danger)] text-white"
                    : done
                      ? "bg-sage-500 text-white"
                      : "bg-sage-100 text-sage-500"
                }`}
              >
                {blocking ? "!" : done ? "✓" : i + 1}
              </div>
              <span
                className={`text-[11px] ${
                  blocking ? "font-semibold text-[var(--color-danger)]" : "text-sage-600"
                }`}
              >
                {step.label}
              </span>
            </div>
            {i < steps.length - 1 && (
              <div className={`h-0.5 flex-1 ${i < currentIndex ? "bg-sage-500" : "bg-sage-100"}`} />
            )}
          </li>
        );
      })}
    </ol>
  );
}
