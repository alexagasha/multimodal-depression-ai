const STEPS = [
  { key: "created", label: "Intake" },
  { key: "scale_responses_recorded", label: "Scales" },
  { key: "audio_uploaded", label: "Audio" },
  { key: "scored", label: "Scored" },
];

export default function SessionStatusStepper({ status }: { status: string }) {
  const currentIndex = Math.max(0, STEPS.findIndex((s) => s.key === status));

  return (
    <ol className="flex items-center gap-2">
      {STEPS.map((step, i) => {
        const done = i <= currentIndex;
        return (
          <li key={step.key} className="flex flex-1 items-center gap-2">
            <div className="flex flex-col items-center gap-1">
              <div
                className={`flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold ${
                  done ? "bg-sage-500 text-white" : "bg-sage-100 text-sage-500"
                }`}
              >
                {done ? "✓" : i + 1}
              </div>
              <span className="text-[11px] text-sage-600">{step.label}</span>
            </div>
            {i < STEPS.length - 1 && (
              <div className={`h-0.5 flex-1 ${i < currentIndex ? "bg-sage-500" : "bg-sage-100"}`} />
            )}
          </li>
        );
      })}
    </ol>
  );
}
