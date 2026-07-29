/**
 * Referral/risk banner. Deliberately NOT themed with the sage/clay palette —
 * this is a safety-critical signal (PHQ-9 item 9 / HAM-D suicide domain) and
 * must stay visually distinct from the rest of the organic aesthetic, and
 * independent of the model prediction / GenUI narrative. See
 * src/safety/risk_flag.py and api/genui.py's docstring for the boundary
 * this UI reflects.
 */
export default function RiskBanner({ compact = false }: { compact?: boolean }) {
  return (
    <div
      role="alert"
      className={`rounded-xl border-2 border-[var(--color-danger-border)] bg-[var(--color-danger-bg)] font-semibold text-[var(--color-danger)] ${
        compact ? "px-3 py-2 text-sm" : "px-4 py-3"
      }`}
    >
      ⚠ Referral flag active — item-level PHQ-9 / HAM-D response indicates
      immediate risk-assessment and referral, per protocol. This is
      independent of any model prediction or AI-generated summary.
    </div>
  );
}
