/** Friendly "nothing here yet" illustration — a leaf sprouting from soil. */
export default function EmptyState({ className = "h-32 w-32" }: { className?: string }) {
  return (
    <svg viewBox="0 0 160 160" className={className} aria-hidden="true">
      <ellipse cx="80" cy="128" rx="52" ry="10" fill="var(--color-sage-100)" />
      <path
        d="M80 128 C80 88 80 60 80 40"
        stroke="var(--color-sage-600)"
        strokeWidth="4"
        strokeLinecap="round"
        fill="none"
      />
      <path
        d="M80 60 C60 55 48 40 50 24 C68 26 80 40 80 60Z"
        fill="var(--color-sage-400)"
      />
      <path
        d="M80 76 C102 70 116 54 114 36 C94 40 80 56 80 76Z"
        fill="var(--color-sage-500)"
      />
      <circle cx="80" cy="128" r="6" fill="var(--color-clay-400)" />
    </svg>
  );
}
