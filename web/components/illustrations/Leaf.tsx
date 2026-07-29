/** Hand-authored leaf mark — used as the logo and (exported separately) the favicon. */
export default function Leaf({ className = "h-6 w-6" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" fill="none" className={className} aria-hidden="true">
      <path
        d="M6 26C4 16 10 6 26 6c1.2 10-4 18-14 19-3 .3-4.8-.2-6-1z"
        fill="var(--color-sage-500)"
      />
      <path
        d="M7 25C11 17 16 11 24 8"
        stroke="var(--color-sage-100)"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  );
}
