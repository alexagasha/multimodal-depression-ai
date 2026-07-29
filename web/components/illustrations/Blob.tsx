/** Soft organic background blob — purely decorative, aria-hidden. */
export default function Blob({
  className = "",
  color = "var(--color-sage-100)",
}: {
  className?: string;
  color?: string;
}) {
  return (
    <svg
      viewBox="0 0 200 200"
      className={className}
      aria-hidden="true"
      focusable="false"
    >
      <path
        fill={color}
        d="M45.4,-58.3C58.6,-48.9,68.6,-33.8,72.1,-17.2C75.6,-0.6,72.6,17.5,63.6,31.8C54.6,46.1,39.6,56.6,22.6,63.1C5.6,69.6,-13.4,72.1,-29.9,66.5C-46.4,60.9,-60.4,47.2,-67.6,30.9C-74.8,14.6,-75.2,-4.3,-68.8,-20.3C-62.4,-36.3,-49.2,-49.4,-34.3,-58.5C-19.4,-67.6,-2.7,-72.7,12.9,-70.8C28.5,-68.9,32.2,-67.7,45.4,-58.3Z"
        transform="translate(100 100)"
      />
    </svg>
  );
}
