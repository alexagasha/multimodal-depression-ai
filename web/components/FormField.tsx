import { ReactNode } from "react";

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-sage-800">{label}</span>
      {hint && <span className="mt-0.5 block text-xs text-sage-600">{hint}</span>}
      <div className="mt-1">{children}</div>
    </label>
  );
}

const controlClass =
  "w-full rounded-xl border border-sage-200 bg-white px-3 py-2 text-sm text-ink-900 shadow-sm focus:border-sage-500 focus:outline-none focus:ring-2 focus:ring-sage-200";

export function Select(props: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={`${controlClass} ${props.className ?? ""}`} />;
}

export function TextInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={`${controlClass} ${props.className ?? ""}`} />;
}

export function TextArea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea {...props} className={`${controlClass} ${props.className ?? ""}`} />;
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-3xl border border-sage-200 bg-white/80 p-5 shadow-sm ${className}`}>
      {children}
    </div>
  );
}
