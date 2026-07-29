"use client";

import { useEffect, useState } from "react";

/**
 * Persists form state to localStorage as the user types, so an accidental
 * tab close mid-interview doesn't lose an in-progress intake/scale form. Not
 * full offline-first sync (see docs/system-roadmap.md) — just draft
 * resilience for the common "browser closed by accident" case.
 */
export function useDraftAutosave<T extends object>(key: string, initial: T) {
  const storageKey = `mizizi:draft:${key}`;

  const [value, setValue] = useState<T>(() => {
    if (typeof window === "undefined") return initial;
    try {
      const raw = window.localStorage.getItem(storageKey);
      return raw ? { ...initial, ...JSON.parse(raw) } : initial;
    } catch {
      return initial;
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(value));
    } catch {
      // localStorage unavailable/full — draft resilience is best-effort only
    }
  }, [storageKey, value]);

  const clearDraft = () => {
    try {
      window.localStorage.removeItem(storageKey);
    } catch {
      // ignore
    }
  };

  return [value, setValue, clearDraft] as const;
}
