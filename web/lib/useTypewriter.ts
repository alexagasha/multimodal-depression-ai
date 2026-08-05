"use client";

import { useEffect, useState } from "react";
import { useReducedMotion } from "motion/react";

/**
 * Reveals `text` character-by-character — used for the GenAI narrative/
 * patient-summary text so the AI feels like it's composing the answer
 * live, not dumping a wall of text. Respects prefers-reduced-motion (shows
 * the full text immediately, per WCAG 2.3.3 — this app already treats
 * accessibility as a first-class concern via ScaleForm's read-aloud).
 */
export function useTypewriter(text: string, speedMs = 12): string {
  const prefersReducedMotion = useReducedMotion();
  const [display, setDisplay] = useState("");
  const [prevText, setPrevText] = useState(text);

  // Reset synchronously during render when `text` changes, rather than via
  // an effect — this is React's documented "adjusting state when a prop
  // changes" pattern, and avoids a setState call directly in an effect body.
  if (text !== prevText) {
    setPrevText(text);
    setDisplay("");
  }

  useEffect(() => {
    if (prefersReducedMotion || !text) return;
    let i = 0;
    const id = setInterval(() => {
      i++;
      setDisplay(text.slice(0, i));
      if (i >= text.length) clearInterval(id);
    }, speedMs);
    return () => clearInterval(id);
  }, [text, speedMs, prefersReducedMotion]);

  return prefersReducedMotion || !text ? text : display;
}
