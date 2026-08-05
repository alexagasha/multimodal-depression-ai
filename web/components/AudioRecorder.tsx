"use client";

import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { api, ApiError, NoteDraft, TranscriptRow } from "@/lib/api";
import { Card } from "@/components/FormField";

/** How much audio to bank before shipping a chunk for live transcription.
 *  Short enough that captions feel live, long enough that base.en on a
 *  CPU-only box keeps up (it transcribes a chunk in well under this). */
const CHUNK_SECONDS = 6;

/**
 * Real live-mic recording: getUserMedia -> AudioContext -> ScriptProcessorNode
 * captures raw Float32 PCM samples, hand-encoded into 16-bit WAV Blobs
 * (browsers' MediaRecorder doesn't encode WAV directly, hence this approach).
 * ScriptProcessorNode is deprecated but universally supported; AudioWorklet
 * is the eventual upgrade (see docs/system-roadmap.md Phase 2).
 *
 * Streams rather than batching: every CHUNK_SECONDS the audio captured since
 * the last send is posted to /audio/chunk, so transcription and the AI note
 * build up *during* the interview instead of after it. On stop it finalizes
 * (server assembles the full WAV + transcript) and the caller auto-scores —
 * no "upload", "transcribe" or "run scoring" button anywhere in the flow.
 *
 * A parallel AnalyserNode drives a live canvas waveform while recording —
 * purely visual, doesn't touch the PCM capture path — so the interview
 * feels like a real recording session, not a black box with a timer.
 *
 * Falls back to a plain whole-file upload when the microphone isn't available.
 */
export default function AudioRecorder({
  visitId,
  onDone,
  onTranscriptRows,
  onNote,
  onRecordingChange,
}: {
  visitId: string;
  onDone: (info: { duration_sec: number; n_transcript_rows: number }) => void;
  /** New transcript lines as they're recognised, for the live transcript. */
  onTranscriptRows?: (rows: TranscriptRow[]) => void;
  /** The running AI note, whenever the server redrafts it. */
  onNote?: (note: NoteDraft) => void;
  /** Lets the page show live-transcript/note affordances while recording. */
  onRecordingChange?: (recording: boolean) => void;
}) {
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<{ duration_sec: number; n_transcript_rows: number } | null>(null);

  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const sampleRateRef = useRef(16000);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  // Samples captured but not yet shipped for live transcription. Kept
  // separate from chunksRef (the full take) so each chunk is sent once.
  const pendingRef = useRef<Float32Array[]>([]);
  const streamingRef = useRef(false);
  const inFlightRef = useRef(false);

  useEffect(() => stopWaveform, []);

  // The <canvas> only mounts once `recording` is true (see JSX below), and
  // that state update isn't reflected in the DOM until after this render
  // commits — so the draw loop has to kick off from an effect (once the
  // canvas ref is actually populated), not synchronously inside
  // startRecording(), or canvasRef.current is still null and drawWaveform()
  // silently no-ops forever.
  useEffect(() => {
    if (recording) drawWaveform();
  }, [recording]);

  /**
   * Renders a scrolling, layered "mesh" waveform modeled on 3D-soundwave
   * artwork: nested glowing contour lines over a dark backdrop, warm sage ->
   * clay gradient, with a bright cream filament tracing the outer crest.
   *
   * Plots loudness over TIME (a scrolling history buffer fed by RMS of the
   * time-domain data), not frequency across x — voice energy sits almost
   * entirely in the low bins, so a spectrum layout leaves most of the width
   * permanently dead, whereas a time plot gives varied peaks the whole way
   * across. Three parallax layers read the same history at slightly
   * different lags, so their crests offset and the wave gains depth.
   */
  function drawWaveform() {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    const canvasCtx = canvas?.getContext("2d");
    if (!canvas || !analyser || !canvasCtx) return;

    const POINTS = 72;
    const MAX_LAG = 10;
    const LAYERS = [
      { scale: 0.52, alpha: 0.22, lag: 10 },
      { scale: 0.76, alpha: 0.38, lag: 5 },
      { scale: 1.0, alpha: 0.85, lag: 0 },
    ];
    const CONTOURS = 8;

    const timeData = new Uint8Array(analyser.fftSize);
    const history = new Array<number>(POINTS + MAX_LAG).fill(0);
    let level = 0;
    let gain = 0.15;
    let frame = 0;
    let phase = 0;

    // Catmull-Rom spline through the points -> smooth, organic curve rather
    // than visibly straight segments between samples.
    const curveThrough = (pts: [number, number][]) => {
      canvasCtx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 0; i < pts.length - 1; i++) {
        const p0 = pts[i - 1] ?? pts[i];
        const p1 = pts[i];
        const p2 = pts[i + 1];
        const p3 = pts[i + 2] ?? pts[i + 1];
        canvasCtx.bezierCurveTo(
          p1[0] + (p2[0] - p0[0]) / 6,
          p1[1] + (p2[1] - p0[1]) / 6,
          p2[0] - (p3[0] - p1[0]) / 6,
          p2[1] - (p3[1] - p1[1]) / 6,
          p2[0],
          p2[1]
        );
      }
    };

    const draw = () => {
      if (!analyserRef.current) return; // stopped mid-frame
      rafRef.current = requestAnimationFrame(draw);
      frame++;
      phase += 0.045;

      // RMS of the time-domain signal = perceived loudness this frame.
      analyser.getByteTimeDomainData(timeData);
      let sumSq = 0;
      for (let i = 0; i < timeData.length; i++) {
        const v = (timeData[i] - 128) / 128;
        sumSq += v * v;
      }
      const raw = Math.sqrt(sumSq / timeData.length);
      level += (raw - level) * 0.35;

      // Auto-gain against a decaying running peak, so quiet and loud mics
      // alike fill the height instead of drawing a permanently flat ribbon.
      gain = Math.max(level, gain * 0.995, 0.06);
      if (frame % 2 === 0) {
        // ^1.6 adds contrast: loud syllables stay tall and quiet passages
        // stay low, giving real peaks and valleys rather than a solid slab.
        history.push(Math.pow(Math.min(1, level / gain), 1.6));
        history.shift();
      }

      const { width, height } = canvas;
      canvasCtx.clearRect(0, 0, width, height);

      const centerY = height / 2;
      const maxHalf = height / 2 - 5;
      const step = width / (POINTS - 1);
      let peak = 0;
      for (let i = history.length - POINTS; i < history.length; i++) {
        if (history[i] > peak) peak = history[i];
      }

      // Canvas can't read CSS custom properties, hence the literal hexes.
      const gradient = canvasCtx.createLinearGradient(0, 0, width, 0);
      gradient.addColorStop(0, "#8bb373"); // sage-400
      gradient.addColorStop(0.45, "#e5b48a"); // clay-300
      gradient.addColorStop(1, "#c17a42"); // clay-500
      // Glow warms from sage toward clay as volume rises, so louder speech
      // visibly lights up rather than only growing taller.
      const glow = mixSageClay(Math.min(1, peak * 1.6));

      for (const layer of LAYERS) {
        const top: [number, number][] = [];
        const bot: [number, number][] = [];
        for (let i = 0; i < POINTS; i++) {
          const t = i / (POINTS - 1);
          // Taper only near the two edges, so mid-width peaks keep full
          // height and the wave still fades gracefully in and out.
          const edge = Math.min(1, Math.min(t, 1 - t) / 0.12);
          // Gentle idle ripple so the wave still undulates during silence;
          // real signal wins the moment there is any input.
          const breathe = 0.05 + Math.sin(phase + i * 0.3 + layer.lag) * 0.022;
          const amp = Math.max(history[i + (MAX_LAG - layer.lag)], breathe) * edge * layer.scale;
          const h = Math.max(1, amp * maxHalf);
          const x = i * step;
          top.push([x, centerY - h]);
          bot.push([x, centerY + h]);
        }

        canvasCtx.save();
        canvasCtx.strokeStyle = gradient;
        canvasCtx.shadowColor = glow;

        // Nested contours (the same wave at fractions of full height) give
        // the layered, luminous wireframe look instead of one filled mass.
        for (let k = 1; k <= CONTOURS; k++) {
          const f = k / CONTOURS;
          const outer = k === CONTOURS;
          canvasCtx.globalAlpha = layer.alpha * (outer ? 1 : 0.16 + 0.5 * f);
          canvasCtx.lineWidth = outer ? 1.4 : 0.9;
          canvasCtx.shadowBlur = outer ? 7 + peak * 13 : 3;
          for (const side of [top, bot]) {
            canvasCtx.beginPath();
            curveThrough(side.map(([x, y]) => [x, centerY + (y - centerY) * f]));
            canvasCtx.stroke();
          }
        }

        // Bright filament along the front layer's outermost crest/trough.
        if (layer.scale === 1) {
          canvasCtx.globalAlpha = 0.9;
          canvasCtx.strokeStyle = "rgba(253, 251, 246, 0.9)"; // cream-50
          canvasCtx.lineWidth = 1.2;
          canvasCtx.shadowBlur = 9;
          for (const side of [top, bot]) {
            canvasCtx.beginPath();
            curveThrough(side);
            canvasCtx.stroke();
          }
        }
        canvasCtx.restore();
      }
    };
    draw();
  }

  function stopWaveform() {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
  }

  /**
   * Ships everything captured since the last send. `flush` ignores the
   * length threshold so the tail of the recording isn't dropped on stop.
   * Skips while a previous chunk is still in flight — on slow hardware two
   * overlapping posts would interleave and corrupt the server-side ordering.
   */
  async function sendPendingChunk(flush = false) {
    if (inFlightRef.current) return;
    const pending = pendingRef.current;
    const samples = pending.reduce((n, c) => n + c.length, 0);
    const minSamples = CHUNK_SECONDS * sampleRateRef.current;
    if (samples === 0 || (!flush && samples < minSamples)) return;

    pendingRef.current = [];
    inFlightRef.current = true;
    setTranscribing(true);
    try {
      const blob = encodeWav(pending, sampleRateRef.current);
      const result = await api.uploadAudioChunk(visitId, blob);
      if (result.new_rows.length) onTranscriptRows?.(result.new_rows);
      if (result.note_updated && result.note) onNote?.(result.note);
    } catch (e) {
      // A dropped chunk costs a few seconds of captions, not the recording —
      // the full audio is still banked locally in chunksRef.
      console.warn("live transcription chunk failed", e);
    } finally {
      inFlightRef.current = false;
      setTranscribing(false);
    }
  }

  async function startRecording() {
    setError(null);
    setLastResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      type WindowWithWebkitAudio = typeof window & { webkitAudioContext?: typeof AudioContext };
      const AudioContextCtor =
        window.AudioContext ?? (window as WindowWithWebkitAudio).webkitAudioContext;
      const ctx = new AudioContextCtor();
      audioCtxRef.current = ctx;
      sampleRateRef.current = ctx.sampleRate;

      const source = ctx.createMediaStreamSource(stream);
      sourceRef.current = source;
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      chunksRef.current = [];

      const analyser = ctx.createAnalyser();
      // 1024 samples (~21ms at 48kHz) is a long enough window for a stable
      // RMS loudness reading; 256 makes the level jitter frame to frame.
      analyser.fftSize = 1024;
      analyserRef.current = analyser;

      processor.onaudioprocess = (e) => {
        const samples = new Float32Array(e.inputBuffer.getChannelData(0));
        chunksRef.current.push(samples); // full take, kept as a local safety net
        if (streamingRef.current) pendingRef.current.push(samples);
      };
      source.connect(processor);
      processor.connect(ctx.destination);
      source.connect(analyser); // parallel tap, visual only — doesn't affect the WAV capture

      pendingRef.current = [];
      streamingRef.current = true;
      setRecording(true);
      onRecordingChange?.(true);
      setSeconds(0);
      timerRef.current = setInterval(() => {
        setSeconds((s) => s + 1);
        void sendPendingChunk();
      }, 1000);
    } catch {
      setError("Microphone unavailable in this browser/session — use the file upload below.");
    }
  }

  function stopRecording() {
    streamingRef.current = false;
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    audioCtxRef.current?.close();
    if (timerRef.current) clearInterval(timerRef.current);
    stopWaveform();
    analyserRef.current = null;
    setRecording(false);
    onRecordingChange?.(false);
  }

  /**
   * Stop -> flush the tail -> finalize server-side -> hand back to the caller,
   * which kicks off scoring automatically. The clinician clicks Stop and
   * nothing else.
   */
  async function handleStop() {
    const hadAudio = chunksRef.current.length > 0;
    stopRecording();
    chunksRef.current = [];
    if (!hadAudio) return;

    setUploading(true);
    setError(null);
    try {
      // Wait out any chunk still in flight so the tail lands after it,
      // keeping the server-side transcript in order.
      while (inFlightRef.current) await new Promise((r) => setTimeout(r, 150));
      await sendPendingChunk(true);
      while (inFlightRef.current) await new Promise((r) => setTimeout(r, 150));

      const result = await api.finalizeAudio(visitId);
      setLastResult(result);
      onDone(result);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setUploading(false);
    }
  }

  async function uploadBlob(blob: Blob) {
    setUploading(true);
    setError(null);
    try {
      const result = await api.uploadAudio(visitId, blob);
      setLastResult(result);
      onDone(result);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setUploading(false);
    }
  }

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) await uploadBlob(file);
  }

  return (
    <Card className="space-y-3">
      <h2 className="font-display text-lg font-semibold text-sage-800">Interview audio</h2>
      <p className="text-sm text-sage-600">
        Transcription and the AI note run as you record. Stop when the interview
        ends and scoring starts on its own — or upload a .wav if the microphone
        isn&apos;t available here.
      </p>

      <div className="flex flex-wrap items-center gap-3">
        {!recording ? (
          <button
            type="button"
            onClick={startRecording}
            disabled={uploading}
            className="flex items-center gap-2 rounded-full bg-clay-500 px-4 py-2 text-sm font-semibold text-white transition-transform hover:bg-clay-600 active:scale-95 disabled:opacity-50"
          >
            <span className="h-2.5 w-2.5 rounded-full bg-white" /> Record
          </button>
        ) : (
          <button
            type="button"
            onClick={handleStop}
            className="flex items-center gap-2 rounded-full bg-sage-600 px-4 py-2 text-sm font-semibold text-white transition-transform hover:bg-sage-700 active:scale-95"
          >
            <span className="relative flex h-2.5 w-2.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-white opacity-75" />
              <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-white" />
            </span>
            Stop ({seconds}s)
          </button>
        )}

        <AnimatePresence>
          {recording && transcribing && (
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-1.5 text-sm text-sage-600"
            >
              <span className="flex gap-0.5">
                {[0, 1, 2].map((i) => (
                  <motion.span
                    key={i}
                    className="h-1 w-1 rounded-full bg-clay-500"
                    animate={{ opacity: [0.2, 1, 0.2] }}
                    transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
                  />
                ))}
              </span>
              Transcribing
            </motion.span>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {uploading && (
            <motion.span
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-1 text-sm text-sage-600"
            >
              Finalising &amp; scoring
              <span className="flex gap-0.5">
                {[0, 1, 2].map((i) => (
                  <motion.span
                    key={i}
                    className="h-1 w-1 rounded-full bg-sage-500"
                    animate={{ opacity: [0.2, 1, 0.2] }}
                    transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
                  />
                ))}
              </span>
            </motion.span>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {!uploading && lastResult && (
            <motion.span
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-1 text-sm font-medium text-sage-700"
            >
              ✓ {lastResult.n_transcript_rows} segment{lastResult.n_transcript_rows === 1 ? "" : "s"} · {lastResult.duration_sec.toFixed(0)}s
            </motion.span>
          )}
        </AnimatePresence>
      </div>

      <AnimatePresence>
        {recording && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 88 }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden rounded-xl bg-ink-900"
          >
            <canvas ref={canvasRef} width={600} height={88} className="h-22 w-full" />
          </motion.div>
        )}
      </AnimatePresence>

      <label className="block text-sm text-sage-700">
        Or upload a .wav file:
        <input
          type="file"
          accept="audio/wav"
          onChange={handleFileUpload}
          disabled={uploading}
          className="mt-1 block text-sm"
        />
      </label>

      {error && <p className="text-sm text-[var(--color-danger)]">{error}</p>}
    </Card>
  );
}

// Canvas can't read CSS custom properties, so sage-500/clay-500 are
// hand-copied here as RGB triplets to interpolate between.
const SAGE_500: [number, number, number] = [107, 153, 80];
const CLAY_500: [number, number, number] = [193, 122, 66];

function mixSageClay(t: number): string {
  const r = Math.round(SAGE_500[0] + (CLAY_500[0] - SAGE_500[0]) * t);
  const g = Math.round(SAGE_500[1] + (CLAY_500[1] - SAGE_500[1]) * t);
  const b = Math.round(SAGE_500[2] + (CLAY_500[2] - SAGE_500[2]) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

function encodeWav(chunks: Float32Array[], sampleRate: number): Blob {
  const length = chunks.reduce((sum, c) => sum + c.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  for (const c of chunks) {
    merged.set(c, offset);
    offset += c.length;
  }

  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);
  const writeStr = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate (mono, 16-bit)
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, "data");
  view.setUint32(40, length * 2, true);

  let o = 44;
  for (let i = 0; i < length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, merged[i]));
    view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([buffer], { type: "audio/wav" });
}
