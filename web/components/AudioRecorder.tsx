"use client";

import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { api, ApiError } from "@/lib/api";
import { Card } from "@/components/FormField";

/**
 * Real live-mic recording: getUserMedia -> AudioContext -> ScriptProcessorNode
 * captures raw Float32 PCM samples, hand-encoded into a 16-bit WAV Blob on
 * stop, then uploaded through the existing, already-tested
 * POST /sessions/{id}/audio endpoint unchanged (it expects a real WAV —
 * browsers' MediaRecorder doesn't encode WAV directly, hence this approach).
 * ScriptProcessorNode is deprecated but universally supported; AudioWorklet
 * is the eventual upgrade (see docs/system-roadmap.md Phase 2).
 *
 * A parallel AnalyserNode drives a live canvas waveform while recording —
 * purely visual, doesn't touch the PCM capture path — so the interview
 * feels like a real recording session, not a black box with a timer.
 *
 * Falls back to a plain file upload when the microphone isn't available.
 */
export default function AudioRecorder({
  visitId,
  onDone,
}: {
  visitId: string;
  onDone: (info: { duration_sec: number; n_transcript_rows: number }) => void;
}) {
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [uploading, setUploading] = useState(false);
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

  useEffect(() => stopWaveform, []);

  function drawWaveform() {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    const canvasCtx = canvas?.getContext("2d");
    if (!canvas || !analyser || !canvasCtx) return;

    const data = new Uint8Array(analyser.frequencyBinCount);

    const draw = () => {
      if (!analyserRef.current) return; // stopped mid-frame
      rafRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(data);

      const { width, height } = canvas;
      canvasCtx.clearRect(0, 0, width, height);
      canvasCtx.lineWidth = 2;
      canvasCtx.strokeStyle = "#c17a42"; // clay-500 (canvas can't read CSS vars)
      canvasCtx.beginPath();
      const sliceWidth = width / data.length;
      let x = 0;
      for (let i = 0; i < data.length; i++) {
        const v = data[i] / 128.0;
        const y = (v * height) / 2;
        if (i === 0) canvasCtx.moveTo(x, y);
        else canvasCtx.lineTo(x, y);
        x += sliceWidth;
      }
      canvasCtx.lineTo(width, height / 2);
      canvasCtx.stroke();
    };
    draw();
  }

  function stopWaveform() {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
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
      analyser.fftSize = 256;
      analyserRef.current = analyser;

      processor.onaudioprocess = (e) => {
        chunksRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      processor.connect(ctx.destination);
      source.connect(analyser); // parallel tap, visual only — doesn't affect the WAV capture

      setRecording(true);
      setSeconds(0);
      timerRef.current = setInterval(() => setSeconds((s) => s + 1), 1000);
      drawWaveform();
    } catch {
      setError("Microphone unavailable in this browser/session — use the file upload below.");
    }
  }

  function stopRecording(): Blob | null {
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    audioCtxRef.current?.close();
    if (timerRef.current) clearInterval(timerRef.current);
    stopWaveform();
    analyserRef.current = null;
    setRecording(false);

    if (chunksRef.current.length === 0) return null;
    const blob = encodeWav(chunksRef.current, sampleRateRef.current);
    chunksRef.current = [];
    return blob;
  }

  async function handleStop() {
    const blob = stopRecording();
    if (blob) await uploadBlob(blob);
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
        Record live, or upload a .wav file if the microphone isn&apos;t available here.
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
          {uploading && (
            <motion.span
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-1 text-sm text-sage-600"
            >
              Uploading &amp; transcribing
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
            animate={{ opacity: 1, height: 56 }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden rounded-xl bg-sage-50"
          >
            <canvas ref={canvasRef} width={600} height={56} className="h-14 w-full" />
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
