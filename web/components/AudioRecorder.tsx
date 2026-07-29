"use client";

import { useRef, useState } from "react";
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
 * Falls back to a plain file upload when the microphone isn't available.
 */
export default function AudioRecorder({
  sessionId,
  onDone,
}: {
  sessionId: string;
  onDone: (info: { duration_sec: number; n_transcript_rows: number }) => void;
}) {
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const sampleRateRef = useRef(16000);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  async function startRecording() {
    setError(null);
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

      processor.onaudioprocess = (e) => {
        chunksRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      processor.connect(ctx.destination);

      setRecording(true);
      setSeconds(0);
      timerRef.current = setInterval(() => setSeconds((s) => s + 1), 1000);
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
      const result = await api.uploadAudio(sessionId, blob);
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
            className="flex items-center gap-2 rounded-full bg-clay-500 px-4 py-2 text-sm font-semibold text-white hover:bg-clay-600 disabled:opacity-50"
          >
            <span className="h-2.5 w-2.5 rounded-full bg-white" /> Record
          </button>
        ) : (
          <button
            type="button"
            onClick={handleStop}
            className="flex items-center gap-2 rounded-full bg-sage-600 px-4 py-2 text-sm font-semibold text-white hover:bg-sage-700"
          >
            <span className="h-2.5 w-2.5 bg-white" /> Stop ({seconds}s)
          </button>
        )}
        {uploading && <span className="text-sm text-sage-600">Uploading + transcribing…</span>}
      </div>

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
