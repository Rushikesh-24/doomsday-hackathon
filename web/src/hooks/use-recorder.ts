"use client";
import { useRef, useState } from "react";

/** Microphone capture via MediaRecorder (webm/ogg/mp4 depending on browser; the backend decodes with ffmpeg). */
export function useRecorder(maxSeconds = 15) {
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const rec = useRef<MediaRecorder | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  async function start(onDone: (blob: Blob) => void) {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser does not expose a microphone (needs https or localhost).");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const r = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      r.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      r.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        setRecording(false);
        if (timer.current) clearTimeout(timer.current);
        onDone(new Blob(chunks, { type: r.mimeType || "audio/webm" }));
      };
      rec.current = r;
      r.start();
      setRecording(true);
      timer.current = setTimeout(() => r.state === "recording" && r.stop(), maxSeconds * 1000);
    } catch (e) {
      setError(`Microphone unavailable: ${(e as Error).message}`);
    }
  }
  function stop() {
    if (rec.current?.state === "recording") rec.current.stop();
  }
  return { recording, error, start, stop };
}

export const LANGUAGES: [string, string][] = [
  ["", "Auto-detect"], ["en", "English"], ["hi", "Hindi"], ["mr", "Marathi"], ["ta", "Tamil"], ["te", "Telugu"],
  ["bn", "Bengali"], ["ur", "Urdu"], ["es", "Spanish"], ["fr", "French"], ["de", "German"],
];
