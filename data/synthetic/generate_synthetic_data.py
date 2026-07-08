"""
Generates synthetic participant sessions matching the structural shape of
DAIC-WOZ / E-DAIC-WOZ, so the rest of the pipeline can be built and tested
before real dataset access is granted.

Each participant session folder contains:
    {pid}_TRANSCRIPT.csv   - start_time, stop_time, speaker, value
    {pid}_AUDIO.wav         - placeholder raw audio (silence + light noise)
    {pid}_VIDEO/frame_*.npy - placeholder video frames (random RGB arrays)
    {pid}_METADATA.json     - demographic/clinical fields
LABELS.csv                  - participant_id, phq8_score, split (train/dev/test)

This mirrors the real dataset's known shape closely enough that swapping
in real files later only requires pointing DATA_ROOT at the new folder —
no pipeline code changes.
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

FILLER_LINES = [
    "I have been feeling okay this week.",
    "Work has been pretty stressful lately.",
    "I sleep around six hours a night.",
    "My family has been supportive.",
    "I do not really enjoy going out much these days.",
    "Some days are harder than others.",
    "I try to exercise when I can.",
    "I have not been very motivated recently.",
]

INTERVIEWER_LINES = [
    "Can you tell me more about that?",
    "How long has that been going on?",
    "What helps you feel better?",
    "How would you describe your mood this week?",
]


def make_transcript(pid, n_turns=20):
    rows = []
    t = 0.0
    for i in range(n_turns):
        speaker = "Ellie" if i % 2 == 0 else "Participant"
        text = random.choice(INTERVIEWER_LINES) if speaker == "Ellie" else random.choice(FILLER_LINES)
        dur = random.uniform(3, 12)
        rows.append({"start_time": round(t, 2), "stop_time": round(t + dur, 2),
                     "speaker": speaker, "value": text})
        t += dur + random.uniform(0.2, 1.0)
    return pd.DataFrame(rows)


def make_audio(duration_sec, sample_rate=16000):
    n_samples = int(duration_sec * sample_rate)
    # light synthetic "voice-like" noise rather than pure silence
    audio = (np.random.randn(n_samples) * 0.01).astype(np.float32)
    return audio, sample_rate


def make_video_frames(duration_sec, fps=1, h=64, w=64):
    n_frames = max(1, int(duration_sec * fps))
    frames = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n_frames)]
    return frames


def make_metadata(pid):
    return {
        "participant_id": pid,
        "age": int(np.random.randint(18, 65)),
        "gender": random.choice(["male", "female"]),
        "education_years": int(np.random.randint(8, 20)),
    }


def generate(n_participants, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    labels = []
    splits = (["train"] * int(n_participants * 0.6)
              + ["dev"] * int(n_participants * 0.2))
    splits += ["test"] * (n_participants - len(splits))
    random.shuffle(splits)

    for i in range(n_participants):
        pid = 300 + i  # DAIC-WOZ-style numeric IDs start around 300
        session_dir = os.path.join(out_dir, str(pid))
        os.makedirs(session_dir, exist_ok=True)

        transcript = make_transcript(pid)
        transcript.to_csv(os.path.join(session_dir, f"{pid}_TRANSCRIPT.csv"), index=False)

        duration = float(transcript["stop_time"].iloc[-1])
        audio, sr = make_audio(duration)
        np.save(os.path.join(session_dir, f"{pid}_AUDIO.npy"), audio)
        with open(os.path.join(session_dir, f"{pid}_AUDIO.meta.json"), "w") as f:
            json.dump({"sample_rate": sr, "duration_sec": duration}, f)

        frames_dir = os.path.join(session_dir, f"{pid}_VIDEO")
        os.makedirs(frames_dir, exist_ok=True)
        frames = make_video_frames(duration)
        for j, frame in enumerate(frames):
            np.save(os.path.join(frames_dir, f"frame_{j:04d}.npy"), frame)

        with open(os.path.join(session_dir, f"{pid}_METADATA.json"), "w") as f:
            json.dump(make_metadata(pid), f)

        phq8 = int(np.clip(np.random.normal(8, 5), 0, 24))
        labels.append({"participant_id": pid, "phq8_score": phq8, "split": splits[i]})

    pd.DataFrame(labels).to_csv(os.path.join(out_dir, "LABELS.csv"), index=False)
    print(f"Generated {n_participants} synthetic sessions in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_participants", type=int, default=5)
    parser.add_argument("--out_dir", type=str,
                         default=os.path.join(os.path.dirname(__file__), "sessions"))
    args = parser.parse_args()
    generate(args.n_participants, args.out_dir)
