"""
Generates synthetic participant sessions matching the structural shape of the
real data-collection tool (QUESTIONAIRE-V2_R: Uganda PHQ-9 + HAM-D instrument),
so the rest of the pipeline can be built and tested before real data arrives.

Each participant session folder contains:
    {pid}_TRANSCRIPT.csv   - start_time, stop_time, speaker, value
    {pid}_AUDIO.npy         - placeholder raw audio (silence + light noise)
    {pid}_AUDIO.meta.json   - sample_rate, duration_sec
    {pid}_METADATA.json     - Section A demographics + site
LABELS.csv                  - participant_id, phq9_score, hamd_score, split

Label scheme (multi-task, agreed):
    phq9_score  0-27   (Section B, self-report)
    hamd_score  0-44   (Section C, 11 items x 0-4; the paper form misprints /52)

This mirrors the real instrument closely enough that swapping in real files
later only requires pointing DATA_ROOT at the new folder — no pipeline changes.
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# --- Section D themed lines (open-ended prompts drive text + audio modelling) ---
FILLER_LINES = [
    "My mood has been low most days this past week.",
    "I do not sleep well, I keep waking up in the night.",
    "I have little energy to do my daily activities.",
    "Things I used to enjoy do not interest me anymore.",
    "When I feel low I pray and talk to my family.",
    "Some days feel very heavy and hard to carry.",
    "My appetite has changed, I am not eating much.",
    "I find it hard to concentrate or make decisions.",
]

INTERVIEWER_LINES = [
    "How are you feeling today as we speak?",
    "How has your sleep been in the past two weeks?",
    "How has your energy been during the day?",
    "Are there activities you used to enjoy but do not now?",
]

# --- Section A categorical value sets (match the paper form wording) ---
AGE_BANDS = ["18-25", "26-35", "36-45", "46-55", "56-65"]
MARITAL = ["married", "divorced", "widowed", "single"]
ETHNICITY = ["bantu", "luo", "other"]
RESIDENCE = ["urban", "semi-urban", "rural"]
EDUCATION = ["none", "primary", "secondary", "tertiary_university"]
EMPLOYMENT = ["employed", "unemployed", "self-employed", "student"]
SITES = ["butabika", "mulago", "other"]


def make_transcript(pid, n_turns=20):
    rows = []
    t = 0.0
    for i in range(n_turns):
        speaker = "Interviewer" if i % 2 == 0 else "Participant"
        text = random.choice(INTERVIEWER_LINES) if speaker == "Interviewer" else random.choice(FILLER_LINES)
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


def make_metadata(pid):
    """Section A demographics + site (provenance)."""
    return {
        "participant_id": pid,
        "age_band": random.choice(AGE_BANDS),
        "sex": random.choice(["male", "female"]),
        "marital_status": random.choice(MARITAL),
        "ethnicity": random.choice(ETHNICITY),
        "residence": random.choice(RESIDENCE),
        "education_level": random.choice(EDUCATION),
        "employment_status": random.choice(EMPLOYMENT),
        "smartphone": random.choice(["yes", "no"]),
        "site": random.choice(SITES),  # kept for stratification, NOT a model feature
    }


def make_labels():
    """Two correlated severity scores from one latent 'true severity' factor."""
    latent = np.random.normal(0.0, 1.0)
    phq9 = int(np.clip(round(8 + latent * 5), 0, 27))   # Section B, 0-27
    hamd = int(np.clip(round(14 + latent * 8), 0, 44))  # Section C, 0-44
    return phq9, hamd


def generate(n_participants, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    labels = []
    splits = (["train"] * int(n_participants * 0.6)
              + ["dev"] * int(n_participants * 0.2))
    splits += ["test"] * (n_participants - len(splits))
    random.shuffle(splits)

    for i in range(n_participants):
        pid = 300 + i  # numeric session IDs (kept DAIC-style for continuity)
        session_dir = os.path.join(out_dir, str(pid))
        os.makedirs(session_dir, exist_ok=True)

        transcript = make_transcript(pid)
        transcript.to_csv(os.path.join(session_dir, f"{pid}_TRANSCRIPT.csv"), index=False)

        duration = float(transcript["stop_time"].iloc[-1])
        audio, sr = make_audio(duration)
        np.save(os.path.join(session_dir, f"{pid}_AUDIO.npy"), audio)
        with open(os.path.join(session_dir, f"{pid}_AUDIO.meta.json"), "w") as f:
            json.dump({"sample_rate": sr, "duration_sec": duration}, f)

        with open(os.path.join(session_dir, f"{pid}_METADATA.json"), "w") as f:
            json.dump(make_metadata(pid), f)

        phq9, hamd = make_labels()
        labels.append({"participant_id": pid, "phq9_score": phq9,
                       "hamd_score": hamd, "split": splits[i]})

    pd.DataFrame(labels).to_csv(os.path.join(out_dir, "LABELS.csv"), index=False)
    print(f"Generated {n_participants} synthetic sessions in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_participants", type=int, default=5)
    parser.add_argument("--out_dir", type=str,
                         default=os.path.join(os.path.dirname(__file__), "sessions"))
    args = parser.parse_args()
    generate(args.n_participants, args.out_dir)
