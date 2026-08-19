"""
Builds the shared "valid participant time ranges" used by every modality.

This is the single source of truth for:
  1. which time spans belong to the participant (interviewer turns removed)
  2. how those spans are sliced into fixed 30s segments

Every modality pipeline (text/audio) slices against the SAME segment
list produced here, so segment N is guaranteed to refer to the same 30s
window across both modalities for a given participant.
"""
import os

import pandas as pd

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthetic", "sessions")
SEGMENT_LEN_SEC = 30.0
INTERVIEWER_SPEAKER_NAMES = {"Ellie", "Interviewer"}


def load_transcript(pid, data_root=DATA_ROOT):
    path = os.path.join(data_root, str(pid), f"{pid}_TRANSCRIPT.csv")
    df = pd.read_csv(path)
    # A row with no text reads back as NaN, and str(NaN) is the literal "nan".
    # Left alone, every untranscribed turn feeds the string "nan" to BERT as if
    # the participant had said it — real risk here, since clips too short to
    # transcribe are deliberately left blank.
    if "value" in df.columns:
        df["value"] = df["value"].fillna("").astype(str)
    return df


def participant_only_ranges(transcript_df):
    """Return list of (start, end, text) for participant-only turns.

    Turns with no text are dropped rather than pooled as empty strings, so a
    segment made up entirely of untranscribed turns yields no text at all
    instead of a run of blanks."""
    mask = ~transcript_df["speaker"].isin(INTERVIEWER_SPEAKER_NAMES)
    rows = transcript_df[mask]
    out = []
    for r in rows.itertuples(index=False):
        text = "" if pd.isna(r.value) else str(r.value).strip()
        if text:
            out.append((r.start_time, r.stop_time, text))
    return out


def build_segments(pid, data_root=DATA_ROOT, segment_len=SEGMENT_LEN_SEC):
    """
    Returns a list of dicts:
        [{"segment_idx": 0, "start": 0.0, "end": 30.0, "text": "..."}, ...]
    Text for a segment is the concatenation of all participant turns whose
    midpoint falls inside that segment's time window.
    """
    transcript = load_transcript(pid, data_root)
    ranges = participant_only_ranges(transcript)
    if not ranges:
        return []

    total_end = max(end for _, end, _ in ranges)
    n_segments = max(1, int(total_end // segment_len) + 1)

    segments = []
    for seg_idx in range(n_segments):
        seg_start = seg_idx * segment_len
        seg_end = seg_start + segment_len
        texts_in_segment = [
            text for (start, end, text) in ranges
            if seg_start <= (start + end) / 2 < seg_end
        ]
        segments.append({
            "segment_idx": seg_idx,
            "start": seg_start,
            "end": seg_end,
            "text": " ".join(texts_in_segment) if texts_in_segment else "",
        })
    return segments


if __name__ == "__main__":
    segs = build_segments(300)
    for s in segs:
        print(s)
