"""
Collection waves - the one vocabulary the KoBo scripts share.

The study is collected in waves. Wave 1 (153 collected, 135 usable) is the
cohort every reported result was computed on, and it is split train/dev/test.
A later wave is held out: its participants get split == their wave name
("wave2") and are never stratified into train/dev/test, so the wave stays a
sample no model has seen until someone deliberately decides to spend it.

This is its own tiny module because clean_kobo.py, adapt_kobo.py and
validate_kobo.py run standalone on Colab, where only data/kobo/*.py is uploaded
and nothing under src/ can be imported. src/eval/cohort.py carries the same
TRAINING_SPLITS for the analysis side; tests/test_kobo_waves.py asserts the two
agree.
"""

BASE_WAVE = "wave1"
TRAINING_SPLITS = ("train", "dev", "test")


def wave_of(entry) -> str:
    """A registry entry's wave. Entries written before waves existed are wave 1,
    because wave 1 was the only cohort there was."""
    return (entry or {}).get("wave") or BASE_WAVE


def is_holdout_wave(wave) -> bool:
    return (wave or BASE_WAVE) != BASE_WAVE
