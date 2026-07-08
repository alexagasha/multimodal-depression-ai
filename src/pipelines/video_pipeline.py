"""
Video pipeline: STUB — not active in Phase 1.

Active modalities: text + audio + metadata.
Video (MobileNetV3) will be integrated in a later phase.

This file exists so the repo structure is complete and run_pipeline.py
has an import path ready. The stub raises NotImplementedError so it is
never silently called without intent.

When video is ready to integrate:
  1. Implement MobileNetV3VideoEncoder.encode()
  2. Implement run_video_pipeline()
  3. Enable the video branch in src/fusion/run_pipeline.py (marked with TODO: VIDEO)
  4. Update FUSION_INPUT_DIM in src/fusion/model.py to include VIDEO_EMBED_DIM
"""

VIDEO_EMBED_DIM = 960  # MobileNetV3-Large final feature dim


class MobileNetV3VideoEncoder:
    def encode(self, frame):
        raise NotImplementedError("Video pipeline not active in Phase 1.")


def run_video_pipeline(pid, segments, encoder, data_root=None):
    raise NotImplementedError("Video pipeline not active in Phase 1.")
