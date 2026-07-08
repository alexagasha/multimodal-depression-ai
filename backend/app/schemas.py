"""
Pydantic schemas for API request/response bodies.

Kept deliberately separate from services.py so the HTTP contract is visible
in one place and doesn't drift silently as pipeline internals change.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    exists: bool
    n_participants: int
    splits: Dict[str, int] = Field(default_factory=dict)
    participant_ids: List[int] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    mock_encoders: bool
    model_seed: Optional[int]
    last_run_at: Optional[str]
    last_n_participants: Optional[int]
    predictions_available: bool
    dataset: DatasetInfo


class GenerateDatasetRequest(BaseModel):
    n_participants: int = Field(default=5, ge=1, le=200)


class RunPipelineRequest(BaseModel):
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class ParticipantResult(BaseModel):
    participant_id: int
    n_segments: int
    phq8_pred: float
    binary_pred: int
    phq8_true: int
    binary_true: int
    split: str


class PipelineRunResponse(BaseModel):
    results: List[ParticipantResult]
    metrics: Optional[dict] = None
    model_seed: int
    ran_at: str


class ConfusionMatrix(BaseModel):
    labels: List[str]
    matrix: List[List[int]]


class MetricsResponse(BaseModel):
    f1_weighted: float
    f1_macro: float
    rmse: float
    mae: float
    n_participants: int
    depression_threshold: float
    confusion_matrix: ConfusionMatrix


class ModalityAttribution(BaseModel):
    baseline_phq8_pred: float
    modality_attributions: Dict[str, float]


class SegmentPreview(BaseModel):
    segment_idx: int
    start: float
    end: float
    text: str


class ParticipantDetail(BaseModel):
    participant_id: int
    metadata: dict
    n_segments: int
    segments_preview: List[SegmentPreview]
    prediction: Optional[dict]
    attribution: Optional[ModalityAttribution]


class TestRunResponse(BaseModel):
    passed: int
    failed: int
    success: bool
    output: str
