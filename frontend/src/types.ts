export interface DatasetInfo {
  exists: boolean;
  n_participants: number;
  splits: Record<string, number>;
  participant_ids: number[];
}

export interface HealthResponse {
  status: string;
  mock_encoders: boolean;
  model_seed: number | null;
  last_run_at: string | null;
  last_n_participants: number | null;
  predictions_available: boolean;
  dataset: DatasetInfo;
}

export interface ParticipantResult {
  participant_id: number;
  n_segments: number;
  phq8_pred: number;
  binary_pred: number;
  phq8_true: number;
  binary_true: number;
  split: string;
}

export interface Metrics {
  f1_weighted: number;
  f1_macro: number;
  rmse: number;
  mae: number;
  n_participants: number;
  depression_threshold: number;
  confusion_matrix?: {
    labels: string[];
    matrix: number[][];
  };
}

export interface PipelineRunResponse {
  results: ParticipantResult[];
  metrics: Metrics | null;
  model_seed: number;
  ran_at: string;
}

export interface ModalityAttribution {
  baseline_phq8_pred: number;
  modality_attributions: Record<string, number>;
}

export interface AttributionEntry extends ModalityAttribution {
  participant_id: number;
}

export interface SegmentPreview {
  segment_idx: number;
  start: number;
  end: number;
  text: string;
}

export interface ParticipantMetadata {
  participant_id: number;
  age: number;
  gender: string;
  education_years: number;
}

export interface ParticipantDetail {
  participant_id: number;
  metadata: ParticipantMetadata;
  n_segments: number;
  segments_preview: SegmentPreview[];
  prediction: ParticipantResult | null;
  attribution: ModalityAttribution | null;
}

export interface TestRunResponse {
  passed: number;
  failed: number;
  success: boolean;
  output: string;
}

export interface ApiErrorBody {
  detail: string;
}
