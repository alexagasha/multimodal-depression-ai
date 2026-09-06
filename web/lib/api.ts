/**
 * Typed client for the FastAPI backend (api/main.py). Local dev only — CORS
 * is wide open there on purpose (see api/main.py's CORSMiddleware comment).
 *
 * Naming: the frontend speaks "patient"/"visit" (clinical framing); the
 * backend's underlying collections stay "participants"/"sessions" (see
 * docs/system-roadmap.md) — this module is the translation layer, calling
 * the same endpoints under clinically-named functions.
 */
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: init?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // no JSON body
    }
    throw new ApiError(res.status, detail);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

// ---- Types --------------------------------------------------------------

export interface PatientIn {
  age_band: string;
  sex: string;
  marital_status: string;
  ethnicity: string;
  residence: string;
  education_level: string;
  employment_status: string;
  smartphone: string;
  site: string;
}

export interface ScaleResponsesIn {
  phq9_total: number;
  hamd_total: number;
  phq9_item9: number;
  hamd_suicide_item: number;
}

export type SubtypeLikelihood = "none" | "possible" | "present";

export interface SubtypeEntry {
  likelihood: SubtypeLikelihood;
  rationale: string;
}

/** Keys match api/subtype_differential.py's SUBTYPE_CRITERIA. */
export interface SubtypeDifferentialResult {
  melancholic: SubtypeEntry;
  atypical: SubtypeEntry;
  anxious_distress: SubtypeEntry;
  psychotic_features: SubtypeEntry;
}

export interface EvidenceResult {
  modality: string;
  quotes: string[];
}

export interface AcousticDriver {
  feature: string;
  contribution: number;
  direction: string;
}

export interface ScoreResult {
  session_id: string;
  phq9_pred: number;
  hamd_pred: number;
  /** Clinician-administered totals (ScaleForm), surfaced alongside the AI
   * estimate above — the AI runs as a second opinion, not a replacement. */
  phq9_clinician: number;
  hamd_clinician: number;
  /** Screening priority, not a diagnosis. At the calibrated operating point
   * positive predictive value is 0.93 but negative predictive value is only
   * 0.46 — so a 0 here is weak evidence and must never be presented as
   * ruling depression out. See docs/model-card.md. */
  binary_pred: number;
  risk_flag: boolean;
  modality_attributions: Record<string, number>;
  /** Named prosodic measures driving this estimate. Present only when the
   * served acoustic features are the interpretable ones; a self-supervised
   * embedding has no nameable dimensions. */
  acoustic_drivers?: AcousticDriver[] | null;
  /** Which weights produced this, so a stored result stays traceable. */
  model?: {
    features: string;
    input_dim: number;
    caseness_threshold: number;
    trained: boolean;
    fitted: string | null;
  };
  narrative: string;
  /** All four below are null when unavailable (no ANTHROPIC_API_KEY) —
   * deliberately no non-LLM fallback for any of these, see each module's
   * docstring under api/. */
  subtype_differential: SubtypeDifferentialResult | null;
  evidence: EvidenceResult | null;
  treatment_suggestions: string[] | null;
  patient_summary: string | null;
}

export interface NoteDraft {
  subjective: string;
  objective: string;
  assessment: string;
  plan: string;
  /** Present on the on-demand draft (POST /note-draft), absent on the live
   * running draft. Carried back on save so the note records which model
   * drafted it and how long it was reviewed for. */
  generated_at?: string;
  ai_model?: string;
}

/** One transcribed line, in the same shape the batch pipeline produces. */
export interface TranscriptRow {
  start_time: number;
  stop_time: number;
  speaker: string;
  value: string;
}

export interface ChunkResult {
  new_rows: TranscriptRow[];
  duration_sec: number;
  n_rows_total: number;
  note: NoteDraft | null;
  note_updated: boolean;
}

export interface LiveState {
  rows: TranscriptRow[];
  duration_sec: number;
  note: NoteDraft | null;
}

export interface TreatmentEventIn {
  event_type: "medication_change" | "therapy_session" | "other";
  description: string;
}

export interface TreatmentEvent extends TreatmentEventIn {
  event_id: string;
  event_date: string;
}

export interface TrendFlag {
  flag: boolean;
  reason: string;
}

export interface CaseloadQueryResult {
  answer: string;
  matching_patient_ids: string[];
}

export interface AnalyticsResult {
  total_patients: number;
  total_visits: number;
  scored_visits: number;
  referral_flag_rate: number | null;
  caseness_rate: number | null;
  phq9_severity_distribution: Record<string, number>;
  narrative: string | null;
}

export interface VisitSummary {
  session_id: string;
  participant_id: string;
  status: string;
  risk_flag: boolean | null;
  created_at: string;
  phq9_pred: number | null;
  hamd_pred: number | null;
  binary_pred: number | null;
  reviewed: boolean;
}

/** A registered patient, enriched with their visit history at a glance —
 * powers the roster (GET /participants) and the patient detail page. */
export interface PatientSummary extends PatientIn {
  participant_id: string;
  visit_count: number;
  last_visit_at: string | null;
  risk_flag: boolean | null;
  phq9_pred: number | null;
  hamd_pred: number | null;
  /** Rule-based (not LLM) trend flags — see api/trends.py. */
  relapse_warning: TrendFlag;
  risk_trajectory: TrendFlag;
}

export type NoteSource = "clinician" | "ai_draft_edited" | "ai_draft_accepted";

/** Provenance of a saved note — what drafted it, what the clinician changed,
 * and how long they had it before signing. See
 * docs/clinical-documentation-plan.md §1.5. */
export interface NoteProvenance {
  source: NoteSource;
  ai_model: string | null;
  draft_generated_at: string | null;
  edited_fields: string[] | null;
  review_seconds: number | null;
}

export interface ClinicalNoteIn {
  author: string;
  author_role?: string | null;
  note_type?: "initial_evaluation" | "progress" | "risk_assessment" | "addendum";
  note_text?: string | null;
  subjective?: string | null;
  objective?: string | null;
  assessment?: string | null;
  plan?: string | null;
  amends?: string | null;
  source?: NoteSource;
  draft_generated_at?: string | null;
  ai_model?: string | null;
  edited_fields?: string[] | null;
}

export interface ClinicalNote {
  note_id: string;
  note_type: string;
  author: string;
  author_role: string | null;
  amends: string | null;
  subjective: string | null;
  objective: string | null;
  assessment: string | null;
  plan: string | null;
  /** Flat rendering derived from the SOAP fields — for display only; the
   * structured fields are what storage treats as authoritative. */
  note_text: string;
  created_at: string;
  signed_at: string;
  signed_by: string;
  provenance: NoteProvenance;
}

/** Where a mental-status domain's content could come from — and therefore how
 * far to trust it. `not_observable` means audio cannot support the domain at
 * all; it stays the clinician's to observe. See api/mse.py. */
export type MSESource = "not_observable" | "measured" | "transcript";

export interface MSEMeasure {
  label: string;
  value: number;
  unit: string;
}

export interface MSEDomain {
  domain: string;
  label: string;
  source: MSESource;
  finding: string | null;
  measures: MSEMeasure[] | null;
  note: string | null;
}

export interface MSEResult {
  domains: MSEDomain[];
}

export type Ideation = "none" | "passive" | "active";

export type Disposition =
  | "routine_follow_up"
  | "urgent_follow_up"
  | "referral"
  | "same_day_referral"
  | "admission";

/** A completed suicide-risk assessment. Every field is the clinician's —
 * nothing here is model-filled. See api/risk_assessment.py. */
export interface RiskAssessmentIn {
  assessor: string;
  assessor_role?: string | null;
  ideation: Ideation;
  intent: boolean;
  plan: boolean;
  plan_description?: string | null;
  means_access: boolean;
  means_description?: string | null;
  prior_attempts: boolean;
  prior_attempts_description?: string | null;
  protective_factors: string[];
  verbatim_quotes: string[];
  safety_plan?: string | null;
  disposition: Disposition;
  clinical_reasoning: string;
}

export interface RiskAssessment extends RiskAssessmentIn {
  assessment_id: string;
  session_id: string;
  assessed_at: string;
}

export interface ReviewIn {
  reviewer: string;
  agrees: boolean;
  adjusted_phq9?: number | null;
  adjusted_hamd?: number | null;
  comment?: string | null;
}

export interface Review extends ReviewIn {
  session_id: string;
  reviewed_at: string;
}

// ---- Calls ----------------------------------------------------------------

export const api = {
  registerPatient: (body: PatientIn) =>
    request<{ participant_id: string }>("/participants", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  listPatients: () => request<PatientSummary[]>("/participants"),

  getPatient: (patientId: string) => request<PatientSummary>(`/participants/${patientId}`),

  startVisit: (patientId: string) =>
    request<{ session_id: string }>("/sessions", {
      method: "POST",
      body: JSON.stringify({ participant_id: patientId }),
    }),

  submitScaleResponses: (visitId: string, body: ScaleResponsesIn) =>
    request<{ risk_flag: boolean }>(`/sessions/${visitId}/scale-responses`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  uploadAudio: (visitId: string, blob: Blob, filename = "visit.wav") => {
    const form = new FormData();
    form.append("file", blob, filename);
    return request<{ duration_sec: number; n_transcript_rows: number }>(
      `/sessions/${visitId}/audio`,
      { method: "POST", body: form }
    );
  },

  /** Live streaming: post ~6s of audio mid-recording for incremental
   *  transcription + a running AI note. See api/main.py's /audio/chunk. */
  uploadAudioChunk: (visitId: string, blob: Blob, filename = "chunk.wav") => {
    const form = new FormData();
    form.append("file", blob, filename);
    return request<ChunkResult>(`/sessions/${visitId}/audio/chunk`, {
      method: "POST",
      body: form,
    });
  },

  finalizeAudio: (visitId: string) =>
    request<{ duration_sec: number; n_transcript_rows: number }>(
      `/sessions/${visitId}/audio/finalize`,
      { method: "POST" }
    ),

  getLiveState: (visitId: string) => request<LiveState>(`/sessions/${visitId}/live`),

  scoreVisit: (visitId: string) =>
    request<ScoreResult>(`/sessions/${visitId}/score`, { method: "POST" }),

  getVisitResults: (visitId: string) =>
    request<ScoreResult>(`/sessions/${visitId}/results`),

  getVisit: (visitId: string) => request<VisitSummary>(`/sessions/${visitId}`),

  listVisits: () => request<VisitSummary[]>("/sessions"),

  listPatientVisits: (patientId: string) =>
    request<VisitSummary[]>(`/participants/${patientId}/sessions`),

  addNote: (visitId: string, body: ClinicalNoteIn) =>
    request<ClinicalNote>(`/sessions/${visitId}/notes`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  listNotes: (visitId: string) =>
    request<ClinicalNote[]>(`/sessions/${visitId}/notes`),

  getMSE: (visitId: string) =>
    request<MSEResult>(`/sessions/${visitId}/mse`, { method: "POST" }),

  addRiskAssessment: (visitId: string, body: RiskAssessmentIn) =>
    request<RiskAssessment>(`/sessions/${visitId}/risk-assessment`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  listRiskAssessments: (visitId: string) =>
    request<RiskAssessment[]>(`/sessions/${visitId}/risk-assessment`),

  /** Verbatim transcript lines the clinician may want to quote. Suggestions
   * only — an empty list is a normal result, not a failure. */
  getRiskQuotes: (visitId: string) =>
    request<{ quotes: string[] }>(`/sessions/${visitId}/risk-quotes`, { method: "POST" }),

  closeVisit: (visitId: string) =>
    request<{ session_id: string; status: string }>(`/sessions/${visitId}/close`, {
      method: "POST",
    }),

  submitReview: (visitId: string, body: ReviewIn) =>
    request<Review>(`/sessions/${visitId}/review`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  getReview: async (visitId: string): Promise<Review | null> => {
    try {
      return await request<Review>(`/sessions/${visitId}/review`);
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) return null;
      throw e;
    }
  },

  getNoteDraft: (visitId: string) =>
    request<NoteDraft>(`/sessions/${visitId}/note-draft`, { method: "POST" }),

  addTreatmentEvent: (patientId: string, body: TreatmentEventIn) =>
    request<TreatmentEvent>(`/participants/${patientId}/treatments`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  listTreatmentEvents: (patientId: string) =>
    request<TreatmentEvent[]>(`/participants/${patientId}/treatments`),

  queryCaseload: (question: string) =>
    request<CaseloadQueryResult>("/query", {
      method: "POST",
      body: JSON.stringify({ question }),
    }),

  getAnalytics: () => request<AnalyticsResult>("/analytics"),
};
