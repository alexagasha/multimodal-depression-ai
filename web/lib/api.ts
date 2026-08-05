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

export interface ScoreResult {
  session_id: string;
  phq9_pred: number;
  hamd_pred: number;
  /** Clinician-administered totals (ScaleForm), surfaced alongside the AI
   * estimate above — the AI runs as a second opinion, not a replacement. */
  phq9_clinician: number;
  hamd_clinician: number;
  binary_pred: number;
  risk_flag: boolean;
  modality_attributions: Record<string, number>;
  narrative: string;
  /** null when unavailable (no ANTHROPIC_API_KEY) — there is deliberately no
   * non-LLM fallback for this one, see api/subtype_differential.py. */
  subtype_differential: SubtypeDifferentialResult | null;
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
}

export interface ClinicalNote {
  note_id: string;
  author: string;
  note_text: string;
  created_at: string;
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

  scoreVisit: (visitId: string) =>
    request<ScoreResult>(`/sessions/${visitId}/score`, { method: "POST" }),

  getVisitResults: (visitId: string) =>
    request<ScoreResult>(`/sessions/${visitId}/results`),

  getVisit: (visitId: string) => request<VisitSummary>(`/sessions/${visitId}`),

  listVisits: () => request<VisitSummary[]>("/sessions"),

  listPatientVisits: (patientId: string) =>
    request<VisitSummary[]>(`/participants/${patientId}/sessions`),

  addNote: (visitId: string, author: string, note_text: string) =>
    request<ClinicalNote>(`/sessions/${visitId}/notes`, {
      method: "POST",
      body: JSON.stringify({ author, note_text }),
    }),

  listNotes: (visitId: string) =>
    request<ClinicalNote[]>(`/sessions/${visitId}/notes`),

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
};
