/**
 * Typed client for the FastAPI backend (api/main.py). Local dev only — CORS
 * is wide open there on purpose (see api/main.py's CORSMiddleware comment).
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

export interface ParticipantIn {
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

export interface ScoreResult {
  session_id: string;
  phq9_pred: number;
  hamd_pred: number;
  binary_pred: number;
  risk_flag: boolean;
  modality_attributions: Record<string, number>;
  narrative: string;
}

export interface SessionSummary {
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
  createParticipant: (body: ParticipantIn) =>
    request<{ participant_id: string }>("/participants", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  createSession: (participant_id: string) =>
    request<{ session_id: string }>("/sessions", {
      method: "POST",
      body: JSON.stringify({ participant_id }),
    }),

  submitScaleResponses: (sessionId: string, body: ScaleResponsesIn) =>
    request<{ risk_flag: boolean }>(`/sessions/${sessionId}/scale-responses`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  uploadAudio: (sessionId: string, blob: Blob, filename = "session.wav") => {
    const form = new FormData();
    form.append("file", blob, filename);
    return request<{ duration_sec: number; n_transcript_rows: number }>(
      `/sessions/${sessionId}/audio`,
      { method: "POST", body: form }
    );
  },

  scoreSession: (sessionId: string) =>
    request<ScoreResult>(`/sessions/${sessionId}/score`, { method: "POST" }),

  getResults: (sessionId: string) =>
    request<ScoreResult>(`/sessions/${sessionId}/results`),

  getSession: (sessionId: string) => request<SessionSummary>(`/sessions/${sessionId}`),

  listSessions: () => request<SessionSummary[]>("/sessions"),

  listParticipantSessions: (participantId: string) =>
    request<SessionSummary[]>(`/participants/${participantId}/sessions`),

  addNote: (sessionId: string, author: string, note_text: string) =>
    request<ClinicalNote>(`/sessions/${sessionId}/notes`, {
      method: "POST",
      body: JSON.stringify({ author, note_text }),
    }),

  listNotes: (sessionId: string) =>
    request<ClinicalNote[]>(`/sessions/${sessionId}/notes`),

  submitReview: (sessionId: string, body: ReviewIn) =>
    request<Review>(`/sessions/${sessionId}/review`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  getReview: async (sessionId: string): Promise<Review | null> => {
    try {
      return await request<Review>(`/sessions/${sessionId}/review`);
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) return null;
      throw e;
    }
  },
};
