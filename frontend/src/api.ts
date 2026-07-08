import type {
  AttributionEntry,
  DatasetInfo,
  HealthResponse,
  Metrics,
  ParticipantDetail,
  ParticipantResult,
  PipelineRunResponse,
  TestRunResponse,
} from "./types";

// Empty string -> relative "/api/..." calls, handled by the Vite dev proxy
// (see vite.config.ts) or by same-origin hosting in production. Set
// VITE_API_BASE_URL at build time to point at a separately-deployed backend
// (e.g. Render, the same split LASA-VOTING-SYSTEM uses).
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    throw new ApiError(
      0,
      "Could not reach the API. Is the backend running (uvicorn app.main:app --port 8000)?"
    );
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // response wasn't JSON — fall back to statusText
    }
    throw new ApiError(res.status, detail);
  }

  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),

  getDataset: () => request<DatasetInfo>("/api/dataset"),

  generateDataset: (n_participants: number) =>
    request<DatasetInfo>("/api/dataset/generate", {
      method: "POST",
      body: JSON.stringify({ n_participants }),
    }),

  runPipeline: (seed: number) =>
    request<PipelineRunResponse>("/api/pipeline/run", {
      method: "POST",
      body: JSON.stringify({ seed }),
    }),

  getPredictions: () => request<ParticipantResult[]>("/api/predictions"),

  getMetrics: () => request<Metrics>("/api/metrics"),

  getAllAttributions: () => request<AttributionEntry[]>("/api/attributions"),

  getParticipant: (pid: number) =>
    request<ParticipantDetail>(`/api/participants/${pid}`),

  runTests: () => request<TestRunResponse>("/api/tests/run", { method: "POST" }),
};
