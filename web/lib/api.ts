// Typed client for the FilingLens API. Types are generated from the FastAPI OpenAPI schema:
//   uv run filinglens api openapi && npm run gen:api
import type { components } from "./api-types";

type S = components["schemas"];
export type CompanyListItem = S["CompanyListItem"];
export type CompanyDetail = S["CompanyDetail"];
export type QuarterValue = S["QuarterValue"];
export type FilingRef = S["FilingRef"];
export type RiskFactorOut = S["RiskFactorOut"];
export type RiskChange = S["RiskChange"];
export type ChunkDetail = S["ChunkDetail"];
export type EvalRunSummary = S["EvalRunSummary"];
export type EvalRunDetail = S["EvalRunDetail"];
export type MetricPoint = S["MetricPoint"];
export type ReviewItemOut = S["ReviewItemOut"];
export type ReviewItemDetail = S["ReviewItemDetail"];
export type Decision = S["Decision"];
export type PricePoint = S["PricePoint"];
export type PriceHistoryResult = S["PriceHistoryResult"];

export const API_BASE = process.env.NEXT_PUBLIC_FILINGLENS_API_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...init });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body: unknown = await res.json();
      if (body && typeof body === "object" && "detail" in body) detail = String((body as { detail: unknown }).detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export const api = {
  companies: () => request<CompanyListItem[]>("/api/companies"),
  company: (ticker: string) => request<CompanyDetail>(`/api/companies/${encodeURIComponent(ticker)}`),
  chunk: (id: number) => request<ChunkDetail>(`/api/chunks/${id}`),
  evals: () => request<EvalRunSummary[]>("/api/evals"),
  evalRun: (id: string) => request<EvalRunDetail>(`/api/evals/${encodeURIComponent(id)}`),
  reviews: (status = "pending") => request<ReviewItemOut[]>(`/api/review?status=${status}`),
  review: (id: number) => request<ReviewItemDetail>(`/api/review/${id}`),
  decide: (id: number, decision: Decision, token: string) =>
    request<ReviewItemDetail>(`/api/review/${id}/decision`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(token ? { "X-API-Token": token } : {}) },
      body: JSON.stringify(decision),
    }),
};
