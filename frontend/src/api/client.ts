// Thin fetch wrapper for the HeReFaNMi backend. Base URL is same-origin "/api"
// by default (nginx/Vite proxy forwards to the backend); override with VITE_API_BASE.

const BASE = import.meta.env.VITE_API_BASE ?? "/api";

export type Verdict = "Trustworthy" | "Doubtful" | "Fake";

export interface MedicalTalkResult {
  data: string; // reasoning text (or the non-medical notice)
  news?: "True" | "False";
  label?: Verdict;
  source?: string[];
  key: string;
}

export interface AuthResult {
  access_token: string;
  token_type: string;
  uid: string;
  points: number;
  is_admin: boolean;
}

export interface SourceConfig {
  name: string;
  base_url: string;
  listing_url: string;
  listing_link_selector: string;
  title_selector: string;
  content_selector: string;
  date_selector: string | null;
  date_attr: string | null;
  enabled: boolean;
  interval_minutes: number;
  last_run_at: string | null;
  last_status: string | null;
}

export interface AdminStats {
  users: number;
  queries: number;
  chunks: number;
  per_source_chunks: Record<string, number>;
  sources: number;
}

export interface AdminUser {
  uid: string;
  email: string;
  points: number;
  is_admin: boolean;
}

const TOKEN_KEY = "hrf_token";
const ADMIN_KEY = "hrf_is_admin";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}
export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}
export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(ADMIN_KEY);
}
export function getAdminFlag(): boolean {
  return localStorage.getItem(ADMIN_KEY) === "1";
}
export function setAdminFlag(isAdmin: boolean): void {
  localStorage.setItem(ADMIN_KEY, isAdmin ? "1" : "0");
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  auth = true,
): Promise<T> {
  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (auth) {
    const token = getToken();
    if (token) headers.Authorization = `Bearer ${token}`;
  }
  const resp = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!resp.ok) {
    let detail = `Request failed (${resp.status})`;
    try {
      const j = await resp.json();
      detail = j.detail ?? detail;
    } catch {
      /* non-JSON error */
    }
    const err = new Error(detail) as Error & { status: number };
    err.status = resp.status;
    throw err;
  }
  return resp.json() as Promise<T>;
}

export const api = {
  signup: (email: string, password: string) =>
    request<AuthResult>("POST", "/auth/signup", { email, password }, false),
  login: (email: string, password: string) =>
    request<AuthResult>("POST", "/auth/login", { email, password }, false),
  medicalTalk: (data: string, opinion: string, backend: string) =>
    request<MedicalTalkResult>("POST", "/medicalTalk", { data, opinion, backend }),
  save: (reference: string, rating: string) =>
    request<{ status: string }>("POST", "/save", { reference, rating }),
  pointCheck: () => request<{ points: number }>("POST", "/pointcheck", {}),

  // ---- admin ----
  adminStats: () => request<AdminStats>("GET", "/admin/stats"),
  adminSources: () => request<{ sources: SourceConfig[] }>("GET", "/admin/sources"),
  adminCreateSource: (body: Partial<SourceConfig> & { name: string; base_url: string; listing_url: string }) =>
    request<SourceConfig>("POST", "/admin/sources", body),
  adminUpdateSource: (name: string, body: Partial<SourceConfig>) =>
    request<SourceConfig>("PATCH", `/admin/sources/${name}`, body),
  adminDeleteSource: (name: string) =>
    request<{ status: string }>("DELETE", `/admin/sources/${name}`),
  adminScrape: (sources: string[] | null) =>
    request<Record<string, unknown>>("POST", "/admin/scrape", { sources }),
  adminUsers: () => request<{ users: AdminUser[] }>("GET", "/admin/users"),
  adminSetPoints: (uid: string, points: number) =>
    request<{ status: string }>("POST", `/admin/users/${uid}/points`, { points }),
  adminQueries: (limit = 50) =>
    request<{ queries: Array<Record<string, string>> }>("GET", `/admin/queries?limit=${limit}`),
};
