import type { AppState, ChallengeDetail, DetectionBox } from "./types";
import { tabSessionHeaders } from "./sessionId";

export type WorkspaceMode = "platform" | "own";

export function workspaceFromState(s: AppState): WorkspaceMode {
  if (s.workspace === "own" || s.review_mode === "local") return "own";
  return "platform";
}

export function reviewModeForWorkspace(w: WorkspaceMode): "challenge" | "local" {
  return w === "own" ? "local" : "challenge";
}

const API = "/api";
const REQUEST_TIMEOUT_MS = 30000;
const DOWNLOAD_GT_TIMEOUT_MS = 20 * 60 * 1000;
const ENTER_TOPIC_TIMEOUT_MS = 120000;
const FIND_BEST_TIMEOUT_MS = 120000;
const FIND_BEST_EVAL_TIMEOUT_MS = 300000;
const MAX_RETRIES = 3;

function isRetryableError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const msg = error.message.toLowerCase();
  return (
    error.name === "TypeError" ||
    error.name === "AbortError" ||
    msg.includes("failed to fetch") ||
    msg.includes("network") ||
    msg.includes("timeout") ||
    msg.includes("aborted")
  );
}

function friendlyRequestError(error: unknown, timeoutMs: number): Error {
  if (error instanceof DOMException && error.name === "AbortError") {
    return new Error(
      `Request timed out after ${Math.round(timeoutMs / 1000)}s (server busy). Wait a moment and try again.`,
    );
  }
  if (error instanceof Error) {
    const msg = error.message.toLowerCase();
    if (error.name === "AbortError" || msg.includes("aborted without reason") || msg.includes("signal is aborted")) {
      return new Error(
        `Request timed out after ${Math.round(timeoutMs / 1000)}s (server busy). Wait a moment and try again.`,
      );
    }
    return error;
  }
  return new Error("Request failed");
}

async function sleep(ms: number) {
  await new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function request<T>(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const timeoutMs = init?.timeoutMs ?? REQUEST_TIMEOUT_MS;
  const { timeoutMs: _ignored, ...fetchInit } = init || {};
  let lastError: unknown = null;
  for (let attempt = 0; attempt < MAX_RETRIES; attempt += 1) {
    if (fetchInit.signal?.aborted) {
      throw friendlyRequestError(new DOMException("Aborted", "AbortError"), timeoutMs);
    }
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), timeoutMs);
    const onExternalAbort = () => controller.abort();
    fetchInit.signal?.addEventListener("abort", onExternalAbort);
    try {
      const res = await fetch(`${API}${path}`, {
        ...fetchInit,
        credentials: "include",
        signal: controller.signal,
        headers: tabSessionHeaders({ "Content-Type": "application/json", ...(fetchInit.headers || {}) }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        const detail = (body as { detail?: string }).detail || res.statusText;
        if (res.status === 401 && !path.startsWith("/auth/") && !path.startsWith("/ultralytics/")) {
          window.dispatchEvent(new CustomEvent("labeler-auth-required"));
        }
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      return res.json();
    } catch (error) {
      lastError = friendlyRequestError(error, timeoutMs);
      if (attempt + 1 >= MAX_RETRIES || !isRetryableError(error)) {
        break;
      }
      await sleep(150 * (attempt + 1));
    } finally {
      window.clearTimeout(timer);
      fetchInit.signal?.removeEventListener("abort", onExternalAbort);
    }
  }
  if (lastError instanceof Error) {
    throw lastError;
  }
  throw new Error("Request failed");
}

export const api = {
  authStatus: () => request<AuthStatus>("/auth/status"),
  requestAccess: (email: string) =>
    request<{ ok: boolean; email: string; message: string; expires_in_seconds: number }>("/auth/request", {
      method: "POST",
      body: JSON.stringify({ email }),
    }),
  verifyAccess: (email: string, code: string) =>
    request<{ ok: boolean; email: string; authenticated: boolean }>("/auth/verify", {
      method: "POST",
      body: JSON.stringify({ email, code }),
    }),
  logout: () => request<{ ok: boolean }>("/auth/logout", { method: "POST" }),
  getState: () => request<AppState>("/state"),
  setSkill: (element_path: string) =>
    request<AppState>("/skill", { method: "POST", body: JSON.stringify({ element_path }) }),
  setReviewMode: (mode: string) =>
    request<AppState>("/review-mode", {
      method: "POST",
      body: JSON.stringify({ mode }),
      timeoutMs: ENTER_TOPIC_TIMEOUT_MS,
    }),
  setWorkspace: (workspace: WorkspaceMode) => api.setReviewMode(reviewModeForWorkspace(workspace)),
  refreshOwnPool: () =>
    request<AppState>("/own/refresh", { method: "POST", timeoutMs: ENTER_TOPIC_TIMEOUT_MS }),
  listOwnExisting: () =>
    request<{
      files: Array<{
        filename: string;
        stem: string;
        block: number;
        size: number;
        digest?: string;
        width: number;
        height: number;
        reviewed: boolean;
      }>;
    }>("/own/existing"),
  uploadOwnImages: async (
    files: File[],
    opts?: {
      onProgress?: (loaded: number, total: number) => void;
      replaceNames?: string[];
    },
  ) => {
    const form = new FormData();
    for (const f of files) form.append("files", f, f.name);
    for (const name of opts?.replaceNames || []) form.append("replace_names", name);
    // XHR so we can report upload progress for large Own batches.
    return new Promise<
      AppState & { upload?: { added: number; skipped: number; deduped?: number; replaced?: number; total: number } }
    >((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API}/own/upload`);
      xhr.withCredentials = true;
      const headers = tabSessionHeaders();
      for (const [k, v] of Object.entries(headers)) xhr.setRequestHeader(k, v);
      xhr.upload.onprogress = (ev) => {
        if (!opts?.onProgress) return;
        opts.onProgress(ev.loaded, ev.lengthComputable ? ev.total : 0);
      };
      xhr.onload = () => {
        let body: unknown = null;
        try {
          body = JSON.parse(xhr.responseText || "{}");
        } catch {
          body = { detail: xhr.statusText };
        }
        if (xhr.status === 401) {
          window.dispatchEvent(new CustomEvent("labeler-auth-required"));
        }
        if (xhr.status < 200 || xhr.status >= 300) {
          const detail = (body as { detail?: unknown })?.detail;
          reject(new Error(typeof detail === "string" ? detail : xhr.statusText || `Upload failed (${xhr.status})`));
          return;
        }
        resolve(body as AppState & { upload?: { added: number; skipped: number; total: number } });
      };
      xhr.onerror = () => reject(new Error("Network error during Own upload"));
      xhr.onabort = () => reject(new Error("Upload cancelled"));
      xhr.send(form);
    });
  },
  deleteOwnRange: (opts?: { blockMin?: number | null; blockMax?: number | null }) =>
    request<AppState & { delete?: { removed: number; total: number } }>("/own/delete-range", {
      method: "POST",
      body: JSON.stringify({
        block_min: opts?.blockMin ?? null,
        block_max: opts?.blockMax ?? null,
      }),
      timeoutMs: ENTER_TOPIC_TIMEOUT_MS,
    }),
  exportOwnDataset: async (reviewedOnly = true) => {
    const res = await fetch(`${API}/own/export?reviewed_only=${reviewedOnly ? "true" : "false"}`, {
      credentials: "include",
      headers: tabSessionHeaders(),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      if (res.status === 401) window.dispatchEvent(new CustomEvent("labeler-auth-required"));
      throw new Error(typeof (body as { detail?: string }).detail === "string" ? (body as { detail: string }).detail : res.statusText);
    }
    const blob = await res.blob();
    const disposition = res.headers.get("Content-Disposition") || "";
    const match = /filename\*?=(?:UTF-8''|")?([^\";]+)/i.exec(disposition);
    const filename = match ? decodeURIComponent(match[1].replace(/"/g, "")) : "own_gt.zip";
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
  },
  startCollect: (continue_mode = false) =>
    request<AppState>(`/collect/start?continue_mode=${continue_mode}`, { method: "POST" }),
  stopCollect: () => request<AppState>("/collect/stop", { method: "POST" }),
  ultralyticsAccount: (api_key: string) =>
    request<{ ok: boolean; owner: string; summary: Record<string, unknown> }>("/ultralytics/account", {
      method: "POST",
      body: JSON.stringify({ api_key }),
      timeoutMs: 45000,
    }),
  ultralyticsListDatasets: (api_key: string) =>
    request<{ ok: boolean; owner: string; datasets: UltralyticsDataset[] }>("/ultralytics/datasets", {
      method: "POST",
      body: JSON.stringify({ api_key }),
      timeoutMs: 60000,
    }),
  ultralyticsUpload: (body: UltralyticsUploadRequest) =>
    request<UltralyticsUploadResult>("/ultralytics/upload", {
      method: "POST",
      body: JSON.stringify(body),
      timeoutMs: 15 * 60_000,
    }),
  dppDefaults: () => request<{ min_gt: number; topic_slug: string; class_names: string[]; reviewed_available: number }>("/dpp/defaults"),
  dppUpload: async (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${API}/dpp/upload`, {
      method: "POST",
      body: fd,
      credentials: "include",
      headers: tabSessionHeaders(),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      if (res.status === 401) window.dispatchEvent(new CustomEvent("labeler-auth-required"));
      throw new Error(typeof (body as { detail?: string }).detail === "string" ? (body as { detail: string }).detail : res.statusText);
    }
    return res.json() as Promise<{ job: DppJob }>;
  },
  dppStatus: (job_id: string) => request<{ job: DppJob }>(`/dpp/status?job_id=${encodeURIComponent(job_id)}`),
  dppRun: (body: DppRunRequest) =>
    request<{ job: DppJob }>("/dpp/run", { method: "POST", body: JSON.stringify(body), timeoutMs: 60_000 }),
  dppStop: (job_id: string) =>
    request<{ job: DppJob }>(`/dpp/stop?job_id=${encodeURIComponent(job_id)}`, { method: "POST" }),
  dppCompare: (body: DppCompareRequest) =>
    request<{ ok: boolean; compare: DppCompareResult; job: DppJob }>("/dpp/compare", {
      method: "POST",
      body: JSON.stringify(body),
      timeoutMs: 20 * 60_000,
    }),
  dppEvaluateImage: async (job_id: string, file: File, hf_repo?: string) => {
    const fd = new FormData();
    fd.append("file", file);
    const qs = new URLSearchParams({ job_id });
    if (hf_repo) qs.set("hf_repo", hf_repo);
    const res = await fetch(`${API}/dpp/evaluate-image?${qs}`, {
      method: "POST",
      body: fd,
      credentials: "include",
      headers: tabSessionHeaders(),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      if (res.status === 401) window.dispatchEvent(new CustomEvent("labeler-auth-required"));
      throw new Error(typeof (body as { detail?: string }).detail === "string" ? (body as { detail: string }).detail : res.statusText);
    }
    return res.json() as Promise<DppEvalImageResult>;
  },
  dppDownload: async (job_id: string) => {
    const res = await fetch(`${API}/dpp/download/${encodeURIComponent(job_id)}`, {
      credentials: "include",
      headers: tabSessionHeaders(),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(typeof (body as { detail?: string }).detail === "string" ? (body as { detail: string }).detail : res.statusText);
    }
    const blob = await res.blob();
    const disposition = res.headers.get("Content-Disposition") || "";
    const match = /filename\*?=(?:UTF-8''|")?([^\";]+)/i.exec(disposition);
    const filename = match ? decodeURIComponent(match[1].replace(/"/g, "")) : `${job_id}_ship.zip`;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  },
  getChallenge: (
    index: number,
    fast = true,
    signal?: AbortSignal,
    activate = true,
    reason?: string,
  ) => {
    const why = reason ? `&why=${encodeURIComponent(reason.slice(0, 32))}` : "";
    return request<ChallengeDetail>(`/challenges/${index}?fast=${fast}&activate=${activate}${why}`, {
      signal,
    });
  },
  getChallengeVariants: (index: number, signal?: AbortSignal) =>
    request<{ reference_variants: Record<string, unknown>[]; reference_boxes: Record<string, unknown>[] }>(
      `/challenges/${index}/variants`,
      { signal, timeoutMs: 8000 },
    ),
  getChallengeFull: (index: number) =>
    request<ChallengeDetail>(`/challenges/${index}?fast=false&activate=false`),
  saveBoxes: (index: number, boxes: DetectionBox[], reviewed = true, itemKey?: string | null) =>
    request<{ ok: boolean }>(`/challenges/${index}/boxes`, {
      method: "PUT",
      body: JSON.stringify({
        boxes,
        reviewed,
        ...(itemKey ? { item_key: itemKey } : {}),
      }),
    }),
  markReviewed: (index: number) => request<AppState>(`/challenges/${index}/mark-reviewed`, { method: "POST" }),
  acceptReference: (index: number, variantIdx?: number, itemKey?: string | null) =>
    request<AppState>(`/challenges/${index}/accept-reference`, {
      method: "POST",
      body: JSON.stringify({
        ...(variantIdx != null ? { variant_idx: variantIdx } : {}),
        ...(itemKey ? { item_key: itemKey } : {}),
      }),
    }),
  reviewAgain: (index: number) =>
    request<{ state: AppState; detail: ChallengeDetail }>(`/challenges/${index}/review-again`, { method: "POST" }),
  downloadReviewedGt: async (
    opts?: { blockMin?: number | null; blockMax?: number | null; quality?: string },
    onProgress?: (received: number, total: number) => void,
  ) => {
    const params = new URLSearchParams();
    if (opts?.blockMin != null && Number.isFinite(opts.blockMin)) params.set("block_min", String(opts.blockMin));
    if (opts?.blockMax != null && Number.isFinite(opts.blockMax)) params.set("block_max", String(opts.blockMax));
    if (opts?.quality && opts.quality !== "all") params.set("quality", opts.quality);
    const qs = params.toString();
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), DOWNLOAD_GT_TIMEOUT_MS);
    let res: Response;
    try {
      res = await fetch(`${API}/export/reviewed-gt${qs ? `?${qs}` : ""}`, {
        credentials: "include",
        signal: controller.signal,
        headers: tabSessionHeaders(),
      });
    } catch (e) {
      throw friendlyRequestError(e, DOWNLOAD_GT_TIMEOUT_MS);
    } finally {
      window.clearTimeout(timer);
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      const detail = (body as { detail?: string }).detail || res.statusText;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    const disposition = res.headers.get("Content-Disposition") || "";
    const match = /filename\*?=(?:UTF-8''|")?([^\";]+)/i.exec(disposition);
    const filename = match ? decodeURIComponent(match[1].replace(/"/g, "")) : "reviewed_gt.zip";
    const total = Number(res.headers.get("Content-Length")) || 0;
    const body = res.body;
    let blob: Blob;
    if (body && onProgress) {
      const reader = body.getReader();
      const chunks: Uint8Array[] = [];
      let received = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) {
          chunks.push(value);
          received += value.byteLength;
          onProgress(received, total);
        }
      }
      blob = new Blob(chunks as BlobPart[], { type: res.headers.get("Content-Type") || "application/zip" });
    } else {
      blob = await res.blob();
    }
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
    const countHeader = res.headers.get("X-Reviewed-Count");
    return { count: countHeader ? Number(countHeader) : null, filename };
  },
  listTopics: () => request<{ topics: TopicInfo[] }>("/topics"),
  enterTopic: (slug: string) =>
    request<AppState & { topic?: TopicInfo }>(
      "/topics/" + encodeURIComponent(slug) + "/enter",
      { method: "POST", timeoutMs: ENTER_TOPIC_TIMEOUT_MS },
    ),
  getChutesStatus: (body: ChutesQuery) =>
    request<ChutesStatus>("/chutes/status", { method: "POST", body: JSON.stringify(body) }),
  getChutesBalanceAlert: (body: ChutesQuery) =>
    request<ChutesBalanceAlert>("/chutes/balance-alert", { method: "POST", body: JSON.stringify(body) }),
  saveChutesMonitor: (body: ChutesQuery) =>
    request<{ configured: boolean; threshold: number; has_api_key: boolean }>("/chutes/monitor", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  getFindBestDefaults: () =>
    request<FindBestDefaults>("/find-best/defaults", { timeoutMs: FIND_BEST_TIMEOUT_MS }),
  getFindBestStatus: () =>
    request<{
      job: FindBestJob | null;
      element?: string;
      job_element?: string | null;
      topic_mismatch?: boolean;
    }>("/find-best/status"),
  getFindBestHistory: () => request<{ history: FindBestHistoryEntry[] }>("/find-best/history"),
  getFindBestHistoryProgress: (historyId: string) =>
    request<FindBestJob>(`/find-best/history/${encodeURIComponent(historyId)}/progress`, {
      timeoutMs: FIND_BEST_TIMEOUT_MS,
    }),
  startFindBest: (body: FindBestStartBody) =>
    request<{ job: FindBestJob; staged: Record<string, unknown> }>("/find-best/start", {
      method: "POST",
      body: JSON.stringify({
        element: body.element,
        days: body.days,
        top_k: body.top_k,
        min_cosine: body.min_cosine,
        n_images: body.n_images,
        block_min: body.blockMin,
        block_max: body.blockMax,
        quality: body.quality,
      }),
      timeoutMs: FIND_BEST_TIMEOUT_MS,
    }),
  stopFindBest: () =>
    request<{ job: FindBestJob }>("/find-best/stop", {
      method: "POST",
      body: JSON.stringify({}),
      timeoutMs: FIND_BEST_TIMEOUT_MS,
    }),
  downloadFindBestOnnx: async (historyId: string) => {
    const res = await fetch(`${API}/find-best/history/${encodeURIComponent(historyId)}/download`, {
      credentials: "include",
      headers: tabSessionHeaders(),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      const detail = (body as { detail?: string }).detail || res.statusText;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    const blob = await res.blob();
    const disposition = res.headers.get("Content-Disposition") || "";
    const match = /filename\*?=(?:UTF-8''|")?([^\";]+)/i.exec(disposition);
    const filename = match
      ? decodeURIComponent(match[1].replace(/"/g, ""))
      : `${historyId}_ship.zip`;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  },
  loadFindBestMiner: (historyId: string) =>
    request<AppState>(`/find-best/history/${encodeURIComponent(historyId)}/load-miner`, {
      method: "POST",
      body: JSON.stringify({}),
      timeoutMs: FIND_BEST_TIMEOUT_MS,
    }),
  evaluateFindBestModel: async (historyId: string, files: File[]) => {
    const fd = new FormData();
    for (const file of files) fd.append("files", file, file.name);
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), FIND_BEST_EVAL_TIMEOUT_MS);
    let res: Response;
    try {
      res = await fetch(`${API}/find-best/history/${encodeURIComponent(historyId)}/evaluate`, {
        method: "POST",
        body: fd,
        credentials: "include",
        signal: controller.signal,
        headers: tabSessionHeaders(),
      });
    } catch (e) {
      throw friendlyRequestError(e, FIND_BEST_EVAL_TIMEOUT_MS);
    } finally {
      window.clearTimeout(timer);
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      const detail = (body as { detail?: string }).detail || res.statusText;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return (await res.json()) as FindBestEvalResult;
  },
  previewLabelImport: (files: LabelImportFileInput[]) =>
    request<LabelImportPreviewResult>("/labels/import/preview", {
      method: "POST",
      body: JSON.stringify({ files }),
      timeoutMs: 120000,
    }),
  applyLabelImport: (body: LabelImportApplyBody) =>
    request<LabelImportApplyResult>("/labels/import/apply", {
      method: "POST",
      body: JSON.stringify(body),
      timeoutMs: 180000,
    }),
};

export type LabelImportFileInput = { name: string; content: string };

export type LabelImportScoreTriple = { s: number | null; m: number | null; f: number | null };

export type LabelImportReadyItem = {
  block: number;
  index: number;
  item_key: string;
  display_name: string;
  filename: string;
  box_count: number;
};

export type LabelImportConflictItem = {
  block: number;
  index: number;
  item_key: string;
  display_name: string;
  filename: string;
  width?: number;
  height?: number;
  existing_box_count: number;
  upload_box_count: number;
  existing_boxes?: DetectionBox[];
  upload_boxes?: DetectionBox[];
  platform: LabelImportScoreTriple;
  existing_t: LabelImportScoreTriple;
  upload_t: LabelImportScoreTriple;
};

export type LabelImportIssueItem = { name: string; reason: string };

export type LabelImportPreviewResult = {
  token: string;
  class_count: number;
  class_names: string[];
  summary: {
    total_files: number;
    matched: number;
    ready: number;
    conflicts: number;
    unmatched: number;
    invalid: number;
  };
  ready: LabelImportReadyItem[];
  conflicts: LabelImportConflictItem[];
  unmatched: LabelImportIssueItem[];
  invalid: LabelImportIssueItem[];
};

export type LabelImportApplyBody = {
  token: string;
  replace_blocks?: number[];
  replace_all?: boolean;
};

export type LabelImportApplyResult = {
  applied: number;
  skipped: number;
  errors: { block: number; reason: string }[];
  applied_keys?: string[];
  state: AppState;
};

export type FindBestStartBody = {
  element?: string;
  days?: number;
  top_k?: number;
  min_cosine?: number;
  n_images?: number;
  blockMin?: number | null;
  blockMax?: number | null;
  quality?: "all" | "pass" | "fail";
};

export type FindBestStepSummary = {
  headline?: string;
  metrics?: { label: string; value: string }[];
  rows?: { title: string; subtitle?: string; meta?: string }[];
};

export type FindBestJobStep = {
  n: number;
  label: string;
  status: string;
  summary?: FindBestStepSummary | null;
};

export type FindBestJob = {
  id: string;
  status: "queued" | "running" | "done" | "error" | string;
  step: number;
  step_label: string;
  element?: string;
  steps: FindBestJobStep[];
  error?: string | null;
  params?: Record<string, unknown>;
  result?: FindBestHistoryEntry | null;
  started_at?: string | null;
  finished_at?: string | null;
  history_id?: string | null;
  min_reviewed?: number;
};

export type FindBestHistoryEntry = {
  id: string;
  created_at: string;
  element: string;
  score?: number | null;
  delta_score?: number | null;
  winner?: string | null;
  top_miner?: string | null;
  has_onnx?: boolean;
  onnx_size?: number;
};

export type FindBestEvalBox = {
  cls: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  conf?: number | null;
};

export type FindBestEvalImage = {
  name: string;
  width: number;
  height: number;
  n_boxes: number;
  boxes: FindBestEvalBox[];
};

export type FindBestEvalResult = {
  history_id: string;
  class_names: string[];
  n_images: number;
  n_boxes: number;
  elapsed_ms: number;
  skipped: string[];
  classes: { cls: number; name: string; count: number }[];
  images: FindBestEvalImage[];
};

export type FindBestElementOption = {
  value: string;
  label: string;
  winner?: string;
};

export type FindBestElementInfo = {
  value: string;
  label: string;
  topic_slug?: string;
  known?: boolean;
  winner?: string;
  winner_short?: string;
  score_id?: string;
  search_queries?: string[];
};

export type FindBestDefaults = {
  element: string;
  element_info?: FindBestElementInfo | null;
  elements: FindBestElementOption[];
  days: number;
  top_k: number;
  min_cosine: number;
  n_images: number;
  param_source?: string;
  history_count?: number;
  best_score?: number | null;
  reviewed_available: number;
  min_reviewed: number;
  job: FindBestJob | null;
  history: FindBestHistoryEntry[];
};

export type ChutesQuery = {
  api_key?: string;
  threshold?: number | null;
  force?: boolean;
};

export type ChutesStatus = {
  username?: string | null;
  user_id?: string | null;
  balance: number | null;
  balance_alert_usd: number;
  low_balance: boolean;
  configured?: boolean;
  fetched_at?: string;
};

export type ChutesBalanceAlert = {
  configured: boolean;
  low_balance: boolean;
  balance: number | null;
  balance_alert_usd: number;
  username?: string | null;
  fetched_at?: string;
};

export type TopicInfo = {
  slug: string;
  title: string;
  subtitle?: string;
  element_id: string;
  element_path: string;
  evaluation_count: number;
  first_block?: number;
  latest_block?: number;
  local_images: number;
  preview_ready?: boolean;
  preview_url: string;
  active?: boolean;
};

export type AuthStatus = {
  enabled: boolean;
  authenticated: boolean;
  email?: string | null;
  admin_email?: string;
  smtp_configured?: boolean;
};

export type UltralyticsDataset = {
  id: string;
  dataset: string;
  name: string;
  task?: string | null;
  imageCount?: number;
  visibility?: string | null;
};

export type UltralyticsUploadRequest = {
  api_key: string;
  owner?: string;
  dataset: string;
  dataset_id?: string;
  create_if_missing?: boolean;
  dataset_name?: string;
  block_min?: number | null;
  block_max?: number | null;
  quality?: string;
  train_pct?: number;
  val_pct?: number;
  test_pct?: number;
  force_split?: string | null;
  conflict_policy?: "skip" | "replace" | "keep_both";
};

export type UltralyticsUploadResult = {
  ok: boolean;
  packed: number;
  filename: string;
  owner: string;
  dataset: string;
  datasetId: string;
  conflictPolicy: string;
  url: string;
  sessionId?: string;
  ingest?: Record<string, unknown>;
};

export type DppJob = {
  id: string;
  topic_slug: string;
  class_names: string[];
  status: string;
  message: string;
  verify?: {
    ok?: boolean;
    error?: string;
    warning?: string | null;
    topic_n_classes?: number;
    model_n_classes?: number | null;
    topic_classes?: string[];
  };
  baseline_score?: { score?: number; map50?: number; ffpi?: number } | null;
  postproc?: {
    score?: number;
    map50?: number;
    ffpi?: number;
    final_live?: { score?: number; map50?: number; ffpi?: number };
  } | null;
  compare?: DppCompareResult | null;
  error?: string | null;
  ship_ready?: boolean;
  n_gt?: number;
};

export type DppRunRequest = {
  job_id: string;
  block_min?: number | null;
  block_max?: number | null;
  quality?: string;
  n_images?: number;
  passes?: number;
  random_n?: number;
};

export type DppCompareRequest = {
  job_id: string;
  hf_repo: string;
  block_min?: number | null;
  block_max?: number | null;
  quality?: string;
  n_images?: number;
};

export type DppCompareResult = {
  hf_repo: string;
  n_images: number;
  ours: { score: number; map50: number; ffpi: number };
  theirs: { score: number; map50: number; ffpi: number };
  delta_score: number;
  winner: string;
};

export type DppEvalImageResult = {
  filename: string;
  width: number;
  height: number;
  class_names: string[];
  ours: { boxes: DetectionBox[]; class_names?: string[]; n_boxes: number };
  theirs?: { boxes: DetectionBox[]; class_names?: string[]; n_boxes: number; hf_repo?: string } | null;
};
