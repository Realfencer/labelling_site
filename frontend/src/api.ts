import type { AppState, ChallengeDetail, DetectionBox } from "./types";
import { tabSessionHeaders, withTabSessionParam } from "./sessionId";
import type { TeamSyncResponse } from "./store/useTeam";

export type WorkspaceMode = "platform" | "own";

export function workspaceFromState(s: AppState): WorkspaceMode {
  if (s.workspace === "own" || s.review_mode === "local") return "own";
  return "platform";
}

export function reviewModeForWorkspace(w: WorkspaceMode): "challenge" | "local" {
  return w === "own" ? "local" : "challenge";
}

export type PeekInfo = {
  mode: string;
  challenge: { item_key: string; block: number; media_rev?: number; display_name: string };
  width: number;
  height: number;
  boxes: Array<Record<string, unknown>>;
  class_names: string[];
};

/** One finding of the label audit. ``i`` = index of the label box it is about (-1 for a drink the model sees unlabeled). */
export type ReviewIssue = {
  kind: "huge" | "tiny" | "stacked" | "nested" | "wrong" | "missed";
  risk: number;
  i: number;
  text: string;
  detail?: string;
  pct?: number;
  px?: number[];
  other?: number;
  iou?: number;
  model_cls?: number;
  conf?: number;
  box?: number[];
};
export type ReviewItemStatus = { risk: number; issues: ReviewIssue[]; model: boolean; done: boolean; dismissed: boolean; open: boolean };
export type ReviewModelCheck = { running: boolean; done: number; total: number; miner?: string | null; error?: string | null };
export type ReviewSummary = {
  ready: boolean;
  labeled: number;
  flagged: number;
  dismissed: number;
  by_kind: Record<string, number>;
  model_check: ReviewModelCheck;
  version: number;
};
export type ReviewStatus = { version: number; summary: ReviewSummary; items?: Record<string, ReviewItemStatus | null> };

const API = "/api";
const REQUEST_TIMEOUT_MS = 30000;
const DOWNLOAD_GT_TIMEOUT_MS = 20 * 60 * 1000;
const ENTER_TOPIC_TIMEOUT_MS = 120000;
const FIND_BEST_TIMEOUT_MS = 120000;
const FIND_BEST_EVAL_TIMEOUT_MS = 300000;
const MAX_RETRIES = 3;

/** Non-2xx response. Never retried: the server already answered. */
export class HttpError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "Error"; // keep String(e) output identical to before
  }
}

/**
 * Retry only transport-level failures (fetch rejects with TypeError). A client timeout is
 * retried for idempotent GETs only: re-sending a timed-out POST (enterTopic, setWorkspace,
 * downloads...) duplicates server work and can stall the UI for minutes.
 */
function isRetryableError(error: unknown, method: string, userAborted: boolean): boolean {
  if (userAborted || error instanceof HttpError) return false;
  if (error instanceof TypeError) return true;
  const isAbort =
    (error instanceof DOMException && error.name === "AbortError") ||
    (error instanceof Error && error.name === "AbortError");
  return isAbort && method === "GET";
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
        throw new HttpError(typeof detail === "string" ? detail : JSON.stringify(detail), res.status);
      }
      return res.json();
    } catch (error) {
      const userAborted = Boolean(fetchInit.signal?.aborted);
      lastError = userAborted ? error : friendlyRequestError(error, timeoutMs);
      const method = (fetchInit.method || "GET").toUpperCase();
      if (attempt + 1 >= MAX_RETRIES || !isRetryableError(error, method, userAborted)) {
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
  login: (password: string, username: string) =>
    request<{ ok: boolean; authenticated: boolean }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ password, username }),
    }),
  /** Read-only look at any image of the topic (another day / the other workspace): what a watched labeler is on. */
  teamPeek: (itemKey: string, workspace: string) =>
    request<PeekInfo>(`/team/peek?item_key=${encodeURIComponent(itemKey)}&workspace=${encodeURIComponent(workspace)}`, { timeoutMs: 15000 }),
  /** Hold G on a zoomed spot: ask the team to look at it (refresh while held, on:false on release). */
  teamHelp: (body: { item_key: string; on: boolean; view?: { cx: number; cy: number; z: number } | null }) =>
    request<{ ok: boolean }>("/team/help", { method: "POST", body: JSON.stringify(body), timeoutMs: 6000 }),
  teamSync: (
    body: {
      item_key: string | null;
      label: string;
      workspace: string;
      idle_s: number;
      watch?: string | null;
      stop_watch?: boolean;
      since: number | null;
    },
    signal?: AbortSignal,
  ) =>
    request<TeamSyncResponse>("/team/sync", {
      method: "POST",
      body: JSON.stringify(body),
      timeoutMs: 8000,
      signal,
    }),
  collectorOverview: () => request<CollectorOverview>("/collector/overview", { timeoutMs: 20000 }),
  collectorList: (status = "", q = "") =>
    request<{ submissions: CollectorSubmission[] }>(
      `/collector/submissions?status=${encodeURIComponent(status)}&q=${encodeURIComponent(q)}`,
      { timeoutMs: 20000 },
    ),
  collectorGet: (id: number) =>
    request<{ submission: CollectorSubmission; evaluations: CollectorEvaluation[] }>(`/collector/submissions/${id}`),
  collectorEval: (id: number) => request<CollectorEvaluation>(`/collector/evaluations/${id}`),
  collectorRefresh: () => request<{ command_id: number }>("/collector/refresh", { method: "POST" }),
  collectorAdd: (model: string, revision = "") =>
    request<{ command_id: number }>("/collector/add", { method: "POST", body: JSON.stringify({ model, revision }) }),
  collectorRetry: (id: number) => request<{ command_id: number }>(`/collector/submissions/${id}/retry`, { method: "POST" }),
  collectorRemeasure: (
    id: number,
    body: { mode: "working" | "latest" | "range"; n?: number; min_block?: number; max_block?: number },
  ) => request<{ command_id: number }>(`/collector/submissions/${id}/remeasure`, { method: "POST", body: JSON.stringify(body) }),
  collectorPush: (id: number, body: { account_id: string; repo_name: string; deep_pp: boolean; nudge: boolean; n: number }) =>
    request<{ command_id: number }>(`/collector/submissions/${id}/push`, { method: "POST", body: JSON.stringify(body) }),
  collectorDelete: (ids: number[]) =>
    request<{ deleted: number[]; skipped: Array<{ id: number; reason: string }> }>("/collector/delete", {
      method: "POST",
      body: JSON.stringify({ ids }),
    }),
  collectorRestore: (id: number) => request<{ ok: boolean }>(`/collector/submissions/${id}/restore`, { method: "POST" }),
  collectorSchedule: () => request<CollectorSchedule>("/collector/schedule"),
  collectorSetSchedule: (body: Partial<{ enabled: boolean; start_min: number; length_min: number; utc_offset_min: number }>) =>
    request<CollectorSchedule>("/collector/schedule", { method: "PUT", body: JSON.stringify(body) }),
  collectorImportMine: (repos: string[] = []) =>
    request<{ command_id: number }>("/collector/import-mine", { method: "POST", body: JSON.stringify({ repos }) }),
  collectorImages: () =>
    request<{ blocks: number[]; count: number; default_n: number; min_images: number }>("/collector/images", { timeoutMs: 30000 }),
  /** No blocks = the working image set. */
  collectorRemeasureSet: (blocks: number[] = []) =>
    request<{ command_id: number; n_images: number }>("/collector/remeasure-set", { method: "POST", body: JSON.stringify({ blocks }) }),
  collectorImageSet: () => request<CollectorImageSet>("/collector/image-set", { timeoutMs: 30000 }),
  collectorSetImageSet: (body: { mode: "latest" | "blocks"; n?: number; blocks?: number[] }) =>
    request<CollectorImageSet>("/collector/image-set", { method: "PUT", body: JSON.stringify(body), timeoutMs: 30000 }),
  collectorVariants: (subId: number) =>
    request<{ variants: CollectorVariant[]; image_set: Partial<CollectorImageSet> }>(`/collector/submissions/${subId}/variants`),
  collectorStage: (subId: number, parentId: number | null, stage: "dpp" | "hash") =>
    request<{ command_id: number }>(`/collector/submissions/${subId}/variants/stage`, {
      method: "POST",
      body: JSON.stringify({ parent_id: parentId, stage }),
    }),
  collectorPushVariant: (vid: number, body: { account_id: string; repo_name: string }) =>
    request<{ command_id: number }>(`/collector/variants/${vid}/push`, { method: "POST", body: JSON.stringify(body) }),
  collectorDeleteVariant: (vid: number) => request<{ ok: boolean }>(`/collector/variants/${vid}`, { method: "DELETE" }),
  collectorCancel: (id: number) => request<{ ok: boolean }>(`/collector/commands/${id}/cancel`, { method: "POST" }),
  collectorRemeasureAll: () => request<{ command_id: number }>("/collector/remeasure-all", { method: "POST" }),
  collectorTry: async (files: File[], ids: number[]): Promise<{ command_id: number }> => {
    const fd = new FormData();
    files.forEach((f) => fd.append("images", f, f.name));
    fd.append("ids", ids.join(","));
    const res = await fetch("/api/collector/try", { method: "POST", body: fd, credentials: "include", headers: tabSessionHeaders() });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(String((body as { detail?: string }).detail || res.statusText));
    }
    return res.json();
  },
  collectorCommand: (id: number) => request<CollectorCommand>(`/collector/commands/${id}`),
  keysGet: () => request<KeysView>("/keys"),
  keysPutHf: (token: string) =>
    request<{ ok: boolean; account: string; orgs: string[]; keys: KeysView }>("/keys/hf", { method: "PUT", body: JSON.stringify({ token }), timeoutMs: 30000 }),
  keysDeleteHf: (id: string) => request<{ ok: boolean; keys: KeysView }>(`/keys/hf/${encodeURIComponent(id)}`, { method: "DELETE" }),
  keysScanHf: (id: string) =>
    request<{ ok: boolean; keys: KeysView }>(`/keys/hf/${encodeURIComponent(id)}/scan`, { method: "POST", timeoutMs: 90000 }),
  keysModels: (account = "") => request<{ models: HfModelRow[] }>(`/keys/hf/models?account=${encodeURIComponent(account)}`),
  keysPut: (service: KeyService, api_key: string) =>
    request<{ ok: boolean; keys: KeysView }>(`/keys/${service}`, { method: "PUT", body: JSON.stringify({ api_key }) }),
  keysDelete: (service: KeyService) => request<{ ok: boolean; keys: KeysView }>(`/keys/${service}`, { method: "DELETE" }),
  keysTest: (service: KeyService) =>
    request<{ ok: boolean; detail: string }>(`/keys/${service}/test`, { method: "POST", timeoutMs: 30000 }),
  keysImport: (body: { chutes_api_key?: string; ultralytics_api_key?: string }) =>
    request<{ ok: boolean; imported: string[]; keys: KeysView }>("/keys/import", { method: "POST", body: JSON.stringify(body) }),
  teamDraft: (itemKey: string, boxes: DetectionBox[]) =>
    request<{ ok: boolean }>("/team/draft", {
      method: "POST",
      body: JSON.stringify({ item_key: itemKey, boxes }),
      timeoutMs: 5000,
    }),
  logout: () => request<{ ok: boolean }>("/auth/logout", { method: "POST" }),
  getState: () => request<AppState>("/state"),
  setSkill: (element_path: string) =>
    request<AppState>("/skill", { method: "POST", body: JSON.stringify({ element_path }) }),
  setReviewMode: (mode: string, preferItemKey?: string | null) =>
    request<AppState>("/review-mode", {
      method: "POST",
      body: JSON.stringify({
        mode,
        ...(preferItemKey ? { prefer_item_key: preferItemKey } : {}),
      }),
      timeoutMs: ENTER_TOPIC_TIMEOUT_MS,
    }),
  setWorkspace: (workspace: WorkspaceMode, preferItemKey?: string | null) =>
    api.setReviewMode(reviewModeForWorkspace(workspace), preferItemKey),
  refreshOwnPool: () =>
    request<AppState>("/own/refresh", { method: "POST", timeoutMs: ENTER_TOPIC_TIMEOUT_MS }),
  /** Label audit: how many labeled Own images look suspicious (the Review card). */
  ownReviewSummary: () => request<ReviewSummary>("/own/review/summary", { timeoutMs: 30000 }),
  /** Live findings for this tab's Review list (``since`` = the version already held: items are left out when unchanged). */
  ownReviewStatus: (since: number) => request<ReviewStatus>(`/own/review/status?since=${since}`, { timeoutMs: 30000 }),
  ownReviewDismiss: (itemKey: string, on: boolean) =>
    request<{ ok: boolean; status: ReviewItemStatus | null; version: number }>("/own/review/dismiss", {
      method: "POST",
      body: JSON.stringify({ item_key: itemKey, on }),
    }),
  ownReviewModelCheck: (action: "start" | "stop") =>
    request<ReviewModelCheck>("/own/review/model-check", { method: "POST", body: JSON.stringify({ action }), timeoutMs: 30000 }),
  listOwnDays: () =>
    request<OwnDaysResponse>("/own/days", { timeoutMs: ENTER_TOPIC_TIMEOUT_MS }),
  setOwnDay: (date: string | null, preferItemKey?: string | null, field?: OwnField | null) =>
    request<AppState>("/own/day", {
      method: "POST",
      body: JSON.stringify({
        date,
        ...(preferItemKey ? { prefer_item_key: preferItemKey } : {}),
        ...(field ? { field } : {}),
      }),
      timeoutMs: ENTER_TOPIC_TIMEOUT_MS,
    }),
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
  deleteOwnRange: (opts?: { blockMin?: number | null; blockMax?: number | null; itemKeys?: string[] }) =>
    request<AppState & { delete?: { removed: number; total: number } }>("/own/delete-range", {
      method: "POST",
      body: JSON.stringify({
        block_min: opts?.blockMin ?? null,
        block_max: opts?.blockMax ?? null,
        item_keys: opts?.itemKeys?.length ? opts.itemKeys : null,
      }),
      timeoutMs: ENTER_TOPIC_TIMEOUT_MS,
    }),
  /** Delete every Own image (New, SAM3 draft and Done) of the given calendar days. */
  deleteOwnDays: (days: string[]) =>
    request<AppState & { delete?: { removed: number; total: number; days: string[] } }>("/own/delete-days", {
      method: "POST",
      body: JSON.stringify({ days }),
      timeoutMs: ENTER_TOPIC_TIMEOUT_MS,
    }),
  clearOwnDayNew: () =>
    request<
      AppState & {
        clear?: { removed: number; total: number; day: string; day_remaining: number; own_day: string | null };
      }
    >("/own/clear-new", {
      method: "POST",
      timeoutMs: ENTER_TOPIC_TIMEOUT_MS,
    }),
  ownHfImportStart: (body: { url: string; hf_token?: string }) =>
    request<{ job: OwnHfImportJob }>("/own/hf-import/start", {
      method: "POST",
      body: JSON.stringify(body),
      timeoutMs: 60_000,
    }),
  ownHfImportStatus: (job_id: string) =>
    request<{ job: OwnHfImportJob }>(`/own/hf-import/status?job_id=${encodeURIComponent(job_id)}`),
  ownHfImportCommit: (body: { job_id: string; replace_names?: string[] }) =>
    request<AppState & { job: OwnHfImportJob; upload?: OwnHfImportResult }>("/own/hf-import/commit", {
      method: "POST",
      body: JSON.stringify(body),
      timeoutMs: 120_000,
    }),
  ownHfImportCancel: (job_id: string) =>
    request<{ job: OwnHfImportJob }>(`/own/hf-import/cancel?job_id=${encodeURIComponent(job_id)}`, {
      method: "POST",
    }),
  ownSam3Label: (body: {
    item_key?: string | null;
    index?: number | null;
    chutes_api_key?: string;
    conf?: number;
  }) =>
    request<AppState & { detail?: ChallengeDetail; sam3?: { n_boxes: number; class_names: string[]; item_key: string; index: number; state?: string } }>(
      "/own/sam3-label",
      {
        method: "POST",
        body: JSON.stringify(body),
        timeoutMs: 10 * 60 * 1000,
      },
    ),
  ownSam3LabelDay: (body?: { chutes_api_key?: string; conf?: number }) =>
    request<
      AppState & {
        sam3_cluster?: {
          day: string;
          requested: number;
          labeled: number;
          failed: Array<{ item_key: string; error: string }>;
          n_boxes: number;
          class_names: string[];
        };
      }
    >("/own/sam3-label-day", {
      method: "POST",
      body: JSON.stringify(body || {}),
      timeoutMs: 60 * 60 * 1000,
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
      timeoutMs: 6 * 60 * 60_000, // big exports go up as several ~9 GB archives, one after the other
    }),
  dppDefaults: () => request<{ min_gt: number; topic_slug: string; class_names: string[]; reviewed_available: number }>("/dpp/defaults"),
  dppHistory: () => request<{ history: DppHistoryEntry[]; topic_slug: string }>("/dpp/history"),
  dppClearHistory: (keep_job_id?: string) =>
    request<DiskClearResult>("/dpp/clear-history", {
      method: "POST",
      body: JSON.stringify({ part: "deep_postproc", keep_job_id: keep_job_id || null }),
      timeoutMs: 10 * 60_000,
    }),
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
    // Native browser download — avoid fetch→blob buffering which hung the ZIP buttons
    // on ~8MB packs (spinner stuck on "…" while the connection drained slowly).
    const url = withTabSessionParam(`${API}/dpp/download/${encodeURIComponent(job_id)}`);
    const a = document.createElement("a");
    a.href = url;
    a.download = `dpp_${job_id}_ship.zip`;
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
  },
  dppPushToHf: (job_id: string, body: { token: string; repo: string }) =>
    request<{
      ok: boolean;
      history_id: string;
      repo_id: string;
      created: boolean;
      url: string;
    }>(`/dpp/${encodeURIComponent(job_id)}/push-hf`, {
      method: "POST",
      body: JSON.stringify(body),
      timeoutMs: 600_000,
    }),
  dppEvaluateShip: async (job_id: string, files: File[]) => {
    const fd = new FormData();
    for (const file of files) fd.append("files", file, file.name);
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), FIND_BEST_EVAL_TIMEOUT_MS);
    let res: Response;
    try {
      res = await fetch(`${API}/dpp/${encodeURIComponent(job_id)}/evaluate`, {
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
      if (res.status === 401) window.dispatchEvent(new CustomEvent("labeler-auth-required"));
      const detail = (body as { detail?: string }).detail || res.statusText;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return (await res.json()) as FindBestEvalResult;
  },
  manualMergeCreate: (body: {
    block_min?: number | null;
    block_max?: number | null;
    item_keys?: string[] | null;
    quality?: string;
    n_images?: number;
  }) =>
    request<{ job: ManualMergeJob }>("/manual-merge/create", {
      method: "POST",
      body: JSON.stringify(body),
      timeoutMs: 120_000,
    }),
  manualMergeCurrent: (topic_slug?: string) =>
    request<{ job: ManualMergeJob | null; topic_slug: string }>(
      `/manual-merge/current${topic_slug ? `?topic_slug=${encodeURIComponent(topic_slug)}` : ""}`,
    ),
  manualMergeHistory: (topic_slug?: string) =>
    request<{ history: ManualMergeHistoryEntry[]; topic_slug: string }>(
      `/manual-merge/history${topic_slug ? `?topic_slug=${encodeURIComponent(topic_slug)}` : ""}`,
    ),
  manualMergeClearHistory: (keep_job_id?: string) =>
    request<DiskClearResult>("/manual-merge/clear-history", {
      method: "POST",
      body: JSON.stringify({ part: "manual_merge", keep_job_id: keep_job_id || null }),
      timeoutMs: 30 * 60_000,
    }),
  /** Manual merge of Collected models, scored on the Collected page's working image set. */
  manualMergeFromCollected: (submission_ids: number[]) =>
    request<{ job: ManualMergeJob; image_set_label: string; n_images: number }>("/manual-merge/create-from-collected", {
      method: "POST",
      body: JSON.stringify({ submission_ids }),
      timeoutMs: 120_000,
    }),
  manualMergeStatus: (job_id: string) =>
    request<{ job: ManualMergeJob }>(`/manual-merge/status?job_id=${encodeURIComponent(job_id)}`),
  manualMergeGtImages: (job_id: string) =>
    request<{ images: ManualMergeGtImage[] }>(`/manual-merge/gt-images?job_id=${encodeURIComponent(job_id)}`),
  manualMergeAddModel: (body: { job_id: string; repo: string; revision?: string; hf_token?: string }) =>
    request<{ job: ManualMergeJob }>("/manual-merge/add-model", {
      method: "POST",
      body: JSON.stringify(body),
      timeoutMs: 600_000,
    }),
  manualMergeRemoveModel: (body: { job_id: string; slug: string }) =>
    request<{ job: ManualMergeJob }>("/manual-merge/remove-model", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  manualMergeScore: (job_id: string) =>
    request<{ job: ManualMergeJob }>("/manual-merge/score", {
      method: "POST",
      body: JSON.stringify({ job_id }),
    }),
  manualMergeStart: (job_id: string, use_remote = true) =>
    request<{ job: ManualMergeJob }>("/manual-merge/start", {
      method: "POST",
      body: JSON.stringify({ job_id, use_remote }),
    }),
  /** Which merge machines are free right now (takes a few seconds: a CPU sample over SSH each). */
  manualMergeMachines: () => request<{ machines: MergeMachine[] }>("/manual-merge/machines", { timeoutMs: 60000 }),
  manualMergeDeepPp: (job_id: string) =>
    request<{ job: ManualMergeJob }>("/manual-merge/deep-pp", {
      method: "POST",
      body: JSON.stringify({ job_id }),
    }),
  manualMergeStop: (job_id: string) =>
    request<{ job: ManualMergeJob }>("/manual-merge/stop", {
      method: "POST",
      body: JSON.stringify({ job_id }),
    }),
  manualMergeEvaluate: async (job_id: string, files: File[]) => {
    const fd = new FormData();
    for (const file of files) fd.append("files", file, file.name);
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), FIND_BEST_EVAL_TIMEOUT_MS);
    let res: Response;
    try {
      res = await fetch(`${API}/manual-merge/${encodeURIComponent(job_id)}/evaluate`, {
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
      if (res.status === 401) window.dispatchEvent(new CustomEvent("labeler-auth-required"));
      const detail = (body as { detail?: string }).detail || res.statusText;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return (await res.json()) as FindBestEvalResult;
  },
  manualMergeEvaluateGt: (body: {
    job_id: string;
    stem: string;
    item_key?: string | null;
    index?: number | null;
  }) =>
    request<ManualMergeEvalGtResult>("/manual-merge/evaluate-gt", {
      method: "POST",
      body: JSON.stringify(body),
      timeoutMs: FIND_BEST_EVAL_TIMEOUT_MS,
    }),
  manualMergeDownload: async (job_id: string) => {
    const url = withTabSessionParam(`${API}/manual-merge/download/${encodeURIComponent(job_id)}`);
    const a = document.createElement("a");
    a.href = url;
    a.download = `manual_${job_id}_ship.zip`;
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
  },
  manualMergePushToHf: (job_id: string, body: { token: string; repo: string }) =>
    request<{ ok: boolean; repo_id: string; created: boolean; url: string }>(
      `/manual-merge/${encodeURIComponent(job_id)}/push-hf`,
      {
        method: "POST",
        body: JSON.stringify(body),
        timeoutMs: 600_000,
      },
    ),
  hfCompareDefaults: () =>
    request<{ min_gt: number; topic_slug: string; class_names: string[]; reviewed_available: number }>(
      "/hf-compare/defaults",
    ),
  hfCompareSearch: (q: string, limit = 20, hf_token?: string) => {
    const qs = new URLSearchParams({ q, limit: String(limit) });
    if (hf_token) qs.set("hf_token", hf_token);
    return request<{ models: HfModelSuggestion[]; query: string }>(`/hf-compare/search?${qs}`, {
      timeoutMs: 15000,
    });
  },
  hfCompareStart: (body: HfCompareStartRequest) =>
    request<{ job: HfCompareJob }>("/hf-compare/start", {
      method: "POST",
      body: JSON.stringify(body),
      timeoutMs: 60_000,
    }),
  hfCompareStatus: (job_id: string) =>
    request<{ job: HfCompareJob }>(`/hf-compare/status?job_id=${encodeURIComponent(job_id)}`),
  hfCompareStop: (job_id: string) =>
    request<{ job: HfCompareJob }>(`/hf-compare/stop?job_id=${encodeURIComponent(job_id)}`, { method: "POST" }),
  hfCompareClearHistory: (keep_job_id?: string) =>
    request<DiskClearResult>("/hf-compare/clear-history", {
      method: "POST",
      body: JSON.stringify({ part: "hf_compare", keep_job_id: keep_job_id || null }),
      timeoutMs: 30 * 60_000,
    }),
  hfCompareDeepPpPush: (job_id: string, body: HfCompareDeepPpPushRequest) =>
    request<{ job: HfCompareJob }>(`/hf-compare/${encodeURIComponent(job_id)}/deep-pp-push`, {
      method: "POST",
      body: JSON.stringify(body),
      timeoutMs: 60_000,
    }),
  hfCompareDeepPpPushStop: (job_id: string) =>
    request<{ job: HfCompareJob }>(
      `/hf-compare/${encodeURIComponent(job_id)}/deep-pp-push/stop`,
      { method: "POST" },
    ),
  baseModelStatus: () => request<BaseModelStatus>("/base-model/status"),
  baseModelSelect: (body: { hf_repo: string; hf_token?: string }) =>
    request<{ candidate: BaseModelCandidate; taxonomy?: string[] }>("/base-model/pick", {
      method: "POST",
      body: JSON.stringify(body),
      timeoutMs: 300_000,
    }),
  baseModelEvaluate: async (candidate_id: string, files: File[]) => {
    const fd = new FormData();
    for (const file of files) fd.append("files", file, file.name);
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), FIND_BEST_EVAL_TIMEOUT_MS);
    let res: Response;
    try {
      res = await fetch(`${API}/base-model/${encodeURIComponent(candidate_id)}/evaluate`, {
        method: "POST",
        body: fd,
        credentials: "include",
        headers: tabSessionHeaders(),
        signal: controller.signal,
      });
    } finally {
      window.clearTimeout(timer);
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      if (res.status === 401) window.dispatchEvent(new CustomEvent("labeler-auth-required"));
      throw new Error(
        typeof (body as { detail?: string }).detail === "string"
          ? (body as { detail: string }).detail
          : res.statusText,
      );
    }
    return (await res.json()) as FindBestEvalResult;
  },
  baseModelEvaluateGt: (body: {
    candidate_id: string;
    block?: number;
    item_key?: string;
    mode?: string;
    index?: number;
  }) =>
    request<BaseModelGtEvalResult>("/base-model/evaluate-gt", {
      method: "POST",
      body: JSON.stringify(body),
      timeoutMs: 120_000,
    }),
  baseModelApply: (candidate_id: string) =>
    request<{ ok: boolean; hf_model_id?: string; miner_folder?: string; custom?: boolean }>(
      "/base-model/apply",
      {
        method: "POST",
        body: JSON.stringify({ candidate_id }),
        timeoutMs: 60_000,
      },
    ),
  baseModelReset: () =>
    request<{ ok: boolean; hf_model_id?: string; miner_folder?: string | null; custom?: boolean }>(
      "/base-model/reset",
      { method: "POST", timeoutMs: 120_000 },
    ),
  baseModelClearHistory: () =>
    request<DiskClearResult>("/base-model/clear-history", {
      method: "POST",
      body: JSON.stringify({}),
      timeoutMs: 10 * 60_000,
    }),
  getChallenge: (
    index: number,
    fast = true,
    signal?: AbortSignal,
    activate = true,
    reason?: string,
    itemKey?: string | null,
  ) => {
    const why = reason ? `&why=${encodeURIComponent(reason.slice(0, 32))}` : "";
    const ik = itemKey ? `&item_key=${encodeURIComponent(itemKey)}` : "";
    return request<ChallengeDetail>(`/challenges/${index}?fast=${fast}&activate=${activate}${why}${ik}`, {
      signal,
    });
  },
  getChallengeVariants: (index: number, signal?: AbortSignal, itemKey?: string | null) => {
    const ik = itemKey ? `?item_key=${encodeURIComponent(itemKey)}` : "";
    return request<{ reference_variants: Record<string, unknown>[]; reference_boxes: Record<string, unknown>[] }>(
      `/challenges/${index}/variants${ik}`,
      { signal, timeoutMs: 8000 },
    );
  },
  getChallengeFull: (index: number) =>
    request<ChallengeDetail>(`/challenges/${index}?fast=false&activate=false`),
  saveBoxes: (
    index: number,
    boxes: DetectionBox[],
    reviewed = true,
    itemKey?: string | null,
    baseRev?: number | null,
  ) =>
    request<{ ok: boolean; rev?: number; labeled_by?: string | null }>(`/challenges/${index}/boxes`, {
      method: "PUT",
      body: JSON.stringify({
        boxes,
        reviewed,
        ...(itemKey ? { item_key: itemKey } : {}),
        ...(baseRev != null ? { base_rev: baseRev } : {}),
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
  sendReviewedGt: (opts?: { blockMin?: number | null; blockMax?: number | null; quality?: string; itemKeys?: string[] }) =>
    request<{
      ok: boolean;
      n_images: number;
      block_min?: number | null;
      block_max?: number | null;
      item_keys?: string[] | null;
      quality?: string;
      remote?: string;
      host?: string;
      path?: string;
    }>("/export/send-gt", {
      method: "POST",
      body: JSON.stringify({
        block_min: opts?.blockMin ?? null,
        block_max: opts?.blockMax ?? null,
        quality: opts?.quality || "all",
        item_keys: opts?.itemKeys?.length ? opts.itemKeys : null,
      }),
      timeoutMs: DOWNLOAD_GT_TIMEOUT_MS,
    }),
  downloadReviewedGt: async (
    opts?: { blockMin?: number | null; blockMax?: number | null; quality?: string; itemKeys?: string[] },
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
      // A hand-picked list can be long: send it as a POST body instead of the URL.
      res = opts?.itemKeys?.length
        ? await fetch(`${API}/export/reviewed-gt`, {
            method: "POST",
            credentials: "include",
            signal: controller.signal,
            headers: { ...tabSessionHeaders(), "Content-Type": "application/json" },
            body: JSON.stringify({ item_keys: opts.itemKeys, quality: "all" }),
          })
        : await fetch(`${API}/export/reviewed-gt${qs ? `?${qs}` : ""}`, {
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
  enterTopic: (
    slug: string,
    opts?: { workspace?: WorkspaceMode | null; preferItemKey?: string | null },
  ) => {
    const q = new URLSearchParams();
    if (opts?.workspace) q.set("workspace", opts.workspace);
    if (opts?.preferItemKey) q.set("prefer_item_key", opts.preferItemKey);
    const qs = q.toString();
    return request<AppState & { topic?: TopicInfo }>(
      "/topics/" + encodeURIComponent(slug) + "/enter" + (qs ? `?${qs}` : ""),
      { method: "POST", timeoutMs: ENTER_TOPIC_TIMEOUT_MS },
    );
  },
  warmMediaCache: (preferIndex?: number) =>
    request<{ ok: boolean; loaded: number; warming: boolean }>(
      "/media/warm-cache" + (preferIndex != null ? `?prefer_index=${preferIndex}` : ""),
      { method: "POST" },
    ),
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
  clearFindBestHistory: () =>
    request<DiskClearResult>("/find-best/clear-history", {
      method: "POST",
      body: JSON.stringify({}),
      timeoutMs: 30 * 60_000,
    }),
  getDiskUsage: () => request<DiskUsageSummary>("/disk/usage", { timeoutMs: 120_000 }),
  clearDiskHistory: (part: string, keep_job_id?: string) =>
    request<DiskClearResult>("/disk/clear-history", {
      method: "POST",
      body: JSON.stringify({ part, keep_job_id: keep_job_id || null }),
      timeoutMs: 30 * 60_000,
    }),
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
        item_keys: body.itemKeys?.length ? body.itemKeys : null,
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
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
  },
  pushFindBestToHf: (historyId: string, body: { token: string; repo: string }) =>
    request<{
      ok: boolean;
      history_id: string;
      repo_id: string;
      created: boolean;
      url: string;
    }>(`/find-best/history/${encodeURIComponent(historyId)}/push-hf`, {
      method: "POST",
      body: JSON.stringify(body),
      timeoutMs: 600_000,
    }),
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
  /** Hand-picked images (replaces the block range). */
  itemKeys?: string[];
  element?: string;
  days?: number;
  top_k?: number;
  min_cosine?: number;
  n_images?: number;
  blockMin?: number | null;
  blockMax?: number | null;
  quality?: "all" | "pass" | "fail";
};

export type DiskUsagePart = {
  path: string;
  bytes: number;
  human: string;
  entries: number;
};

export type DiskUsageSummary = {
  ok: boolean;
  total_bytes: number;
  total_human: string;
  parts: Record<string, DiskUsagePart>;
};

export type DiskClearResult = {
  ok: boolean;
  part?: string;
  removed?: number;
  kept?: number;
  history_removed?: number;
  history_kept?: number;
  runs_removed?: number;
  runs_kept?: number;
  freed_bytes?: number;
  freed_human?: string;
  skipped_busy?: string[];
  usage?: DiskUsageSummary;
};

export type FindBestStepSummary = {
  headline?: string;
  metrics?: { label: string; value: string }[];
  rows?: {
    title: string;
    subtitle?: string;
    meta?: string;
    rank?: number;
    score?: number;
    bar?: number | null;
  }[];
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
  label?: string | null;
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
  /** The network's own confidence before the miner's gate / cluster boost. */
  raw_conf?: number | null;
};

export type FindBestEvalClassCount = {
  cls: number;
  name: string;
  count: number;
};

export type FindBestEvalImage = {
  name: string;
  width: number;
  height: number;
  n_boxes: number;
  boxes: FindBestEvalBox[];
  class_counts?: FindBestEvalClassCount[];
};

export type FindBestEvalResult = {
  history_id?: string;
  candidate_id?: string;
  hf_repo?: string;
  class_names: string[];
  inference?: string;
  inference_path?: string;
  n_images: number;
  n_boxes: number;
  elapsed_ms: number;
  skipped: string[];
  classes: { cls: number; name: string; count: number }[];
  images: FindBestEvalImage[];
};

export type BaseModelCandidate = {
  id: string;
  hf_repo: string;
  folder?: string;
  topic_slug?: string;
  ok?: boolean;
  created_at?: string;
  inference?: string;
  miner_class_names?: string[];
  verify?: {
    ok?: boolean;
    error?: string | null;
    warnings?: string[];
    class_names?: string[];
    has_chute_config?: boolean;
    inference?: string;
    miner?: { ok?: boolean; class_names?: string[]; n_classes?: number };
    onnx?: { ok?: boolean; size_bytes?: number; inputs?: unknown[]; outputs?: unknown[] };
  };
};

export type BaseModelPassImage = {
  mode: "challenge" | "local" | string;
  index: number;
  block: number;
  item_key?: string | null;
  checks_passed?: number | null;
  checks_total?: number | null;
  has_r123?: boolean;
};

export type BaseModelStatus = {
  hf_model_id?: string | null;
  miner_folder?: string | null;
  custom?: boolean;
  override?: { hf_repo?: string; folder?: string; applied_at?: string; custom?: boolean } | null;
  topic_slug?: string;
  class_names?: string[];
  pass_blocks?: number[];
  pass_images?: BaseModelPassImage[];
  review_mode?: string;
};

export type BaseModelRVariant = {
  rank: number;
  label: string;
  source?: string;
  n_boxes: number;
  boxes: FindBestEvalBox[];
  is_winner?: boolean;
  score?: {
    score?: number | null;
    composite?: number | null;
    map50?: number | null;
    ffpi?: number | null;
    false_positive?: number | null;
  };
  platform?: {
    score?: number | null;
    total_weighted?: number | null;
    map50?: number | null;
    false_positive?: number | null;
    composite_score?: number | null;
    source?: string | null;
  };
};

export type BaseModelGtEvalResult = {
  candidate_id: string;
  hf_repo: string;
  block: number;
  width: number;
  height: number;
  elapsed_ms: number;
  class_names: string[];
  miner_class_names?: string[];
  inference?: string;
  inference_path?: string;
  n_boxes: number;
  boxes: FindBestEvalBox[];
  gt_boxes?: FindBestEvalBox[];
  n_gt?: number;
  class_counts?: FindBestEvalClassCount[];
  image_data_url?: string;
  image_url?: string;
  index?: number;
  r123?: BaseModelRVariant[];
  mode?: string;
  item_key?: string | null;
  checks_passed?: number | null;
  checks_total?: number | null;
  score?: {
    score?: number | null;
    composite?: number | null;
    map50?: number | null;
    ffpi?: number | null;
    false_positive?: number | null;
    platform_score?: number | null;
    platform_map50?: number | null;
    platform_fp?: number | null;
  };
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
  authors_store_count?: number;
  class_match?: string;
  job: FindBestJob | null;
  history: FindBestHistoryEntry[];
};

export type TopicInfo = {
  slug: string;
  title: string;
  subtitle?: string;
  element_id: string;
  element_path: string;
  objects?: string[];
  evaluation_count: number;
  first_block?: number;
  latest_block?: number;
  local_images: number;
  platform_images?: number;
  preview_ready?: boolean;
  preview_url: string;
  active?: boolean;
  status?: "active" | "completed" | "upcoming" | string;
  label_ready?: boolean;
  can_label?: boolean;
};

export type CollectorSubmission = {
  id: number;
  model: string;
  revision: string;
  element_id: string | null;
  hotkeys: string[];
  first_block: number;
  commit_ts: number;
  first_seen: number;
  status: "waiting_public" | "queued_eval" | "evaluating" | "passed" | "rejected" | "expired" | "duplicate" | "error" | "deleted";
  detail: string;
  deadline: number;
  attempts: number;
  onnx_size: number | null;
  onnx_sha: string | null;
  duplicate_of: number | null;
  best_s: number | null;
  updated_at: number;
  eval_id: number | null;
  mean_s: number | null;
  mean_m: number | null;
  mean_f: number | null;
  p50_ms: number | null;
  eval_p95: number | null;
  /** p95 as measured, and the reference winner's time in the same run (set when the box was busy). */
  p95_raw?: number | null;
  ref_p95?: number | null;
  baseline_p: number | null;
  n_images: number | null;
  blocks_min: number | null;
  blocks_max: number | null;
  eval_reason: string | null;
  eval_kind: string | null;
  eval_at: number | null;
  eval_passed: number | null;
  n_evals: number;
};

export type CollectorEvaluation = {
  id: number;
  kind: string;
  set_spec: Record<string, unknown>;
  set_hash: string;
  n_images: number;
  blocks_min: number;
  blocks_max: number;
  mean_s: number | null;
  mean_m: number | null;
  mean_f: number | null;
  p50_ms: number | null;
  p95_ms: number | null;
  max_ms: number | null;
  load_s: number | null;
  baseline_p: number | null;
  passed: number;
  reason: string;
  runtime_s: number | null;
  created_at: number;
  per_image?: Record<string, { s: number; m: number; f: number; n_gt: number; n_pred: number; ms: number | null }>;
};

export type CollectorOverview = {
  counts: Record<string, number>;
  total: number;
  service_alive: boolean;
  service_seen_s: number | null;
  chain_head: number;
  chain_polled_at: number | null;
  gate: { min_s: number; max_p95_ms: number; ref_idle_ms?: number; n_images: number; max_onnx_mb: number; public_wait_min: number; backfill_days: number };
  gt: { count: number; newest: number | null; oldest: number | null; latest_n_range: [number, number] | null };
  best: { id: number; model: string; best_s: number; p95_ms: number } | null;
  /** What the scores in the table were last measured on. */
  scores_basis: { kind: "latest" | "custom"; label: string; n: number; blocks_min: number | null; blocks_max: number | null; complete: boolean; at: number } | null;
  /** The working image set: every measurement, deep post-process and push uses it. */
  image_set: Partial<CollectorImageSet>;
  /** A bulk re-measure (chosen-set run or the nightly one) that is running right now. */
  bulk: {
    kind: "custom" | "daily";
    id: number | null;
    progress: string;
    status: string;
    current_id: number | null;
    done: number;
    total: number;
    label: string;
  } | null;
};

export type CollectorImageSet = {
  mode: "latest" | "blocks";
  n: number;
  label: string;
  blocks_min: number | null;
  blocks_max: number | null;
  blocks: number[];
  custom: boolean;
  default_n: number;
};

export type CollectorVariant = {
  id: number;
  submission_id: number;
  parent_id: number | null;
  stage: "original" | "dpp" | "hash";
  label: string;
  onnx_sha: string | null;
  n_images: number | null;
  set_label: string;
  s: number | null;
  p95_ms: number | null;
  ok: number;
  reason: string;
  info: {
    s_before?: number;
    p95_before?: number;
    from?: number;
    nudge?: unknown;
    dpp?: {
      baseline_score?: number;
      final_score?: number;
      postproc_applied?: boolean;
      params?: unknown;
      runtime_s?: number;
      latency_reverted?: boolean;
      latency_rule?: { limit_ms?: number; baseline_p95_ms?: number; after_p95_ms?: number; ok?: boolean; reverted?: boolean };
    };
  };
  pushed: Array<{ repo: string; url: string; account: string; at: number; s?: number; p95_ms?: number }>;
  has_files: boolean;
  created_at: number;
};

export type CollectorSchedule = {
  settings: { enabled: number; start_min: number; length_min: number; utc_offset_min: number };
  in_window: boolean;
  next_start: number;
  local_time: string;
  n_passed: number;
  state: {
    date?: string;
    total?: number;
    done?: number[];
    failed?: Array<{ id: number; error: string }>;
    running?: boolean;
    complete?: boolean;
    current?: string;
    started_at?: number;
    finished_at?: number | null;
    manual?: boolean;
  };
};

export type TryBox = { x1: number; y1: number; x2: number; y2: number; cls_id: number; conf: number | null };
export type TryModelResult = {
  id: number;
  model: string;
  revision?: string;
  score?: number | null;
  ok: boolean;
  error: string;
  load_s?: number | null;
  boxes?: Record<string, TryBox[] | null>;
  ms?: number[];
};
export type TryResult = {
  images: Array<{ name: string; file: string; w: number; h: number; scale: number }>;
  classes: string[];
  models: TryModelResult[];
  total: number;
  done: number;
};

export type CollectorCommand = {
  id: number;
  kind: string;
  status: "queued" | "running" | "done" | "error";
  progress: string;
  result: Record<string, unknown>;
};

export type KeysView = {
  hf: Array<{
    id: string;
    account: string;
    orgs: string[];
    masked: string;
    verified_at: number | null;
    scan: { at?: number; n?: number; error?: string };
    n_models: number;
  }>;
  chutes: { configured: boolean; masked: string };
  ultralytics: { configured: boolean; masked: string };
  openai?: { configured: boolean; masked: string };
  gemini?: { configured: boolean; masked: string };
  openrouter?: { configured: boolean; masked: string };
  updated_at: number | null;
  updated_by: string | null;
};

export type KeyService = "chutes" | "ultralytics" | "openai" | "gemini" | "openrouter";

export type HfModelRow = {
  repo_id: string;
  private: boolean;
  last_modified: string | null;
  account_id: string;
  account: string;
};

export type AuthStatus = {
  enabled: boolean;
  authenticated: boolean;
  email?: string | null;
  username?: string | null;
  color?: string | null;
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
  item_keys?: string[] | null;
  /** Own: whole calendar days (YYYY-MM-DD) — send several days as one dataset. */
  own_days?: string[] | null;
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
  /** One entry per archive: exports over the 10 GB limit are split into several. */
  parts?: Array<{ part: number; packed: number; bytes?: number; filename: string }>;
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
    style?: string;
    mode?: string;
    n_trials?: number;
    final_live?: { score?: number; map50?: number; ffpi?: number };
    /** Robust search: scores on images the search never tuned on (the honest gain). */
    holdout?: {
      n_train: number;
      n_holdout: number;
      before: number;
      after: number;
      delta: number;
      delta_se?: number | null;
      candidate_delta: number;
    };
    warning?: string | null;
    postproc_applied?: boolean;
    reverted?: boolean;
    latency?: {
      p95_ms?: number;
      p50_ms?: number;
      mean_ms?: number;
      gate_ms?: number;
      passed?: boolean;
      /** Load-aware timing: p95_ms is then the idle-equivalent; this is what was measured while busy. */
      normalized?: boolean;
      p95_raw_ms?: number;
      ref_ms?: number;
    };
  } | null;
  progress?: {
    phase?: string;
    trial?: number;
    total?: number;
    pct?: number;
    score?: number | null;
    best?: number | null;
    style?: string;
    message?: string;
    map50?: number;
    ffpi?: number;
  } | null;
  compare?: DppCompareResult | null;
  error?: string | null;
  ship_ready?: boolean;
  n_gt?: number;
};

export type ManualMergeHistoryEntry = {
  id: string;
  topic_slug?: string | null;
  status?: string;
  message?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  score?: number | null;
  map50?: number | null;
  n_gt?: number | null;
  block_min?: number | null;
  block_max?: number | null;
  item_keys?: string[] | null;
  quality?: string | null;
  n_models?: number;
  has_ship?: boolean;
  onnx_size?: number;
  error?: string | null;
};

export type ManualMergeModel = {
  slug: string;
  repo_id: string;
  revision: string;
  path: string;
  score?: number | null;
  map50?: number | null;
  ffpi?: number | null;
  error?: string | null;
  status?: string;
  n_images?: number;
};

export type ManualMergePairResult = {
  a: string;
  b: string;
  a_score?: number | null;
  b_score?: number | null;
  n_tried: number;
  n_ok: number;
  best_score?: number | null;
  best_name?: string | null;
  best_path?: string | null;
  best_method?: string | null;
  best_weights?: number[] | null;
  top_mixes?: {
    name?: string;
    score?: number;
    map50?: number;
    method?: string;
    weights?: number[];
  }[];
  errors?: string[];
};

export type MergeMachine = {
  host: string;
  state: string; // free | busy | offline | setup | checking | running | standby
  cores?: number;
  busy_pct?: number | null;
  slots?: number;
  running?: number;
  done?: number;
  note?: string;
};

export type ManualMergeJob = {
  id: string;
  topic_slug: string;
  class_names: string[];
  status: string;
  message: string;
  error?: string | null;
  block_min?: number | null;
  block_max?: number | null;
  item_keys?: string[] | null;
  quality?: string;
  n_gt: number;
  models: ManualMergeModel[];
  pair_results: ManualMergePairResult[];
  best?: {
    score?: number;
    map50?: number;
    ffpi?: number;
    name?: string;
    members?: string[];
    method?: string;
    weights?: number[];
    template?: string;
    path?: string;
    baseline?: number;
    recipe?: string;
    best_solo_repo?: string | null;
    best_solo_score?: number | null;
    delta_vs_best_solo?: number | null;
    models_before?: {
      repo_id?: string;
      revision?: string;
      score?: number | null;
      map50?: number | null;
      ffpi?: number | null;
    }[];
    postproc?: boolean;
    score_before_dpp?: number;
    delta_dpp?: number;
  } | null;
  postproc?: Record<string, unknown> | null;
  machines?: MergeMachine[];
  progress?: {
    phase?: string;
    i?: number;
    total?: number;
    pct?: number;
    pair?: string[];
    pair_scores?: Array<number | null | undefined>;
    pair_best?: number | null;
    best_so_far?: number | null;
    best_so_far_name?: string | null;
    best_solo?: number | null;
    headline?: string;
    best?: number;
    message?: string;
  } | null;
  ship_ready?: boolean;
  n_pairs_planned?: number;
};

export type ManualMergeGtImage = {
  stem: string;
  filename: string;
  block?: number | null;
  has_label?: boolean;
};

export type ManualMergeImageScore = {
  composite: number;
  map50: number;
  false_positive: number;
  ffpi: number;
  tp: number;
  fp: number;
  fn: number;
  n_gt: number;
  n_pred: number;
};

export type ManualMergeEvalGtResult = {
  stem: string;
  width: number;
  height: number;
  class_names: string[];
  n_gt?: number;
  ours: { boxes: DetectionBox[]; n_boxes: number; label: string; score?: ManualMergeImageScore };
  r1: {
    boxes: DetectionBox[];
    n_boxes: number;
    label: string;
    score?: ManualMergeImageScore;
    platform?: { score?: number | null; map50?: number | null; fp?: number | null; source?: string | null };
  };
  delta_composite?: number;
  best?: ManualMergeJob["best"];
};

export type HfModelSuggestion = {
  id: string;
  downloads?: number | null;
  likes?: number | null;
  source?: string;
  private?: boolean | null;
};

export type HfCompareResultRow = {
  hf_repo: string;
  ok?: boolean;
  error?: string;
  score?: number | null;
  map50?: number | null;
  ffpi?: number | null;
  fp_pillar?: number | null;
  n_pred?: number | null;
  n_images?: number | null;
  latency_mean_ms?: number | null;
  latency_p50_ms?: number | null;
  latency_p95_ms?: number | null;
  latency_passed?: boolean | null;
  latency_gate_ms?: number | null;
  pack_dir?: string | null;
};

/** Own "field": the images one generator made. */
export type OwnField = "openai" | "gemini";
export type OwnDayCounts = { total: number; new: number; sam3?: number; done: number };

export type OwnDayRow = OwnDayCounts & {
  date: string;
  /** Per generator ("openai" / "gemini" / "other"). */
  fields?: Record<string, OwnDayCounts>;
};

export type OwnDaysResponse = {
  days: OwnDayRow[];
  timezone: string;
  own_day?: string | null;
  own_field?: OwnField | null;
  today?: string;
};

export type OwnHfImportConflict = {
  dest_name: string;
  stem: string;
  existing_name?: string | null;
  has_label?: boolean;
};

export type OwnHfImportResult = {
  added: number;
  replaced: number;
  skipped: number;
  labels_written: number;
  total: number;
};

export type OwnHfImportJob = {
  id: string;
  url: string;
  status: string;
  message: string;
  error?: string | null;
  running?: boolean;
  archive_name?: string;
  n_images?: number;
  n_labels?: number;
  classes?: string[];
  conflicts?: OwnHfImportConflict[];
  new_count?: number;
  conflict_count?: number;
  import_result?: OwnHfImportResult | null;
};

export type HfComparePpPush = {
  status?: string;
  message?: string;
  error?: string | null;
  hf_repo?: string;
  target_repo?: string;
  baseline_score?: number | null;
  pp_score?: number | null;
  nudge?: {
    ship_id?: string;
    sha256_before?: string;
    sha256_after?: string;
    date?: string;
  } | null;
  url?: string | null;
  created?: boolean;
  progress?: {
    phase?: string;
    pct?: number;
    message?: string;
    trial?: number;
    total?: number;
    score?: number | null;
    best?: number | null;
  };
};

export type HfCompareJob = {
  id: string;
  topic_slug: string;
  class_names: string[];
  repos: string[];
  status: string;
  message: string;
  error?: string | null;
  block_min?: number | null;
  block_max?: number | null;
  item_keys?: string[] | null;
  quality?: string;
  n_gt?: number;
  results: HfCompareResultRow[];
  running?: boolean;
  has_token?: boolean;
  pp_push?: HfComparePpPush;
  pp_running?: boolean;
};

export type HfCompareStartRequest = {
  repos: string[];
  block_min?: number | null;
  block_max?: number | null;
  item_keys?: string[] | null;
  quality?: "all" | "pass" | "fail" | string;
  n_images?: number;
  hf_token?: string;
};

export type HfCompareDeepPpPushRequest = {
  hf_repo: string;
  target_repo: string;
  hf_token?: string;
  passes?: number;
  random_n?: number;
};
export type DppRunRequest = {
  job_id: string;
  block_min?: number | null;
  block_max?: number | null;
  item_keys?: string[] | null;
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
  item_keys?: string[] | null;
  quality?: string;
  n_images?: number;
};

export type DppHistoryEntry = {
  id: string;
  topic_slug?: string;
  status?: string;
  created_at?: string;
  updated_at?: string;
  score?: number | null;
  baseline_score?: number | null;
  delta?: number | null;
  map50?: number | null;
  n_gt?: number;
  block_min?: number | null;
  block_max?: number | null;
  item_keys?: string[] | null;
  quality?: string;
  has_ship?: boolean;
  onnx_size?: number;
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
