import type {
  AnalyzeJob,
  AnalyzeNeeded,
  CkptJob,
  ClearPreview,
  Generator,
  GtCandidate,
  Distribution,
  Gallery,
  GenItem,
  GenJob,
  GenMode,
  Health,
  MiningReport,
  ModelErrors,
  Package,
  RecordRow,
  RunInsight,
  Run,
  SendJob,
  StudioStatus,
  Summary,
  SwitchName,
} from "./types";

/** Studio API through the labeler: /api/studio/<path> → Studio /api/<path> (login + keys handled server-side). */
const BASE = "/api/studio";

async function call<T>(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), init?.timeoutMs ?? 60_000);
  try {
    const res = await fetch(path, {
      ...init,
      credentials: "include",
      signal: controller.signal,
      headers: init?.body instanceof FormData ? init?.headers : { "Content-Type": "application/json", ...(init?.headers || {}) },
    });
    const text = await res.text();
    let body: unknown = {};
    try {
      body = text ? JSON.parse(text) : {};
    } catch {
      body = { detail: text.slice(0, 300) };
    }
    if (!res.ok) {
      if (res.status === 401) window.dispatchEvent(new CustomEvent("labeler-auth-required"));
      const detail = (body as { detail?: unknown }).detail;
      throw new Error(typeof detail === "string" ? detail : `HTTP ${res.status}`);
    }
    return body as T;
  } catch (e) {
    if ((e as Error).name === "AbortError") throw new Error("The Studio did not answer in time");
    throw e;
  } finally {
    window.clearTimeout(timer);
  }
}

const get = <T,>(p: string, timeoutMs?: number) => call<T>(`${BASE}${p}`, { timeoutMs });
const post = <T,>(p: string, body?: unknown, timeoutMs?: number) =>
  call<T>(`${BASE}${p}`, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body), timeoutMs });

/** Image URL helpers (the server snaps w to 320/480/640/800/1024). */
export const img = {
  gt: (block: number, kind: "thumb" | "preview") => `/api/studio-admin/gt-image/${block}?kind=${kind}`,
  dataset: (fn: string, w?: number) => `${BASE}/dataset/image/${encodeURIComponent(fn)}${w ? `?w=${w}` : ""}`,
  annotated: (fn: string, w?: number) => `${BASE}/image/${encodeURIComponent(fn)}/annotated${w ? `?w=${w}` : ""}`,
  overlay: (fn: string, version: string, w?: number) =>
    `${BASE}/model-errors/${encodeURIComponent(fn)}/overlay?v=${encodeURIComponent(version)}${w ? `&w=${w}` : ""}`,
  gen: (run: string, fn: string, w?: number) => `${BASE}/gen/image/${encodeURIComponent(fn)}?run=${encodeURIComponent(run)}${w ? `&w=${w}` : ""}`,
  raw: (run: string, fn: string, w?: number) => `${BASE}/gen/raw/${encodeURIComponent(fn)}?run=${encodeURIComponent(run)}${w ? `&w=${w}` : ""}`,
  labels: (run: string, fn: string) => `${BASE}/gen/labels/${encodeURIComponent(fn)}?run=${encodeURIComponent(run)}`,
  source: (run: string, fn: string, w?: number) =>
    `${BASE}/gen/source/${encodeURIComponent(fn)}?run=${encodeURIComponent(run)}${w ? `&w=${w}` : ""}`,
};

export const studio = {
  status: () => call<StudioStatus>("/api/studio-admin/status", { timeoutMs: 20_000 }),
  saveSettings: (body: { allowed_users?: string[]; hf_repo?: string }) =>
    call<StudioStatus>("/api/studio-admin/settings", { method: "PUT", body: JSON.stringify(body) }),
  restart: () => call<{ ok: boolean }>("/api/studio-admin/restart", { method: "POST" }),
  /** latest_n: the newest N Done images; item_keys: exactly these images (from gtCandidates). */
  syncGt: (analyze: boolean, pick: { latest_n: number } | { item_keys: string[] }) =>
    call<{ ok: boolean; n_images: number; gt_images: number; gt_updated_at: number | null; analyze?: { error?: string } }>("/api/studio-admin/sync-gt", {
      method: "POST",
      body: JSON.stringify({ analyze, ...pick }),
      timeoutMs: 10 * 60_000,
    }),
  gtCandidates: () => call<{ items: GtCandidate[] }>("/api/studio-admin/gt-candidates", { timeoutMs: 120_000 }),
  /** {block: openai | gemini} for every classified Platform frame. */
  generators: () => call<{ blocks: Record<string, Generator> }>("/api/studio-admin/generators", { timeoutMs: 30_000 }),

  health: () => get<Health>("/health", 30_000),
  analyzeJob: () => get<AnalyzeJob>("/job"),
  analyze: (mode: GenMode) => post<AnalyzeJob>("/analyze", { mode }),
  analyzeNeeded: () => get<AnalyzeNeeded>("/analyze/needed", 30_000),
  setMode: (mode: GenMode) => post<{ gen_mode: GenMode }>("/gen-mode", { mode }),
  summary: () => get<Summary>("/summary"),
  galleries: () => get<Record<string, Gallery>>("/galleries"),
  overlaps: () => get<Record<string, Gallery>>("/overlaps"),
  mining: () => get<MiningReport>("/mining-report"),
  records: (regime?: string | null, q?: string) => {
    const p = new URLSearchParams({ limit: "120" });
    if (regime) p.set("regime", regime);
    if (q) p.set("q", q);
    return get<{ items: RecordRow[]; total?: number }>(`/records?${p}`);
  },
  modelErrors: () => get<ModelErrors>("/model-errors", 60_000),

  ckptJob: () => get<CkptJob>("/detector/job"),
  uploadCkpt: (mode: GenMode, file: File) => {
    const fd = new FormData();
    fd.append("file", file, file.name);
    return call<CkptJob>(`${BASE}/detector/${mode}/upload`, { method: "POST", body: fd, timeoutMs: 15 * 60_000 });
  },
  resetCkpt: (mode: GenMode) => post<{ ok: boolean }>(`/detector/${mode}/reset`),
  setSwitch: (name: SwitchName, enabled: boolean) => post<Record<SwitchName, boolean>>(`/checkpoints/${name}`, { enabled }),

  importChunk: (files: { file: File; rel: string }[], opts: { replace: boolean }) => {
    const fd = new FormData();
    for (const f of files) fd.append("files", f.file, f.rel);
    const q = new URLSearchParams({ upload_id: "beverage_gt", workers: "8", target: "beverage_gt", replace: String(opts.replace) });
    return call<{ images: number; labels: number }>(`${BASE}/import-upload?${q}`, { method: "POST", body: fd, timeoutMs: 10 * 60_000 });
  },
  importFinalize: () => {
    const q = new URLSearchParams({ upload_id: "beverage_gt", workers: "8", target: "beverage_gt", finalize: "true" });
    return call<{ images: number; labels: number; job: AnalyzeJob }>(`${BASE}/import-upload?${q}`, { method: "POST", timeoutMs: 10 * 60_000 });
  },

  distribution: () => get<Distribution>("/gen/distribution"),
  prepare: (per_source: number) => post<Package & { quotas?: unknown[] }>("/gen/prepare", { per_source }, 10 * 60_000),
  pkg: (blockOnly = true) => get<Package>(`/gen/package?block_only=${blockOnly}`),
  genJob: () => get<GenJob>("/gen/job"),
  start: (body: {
    mode: "test" | "final";
    model: string;
    per_source: number;
    source_image: string | null;
    /** Final only: run on just these sources (null → every source in the package). */
    source_images?: string[] | null;
    quality?: string | null;
    input_fidelity?: string | null;
  }) => post<GenJob>("/gen/start", { ...body, source_index: null, cctv_degrade: false }, 120_000),
  collect: (run: string) => post<GenJob>("/gen/collect", { run }, 120_000),
  cancel: (run: string) => post<{ status: string }>("/gen/cancel", { run }, 120_000),
  autolabel: (run: string) => post<GenJob>("/gen/autolabel", { run }),
  runs: () => get<{ runs: Run[] }>("/gen/runs"),
  insight: (run: string) => get<RunInsight>(`/gen/insight?run=${encodeURIComponent(run)}`, 60_000),
  runImages: (run: string) => get<{ run: string; images: GenItem[] }>(`/gen/images?run=${encodeURIComponent(run)}`),
  deleteImages: (run: string, filenames: string[]) =>
    post<{ deleted: number; remaining: number }>("/gen/delete-images", { run, filenames }),
  uploadHf: (run: string | null) =>
    post<{ ok: boolean; url: string; repo_id: string; path_in_repo: string; n_images: number }>("/gen/upload-hf", { run }, 30 * 60_000),
  send: (run: string, filenames: string[] | null) => post<SendJob>("/gen/send-remote", { run, filenames }),
  sendJob: () => get<SendJob>("/gen/send-job"),
  clearPreview: () => get<ClearPreview>("/studio/clear-preview"),
  clear: () => post<{ ok: boolean }>("/studio/clear", { beverage_gt: true, history: true, analysis: true, confirm: "CLEAR" }),
};
