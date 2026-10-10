import HfPushTarget from "./HfPushTarget";
import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import {
  Check,
  GitCompareArrows,
  Loader2,
  Plus,
  Rocket,
  Search,
  Square,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { api, type HfCompareJob, type HfModelSuggestion } from "../api";
import type { ChallengeItem } from "../types";
import { checksFailComplete, checksPassComplete } from "../labelChecks";
import { cn } from "../lib/utils";
import ImageSelectionField from "./ImageSelectionField";

const HIST_BINS = 48;
const RECENT_KEY = "labeler_hf_compare_recent";

function normalizeRepo(raw: string): string {
  let value = raw.trim().replace(/^https?:\/\//i, "");
  if (value.includes("huggingface.co/")) {
    value = value.split("huggingface.co/")[1] || value;
    value = value.split("/tree/")[0].split("/blob/")[0].split("/resolve/")[0];
  }
  if (value.startsWith("models/")) value = value.slice("models/".length);
  return value.replace(/^\/+|\/+$/g, "");
}

function validRepo(id: string): boolean {
  if (!id.includes("/")) return false;
  const [owner, name] = id.split("/", 2);
  if (!owner || !name) return false;
  if (/\s/.test(id)) return false;
  return /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(id);
}

function formatMs(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return `${n.toFixed(1)} ms`;
}

function formatScore(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return n.toFixed(4);
}

function loadRecent(): string[] {
  try {
    const raw = localStorage.getItem(RECENT_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((x) => typeof x === "string" && validRepo(x)).slice(0, 24) : [];
  } catch {
    return [];
  }
}

function saveRecent(repos: string[]) {
  try {
    const prev = loadRecent();
    const merged = [...repos, ...prev.filter((r) => !repos.includes(r))].slice(0, 24);
    localStorage.setItem(RECENT_KEY, JSON.stringify(merged));
  } catch {
    /* ignore */
  }
}

function suggestTargetRepo(source: string): string {
  const id = normalizeRepo(source);
  if (!validRepo(id)) return "";
  const [owner, name] = id.split("/", 2);
  const base = name.replace(/-pp$/i, "");
  return `${owner}/${base}-pp`;
}

export default function CompareHfModelsModal({
  open,
  onClose,
  challenges,
  ownMode = false,
}: {
  open: boolean;
  onClose: () => void;
  challenges: ChallengeItem[];
  ownMode?: boolean;
}) {
  const labeled = useMemo(
    () => challenges.filter((c) => c.loaded && c.reviewed),
    [challenges],
  );
  const blocks = useMemo(
    () => labeled.map((c) => c.block).filter((b) => b > 0).sort((a, b) => a - b),
    [labeled],
  );
  const minBlock = blocks[0] ?? 0;
  const maxBlock = blocks[blocks.length - 1] ?? 0;

  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [quality, setQuality] = useState<"all" | "pass" | "fail">("all");
  const [picked, setPicked] = useState<string[] | null>(null);
  const [repos, setRepos] = useState<string[]>([]);
  const [draft, setDraft] = useState("");
  const [suggestions, setSuggestions] = useState<HfModelSuggestion[]>([]);
  const [recent, setRecent] = useState<string[]>([]);
  const [activeSuggest, setActiveSuggest] = useState(0);
  const [suggestBusy, setSuggestBusy] = useState(false);
  // Hugging Face tokens come from the shared key vault on the server (nothing to type here).
  const [ppAccount, setPpAccount] = useState("");
  const [ppRepoName, setPpRepoName] = useState("");
  const [job, setJob] = useState<HfCompareJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [minGt, setMinGt] = useState(10);
  const [ppSource, setPpSource] = useState("");
  const [ppTarget, setPpTarget] = useState("");
  const [ppBusy, setPpBusy] = useState(false);
  const [clearBusy, setClearBusy] = useState(false);
  const [clearMsg, setClearMsg] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const suggestTimer = useRef<number | null>(null);

  useEffect(() => {
    if (!open) return;
    setFrom(minBlock ? String(minBlock) : "");
    setTo(maxBlock ? String(maxBlock) : "");
    setQuality("all");
    setPicked(null);
    setRepos([]);
    setDraft("");
    setSuggestions([]);
    setJob(null);
    setError("");
    setBusy(false);
    setPpSource("");
    setPpTarget("");
    setPpRepoName("");
    setPpBusy(false);
    setRecent(loadRecent());
    void api.hfCompareDefaults().then((d) => setMinGt(d.min_gt)).catch(() => undefined);
    // Seed browse list with a broad query so the picker isn't empty.
    void api
      .hfCompareSearch("miner", 24)
      .then((res) => setSuggestions(res.models || []))
      .catch(() => undefined);
  }, [open, minBlock, maxBlock]);

  const fromN = Number(from);
  const toN = Number(to);
  const rawLo = Number.isFinite(fromN) ? fromN : minBlock;
  const rawHi = Number.isFinite(toN) ? toN : maxBlock;
  const lo = Math.min(rawLo, rawHi);
  const hi = Math.max(rawLo, rawHi);
  const span = Math.max(1, maxBlock - minBlock);
  const inRange = labeled.filter((c) => c.block >= lo && c.block <= hi);
  const passCount = inRange.filter((c) => checksPassComplete(c)).length;
  const failCount = inRange.filter((c) => checksFailComplete(c)).length;
  const rangeCount = ownMode || quality === "all" ? inRange.length : quality === "pass" ? passCount : failCount;
  const count = picked ? picked.length : rangeCount;

  const hist = useMemo(() => {
    const bins = new Array(HIST_BINS).fill(0);
    for (const c of labeled) {
      const t = (c.block - minBlock) / span;
      const i = Math.min(HIST_BINS - 1, Math.max(0, Math.floor(t * HIST_BINS)));
      bins[i] += 1;
    }
    const peak = Math.max(1, ...bins);
    return bins.map((n) => n / peak);
  }, [labeled, minBlock, span]);

  const setLo = (n: number) => setFrom(String(Math.min(Math.max(n, minBlock), hi)));
  const setHi = (n: number) => setTo(String(Math.max(Math.min(n, maxBlock), lo)));

  const addRepo = useCallback((raw: string) => {
    const id = normalizeRepo(raw);
    if (!validRepo(id)) {
      setError(`Invalid model id "${raw}". Use owner/name (e.g. Realfencer/car1).`);
      return false;
    }
    setRepos((prev) => (prev.includes(id) ? prev : [...prev, id]));
    setDraft("");
    setError("");
    setRecent((prev) => [id, ...prev.filter((r) => r !== id)].slice(0, 24));
    saveRecent([id]);
    return true;
  }, []);

  const removeRepo = (id: string) => setRepos((prev) => prev.filter((r) => r !== id));

  const fetchSuggestions = useCallback(
    (query: string) => {
      if (suggestTimer.current) window.clearTimeout(suggestTimer.current);
      const q = query.trim() || "miner";
      suggestTimer.current = window.setTimeout(() => {
        setSuggestBusy(true);
        void api
          .hfCompareSearch(q, 30)
          .then((res) => {
            setSuggestions(res.models || []);
            setActiveSuggest(0);
          })
          .catch(() => setSuggestions([]))
          .finally(() => setSuggestBusy(false));
      }, 200);
    },
    [],
  );

  useEffect(() => {
    if (!job?.running && !job?.pp_running) return;
    let inflight = false;
    const timer = window.setInterval(() => {
      if (inflight) return;
      inflight = true;
      void api
        .hfCompareStatus(job.id)
        .then((res) => setJob(res.job))
        .catch(() => undefined)
        .finally(() => {
          inflight = false;
        });
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job?.id, job?.running, job?.pp_running]);

  const startCompare = async () => {
    if (repos.length === 0) {
      setError("Add at least one Hugging Face model.");
      return;
    }
    if (count < minGt) {
      setError(`Need at least ${minGt} Done images in range (selected ${count}).`);
      return;
    }
    setBusy(true);
    setError("");
    saveRecent(repos);
    try {
      const res = await api.hfCompareStart({
        repos,
        block_min: lo,
        block_max: hi,
        item_keys: picked ?? undefined,
        quality: ownMode || picked ? "all" : quality,
        n_images: 0,
      });
      setJob(res.job);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setBusy(false);
    }
  };

  const stopCompare = async () => {
    if (!job?.id) return;
    try {
      const res = await api.hfCompareStop(job.id);
      setJob(res.job);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    }
  };

  const pickPpSource = (repo: string) => {
    setPpSource(repo);
    setPpRepoName(suggestTargetRepo(repo).split("/").pop() || "");
    setError("");
  };

  const startDeepPpPush = async () => {
    if (!job?.id) {
      setError("Run compare first.");
      return;
    }
    if (!ppSource) {
      setError("Pick a scored model from the results.");
      return;
    }
    const target = normalizeRepo(ppTarget);
    if (!validRepo(target)) {
      setError("Target repo must be owner/name.");
      return;
    }
    setPpBusy(true);
    setError("");
    try {
      const res = await api.hfCompareDeepPpPush(job.id, {
        hf_repo: ppSource,
        target_repo: target,
      });
      setJob(res.job);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setPpBusy(false);
    }
  };

  const stopDeepPpPush = async () => {
    if (!job?.id) return;
    try {
      const res = await api.hfCompareDeepPpPushStop(job.id);
      setJob(res.job);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    }
  };

  const clearHistoryDisk = async () => {
    if (clearBusy) return;
    const ok = window.confirm(
      "Clear HF Compare history on disk?\n\n" +
        "Deletes finished compare job folders (downloaded models + score caches). Running jobs are kept.\n" +
        "Labels and Own images are never deleted.",
    );
    if (!ok) return;
    setClearBusy(true);
    setClearMsg("");
    try {
      const res = await api.hfCompareClearHistory(job?.id);
      setClearMsg(`Freed ${res.freed_human || "disk"} · removed ${res.removed ?? 0} jobs`);
    } catch (e) {
      setClearMsg(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setClearBusy(false);
    }
  };

  const onDraftKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    const visible = suggestions.filter((s) => !repos.includes(s.id));
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveSuggest((i) => Math.min(i + 1, Math.max(0, visible.length - 1)));
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveSuggest((i) => Math.max(i - 1, 0));
      return;
    }
    if (e.key === "Enter" || e.key === "," || e.key === "Tab") {
      if (visible[activeSuggest]) {
        e.preventDefault();
        addRepo(visible[activeSuggest].id);
        return;
      }
      if (draft.trim()) {
        e.preventDefault();
        const parts = draft.split(/[\s,;]+/).filter(Boolean);
        if (parts.length > 1) parts.forEach((p) => addRepo(p));
        else addRepo(draft);
      }
    }
    if (e.key === "Backspace" && !draft && repos.length) removeRepo(repos[repos.length - 1]);
  };

  if (!open) return null;

  const running = Boolean(job?.running);
  const ppRunning = Boolean(job?.pp_running);
  const pp = job?.pp_push;
  const results = job?.results || [];
  const browse = suggestions.filter((s) => !repos.includes(s.id));
  const recentAvailable = recent.filter((r) => !repos.includes(r));
  const compareDone = Boolean(job && !running && results.some((r) => r.ok));

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/55 p-3 backdrop-blur-[2px]">
      <div className="flex max-h-[94vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-ink-900">
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-700">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-500 text-white">
              <GitCompareArrows className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-sm font-extrabold text-slate-900 dark:text-white">Compare HF models</h2>
              <p className="text-[11px] text-slate-500">
                Done range · pick several packs · score + latency ranked by score
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={clearBusy}
              onClick={() => void clearHistoryDisk()}
              className="inline-flex items-center gap-1 rounded-md border border-rose-200 px-2 py-1 text-[10px] font-bold text-rose-700 hover:bg-rose-50 disabled:opacity-40 dark:border-rose-900/50 dark:text-rose-300"
              title="Delete finished HF compare jobs to free disk"
            >
              {clearBusy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
              Clear history
            </button>
            <button type="button" onClick={onClose} className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
        {clearMsg && (
          <div className="border-b border-slate-200 px-4 py-1.5 text-[10px] text-slate-600 dark:border-slate-700 dark:text-slate-300">
            {clearMsg}
          </div>
        )}

        <div className="min-h-0 flex-1 space-y-4 overflow-auto px-4 py-4">
          <ImageSelectionField suggest={inRange.map((c) => c.item_key)} challenges={challenges} ownMode={ownMode} value={picked} onChange={setPicked} verb="score on" disabled={running} />
          <div className={cn("space-y-4", picked && "pointer-events-none opacity-40")}>
          <div className="grid grid-cols-2 gap-2">
            <label className="block">
              <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">
                {ownMode ? "From #" : "From block"}
              </span>
              <input
                value={from}
                onChange={(e) => setFrom(e.target.value.replace(/[^\d]/g, ""))}
                disabled={running}
                className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 font-mono text-sm tabular-nums dark:border-slate-700 dark:bg-ink-800"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">
                {ownMode ? "To #" : "To block"}
              </span>
              <input
                value={to}
                onChange={(e) => setTo(e.target.value.replace(/[^\d]/g, ""))}
                disabled={running}
                className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 font-mono text-sm tabular-nums dark:border-slate-700 dark:bg-ink-800"
              />
            </label>
          </div>

          <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 pb-3 pt-3 dark:border-slate-700 dark:bg-ink-950">
            <div className="mb-2 flex h-8 items-end gap-px">
              {hist.map((h, i) => {
                const binLo = minBlock + (i / HIST_BINS) * span;
                const binHi = minBlock + ((i + 1) / HIST_BINS) * span;
                const active = binHi >= lo && binLo <= hi;
                return (
                  <div
                    key={i}
                    className={cn("min-h-[2px] flex-1 rounded-sm", active ? "bg-indigo-500" : "bg-slate-300 dark:bg-slate-700")}
                    style={{ height: `${Math.max(8, h * 100)}%` }}
                  />
                );
              })}
            </div>
            <label className="mb-2 block">
              <span className="mb-0.5 flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-500">
                From <span className="font-mono tabular-nums text-slate-700 dark:text-slate-200">{lo}</span>
              </span>
              <input type="range" min={minBlock} max={maxBlock} value={lo} disabled={running} onChange={(e) => setLo(Number(e.target.value))} className="w-full accent-indigo-500" />
            </label>
            <label className="block">
              <span className="mb-0.5 flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-500">
                To <span className="font-mono tabular-nums text-slate-700 dark:text-slate-200">{hi}</span>
              </span>
              <input type="range" min={minBlock} max={maxBlock} value={hi} disabled={running} onChange={(e) => setHi(Number(e.target.value))} className="w-full accent-indigo-500" />
            </label>
          </div>

          {!ownMode && (
            <div>
              <div className="mb-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-500">Which Done images</div>
              <div className="grid grid-cols-3 gap-1.5">
                {(
                  [
                    ["all", "All labeled", inRange.length],
                    ["pass", "Pass R1–R3", passCount],
                    ["fail", "Fail 0/3", failCount],
                  ] as const
                ).map(([id, label, n]) => (
                  <button
                    key={id}
                    type="button"
                    disabled={running}
                    onClick={() => setQuality(id)}
                    className={cn(
                      "rounded-lg border px-2 py-2 text-[11px] font-bold",
                      quality === id
                        ? id === "fail"
                          ? "border-rose-400 bg-rose-500 text-white"
                          : id === "pass"
                            ? "border-emerald-400 bg-emerald-500 text-white"
                            : "border-indigo-400 bg-indigo-500 text-white"
                        : "border-slate-200 text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800",
                    )}
                  >
                    <div>{label}</div>
                    <div className="mt-0.5 text-[10px] font-black tabular-nums opacity-80">{n}</div>
                  </button>
                ))}
              </div>
            </div>
          )}

          </div>

          <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 dark:border-slate-700 dark:bg-ink-950">
            <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Will score</div>
            <div className="mt-0.5 text-2xl font-black tabular-nums text-slate-900 dark:text-white">{count}</div>
            <div className="text-[11px] text-slate-500">
              {picked ? "hand-picked images" : `${ownMode ? "index" : "blocks"} ${lo}–${hi}`}
              {picked ? " · exactly these" : ownMode ? " · every Done" : quality === "pass" ? " · pass" : quality === "fail" ? " · fail" : " · all Done"}
              {` · min ${minGt}`}
            </div>
          </div>

          <p className="text-[11px] text-slate-500 dark:text-slate-400">
            Private models and pushing use your saved Hugging Face accounts (API keys).
          </p>

          <div>
            <div className="mb-1.5 flex items-center justify-between gap-2">
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Hugging Face models</div>
              <div className="text-[10px] text-slate-400">{repos.length} selected · click browse list or type</div>
            </div>

            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-xl border border-slate-200 dark:border-slate-700">
                <div className="border-b border-slate-200 px-3 py-2 dark:border-slate-700">
                  <div className="relative">
                    <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
                    <input
                      ref={inputRef}
                      value={draft}
                      disabled={running}
                      onChange={(e) => {
                        setDraft(e.target.value);
                        fetchSuggestions(e.target.value);
                      }}
                      onKeyDown={onDraftKeyDown}
                      onPaste={(e) => {
                        const text = e.clipboardData.getData("text");
                        if (/[,\s;]/.test(text)) {
                          e.preventDefault();
                          text
                            .split(/[\s,;]+/)
                            .map((p) => p.trim())
                            .filter(Boolean)
                            .forEach((p) => addRepo(p));
                        }
                      }}
                      placeholder="Search or type owner/name…"
                      className="w-full rounded-lg border border-slate-200 bg-white py-2 pl-8 pr-8 font-mono text-sm dark:border-slate-700 dark:bg-ink-800"
                    />
                    {suggestBusy ? (
                      <Loader2 className="absolute right-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 animate-spin text-indigo-500" />
                    ) : (
                      <button
                        type="button"
                        disabled={running || !draft.trim()}
                        onClick={() => addRepo(draft)}
                        className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded-md bg-indigo-600 p-1 text-white disabled:opacity-40"
                        title="Add typed model"
                      >
                        <Plus className="h-3.5 w-3.5" />
                      </button>
                    )}
                  </div>
                </div>

                {recentAvailable.length > 0 && (
                  <div className="border-b border-slate-100 px-3 py-2 dark:border-slate-800">
                    <div className="mb-1 text-[9px] font-bold uppercase tracking-wider text-slate-400">Recent</div>
                    <div className="flex flex-wrap gap-1">
                      {recentAvailable.slice(0, 8).map((r) => (
                        <button
                          key={r}
                          type="button"
                          disabled={running}
                          onClick={() => addRepo(r)}
                          className="rounded-full border border-slate-200 bg-white px-2 py-0.5 font-mono text-[10px] text-slate-700 hover:border-indigo-300 hover:bg-indigo-50 dark:border-slate-700 dark:bg-ink-900 dark:text-slate-200"
                        >
                          {r}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                <ul className="max-h-56 overflow-auto py-1">
                  {browse.length === 0 && (
                    <li className="px-3 py-6 text-center text-[11px] text-slate-400">
                      Type to search Hugging Face models…
                    </li>
                  )}
                  {browse.map((s, i) => (
                    <li key={s.id}>
                      <button
                        type="button"
                        disabled={running}
                        onClick={() => addRepo(s.id)}
                        className={cn(
                          "flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-[12px]",
                          i === activeSuggest
                            ? "bg-indigo-50 dark:bg-indigo-950/40"
                            : "hover:bg-slate-50 dark:hover:bg-slate-800/60",
                        )}
                      >
                        <span className="min-w-0 truncate font-mono font-semibold text-slate-800 dark:text-slate-100">
                          {s.id}
                        </span>
                        <span className="shrink-0 text-[10px] uppercase tracking-wide text-slate-400">
                          {s.private ? "private" : s.source === "local" ? "cached" : s.source === "typed" ? "typed" : s.downloads != null ? `${s.downloads} dl` : "hub"}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="rounded-xl border border-indigo-200 bg-indigo-50/40 dark:border-indigo-900/50 dark:bg-indigo-950/20">
                <div className="flex items-center justify-between border-b border-indigo-200/70 px-3 py-2 dark:border-indigo-900/50">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-indigo-700 dark:text-indigo-300">
                    Selected for compare
                  </div>
                  {repos.length > 0 && (
                    <button
                      type="button"
                      disabled={running}
                      onClick={() => setRepos([])}
                      className="text-[10px] font-semibold text-indigo-600 hover:underline dark:text-indigo-300"
                    >
                      Clear all
                    </button>
                  )}
                </div>
                <ul className="max-h-[18.5rem] space-y-1 overflow-auto p-2">
                  {repos.length === 0 && (
                    <li className="px-2 py-8 text-center text-[11px] text-indigo-700/60 dark:text-indigo-200/50">
                      Click models on the left, or type owner/name and press Enter / +
                    </li>
                  )}
                  {repos.map((repo, idx) => (
                    <li
                      key={repo}
                      className="flex items-center gap-2 rounded-lg border border-indigo-200 bg-white px-2.5 py-2 dark:border-indigo-800 dark:bg-ink-900"
                    >
                      <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-[10px] font-bold text-white">
                        {idx + 1}
                      </span>
                      <span className="min-w-0 flex-1 truncate font-mono text-[12px] font-semibold text-slate-900 dark:text-slate-100">
                        {repo}
                      </span>
                      <Check className="h-3.5 w-3.5 shrink-0 text-emerald-500" />
                      <button
                        type="button"
                        disabled={running}
                        onClick={() => removeRepo(repo)}
                        className="rounded p-0.5 text-slate-400 hover:bg-rose-50 hover:text-rose-500 dark:hover:bg-rose-950/40"
                        aria-label={`Remove ${repo}`}
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          {job && (
            <div className="rounded-xl border border-slate-200 dark:border-slate-700">
              <div className="flex items-center justify-between gap-2 border-b border-slate-200 px-3 py-2 dark:border-slate-700">
                <div className="min-w-0">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Results · sorted by score</div>
                  <div className="truncate text-[12px] text-slate-600 dark:text-slate-300">
                    {running && <Loader2 className="mr-1 inline h-3 w-3 animate-spin text-indigo-500" />}
                    {job.message || job.status}
                    {job.n_gt ? ` · ${job.n_gt} images` : ""}
                  </div>
                </div>
                {running && (
                  <button
                    type="button"
                    onClick={() => void stopCompare()}
                    className="inline-flex items-center gap-1 rounded border border-rose-200 px-2 py-1 text-[10px] font-bold text-rose-600 dark:border-rose-800 dark:text-rose-300"
                  >
                    <Square className="h-3 w-3" />
                    Stop
                  </button>
                )}
              </div>
              <div className="overflow-auto">
                <table className="w-full min-w-[40rem] text-left text-[11px]">
                  <thead className="bg-slate-50 text-[10px] uppercase tracking-wider text-slate-400 dark:bg-ink-950">
                    <tr>
                      <th className="px-3 py-2 font-bold">#</th>
                      <th className="px-3 py-2 font-bold">Model</th>
                      <th className="px-3 py-2 font-bold">Score</th>
                      <th className="px-3 py-2 font-bold">mAP50</th>
                      <th className="px-3 py-2 font-bold">FFPI</th>
                      <th className="px-3 py-2 font-bold">Latency p95</th>
                      <th className="px-3 py-2 font-bold">Mean</th>
                      <th className="px-3 py-2 font-bold">PP</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.length === 0 && (
                      <tr>
                        <td colSpan={8} className="px-3 py-6 text-center text-slate-400">
                          {running ? "Scoring models…" : "No results yet"}
                        </td>
                      </tr>
                    )}
                    {results.map((row, i) => (
                      <tr
                        key={row.hf_repo}
                        className={cn(
                          "border-t border-slate-100 dark:border-slate-800",
                          row.ok && i === 0 && "bg-emerald-50/70 dark:bg-emerald-950/20",
                          !row.ok && "bg-rose-50/40 dark:bg-rose-950/10",
                          ppSource === row.hf_repo && "ring-1 ring-inset ring-indigo-400/60",
                        )}
                      >
                        <td className="px-3 py-2 font-mono tabular-nums text-slate-400">{i + 1}</td>
                        <td className="px-3 py-2">
                          <div className="font-mono font-semibold text-slate-900 dark:text-slate-100">{row.hf_repo}</div>
                          {row.error && <div className="text-[10px] text-rose-500">{row.error}</div>}
                        </td>
                        <td className="px-3 py-2 font-mono font-bold tabular-nums text-slate-900 dark:text-white">
                          {formatScore(row.score)}
                        </td>
                        <td className="px-3 py-2 font-mono tabular-nums">{formatScore(row.map50)}</td>
                        <td className="px-3 py-2 font-mono tabular-nums">{formatScore(row.ffpi)}</td>
                        <td className="px-3 py-2 font-mono tabular-nums">
                          <span className={row.latency_passed === false ? "text-rose-500" : ""}>
                            {formatMs(row.latency_p95_ms)}
                          </span>
                        </td>
                        <td className="px-3 py-2 font-mono tabular-nums">{formatMs(row.latency_mean_ms)}</td>
                        <td className="px-3 py-2">
                          {row.ok ? (
                            <button
                              type="button"
                              disabled={ppRunning || running}
                              onClick={() => pickPpSource(row.hf_repo)}
                              className={cn(
                                "inline-flex items-center gap-1 rounded-md border px-2 py-1 text-[10px] font-bold",
                                ppSource === row.hf_repo
                                  ? "border-indigo-500 bg-indigo-50 text-indigo-700 dark:bg-indigo-950/40 dark:text-indigo-200"
                                  : "border-slate-200 text-slate-600 hover:border-indigo-300 dark:border-slate-700 dark:text-slate-300",
                              )}
                            >
                              <Rocket className="h-3 w-3" />
                              {ppSource === row.hf_repo ? "Selected" : "Deep PP"}
                            </button>
                          ) : (
                            <span className="text-slate-300">—</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {compareDone && (
            <div className="rounded-xl border border-indigo-200 bg-indigo-50/50 p-3 dark:border-indigo-900/50 dark:bg-indigo-950/20">
              <div className="mb-2 flex items-center gap-2 text-[12px] font-bold text-indigo-800 dark:text-indigo-200">
                <Upload className="h-3.5 w-3.5" />
                Deep PP → metadata nudge → private HF push
              </div>
              <p className="mb-3 text-[11px] text-slate-600 dark:text-slate-400">
                Uses the same GT range as this compare. ONNX weights stay identical; only metadata changes the file hash.
              </p>
              <div className="grid gap-2 sm:grid-cols-2">
                <label className="block text-[11px]">
                  <span className="mb-1 block font-semibold text-slate-600 dark:text-slate-300">Source model</span>
                  <input
                    value={ppSource}
                    readOnly
                    placeholder="Pick a row above"
                    className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 font-mono text-[12px] dark:border-slate-700 dark:bg-ink-950"
                  />
                </label>
                <div className="block text-[11px]">
                  <span className="mb-1 block font-semibold text-slate-600 dark:text-slate-300">Push to (private repo)</span>
                  <HfPushTarget
                    accountId={ppAccount}
                    repoName={ppRepoName}
                    onChange={(v) => {
                      setPpAccount(v.accountId);
                      setPpRepoName(v.repoName);
                      setPpTarget(v.repo);
                    }}
                  />
                </div>
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  disabled={ppBusy || ppRunning || !ppSource || !ppTarget.trim()}
                  onClick={() => void startDeepPpPush()}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-[12px] font-bold text-white hover:bg-indigo-500 disabled:opacity-40"
                >
                  {ppBusy || ppRunning ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Rocket className="h-3.5 w-3.5" />}
                  Deep PP &amp; push
                </button>
                {ppRunning && (
                  <button
                    type="button"
                    onClick={() => void stopDeepPpPush()}
                    className="inline-flex items-center gap-1 rounded-lg border border-rose-300 px-3 py-1.5 text-[12px] font-semibold text-rose-600 dark:border-rose-800 dark:text-rose-300"
                  >
                    <Square className="h-3 w-3" />
                    Stop PP
                  </button>
                )}
              </div>
              {pp && (pp.status || pp.message) && (
                <div className="mt-3 rounded-lg border border-slate-200 bg-white/80 px-3 py-2 text-[11px] dark:border-slate-700 dark:bg-ink-950/60">
                  <div className="font-semibold text-slate-800 dark:text-slate-100">
                    {pp.status || "…"} · {pp.message || ""}
                  </div>
                  {(pp.baseline_score != null || pp.pp_score != null) && (
                    <div className="mt-1 font-mono tabular-nums text-slate-600 dark:text-slate-300">
                      baseline {formatScore(pp.baseline_score)} → PP {formatScore(pp.pp_score)}
                      {pp.baseline_score != null && pp.pp_score != null
                        ? ` (Δ ${(Number(pp.pp_score) - Number(pp.baseline_score)).toFixed(4)})`
                        : ""}
                    </div>
                  )}
                  {pp.nudge?.sha256_after && (
                    <div className="mt-1 truncate font-mono text-[10px] text-slate-500">
                      hash {pp.nudge.sha256_before?.slice(0, 12)}… → {pp.nudge.sha256_after.slice(0, 12)}…
                    </div>
                  )}
                  {pp.url && (
                    <a
                      href={pp.url}
                      target="_blank"
                      rel="noreferrer"
                      className="mt-1 inline-block font-semibold text-indigo-600 hover:underline dark:text-indigo-300"
                    >
                      {pp.url}
                    </a>
                  )}
                  {pp.error && <div className="mt-1 text-rose-600 dark:text-rose-300">{pp.error}</div>}
                  {pp.progress?.pct != null && ppRunning && (
                    <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                      <div
                        className="h-full rounded-full bg-indigo-500 transition-all"
                        style={{ width: `${Math.max(2, Math.min(100, Number(pp.progress.pct) || 0))}%` }}
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-[12px] text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200">
              {error}
            </div>
          )}
        </div>

        <div className="flex justify-end gap-2 border-t border-slate-200 px-4 py-3 dark:border-slate-700">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-600 dark:border-slate-700 dark:text-slate-300"
          >
            Close
          </button>
          <button
            type="button"
            disabled={busy || running || repos.length === 0 || count < minGt}
            onClick={() => void startCompare()}
            className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-bold text-white hover:bg-indigo-500 disabled:opacity-40"
          >
            {busy || running ? <Loader2 className="h-4 w-4 animate-spin" /> : <GitCompareArrows className="h-4 w-4" />}
            Compare {repos.length || ""} model{repos.length === 1 ? "" : "s"}
          </button>
        </div>
      </div>
    </div>
  );
}
