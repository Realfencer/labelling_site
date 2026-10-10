import HfPushTarget from "./HfPushTarget";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Check,
  Download,
  FlaskConical,
  Loader2,
  Lock,
  Play,
  Rocket,
  Square,
  Trash2,
  Trophy,
  Upload,
  X,
} from "lucide-react";
import {
  api,
  type FindBestElementInfo,
  type FindBestElementOption,
  type FindBestHistoryEntry,
  type FindBestJob,
  type FindBestJobStep,
} from "../api";
import EvaluateMergedModal from "./EvaluateMergedModal";
import type { ChallengeItem } from "../types";
import { checksFailComplete, checksPassComplete } from "../labelChecks";
import { cn } from "../lib/utils";
import ImageSelectionField from "./ImageSelectionField";

const FALLBACK_MIN_REVIEWED = 1;

function formatBytes(n: number | undefined): string {
  if (!n || n <= 0) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function formatHistoryWhen(iso: string | undefined): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  } catch {
    return iso.length > 19 ? iso.slice(0, 19) : iso;
  }
}

function ElementTopicCard({
  info,
  historyCount,
  bestScore,
  paramSource,
}: {
  info: FindBestElementInfo | null;
  historyCount: number;
  bestScore: number | null;
  paramSource: string;
}) {
  const label = info?.label || "—";
  const queries = (info?.search_queries || []).slice(0, 4);
  return (
    <div className="rounded-xl border border-slate-200 bg-gradient-to-br from-slate-50 to-white p-3 dark:border-slate-700 dark:from-ink-950/80 dark:to-ink-900">
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Element</span>
        <span className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-white px-1.5 py-0.5 text-[10px] font-semibold text-slate-500 dark:border-slate-700 dark:bg-ink-950 dark:text-slate-400">
          <Lock className="h-3 w-3" />
          From this topic
        </span>
      </div>
      <div className="text-base font-extrabold text-slate-900 dark:text-white">{label}</div>
      {info?.topic_slug && (
        <div className="mt-0.5 font-mono text-[10px] text-slate-400">{info.topic_slug}</div>
      )}
      <div className="mt-2 flex items-start gap-1.5 text-[11px] text-slate-600 dark:text-slate-300">
        <Trophy className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-500" />
        <div className="min-w-0">
          <div className="font-semibold text-slate-700 dark:text-slate-200">Final compare vs step-4 top miner</div>
          <div className="text-[10px] text-slate-500">Class filter: count only · authors: site-wide store</div>
        </div>
      </div>
      {queries.length > 0 && (
        <div className="mt-2">
          <div className="mb-1 text-[9px] font-bold uppercase tracking-wide text-slate-400">HF search</div>
          <div className="flex flex-wrap gap-1">
            {queries.map((q) => (
              <span
                key={q}
                className="rounded border border-slate-200/90 bg-white px-1.5 py-0.5 font-mono text-[10px] text-slate-600 dark:border-slate-700 dark:bg-ink-950 dark:text-slate-300"
              >
                {q}
              </span>
            ))}
          </div>
        </div>
      )}
      <div className="mt-2.5 flex flex-wrap gap-x-3 gap-y-1 border-t border-slate-200/80 pt-2 text-[10px] text-slate-500 dark:border-slate-700">
        <span>
          History <span className="font-bold text-slate-700 dark:text-slate-200">{historyCount}</span>
        </span>
        {bestScore != null && (
          <span>
            Best <span className="font-bold tabular-nums text-emerald-600 dark:text-emerald-300">{bestScore.toFixed(4)}</span>
          </span>
        )}
        <span>
          Params{" "}
          <span className="font-semibold text-slate-700 dark:text-slate-200">
            {paramSource === "last_run" ? "from last run" : "defaults"}
          </span>
        </span>
      </div>
      {!info?.known && (
        <p className="mt-2 text-[10px] text-amber-600 dark:text-amber-300">
          Topic not in the known element list — find_best will use fuzzy HF matching.
        </p>
      )}
    </div>
  );
}

function StepCard({
  step,
  stepTotal,
  selected,
  onSelect,
}: {
  step: FindBestJobStep;
  stepTotal: number;
  selected: boolean;
  onSelect: () => void;
}) {
  const done = step.status === "done";
  const active = step.status === "running";
  const failed = step.status === "error";
  const summary = step.summary;
  const canInspect = done || active || failed || Boolean(summary?.headline || summary?.rows?.length);

  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={!canInspect && step.status === "pending"}
      title={
        canInspect
          ? "Click to inspect this step (stays pinned while later steps run)"
          : "Waiting for this step to start"
      }
      className={cn(
        "w-full rounded-xl border px-3 py-2.5 text-left transition",
        selected && "ring-2 ring-sky-400/70",
        done && "border-emerald-300/80 bg-emerald-50/70 dark:border-emerald-800/60 dark:bg-emerald-950/25",
        active && "border-sky-400 bg-sky-50 dark:border-sky-700 dark:bg-sky-950/40",
        failed && "border-rose-300 bg-rose-50 dark:border-rose-800/60 dark:bg-rose-950/30",
        !done && !active && !failed && "border-slate-200 bg-white dark:border-slate-700 dark:bg-ink-900",
        !canInspect && step.status === "pending" && "cursor-not-allowed opacity-60",
      )}
    >
      <div className="flex items-start gap-2.5">
        <span
          className={cn(
            "mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] font-bold",
            done && "bg-emerald-500 text-white",
            active && "bg-sky-500 text-white",
            failed && "bg-rose-500 text-white",
            !done && !active && !failed && "bg-slate-200 text-slate-600 dark:bg-slate-700 dark:text-slate-300",
          )}
        >
          {done ? <Check className="h-3.5 w-3.5" /> : active ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : step.n}
        </span>
        <div className="min-w-0 flex-1">
          <div className="text-[12px] font-bold text-slate-900 dark:text-slate-50">
            [{step.n}/{stepTotal}] {step.label}
          </div>
          {summary?.headline ? (
            <div className="mt-0.5 truncate text-[11px] text-slate-600 dark:text-slate-300">{summary.headline}</div>
          ) : active ? (
            <div className="mt-0.5 text-[11px] text-sky-600 dark:text-sky-300">Working…</div>
          ) : failed ? (
            <div className="mt-0.5 text-[11px] text-rose-500">Failed</div>
          ) : (
            <div className="mt-0.5 text-[11px] text-slate-400">Waiting</div>
          )}
          {summary?.metrics && summary.metrics.length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {summary.metrics.map((m) => (
                <span
                  key={m.label}
                  className="inline-flex items-center gap-1 rounded-md border border-slate-200/80 bg-white/80 px-1.5 py-0.5 text-[10px] dark:border-slate-700 dark:bg-ink-950/60"
                >
                  <span className="font-semibold uppercase tracking-wide text-slate-400">{m.label}</span>
                  <span className="font-bold tabular-nums text-slate-800 dark:text-slate-100">{m.value}</span>
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </button>
  );
}

const FALLBACK_STEP_LABELS = [
  "Commitments + known-author scan",
  "Historical unique ONNX snaps",
  "Download miner packs",
  "Score packs on GT",
  "Cluster by ONNX cosine",
  "N-way soup / merge",
  "Force top-1 × top-7",
  "Deep post-process",
  "Compare vs top scored miner",
  "Result report",
];

function StepDetailPanel({
  step,
  liveStep,
  stepTotal,
  pinned,
  onFollowLive,
}: {
  step: FindBestJobStep | null;
  liveStep: number;
  stepTotal: number;
  pinned: boolean;
  onFollowLive: () => void;
}) {
  if (!step) {
    return (
      <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-slate-300 p-6 text-sm text-slate-400 dark:border-slate-700">
        Select a step to inspect results
      </div>
    );
  }
  const rows = step.summary?.rows || [];
  return (
    <div className="flex h-full min-h-0 flex-col rounded-xl border border-slate-200 bg-slate-50/80 dark:border-slate-700 dark:bg-ink-950/40">
      <div className="border-b border-slate-200 px-4 py-3 dark:border-slate-700">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Step {step.n}/{stepTotal} result
              {step.status === "done" ? " · complete" : step.status === "running" ? " · live" : ""}
            </div>
            <div className="mt-0.5 text-sm font-extrabold text-slate-900 dark:text-white">{step.label}</div>
            {step.summary?.headline && (
              <div className="mt-1 text-[12px] text-slate-600 dark:text-slate-300">{step.summary.headline}</div>
            )}
          </div>
          {pinned && liveStep > 0 && liveStep !== step.n && (
            <button
              type="button"
              onClick={onFollowLive}
              className="shrink-0 rounded-md border border-sky-300 bg-sky-50 px-2 py-1 text-[10px] font-bold text-sky-700 hover:bg-sky-100 dark:border-sky-700 dark:bg-sky-950/50 dark:text-sky-300"
            >
              Follow live · {liveStep}/{stepTotal}
            </button>
          )}
        </div>
        {pinned && liveStep > 0 && liveStep !== step.n && (
          <p className="mt-1.5 text-[10px] text-slate-500">
            Viewing a previous step. Pipeline still running on step {liveStep}.
          </p>
        )}
      </div>
      {step.summary?.metrics && step.summary.metrics.length > 0 && (
        <div className="grid grid-cols-3 gap-2 border-b border-slate-200 px-4 py-3 dark:border-slate-700">
          {step.summary.metrics.map((m) => (
            <div key={m.label} className="rounded-lg border border-slate-200 bg-white px-2 py-1.5 dark:border-slate-700 dark:bg-ink-900">
              <div className="text-[9px] font-bold uppercase tracking-wide text-slate-400">{m.label}</div>
              <div className="mt-0.5 truncate text-sm font-extrabold tabular-nums text-slate-900 dark:text-slate-50">{m.value}</div>
            </div>
          ))}
        </div>
      )}
      <div className="min-h-0 flex-1 overflow-auto px-3 py-2">
        {rows.length === 0 ? (
          <div className="px-1 py-6 text-center text-[12px] text-slate-400">
            {step.status === "running"
              ? (step.summary?.headline || "Working — results appear here as they come in…")
              : step.status === "pending"
                ? "Not started yet"
                : "No detail rows for this step"}
          </div>
        ) : (
          <ul className="space-y-1.5">
            {rows.map((r, i) => {
              const hasScore = typeof r.score === "number";
              const rank = r.rank ?? (hasScore ? i + 1 : null);
              const bar = typeof r.bar === "number" ? Math.max(0, Math.min(100, r.bar)) : null;
              return (
                <li
                  key={`${r.title}-${i}`}
                  className="rounded-lg border border-slate-200 bg-white px-3 py-2 dark:border-slate-700 dark:bg-ink-900"
                >
                  <div className="flex items-baseline gap-2">
                    {rank != null && (
                      <span className="w-5 shrink-0 text-[10px] font-bold tabular-nums text-slate-400">
                        {rank}
                      </span>
                    )}
                    <div className="min-w-0 flex-1">
                      <div className="flex items-baseline justify-between gap-2">
                        <div className="truncate text-[12px] font-bold text-slate-900 dark:text-slate-50">
                          {r.title}
                        </div>
                        {hasScore && (
                          <div className="shrink-0 text-[12px] font-extrabold tabular-nums text-sky-700 dark:text-sky-300">
                            {r.score!.toFixed(4)}
                          </div>
                        )}
                      </div>
                      {r.subtitle && (
                        <div className="truncate text-[11px] text-slate-500">{r.subtitle}</div>
                      )}
                      {bar != null && (
                        <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-sky-500 to-emerald-400"
                            style={{ width: `${bar}%` }}
                          />
                        </div>
                      )}
                      {r.meta && (
                        <div className="mt-0.5 text-[10px] font-semibold text-slate-500 dark:text-slate-400">
                          {r.meta}
                        </div>
                      )}
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}

export default function FindBestModal({
  open,
  onClose,
  reviewedCount,
  defaultElement,
  challenges = [],
  onMinerLoaded,
  ownMode = false,
}: {
  open: boolean;
  onClose: () => void;
  reviewedCount: number;
  defaultElement: string;
  challenges?: ChallengeItem[];
  onMinerLoaded?: (state: import("../types").AppState) => void;
  ownMode?: boolean;
}) {
  const [elements, setElements] = useState<FindBestElementOption[]>([]);
  const [element, setElement] = useState(defaultElement);
  const [elementInfo, setElementInfo] = useState<FindBestElementInfo | null>(null);
  const [historyCount, setHistoryCount] = useState(0);
  const [bestScore, setBestScore] = useState<number | null>(null);
  const [paramSource, setParamSource] = useState("defaults");
  const [authorsStoreCount, setAuthorsStoreCount] = useState<number | null>(null);
  const [days, setDays] = useState("7");
  const [topK, setTopK] = useState("30");
  const [minCosine, setMinCosine] = useState("0.9");
  const [fromBlock, setFromBlock] = useState("");
  const [toBlock, setToBlock] = useState("");
  const [quality, setQuality] = useState<"all" | "pass" | "fail">("all");
  const [picked, setPicked] = useState<string[] | null>(null);
  const [reviewedAvailable, setReviewedAvailable] = useState(reviewedCount);
  const [minReviewed, setMinReviewed] = useState(FALLBACK_MIN_REVIEWED);
  const [job, setJob] = useState<FindBestJob | null>(null);
  const [history, setHistory] = useState<FindBestHistoryEntry[]>([]);
  const [clearBusy, setClearBusy] = useState(false);
  const [clearMsg, setClearMsg] = useState("");
  const [error, setError] = useState("");
  const [alarm, setAlarm] = useState("");
  const [busy, setBusy] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [downloadId, setDownloadId] = useState<string | null>(null);
  const [loadId, setLoadId] = useState<string | null>(null);
  const [pushFormId, setPushFormId] = useState<string | null>(null);
  const [pushId, setPushId] = useState<string | null>(null);
  const [pushAccount, setPushAccount] = useState("");
  const [pushFull, setPushFull] = useState("");
  const [pushRepo, setPushRepo] = useState(() => {
    try {
      return sessionStorage.getItem("find_best_hf_repo") || "";
    } catch {
      return "";
    }
  });
  const [pushMsg, setPushMsg] = useState("");
  const [evalEntry, setEvalEntry] = useState<FindBestHistoryEntry | null>(null);
  const [selectedStep, setSelectedStep] = useState<number>(1);
  /** When set, keep showing this step even while a later step runs. */
  const [pinnedStep, setPinnedStep] = useState<number | null>(null);
  const [viewSource, setViewSource] = useState<"live" | "history">("live");
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);
  const [historyProgress, setHistoryProgress] = useState<FindBestJob | null>(null);
  const [loadingHistoryId, setLoadingHistoryId] = useState<string | null>(null);
  const [topicMismatch, setTopicMismatch] = useState(false);

  const labeled = useMemo(
    () => challenges.filter((c) => c.loaded && c.reviewed),
    [challenges],
  );
  const labeledBlocks = useMemo(
    () => labeled.map((c) => c.block).filter((b) => b > 0).sort((a, b) => a - b),
    [labeled],
  );
  const minBlock = labeledBlocks[0] ?? 0;
  const maxBlock = labeledBlocks[labeledBlocks.length - 1] ?? 0;
  const fromN = Number(fromBlock);
  const toN = Number(toBlock);
  const rawLo = Number.isFinite(fromN) ? fromN : minBlock;
  const rawHi = Number.isFinite(toN) ? toN : maxBlock;
  const lo = Math.min(rawLo, rawHi);
  const hi = Math.max(rawLo, rawHi);
  const span = Math.max(1, maxBlock - minBlock);
  const inRange = labeled.filter((c) => c.block >= lo && c.block <= hi);
  const passCount = inRange.filter((c) => checksPassComplete(c)).length;
  const failCount = inRange.filter((c) => checksFailComplete(c)).length;
  const selectedCount = picked ? picked.length : quality === "pass" ? passCount : quality === "fail" ? failCount : inRange.length;
  const hist = useMemo(() => {
    const bins = new Array(48).fill(0);
    for (const c of labeled) {
      const t = (c.block - minBlock) / span;
      const i = Math.min(47, Math.max(0, Math.floor(t * 48)));
      bins[i] += 1;
    }
    const peak = Math.max(1, ...bins);
    return bins.map((n) => n / peak);
  }, [labeled, minBlock, span]);

  const syncJob = useCallback(async () => {
    const data = await api.getFindBestStatus();
    setTopicMismatch(Boolean(data.topic_mismatch));
    const j = data.job;
    if (j) {
      setJob(j);
      const stepN = j.step ?? 0;
      const isRunning = j.status === "running" || j.status === "queued";
      if (isRunning && pinnedStep === null && stepN > 0) {
        setSelectedStep(stepN);
      }
    } else {
      setJob(null);
    }
    return j;
  }, [pinnedStep]);

  const refresh = useCallback(async () => {
    try {
      const data = await api.getFindBestDefaults();
      setReviewedAvailable(data.reviewed_available);
      setMinReviewed(data.min_reviewed || FALLBACK_MIN_REVIEWED);
      setElements(data.elements || []);
      setHistory(data.history || []);
      if (data.element) setElement(data.element);
      setElementInfo(data.element_info || null);
      setHistoryCount(data.history_count ?? (data.history || []).length);
      setBestScore(typeof data.best_score === "number" ? data.best_score : null);
      setParamSource(data.param_source || "defaults");
      if (typeof data.authors_store_count === "number") setAuthorsStoreCount(data.authors_store_count);
      if (data.days != null) setDays(String(data.days));
      if (data.top_k != null) setTopK(String(data.top_k));
      if (data.min_cosine != null) setMinCosine(String(data.min_cosine));
      await syncJob();
      setPinnedStep(null);
      setViewSource("live");
      setSelectedHistoryId(null);
      setHistoryProgress(null);
    } catch (e) {
      setError(String(e));
    }
  }, [syncJob]);

  const viewHistoryProgress = useCallback(async (entry: FindBestHistoryEntry) => {
    setViewSource("history");
    setSelectedHistoryId(entry.id);
    setPinnedStep(null);
    setLoadingHistoryId(entry.id);
    setError("");
    try {
      const progress = await api.getFindBestHistoryProgress(entry.id);
      setHistoryProgress(progress);
      const lastDone =
        [...(progress.steps || [])].reverse().find((s) => s.status === "done")?.n ||
        progress.step ||
        8;
      setSelectedStep(Math.max(1, Math.min(8, lastDone)));
    } catch (e) {
      setError(String(e));
      setHistoryProgress(null);
    } finally {
      setLoadingHistoryId(null);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    setAlarm("");
    setError("");
    setPinnedStep(null);
    void (async () => {
      const live = await syncJob().catch(() => null);
      const isRunning = live?.status === "running" || live?.status === "queued";
      if (!isRunning) {
        setQuality("all");
        setPicked(null);
        setFromBlock("");
        setToBlock("");
      }
      await refresh();
    })();
  }, [open, refresh, syncJob]);

  useEffect(() => {
    if (!open) return;
    setFromBlock((prev) => prev || (minBlock ? String(minBlock) : ""));
    setToBlock((prev) => prev || (maxBlock ? String(maxBlock) : ""));
  }, [open, minBlock, maxBlock]);

  useEffect(() => {
    if (!open) return;
    setReviewedAvailable(reviewedCount);
  }, [open, reviewedCount]);

  useEffect(() => {
    if (!open) return;
    const fast = job?.status === "queued" || job?.status === "running" || !job;
    const t = window.setInterval(() => {
      void (async () => {
        try {
          const data = await api.getFindBestStatus();
          setTopicMismatch(Boolean(data.topic_mismatch));
          setJob(data.job);
          const stepN = data.job?.step ?? 0;
          const isRunning = data.job?.status === "running" || data.job?.status === "queued";
          // Auto-follow live step only when viewing the live pipeline.
          if (viewSource === "live" && isRunning && pinnedStep === null && stepN > 0) {
            setSelectedStep(stepN);
          }
          if (data.job?.status === "done" || data.job?.status === "error" || data.job?.status === "stopped") {
            setPinnedStep(null);
            const hist = await api.getFindBestHistory();
            setHistory(hist.history || []);
          }
        } catch {
          /* ignore */
        }
      })();
    }, fast ? 800 : 2500);
    return () => window.clearInterval(t);
  }, [open, job?.status, pinnedStep, viewSource]);

  const clearHistoryDisk = async () => {
    if (clearBusy) return;
    const ok = window.confirm(
      "Clear Find Best history and pipeline run files?\n\n" +
        "This frees large disk usage (runs/soups). Active running jobs are kept.\n" +
        "Labels and Own images are never deleted.",
    );
    if (!ok) return;
    setClearBusy(true);
    setClearMsg("");
    try {
      const res = await api.clearFindBestHistory();
      setHistory([]);
      setSelectedHistoryId(null);
      setHistoryProgress(null);
      setHistoryCount(0);
      setClearMsg(`Freed ${res.freed_human || "disk"} · removed ${res.runs_removed ?? 0} runs + ${res.history_removed ?? 0} history packs`);
    } catch (e) {
      setClearMsg(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setClearBusy(false);
    }
  };

  const topicHistory = useMemo(() => {
    return history.filter((h) => {
      const a = (h.element || "").trim().toLowerCase().replace(/-/g, " ");
      const b = (element || "").trim().toLowerCase().replace(/-/g, " ");
      if (!a || !b) return false;
      return a === b || a.replace(/s$/, "") === b.replace(/s$/, "");
    });
  }, [history, element]);

  const belowMin = selectedCount < minReviewed;
  const running = job?.status === "queued" || job?.status === "running";
  const viewingHistory = viewSource === "history" && Boolean(selectedHistoryId);
  const displayJob = viewingHistory ? historyProgress : job;

  const steps: FindBestJobStep[] = useMemo(() => {
    const fromJob = displayJob?.steps;
    if (fromJob && fromJob.length > 0) return fromJob;
    return Array.from({ length: FALLBACK_STEP_LABELS.length }, (_, i) => ({
      n: i + 1,
      label: FALLBACK_STEP_LABELS[i],
      status: "pending" as const,
      summary: null,
    }));
  }, [displayJob?.steps]);

  const stepTotal = steps.length || FALLBACK_STEP_LABELS.length;
  const activeStep = steps.find((s) => s.n === selectedStep) || steps[0] || null;
  const liveStepN = viewingHistory ? displayJob?.step ?? 0 : job?.step ?? 0;
  const isPinnedAway = !viewingHistory && pinnedStep !== null && liveStepN > 0 && pinnedStep !== liveStepN;
  const selectedHistoryEntry = topicHistory.find((h) => h.id === selectedHistoryId) || null;

  const selectStep = (n: number, status: string) => {
    setSelectedStep(n);
    if (viewingHistory) {
      setPinnedStep(n);
      return;
    }
    // Pin when inspecting a non-live step so polling won't yank the panel away.
    if (status === "running" || (liveStepN > 0 && n === liveStepN)) {
      setPinnedStep(null);
    } else {
      setPinnedStep(n);
    }
  };

  const followLive = () => {
    setViewSource("live");
    setSelectedHistoryId(null);
    setHistoryProgress(null);
    setPinnedStep(null);
    if (liveStepN > 0) setSelectedStep(liveStepN);
  };

  const start = async () => {
    setBusy(true);
    setError("");
    setAlarm("");
    if (selectedCount < minReviewed) {
      setAlarm(
        `Need at least ${minReviewed} selected Done images before running find_best_model. Currently ${selectedCount}.`,
      );
      setBusy(false);
      return;
    }
    try {
      const res = await api.startFindBest({
        element: element.trim(),
        days: Number(days) || 7,
        top_k: Number(topK) || 30,
        min_cosine: Number(minCosine) || 0.9,
        n_images: picked ? picked.length : ownMode ? inRange.length : selectedCount,
        blockMin: lo,
        blockMax: hi,
        itemKeys: picked ?? undefined,
        quality: ownMode || picked ? "all" : quality,
      });
      setJob(res.job);
      setViewSource("live");
      setSelectedHistoryId(null);
      setHistoryProgress(null);
      setPinnedStep(null);
      setSelectedStep(1);
      const staged = res.staged || {};
      const avail = staged.reviewed_available;
      if (typeof avail === "number") setReviewedAvailable(avail);
    } catch (e) {
      const msg = String(e).replace(/^Error:\s*/i, "");
      if (msg.toLowerCase().includes("need at least") || msg.toLowerCase().includes("reviewed")) {
        setAlarm(msg);
      } else if (msg.toLowerCase().includes("already running")) {
        const live = await syncJob().catch(() => null);
        if (live && (live.status === "running" || live.status === "queued")) {
          setError("");
          setAlarm(
            `Pipeline already running (step ${live.step || "?"}/${stepTotal}). Progress below — use Stop to cancel.`,
          );
        } else {
          setError(msg);
        }
      } else {
        setError(msg);
      }
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    setStopping(true);
    setError("");
    try {
      const res = await api.stopFindBest();
      setJob(res.job);
      setPinnedStep(null);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setStopping(false);
    }
  };

  const download = async (id: string) => {
    setDownloadId(id);
    setError("");
    try {
      await api.downloadFindBestOnnx(id);
    } catch (e) {
      setError(String(e));
    } finally {
      setDownloadId(null);
    }
  };

  const loadMiner = async (id: string) => {
    setLoadId(id);
    setError("");
    try {
      const state = await api.loadFindBestMiner(id);
      onMinerLoaded?.(state);
      onClose();
    } catch (e) {
      setError(String(e));
    } finally {
      setLoadId(null);
    }
  };

  const pushToHf = async (id: string) => {
    const repo = pushFull.trim();
    if (!repo) {
      setError("Choose a Hugging Face account and a repo name");
      return;
    }
    setPushId(id);
    setError("");
    setPushMsg("");
    try {
      try {
        sessionStorage.setItem("find_best_hf_repo", pushRepo.trim());
      } catch {
        /* ignore */
      }
      const res = await api.pushFindBestToHf(id, { token: "", repo });
      setPushMsg(
        res.created
          ? `Created and pushed → ${res.url}`
          : `Updated → ${res.url}`,
      );
      setPushFormId(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setPushId(null);
    }
  };

  if (!open) return null;

  const displayInfo: FindBestElementInfo =
    elementInfo ||
    ({
      value: element,
      label: (element || defaultElement || "—").replace(/\b\w/g, (c) => c.toUpperCase()),
      known: elements.some((e) => {
        const a = e.value.trim().toLowerCase().replace(/-/g, " ");
        const b = (element || "").trim().toLowerCase().replace(/-/g, " ");
        return a === b || a.replace(/s$/, "") === b.replace(/s$/, "") || a.replace(/\s+/g, "") === b.replace(/\s+/g, "");
      }),
      winner: elements.find((e) => e.value === element)?.winner || "",
      winner_short: (elements.find((e) => e.value === element)?.winner || "").split("/").pop() || "",
      search_queries: [],
    } satisfies FindBestElementInfo);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/55 p-2 backdrop-blur-[2px] sm:p-4">
      <div className="flex h-[96vh] w-full max-w-7xl flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-ink-900">
        <div className="flex items-center justify-between border-b border-slate-200 px-5 py-3.5 dark:border-slate-700">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-sky-500 text-white">
              <Rocket className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-base font-extrabold text-slate-900 dark:text-white">Submit · find best model</h2>
              <p className="text-[12px] text-slate-500 dark:text-slate-400">
                GT = selected Done labels ·{" "}
                <span className={cn("font-semibold", belowMin ? "text-amber-600 dark:text-amber-300" : "text-emerald-600 dark:text-emerald-300")}>
                  {selectedCount} selected
                </span>
                <span className="text-slate-400"> · {reviewedAvailable} reviewed · need ≥ {minReviewed}</span>
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-1.5 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {running && (
          <div className="mx-4 mt-0 flex items-start gap-2 rounded-lg border border-sky-300 bg-sky-50 px-3 py-2.5 text-[12px] text-sky-950 dark:border-sky-800/60 dark:bg-sky-950/40 dark:text-sky-100">
            <Loader2 className="mt-0.5 h-4 w-4 shrink-0 animate-spin" />
            <div>
              {topicMismatch ? (
                <>
                  Merge running for <span className="font-bold">{job?.element || "another topic"}</span> (step{" "}
                  {liveStepN || "?"}/{stepTotal}) while you are on <span className="font-bold">{element || defaultElement}</span>.
                  Progress below still updates. Labeling may feel slow until the merge finishes.
                </>
              ) : (
                <>
                  Merge running — step {liveStepN || "?"}/{stepTotal}. Labeling may feel slow until this finishes.
                </>
              )}
            </div>
          </div>
        )}

        <div className="grid min-h-0 flex-1 gap-4 overflow-hidden p-4 lg:grid-cols-[320px_minmax(0,1fr)]">
          {/* Left: controls */}
          <div className="flex min-h-0 flex-col gap-3 overflow-auto pr-1">
            <ElementTopicCard
              info={displayInfo}
              historyCount={historyCount || topicHistory.length}
              bestScore={bestScore}
              paramSource={paramSource}
            />
            {authorsStoreCount != null && (
              <p className="text-xs text-zinc-500 mt-1">
                Known SN44 authors: {authorsStoreCount} (shared across topics; class filter = count only)
              </p>
            )}

            <div className="grid grid-cols-2 gap-2">
              <label className="block">
                <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">Days</span>
                <input
                  value={days}
                  onChange={(e) => setDays(e.target.value)}
                  disabled={running}
                  className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-sm dark:border-slate-700 dark:bg-ink-800"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">
                  Top-k (merge)
                </span>
                <input
                  value={topK}
                  onChange={(e) => setTopK(e.target.value)}
                  disabled={running}
                  className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-sm dark:border-slate-700 dark:bg-ink-800"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">Min cosine</span>
                <input
                  value={minCosine}
                  onChange={(e) => setMinCosine(e.target.value)}
                  disabled={running}
                  className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-sm dark:border-slate-700 dark:bg-ink-800"
                />
              </label>
            </div>

            <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-3 dark:border-slate-700 dark:bg-ink-950">
              <div className="mb-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-500">GT images for merge</div>
              <ImageSelectionField suggest={inRange.map((c) => c.item_key)} challenges={challenges} ownMode={ownMode} value={picked} onChange={setPicked} verb="use" minCount={minReviewed} disabled={running} className="mb-2" />
              <div className={cn(picked && "pointer-events-none opacity-40")}>
              <div className="mb-2 flex h-8 items-end gap-px">
                {hist.map((h, i) => {
                  const binLo = minBlock + (i / 48) * span;
                  const binHi = minBlock + ((i + 1) / 48) * span;
                  const active = binHi >= lo && binLo <= hi;
                  return (
                    <div
                      key={i}
                      className={cn("min-h-[2px] flex-1 rounded-sm", active ? "bg-sky-500" : "bg-slate-300 dark:bg-slate-700")}
                      style={{ height: `${Math.max(8, h * 100)}%` }}
                    />
                  );
                })}
              </div>
              <label className="mb-2 block">
                <span className="mb-0.5 flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-500">
                  From <span className="font-mono tabular-nums text-slate-700 dark:text-slate-200">{lo}</span>
                </span>
                <input
                  type="range"
                  min={minBlock || 0}
                  max={maxBlock || 0}
                  value={lo}
                  disabled={running || !minBlock}
                  onChange={(e) => setFromBlock(e.target.value)}
                  className="w-full accent-sky-500"
                />
              </label>
              <label className="block">
                <span className="mb-0.5 flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-500">
                  To <span className="font-mono tabular-nums text-slate-700 dark:text-slate-200">{hi}</span>
                </span>
                <input
                  type="range"
                  min={minBlock || 0}
                  max={maxBlock || 0}
                  value={hi}
                  disabled={running || !minBlock}
                  onChange={(e) => setToBlock(e.target.value)}
                  className="w-full accent-sky-500"
                />
              </label>
              <button
                type="button"
                disabled={running}
                onClick={() => {
                  setFromBlock(String(minBlock));
                  setToBlock(String(maxBlock));
                }}
                className="mt-2 text-[11px] font-bold text-sky-600 hover:underline dark:text-sky-300"
              >
                Reset to all labeled ({minBlock}–{maxBlock})
              </button>
              {!ownMode && (
                <div className="mt-2 grid grid-cols-3 gap-1.5">
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
                              : "border-sky-400 bg-sky-500 text-white"
                          : "border-slate-200 text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800",
                      )}
                    >
                      <div>{label}</div>
                      <div className="mt-0.5 text-[10px] font-black tabular-nums opacity-80">{n}</div>
                    </button>
                  ))}
                </div>
              )}
              </div>
              <div className="mt-2 text-[11px] text-slate-500">
                Will use <span className="font-bold tabular-nums text-slate-800 dark:text-slate-100">{picked ? picked.length : ownMode ? inRange.length : selectedCount}</span> images
                {picked
                  ? " · hand-picked"
                  : ownMode
                  ? " · every Done in range"
                  : quality === "pass"
                    ? " · R1–R3 all close to p"
                    : quality === "fail"
                      ? " · no miner close to p"
                      : " · every Done in range"}
              </div>
            </div>

            <p className="text-[11px] leading-relaxed text-slate-500 dark:text-slate-400">
              Same picker as Download labels. Uses every selected Done image in range (no max cap).
            </p>

            {belowMin && !running && (
              <div className="flex items-start gap-2 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-[12px] text-amber-900 dark:border-amber-800/60 dark:bg-amber-950/40 dark:text-amber-100">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                <div>
                  Only <span className="font-bold">{selectedCount}</span> images in this range. Select at least{" "}
                  <span className="font-bold">{minReviewed}</span> before Submit can run.
                </div>
              </div>
            )}

            {alarm && (
              <div className="flex items-start gap-2 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-[12px] font-semibold text-amber-900 dark:border-amber-800/60 dark:bg-amber-950/40 dark:text-amber-100">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                <span>{alarm}</span>
              </div>
            )}

            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                disabled={busy || running || stopping}
                onClick={() => void start()}
                className="inline-flex w-full items-center justify-center gap-1.5 rounded-lg bg-sky-600 px-3 py-2.5 text-sm font-bold text-white hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                {running ? "Running…" : busy ? "Starting…" : "Start"}
              </button>
              <button
                type="button"
                disabled={!running || stopping}
                onClick={() => void stop()}
                className="inline-flex w-full items-center justify-center gap-1.5 rounded-lg border border-rose-300 bg-rose-50 px-3 py-2.5 text-sm font-bold text-rose-700 hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-40 dark:border-rose-800 dark:bg-rose-950/40 dark:text-rose-200 dark:hover:bg-rose-950/70"
              >
                {stopping ? <Loader2 className="h-4 w-4 animate-spin" /> : <Square className="h-3.5 w-3.5 fill-current" />}
                {stopping ? "Stopping…" : "Stop"}
              </button>
            </div>

            {error && (
              <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-[12px] text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200">
                {error}
              </div>
            )}
            {(job?.status === "error" || job?.status === "stopped") && job.error && (
              <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-[12px] text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200">
                {job.error}
              </div>
            )}
            {job?.status === "done" && job.result && (
              <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-[12px] text-emerald-800 dark:border-emerald-900/50 dark:bg-emerald-950/30 dark:text-emerald-200">
                Done
                {job.result.score != null && <> · score {Number(job.result.score).toFixed(4)}</>}
                {job.history_id && (
                  <button type="button" className="ml-2 font-bold underline" onClick={() => void download(job.history_id!)}>
                    Download ZIP
                  </button>
                )}
              </div>
            )}

            <div className="min-h-0 flex-1">
              <div className="mb-1.5 flex items-center gap-2">
                <h3 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Merged model history</h3>
                <button
                  type="button"
                  disabled={clearBusy}
                  onClick={() => void clearHistoryDisk()}
                  className="ml-auto inline-flex items-center gap-1 rounded-md border border-rose-200 px-1.5 py-0.5 text-[10px] font-bold text-rose-700 hover:bg-rose-50 disabled:opacity-40 dark:border-rose-900/50 dark:text-rose-300 dark:hover:bg-rose-950/40"
                  title="Delete Find Best history + pipeline run intermediates to free disk"
                >
                  {clearBusy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
                  Clear history
                </button>
              </div>
              <p className="mb-1.5 text-[10px] text-slate-400">Click a run to inspect its merge pipeline progress.</p>
              {clearMsg && (
                <p className="mb-1.5 text-[10px] text-slate-600 dark:text-slate-300">{clearMsg}</p>
              )}
              <ul className="max-h-52 space-y-1.5 overflow-auto">
                {topicHistory.length === 0 && <li className="text-[11px] text-slate-400">No merged models yet for this topic.</li>}
                {topicHistory.map((h) => (
                  <li
                    key={h.id}
                    className={cn(
                      "rounded-lg border px-2.5 py-2 text-[11px] transition",
                      selectedHistoryId === h.id
                        ? "border-sky-400 bg-sky-50 ring-1 ring-sky-300 dark:border-sky-700 dark:bg-sky-950/30 dark:ring-sky-800"
                        : "border-slate-200 dark:border-slate-700",
                    )}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <button
                        type="button"
                        disabled={loadingHistoryId === h.id}
                        onClick={() => void viewHistoryProgress(h)}
                        className="min-w-0 flex-1 text-left disabled:opacity-60"
                        title="View merge pipeline progress for this run"
                      >
                        <div className="flex items-center gap-1.5">
                          {loadingHistoryId === h.id && <Loader2 className="h-3 w-3 shrink-0 animate-spin text-sky-500" />}
                          <div className="truncate font-semibold text-slate-800 dark:text-slate-100">
                            {h.label || h.element}
                          </div>
                        </div>
                        <div className="truncate text-slate-500">
                          {formatHistoryWhen(h.created_at)}
                          {h.score != null && ` · ${Number(h.score).toFixed(4)}`}
                          {h.label ? ` · ${h.element}` : ""}
                          {` · ${formatBytes(h.onnx_size)}`}
                        </div>
                      </button>
                      <div className="flex shrink-0 items-center gap-1">
                        <button
                          type="button"
                          disabled={!h.has_onnx}
                          onClick={() => setEvalEntry(h)}
                          className="inline-flex items-center gap-1 rounded border border-violet-200 bg-violet-50 px-1.5 py-0.5 font-semibold text-violet-700 disabled:opacity-40 dark:border-violet-800 dark:bg-violet-950/40 dark:text-violet-300"
                          title="Evaluate on uploaded test images"
                        >
                          <FlaskConical className="h-3 w-3" />
                          Eval
                        </button>
                        <button
                          type="button"
                          disabled={!h.has_onnx || loadId === h.id}
                          onClick={() => void loadMiner(h.id)}
                          className="inline-flex items-center gap-1 rounded border border-emerald-200 bg-emerald-50 px-1.5 py-0.5 font-semibold text-emerald-700 disabled:opacity-40 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300"
                          title="Load merged model in labeler (keeps current images)"
                        >
                          <Play className="h-3 w-3" />
                          {loadId === h.id ? "…" : "Load"}
                        </button>
                        <button
                          type="button"
                          disabled={!h.has_onnx || downloadId === h.id}
                          onClick={() => void download(h.id)}
                          className="inline-flex items-center gap-1 rounded border border-sky-200 bg-sky-50 px-1.5 py-0.5 font-semibold text-sky-700 disabled:opacity-40 dark:border-sky-800 dark:bg-sky-950/40 dark:text-sky-300"
                        >
                          <Download className="h-3 w-3" />
                          {downloadId === h.id ? "…" : "ZIP"}
                        </button>
                        <button
                          type="button"
                          disabled={!h.has_onnx || pushId === h.id}
                          onClick={() => {
                            setPushMsg("");
                            setPushFormId((cur) => (cur === h.id ? null : h.id));
                          }}
                          className="inline-flex items-center gap-1 rounded border border-amber-200 bg-amber-50 px-1.5 py-0.5 font-semibold text-amber-800 disabled:opacity-40 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-200"
                          title="Push ship pack to Hugging Face"
                        >
                          <Upload className="h-3 w-3" />
                          {pushId === h.id ? "…" : "Push"}
                        </button>
                      </div>
                    </div>
                    {pushFormId === h.id && (
                      <div className="mt-2 space-y-1.5 rounded-md border border-amber-200/80 bg-amber-50/60 p-2 dark:border-amber-900 dark:bg-amber-950/30">
                        <HfPushTarget
                          accountId={pushAccount}
                          repoName={pushRepo}
                          onChange={(v) => {
                            setPushAccount(v.accountId);
                            setPushRepo(v.repoName);
                            setPushFull(v.repo);
                          }}
                        />
                        <div className="flex items-center justify-end gap-1.5">
                          <button
                            type="button"
                            onClick={() => setPushFormId(null)}
                            className="rounded px-2 py-0.5 text-[10px] font-semibold text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"
                          >
                            Cancel
                          </button>
                          <button
                            type="button"
                            disabled={pushId === h.id || !pushFull.trim()}
                            onClick={() => void pushToHf(h.id)}
                            className="inline-flex items-center gap-1 rounded border border-amber-300 bg-amber-100 px-2 py-0.5 text-[10px] font-bold text-amber-900 disabled:opacity-40 dark:border-amber-700 dark:bg-amber-900/50 dark:text-amber-100"
                          >
                            <Upload className="h-3 w-3" />
                            {pushId === h.id ? "Pushing…" : "Push to HF"}
                          </button>
                        </div>
                      </div>
                    )}
                  </li>
                ))}
              </ul>
              {pushMsg && (
                <p className="mt-1.5 truncate text-[10px] font-medium text-emerald-700 dark:text-emerald-300" title={pushMsg}>
                  {pushMsg}
                </p>
              )}
            </div>
          </div>

          {/* Right: smart step results */}
          <div className="grid min-h-0 gap-3 overflow-hidden lg:grid-cols-[minmax(280px,0.95fr)_minmax(0,1.15fr)]">
            {viewingHistory && (
              <div className="lg:col-span-2 flex items-start justify-between gap-2 rounded-xl border border-violet-300/80 bg-violet-50/80 px-3 py-2 text-[11px] text-violet-950 dark:border-violet-800/60 dark:bg-violet-950/30 dark:text-violet-100">
                <div className="min-w-0">
                  <div className="font-bold">History run · {selectedHistoryEntry?.element || "merged model"}</div>
                  <div className="truncate text-violet-800/90 dark:text-violet-200/90">
                    {formatHistoryWhen(selectedHistoryEntry?.created_at)}
                    {selectedHistoryEntry?.score != null && ` · score ${Number(selectedHistoryEntry.score).toFixed(4)}`}
                    {displayJob?.status && ` · ${displayJob.status}`}
                  </div>
                </div>
                {(running || job) && (
                  <button
                    type="button"
                    onClick={followLive}
                    className="shrink-0 rounded-md border border-violet-300 bg-white px-2 py-1 text-[10px] font-bold text-violet-800 hover:bg-violet-100 dark:border-violet-700 dark:bg-ink-900 dark:text-violet-200"
                  >
                    {running ? "Back to live run" : "Back to current"}
                  </button>
                )}
              </div>
            )}
            <div className="flex min-h-0 flex-col">
              <div className="mb-2 flex items-center justify-between gap-2">
                <h3 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                  {viewingHistory ? "Saved pipeline" : "Pipeline"}
                </h3>
                <div className="flex items-center gap-2">
                  {isPinnedAway && (
                    <button
                      type="button"
                      onClick={followLive}
                      className="text-[10px] font-bold text-sky-600 hover:underline dark:text-sky-300"
                    >
                      Follow live
                    </button>
                  )}
                  {displayJob && (
                    <span className="text-[10px] font-semibold text-slate-500">
                      {displayJob.status}
                      {displayJob.step > 0 ? ` · ${displayJob.step}/${stepTotal}` : ""}
                    </span>
                  )}
                </div>
              </div>
              <p className="mb-2 text-[10px] text-slate-400">
                {viewingHistory
                  ? "Click any step to inspect that merge stage for this saved run."
                  : "Click any finished step to inspect its results — it stays pinned while later steps run."}
              </p>
              <div className="min-h-0 flex-1 space-y-1.5 overflow-auto pr-1">
                {steps.map((s) => (
                  <StepCard
                    key={s.n}
                    step={s}
                    stepTotal={stepTotal}
                    selected={selectedStep === s.n}
                    onSelect={() => selectStep(s.n, s.status)}
                  />
                ))}
              </div>
            </div>
            <StepDetailPanel
              step={activeStep}
              liveStep={liveStepN}
              stepTotal={stepTotal}
              pinned={viewingHistory ? false : isPinnedAway}
              onFollowLive={followLive}
            />
          </div>
        </div>
      </div>
      <EvaluateMergedModal
        open={evalEntry != null}
        historyId={evalEntry?.id || ""}
        title={evalEntry?.label || evalEntry?.element || "merged model"}
        score={evalEntry?.score}
        onClose={() => setEvalEntry(null)}
      />
    </div>
  );
}
