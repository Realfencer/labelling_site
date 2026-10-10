import HfPushTarget from "./HfPushTarget";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Check,
  ChevronRight,
  Download,
  FlaskConical,
  GitBranch,
  History,
  ImagePlus,
  Loader2,
  Plus,
  Sparkles,
  Square,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import {
  api,
  type ManualMergeEvalGtResult,
  type ManualMergeGtImage,
  type ManualMergeHistoryEntry,
  type ManualMergeJob,
  type MergeMachine,
} from "../api";
import type { ChallengeItem, DetectionBox } from "../types";
import { checksFailComplete, checksPassComplete } from "../labelChecks";
import { colorForClass } from "../utils/colors";
import { cn } from "../lib/utils";
import ImageSelectionField from "./ImageSelectionField";
import EvaluateMergedModal from "./EvaluateMergedModal";

const HIST_BINS = 48;
const JOB_LS_KEY = "manual_merge_job_id";

type Step = "gt" | "models" | "merge" | "ship" | "history";
type Quality = "all" | "pass" | "fail";

const STEPS: { id: Step; label: string }[] = [
  { id: "gt", label: "GT range" },
  { id: "models", label: "Models" },
  { id: "merge", label: "Merge" },
  { id: "ship", label: "Ship" },
  { id: "history", label: "History" },
];

function stepForJob(job: ManualMergeJob): Step {
  if (job.ship_ready || job.status === "done") return "ship";
  if (job.best || job.status === "merging" || job.status === "dpp") return "merge";
  if ((job.models?.length || 0) > 0 || job.status === "scoring" || job.status === "ready") return "models";
  return "gt";
}

function rememberJobId(id: string | null | undefined) {
  try {
    if (id) localStorage.setItem(JOB_LS_KEY, id);
    else localStorage.removeItem(JOB_LS_KEY);
  } catch {
    /* ignore */
  }
}

function formatHistoryWhen(iso?: string | null): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString();
  } catch {
    return iso;
  }
}

function formatBytes(n?: number | null): string {
  if (n == null || n <= 0) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}
function OverlayCompare({
  fileUrl,
  width,
  height,
  left,
  right,
  classNames,
  leftTitle,
  rightTitle,
}: {
  fileUrl: string;
  width: number;
  height: number;
  left: DetectionBox[];
  right: DetectionBox[];
  classNames: string[];
  leftTitle: string;
  rightTitle: string;
}) {
  const leftRef = useRef<HTMLCanvasElement>(null);
  const rightRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const paint = (canvas: HTMLCanvasElement | null, boxes: DetectionBox[]) => {
      if (!canvas) return;
      const img = new Image();
      img.onload = () => {
        const maxW = 420;
        const maxH = 320;
        const s = Math.min(maxW / img.width, maxH / img.height, 1);
        const w = Math.max(1, Math.round(img.width * s));
        const h = Math.max(1, Math.round(img.height * s));
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(img, 0, 0, w, h);
        const sx = w / width;
        const sy = h / height;
        for (const box of boxes) {
          const cls = Number((box as DetectionBox & { cls_id?: number }).cls ?? (box as DetectionBox & { cls_id?: number }).cls_id ?? -1);
          const x1 = Math.min(box.x1, box.x2) * sx;
          const y1 = Math.min(box.y1, box.y2) * sy;
          const bw = Math.abs(box.x2 - box.x1) * sx;
          const bh = Math.abs(box.y2 - box.y1) * sy;
          const color = colorForClass(cls >= 0 ? cls : 0);
          ctx.strokeStyle = color;
          ctx.lineWidth = Math.max(2, Math.round(w / 400));
          ctx.strokeRect(x1, y1, bw, bh);
          const name = cls >= 0 ? classNames[cls] ?? `cls ${cls}` : "cls ?";
          const label = (box.raw_conf ?? box.conf) != null ? `${name} ${Math.round(((box.raw_conf ?? box.conf) as number) * 100)}%` : name;
          const fontPx = Math.max(10, Math.round(w / 70));
          ctx.font = `600 ${fontPx}px Inter, Segoe UI, sans-serif`;
          const tw = ctx.measureText(label).width;
          const lh = fontPx + 4;
          ctx.fillStyle = color;
          ctx.fillRect(x1, Math.max(0, y1 - lh), tw + 8, lh);
          ctx.fillStyle = "#fff";
          ctx.fillText(label, x1 + 4, Math.max(fontPx, y1 - 4));
        }
      };
      img.src = fileUrl;
    };
    paint(leftRef.current, left);
    paint(rightRef.current, right);
  }, [fileUrl, width, height, left, right, classNames]);

  return (
    <div className="grid gap-2 md:grid-cols-2">
      <div className="rounded-xl border border-emerald-200 bg-emerald-50/40 p-2 dark:border-emerald-900 dark:bg-emerald-950/20">
        <div className="mb-1 text-[10px] font-bold uppercase tracking-wider text-emerald-700 dark:text-emerald-300">
          {leftTitle} · {left.length}
        </div>
        <canvas ref={leftRef} className="mx-auto max-w-full rounded-md" />
      </div>
      <div className="rounded-xl border border-sky-200 bg-sky-50/40 p-2 dark:border-sky-900 dark:bg-sky-950/20">
        <div className="mb-1 text-[10px] font-bold uppercase tracking-wider text-sky-700 dark:text-sky-300">
          {rightTitle} · {right.length}
        </div>
        <canvas ref={rightRef} className="mx-auto max-w-full rounded-md" />
      </div>
    </div>
  );
}

export default function ManualMergeModal({
  open,
  onClose,
  challenges,
  ownMode = false,
  attachJobId,
}: {
  open: boolean;
  onClose: () => void;
  challenges: ChallengeItem[];
  ownMode?: boolean;
  /** Open this existing job straight away (e.g. one made from Collected models). */
  attachJobId?: string;
}) {
  const [step, setStep] = useState<Step>("gt");
  const [job, setJob] = useState<ManualMergeJob | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [useRemote, setUseRemote] = useState(() => {
    try {
      return localStorage.getItem("sn44-merge-remote") !== "0";
    } catch {
      return true;
    }
  });
  const [machines, setMachines] = useState<MergeMachine[] | null>(null);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [quality, setQuality] = useState<Quality>("all");
  const [picked, setPicked] = useState<string[] | null>(null);
  const [repo, setRepo] = useState("");
  const [revision, setRevision] = useState("main");
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
  const [gtImages, setGtImages] = useState<ManualMergeGtImage[]>([]);
  const [gtStem, setGtStem] = useState("");
  const [gtCompare, setGtCompare] = useState<ManualMergeEvalGtResult | null>(null);
  const [gtUrl, setGtUrl] = useState<string | null>(null);
  const [evalOpen, setEvalOpen] = useState(false);
  const [history, setHistory] = useState<ManualMergeHistoryEntry[]>([]);
  const [historyTopic, setHistoryTopic] = useState("");
  const [historyLoadId, setHistoryLoadId] = useState("");
  const [historyDownloadId, setHistoryDownloadId] = useState("");
  const [clearBusy, setClearBusy] = useState(false);
  const [clearMsg, setClearMsg] = useState("");
  const [booting, setBooting] = useState(false);

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

  const refreshHistory = () => {
    void api
      .manualMergeHistory()
      .then((r) => {
        setHistory(r.history || []);
        setHistoryTopic(r.topic_slug || "");
      })
      .catch(() => {
        /* ignore */
      });
  };

  const clearHistoryDisk = async () => {
    if (clearBusy) return;
    const ok = window.confirm(
      "Clear Manual Merge history on disk?\n\n" +
        "Deletes finished job folders (including large soups/). Running jobs are kept.\n" +
        "Labels and Own images are never deleted.",
    );
    if (!ok) return;
    setClearBusy(true);
    setClearMsg("");
    try {
      const res = await api.manualMergeClearHistory(job?.id);
      refreshHistory();
      setClearMsg(`Freed ${res.freed_human || "disk"} · removed ${res.removed ?? 0} jobs`);
    } catch (e) {
      setClearMsg(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setClearBusy(false);
    }
  };

  const adoptJob = (next: ManualMergeJob) => {
    setJob(next);
    setStep(stepForJob(next));
    rememberJobId(next.id);
    // Don't paste Platform Score block ids into Own mode (or vice versa) — that yields
    // "No Done GT match that range" after a workspace switch / server restart.
    if (!ownMode && next.block_min != null) setFrom(String(next.block_min));
    if (!ownMode && next.block_max != null) setTo(String(next.block_max));
    if (next.quality === "pass" || next.quality === "fail" || next.quality === "all") {
      setQuality(ownMode ? "all" : next.quality);
    }
  };

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setError("");
    setBusy(false);
    setRepo("");
    setRevision("main");
    setPushMsg("");
    setGtImages([]);
    setGtStem("");
    setGtCompare(null);
    setGtUrl(null);
    setEvalOpen(false);
    setFrom(minBlock ? String(minBlock) : "");
    setTo(maxBlock ? String(maxBlock) : "");
    setQuality("all");
    setPicked(null);
    setBooting(true);
    setJob(null);
    setStep("gt");

    const boot = async () => {
      try {
        refreshHistory();
        if (attachJobId) {
          const res = await api.manualMergeStatus(attachJobId);
          if (cancelled) return;
          adoptJob(res.job);
          return;
        }
        const cur = await api.manualMergeCurrent();
        if (cancelled) return;
        if (cur.job && ["scoring", "merging", "dpp"].includes(cur.job.status)) {
          adoptJob(cur.job);
          return;
        }
        let stored = "";
        try {
          stored = localStorage.getItem(JOB_LS_KEY) || "";
        } catch {
          stored = "";
        }
        if (stored) {
          try {
            const res = await api.manualMergeStatus(stored);
            if (cancelled) return;
            if (res.job) {
              adoptJob(res.job);
              return;
            }
          } catch {
            rememberJobId(null);
          }
        }
      } catch {
        /* no live job */
      } finally {
        if (!cancelled) setBooting(false);
      }
    };
    void boot();
    return () => {
      cancelled = true;
    };
  }, [open, minBlock, maxBlock, attachJobId]);

  useEffect(() => {
    if (!open || !job?.id) return;
    if (!["scoring", "merging", "dpp"].includes(job.status)) return;
    let inflight = false; // one status request at a time (the server may be busy running the job itself)
    const t = window.setInterval(() => {
      if (inflight) return;
      inflight = true;
      void api
        .manualMergeStatus(job.id)
        .then((r) => {
          setJob(r.job);
          setError((cur) => (/timed out/i.test(cur) ? "" : cur));
          if (r.job.status === "done" || r.job.status === "error") refreshHistory();
        })
        .catch((e) => {
          const msg = String(e).replace(/^Error:\s*/i, "");
          if (!/timed out|Failed to fetch|NetworkError|aborted/i.test(msg)) setError(msg);
        })
        .finally(() => {
          inflight = false;
        });
    }, 900);
    return () => window.clearInterval(t);
  }, [open, job?.id, job?.status]);

  useEffect(() => {
    if (!job?.id || step !== "ship") return;
    void api
      .manualMergeGtImages(job.id)
      .then((r) => setGtImages(r.images || []))
      .catch(() => setGtImages([]));
  }, [job?.id, step, job?.ship_ready]);

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
  const rangeCount = quality === "pass" ? passCount : quality === "fail" ? failCount : inRange.length;
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

  useEffect(() => {
    if (step !== "models" || !useRemote) return;
    let alive = true;
    setMachines(null);
    api.manualMergeMachines().then((r) => alive && setMachines(r.machines)).catch(() => alive && setMachines([]));
    return () => {
      alive = false;
    };
  }, [step, useRemote]);

  const running = job?.status === "scoring" || job?.status === "merging" || job?.status === "dpp";
  const pairCount = job ? (job.models.length * (job.models.length - 1)) / 2 : 0;

  const createJob = async () => {
    setBusy(true);
    setError("");
    try {
      const res = await api.manualMergeCreate({
        block_min: lo,
        block_max: hi,
        item_keys: picked ?? undefined,
        quality: picked ? "all" : quality,
        n_images: 0,
      });
      adoptJob(res.job);
      setStep("models");
      refreshHistory();
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setBusy(false);
    }
  };

  const loadHistoryJob = async (id: string) => {
    setHistoryLoadId(id);
    setError("");
    try {
      const res = await api.manualMergeStatus(id);
      adoptJob(res.job);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setHistoryLoadId("");
    }
  };

  const downloadHistory = async (id: string) => {
    setHistoryDownloadId(id);
    try {
      await api.manualMergeDownload(id);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setHistoryDownloadId("");
    }
  };

  const addModel = async () => {
    if (!job?.id || !repo.trim()) return;
    setBusy(true);
    setError("");
    try {
      const res = await api.manualMergeAddModel({
        job_id: job.id,
        repo: repo.trim(),
        revision: revision.trim() || "main",
        hf_token: "", // the saved key for the repo owner is used on the server
      });
      setJob(res.job);
      setRepo("");
      setRevision("main");
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setBusy(false);
    }
  };

  const removeModel = async (slug: string) => {
    if (!job?.id) return;
    setBusy(true);
    setError("");
    try {
      const res = await api.manualMergeRemoveModel({ job_id: job.id, slug });
      setJob(res.job);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setBusy(false);
    }
  };

  const scoreModels = async () => {
    if (!job?.id) return;
    setError("");
    try {
      const res = await api.manualMergeScore(job.id);
      setJob(res.job);
      rememberJobId(res.job.id);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    }
  };

  const startMerge = async () => {
    if (!job?.id) return;
    setError("");
    setStep("merge");
    try {
      const res = await api.manualMergeStart(job.id, useRemote);
      setJob(res.job);
      rememberJobId(res.job.id);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    }
  };

  const startDpp = async () => {
    if (!job?.id) return;
    setError("");
    try {
      const res = await api.manualMergeDeepPp(job.id);
      setJob(res.job);
      rememberJobId(res.job.id);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    }
  };

  const pushHf = async () => {
    if (!job?.id) return;
    const repoName = pushFull.trim();
    if (!repoName) {
      setError("Choose a Hugging Face account and a repo name");
      return;
    }
    setBusy(true);
    setError("");
    setPushMsg("");
    try {
      try {
        sessionStorage.setItem("find_best_hf_repo", pushRepo.trim());
      } catch {
        /* ignore */
      }
      const res = await api.manualMergePushToHf(job.id, { token: "", repo: repoName });
      setPushMsg(res.created ? `Created (private) → ${res.url}` : `Updated (private) → ${res.url}`);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setBusy(false);
    }
  };

  const runGtCompare = async (stem: string) => {
    if (!job?.id) return;
    setBusy(true);
    setError("");
    setGtCompare(null);
    if (gtUrl) URL.revokeObjectURL(gtUrl);
    setGtUrl(null);
    try {
      const match = challenges.find((c) => {
        const key = c.item_key || "";
        return key.includes(stem) || String(c.block) === stem || key.endsWith(`:${stem}`);
      });
      const res = await api.manualMergeEvaluateGt({
        job_id: job.id,
        stem,
        item_key: match?.item_key,
        index: match ? challenges.indexOf(match) : undefined,
      });
      setGtCompare(res);
      setGtUrl(`/api/manual-merge/gt-image?job_id=${encodeURIComponent(job.id)}&stem=${encodeURIComponent(stem)}`);
      setGtStem(stem);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (job?.status === "done" && job.ship_ready && step === "merge") {
      setStep("ship");
    }
  }, [job?.status, job?.ship_ready, step]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-2 backdrop-blur-[3px] sm:p-4">
      <div className="flex h-[96vh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-ink-900">
        <div className="flex items-center justify-between border-b border-slate-200 px-5 py-3.5 dark:border-slate-700">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 text-white shadow-lg shadow-violet-500/25">
              <GitBranch className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-base font-extrabold text-slate-900 dark:text-white">Manual merge</h2>
              <p className="text-[12px] text-slate-500">
                Force all-pair soups · {attachJobId ? "images from the Collected working set" : `${count} Done in range`}
                {job?.n_gt ? ` · staged ${job.n_gt}` : ""}
              </p>
            </div>
          </div>
          <button type="button" onClick={onClose} className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex items-center gap-1 border-b border-slate-200 px-4 py-2 dark:border-slate-700">
          {STEPS.map((s, i) => {
            const active = step === s.id;
            const done =
              s.id === "history" ||
              (s.id === "gt" && Boolean(job)) ||
              (s.id === "models" && (job?.models.length || 0) >= 2) ||
              (s.id === "merge" && Boolean(job?.best)) ||
              (s.id === "ship" && Boolean(job?.ship_ready));
            return (
              <button
                key={s.id}
                type="button"
                disabled={s.id !== "gt" && s.id !== "history" && !job}
                onClick={() => {
                  setStep(s.id);
                  if (s.id === "history") refreshHistory();
                }}
                className={cn(
                  "flex flex-1 items-center justify-center gap-1.5 rounded-lg px-2 py-1.5 text-[11px] font-bold transition",
                  active
                    ? "bg-violet-600 text-white shadow-sm"
                    : done
                      ? "bg-violet-50 text-violet-700 dark:bg-violet-950/40 dark:text-violet-200"
                      : "text-slate-400 hover:bg-slate-50 dark:hover:bg-ink-950",
                )}
              >
                {done && !active ? <Check className="h-3 w-3" /> : <span className="opacity-70">{i + 1}</span>}
                {s.label}
              </button>
            );
          })}
        </div>

        <div className="min-h-0 flex-1 overflow-auto p-4">
          {step === "history" && (
            <div className="space-y-3">
              <div className="rounded-xl border border-slate-200 bg-white px-3 py-3 dark:border-slate-700 dark:bg-ink-950">
                <div className="mb-1.5 flex items-center gap-2">
                  <History className="h-3.5 w-3.5 text-violet-500" />
                  <h3 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                    Manual merge history{historyTopic ? ` · ${historyTopic}` : ""}
                  </h3>
                  <button
                    type="button"
                    disabled={clearBusy}
                    onClick={() => void clearHistoryDisk()}
                    className="inline-flex items-center gap-1 rounded-md border border-rose-200 px-1.5 py-0.5 text-[10px] font-bold text-rose-700 hover:bg-rose-50 disabled:opacity-40 dark:border-rose-900/50 dark:text-rose-300"
                    title="Delete finished manual-merge jobs to free disk"
                  >
                    {clearBusy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
                    Clear history
                  </button>
                  <button
                    type="button"
                    onClick={() => refreshHistory()}
                    className="ml-auto text-[10px] font-bold text-violet-600 hover:underline"
                  >
                    Refresh
                  </button>
                </div>
                <p className="mb-2 text-[10px] text-slate-400">
                  Reopen any past run (including live progress after closing the browser). Click a row to load it.
                </p>
                {clearMsg && <p className="mb-2 text-[10px] text-slate-600 dark:text-slate-300">{clearMsg}</p>}
                <ul className="max-h-[60vh] space-y-1.5 overflow-auto">
                  {history.length === 0 && (
                    <li className="text-[11px] text-slate-400">No manual merge jobs yet for this topic.</li>
                  )}
                  {history.map((h) => (
                    <li
                      key={h.id}
                      className={cn(
                        "rounded-lg border px-2.5 py-2 text-[11px]",
                        job?.id === h.id
                          ? "border-violet-400 bg-violet-50 ring-1 ring-violet-300 dark:border-violet-700 dark:bg-violet-950/30 dark:ring-violet-800"
                          : "border-slate-200 dark:border-slate-700",
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <button
                          type="button"
                          disabled={historyLoadId === h.id}
                          onClick={() => void loadHistoryJob(h.id)}
                          className="min-w-0 flex-1 text-left disabled:opacity-60"
                          title="Load this run"
                        >
                          <div className="flex items-center gap-1.5">
                            {historyLoadId === h.id && <Loader2 className="h-3 w-3 shrink-0 animate-spin text-violet-500" />}
                            {["scoring", "merging", "dpp"].includes(String(h.status || "")) && (
                              <span className="rounded bg-amber-100 px-1 py-0.5 text-[9px] font-bold uppercase text-amber-700 dark:bg-amber-950 dark:text-amber-200">
                                live
                              </span>
                            )}
                            <div className="truncate font-mono font-semibold text-slate-800 dark:text-slate-100">{h.id}</div>
                            <span className="shrink-0 text-[10px] uppercase text-slate-400">{h.status}</span>
                          </div>
                          <div className="truncate text-slate-500">
                            {formatHistoryWhen(h.updated_at)}
                            {h.score != null && ` · ${Number(h.score).toFixed(4)}`}
                            {h.n_models != null && ` · ${h.n_models} models`}
                            {h.n_gt ? ` · ${h.n_gt} imgs` : ""}
                            {h.block_min != null && h.block_max != null && ` · ${h.block_min}–${h.block_max}`}
                            {h.has_ship ? ` · ${formatBytes(h.onnx_size)}` : ""}
                          </div>
                        </button>
                        <button
                          type="button"
                          disabled={!h.has_ship || historyDownloadId === h.id}
                          onClick={() => void downloadHistory(h.id)}
                          className="inline-flex shrink-0 items-center gap-1 rounded border border-emerald-200 bg-emerald-50 px-1.5 py-0.5 font-semibold text-emerald-700 disabled:opacity-40 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300"
                          title="Download ship zip"
                        >
                          <Download className="h-3 w-3" />
                          {historyDownloadId === h.id ? "…" : "ZIP"}
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {step === "gt" && (
            <div className="space-y-4">
              {booting && (
                <div className="flex items-center gap-2 rounded-xl border border-violet-200 bg-violet-50 px-3 py-2 text-[12px] text-violet-700 dark:border-violet-900 dark:bg-violet-950/30 dark:text-violet-200">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Checking for a running or recent manual merge…
                </div>
              )}
              {job && ["scoring", "merging", "dpp"].includes(job.status) && (
                <div className="rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-[12px] text-amber-800 dark:border-amber-900 dark:bg-amber-950/30 dark:text-amber-200">
                  Live job <span className="font-mono font-bold">{job.id}</span> · {job.status}
                  {job.message ? ` — ${job.message}` : ""}. Progress continues on the Merge tab.
                  <button
                    type="button"
                    className="ml-2 font-bold underline"
                    onClick={() => setStep(stepForJob(job))}
                  >
                    Resume
                  </button>
                </div>
              )}
              <div className="rounded-2xl border border-slate-200 bg-gradient-to-br from-slate-50 to-violet-50/40 p-4 dark:border-slate-700 dark:from-ink-950 dark:to-violet-950/20">
                <div className="mb-2 text-[10px] font-bold uppercase tracking-wider text-slate-500">Done GT density</div>
                <div className="flex h-12 items-end gap-px">
                  {hist.map((v, i) => (
                    <div key={i} className="flex-1 rounded-t bg-violet-400/80 dark:bg-violet-500/70" style={{ height: `${Math.max(8, v * 100)}%` }} />
                  ))}
                </div>
                <ImageSelectionField suggest={inRange.map((c) => c.item_key)} challenges={challenges} ownMode={ownMode} value={picked} onChange={setPicked} verb="use" className="mt-3" />
                <div className={cn(picked && "pointer-events-none opacity-40")}>
                <div className="mt-3 grid grid-cols-2 gap-2">
                  <label className="block">
                    <span className="mb-1 block text-[10px] font-bold uppercase text-slate-500">From block</span>
                    <input value={from} onChange={(e) => setFrom(e.target.value)} className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-sm dark:border-slate-700 dark:bg-ink-900" />
                  </label>
                  <label className="block">
                    <span className="mb-1 block text-[10px] font-bold uppercase text-slate-500">To block</span>
                    <input value={to} onChange={(e) => setTo(e.target.value)} className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-sm dark:border-slate-700 dark:bg-ink-900" />
                  </label>
                </div>
                <input
                  type="range"
                  min={minBlock}
                  max={maxBlock}
                  value={lo}
                  onChange={(e) => setLo(Number(e.target.value))}
                  className="mt-3 w-full"
                  disabled={!minBlock}
                />
                <input
                  type="range"
                  min={minBlock}
                  max={maxBlock}
                  value={hi}
                  onChange={(e) => setHi(Number(e.target.value))}
                  className="w-full"
                  disabled={!minBlock}
                />
                {!ownMode && (
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {(
                      [
                        ["all", "All", inRange.length],
                        ["pass", "Pass R1–R3", passCount],
                        ["fail", "Fail checks", failCount],
                      ] as const
                    ).map(([k, label, n]) => (
                      <button
                        key={k}
                        type="button"
                        onClick={() => setQuality(k)}
                        className={cn(
                          "rounded-full px-2.5 py-1 text-[11px] font-bold",
                          quality === k
                            ? "bg-violet-600 text-white"
                            : "bg-white text-slate-600 ring-1 ring-slate-200 dark:bg-ink-900 dark:text-slate-300 dark:ring-slate-700",
                        )}
                      >
                        {label} · {n}
                      </button>
                    ))}
                  </div>
                )}
                </div>
                <p className="mt-3 text-[12px] text-slate-500">
                  <span className="font-bold text-violet-700 dark:text-violet-300">{count}</span> {picked ? "hand-picked " : ""}images will be staged for scoring & merging.
                  {ownMode
                    ? " (Own Done labels — synthetic block ids from your local pool.)"
                    : " (Platform Done GT — Score block ids.)"}
                  {" "}
                  <button type="button" className="font-bold text-violet-600 underline" onClick={() => { setStep("history"); refreshHistory(); }}>
                    Open history →
                  </button>
                </p>
                {ownMode && labeled.length === 0 && (
                  <p className="mt-2 rounded-lg border border-amber-200 bg-amber-50 px-2 py-1.5 text-[11px] text-amber-800 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
                    No Own Done labels in this session yet. Hard-refresh Own, wait for Done badges, then Start.
                    For Score-subnet GT, switch to <span className="font-bold">Platform</span> first.
                  </p>
                )}
                {!ownMode && labeled.length === 0 && (
                  <p className="mt-2 rounded-lg border border-amber-200 bg-amber-50 px-2 py-1.5 text-[11px] text-amber-800 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
                    No Platform Done labels loaded. Hard-refresh Platform, wait for the strip, then Start
                    (counts can be stale right after a server restart).
                  </p>
                )}
              </div>
            </div>
          )}

          {step === "models" && job && (
            <div className="space-y-4">
              <div className="rounded-2xl border border-violet-200/80 bg-violet-50/50 p-4 dark:border-violet-900 dark:bg-violet-950/20">
                <div className="mb-2 text-[10px] font-bold uppercase tracking-wider text-violet-600">Add Hugging Face pack</div>
                <div className="grid gap-2 sm:grid-cols-[1.4fr_0.8fr_auto]">
                  <input
                    value={repo}
                    onChange={(e) => setRepo(e.target.value)}
                    placeholder="user/repo"
                    className="rounded-lg border border-slate-200 bg-white px-3 py-2 font-mono text-sm dark:border-slate-700 dark:bg-ink-900"
                  />
                  <input
                    value={revision}
                    onChange={(e) => setRevision(e.target.value)}
                    placeholder="revision (main)"
                    className="rounded-lg border border-slate-200 bg-white px-3 py-2 font-mono text-sm dark:border-slate-700 dark:bg-ink-900"
                  />
                  <button
                    type="button"
                    disabled={busy || !repo.trim()}
                    onClick={() => void addModel()}
                    className="inline-flex items-center justify-center gap-1 rounded-lg bg-violet-600 px-3 py-2 text-sm font-bold text-white disabled:opacity-40"
                  >
                    {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
                    Add
                  </button>
                </div>
                <p className="mt-2 text-[11px] text-slate-400">Private repos work automatically with your saved Hugging Face accounts.</p>
              </div>

              <ul className="space-y-2">
                {job.models.length === 0 && (
                  <li className="rounded-xl border border-dashed border-slate-300 px-4 py-8 text-center text-[12px] text-slate-400 dark:border-slate-700">
                    Add at least 2 models. Example: Realfencer/bev1, WhiteCrow323/bev1, SuperBitDev/bev1 → 3 pairs.
                  </li>
                )}
                {job.models.map((m, idx) => (
                  <li
                    key={m.slug}
                    className="flex items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white px-3 py-2.5 dark:border-slate-700 dark:bg-ink-950"
                  >
                    <div className="min-w-0">
                      <div className="truncate font-mono text-[12px] font-bold text-slate-800 dark:text-slate-100">
                        <span className="mr-2 rounded bg-violet-100 px-1.5 py-0.5 text-[10px] text-violet-700 dark:bg-violet-950 dark:text-violet-200">
                          #{idx + 1}
                        </span>
                        {m.repo_id}
                        <span className="text-slate-400">@{m.revision === "main" ? "main" : m.revision.slice(0, 7)}</span>
                      </div>
                      <div className="text-[11px] text-slate-500">
                        {m.score != null ? (
                          <>
                            score <span className="font-bold tabular-nums text-emerald-600">{Number(m.score).toFixed(4)}</span>
                            {m.map50 != null && ` · mAP50 ${Number(m.map50).toFixed(4)}`}
                          </>
                        ) : m.error ? (
                          <span className="text-rose-500">{m.error}</span>
                        ) : (
                          m.status || "downloaded"
                        )}
                      </div>
                    </div>
                    <button
                      type="button"
                      disabled={busy || running}
                      onClick={() => void removeModel(m.slug)}
                      className="rounded-lg p-1.5 text-slate-400 hover:bg-rose-50 hover:text-rose-600 dark:hover:bg-rose-950/40"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </li>
                ))}
              </ul>

              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  disabled={running || job.models.length < 1}
                  onClick={() => void scoreModels()}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-sky-300 bg-sky-50 px-3 py-2 text-xs font-bold text-sky-800 disabled:opacity-40 dark:border-sky-800 dark:bg-sky-950/40 dark:text-sky-200"
                >
                  {job.status === "scoring" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FlaskConical className="h-3.5 w-3.5" />}
                  Brief test on GT
                </button>
                <span className="text-[11px] text-slate-400">
                  {job.models.length} models → {pairCount} forced pairs
                </span>
              </div>
              <p className="rounded-lg bg-slate-100 px-3 py-2 text-[11px] leading-relaxed text-slate-500 dark:bg-slate-800/60 dark:text-slate-400">
                <b>About these scores:</b> here all images are scored together as one pool. The Collected table scores every block on its own and
                averages them, like the subnet does, so its numbers run about 0.03–0.05 higher. Use the scores here to compare candidates with each other;
                for the exact subnet-style score of a merged result, push it and add it to Collected.
              </p>
            </div>
          )}

          {step === "merge" && job && (
            <div className="space-y-4">
              <div className="rounded-2xl border border-slate-200 bg-gradient-to-br from-slate-50 via-white to-violet-50 p-4 dark:border-slate-700 dark:from-ink-950 dark:via-ink-900 dark:to-violet-950/30">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-[10px] font-bold uppercase tracking-wider text-violet-600">Live merge progress</div>
                    <div className="mt-1 text-sm font-bold text-slate-800 dark:text-slate-100">
                      {job.progress?.headline ||
                        (job.status === "ready" && !job.pair_results?.length && !job.best
                          ? "Merge not started yet — press “Start merge” on the Models step."
                          : job.message)}
                    </div>
                    <div className="mt-1 text-[11px] text-slate-500">
                      {job.progress?.i != null && job.progress?.total != null
                        ? `Pair ${job.progress.i}/${job.progress.total}`
                        : job.progress?.phase || "…"}
                      {job.progress?.pair?.length === 2 && ` · ${job.progress.pair[0]} × ${job.progress.pair[1]}`}
                    </div>
                  </div>
                  {running && <Loader2 className="h-5 w-5 shrink-0 animate-spin text-violet-500" />}
                </div>
                {job.machines && job.machines.length > 0 && running && (
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {job.machines.map((m) => (
                      <span
                        key={m.host}
                        title={m.note || undefined}
                        className={cn(
                          "rounded-md border px-2 py-0.5 text-[10px] font-semibold",
                          (m.running ?? 0) > 0
                            ? "border-emerald-500/60 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                            : m.state === "busy" || m.state === "offline"
                              ? "border-amber-500/50 text-amber-700 dark:text-amber-300"
                              : "border-slate-300 text-slate-500 dark:border-slate-600",
                        )}
                      >
                        {m.host}: {(m.running ?? 0) > 0 ? `running ${m.running}` : m.state}
                        {m.busy_pct != null ? ` · CPU ${m.busy_pct}%` : ""}
                        {(m.done ?? 0) > 0 ? ` · ${m.done} done` : ""}
                      </span>
                    ))}
                  </div>
                )}
                <div className="mt-3 h-2.5 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500 transition-all duration-500"
                    style={{ width: `${Math.min(100, Number(job.progress?.pct || 0))}%` }}
                  />
                </div>
                <div className="mt-3 grid gap-2 sm:grid-cols-3">
                  <div className="rounded-xl border border-slate-200 bg-white/80 px-3 py-2 dark:border-slate-700 dark:bg-ink-950/60">
                    <div className="text-[9px] font-bold uppercase text-slate-400">Best solo (before)</div>
                    <div className="font-mono text-lg font-black tabular-nums text-slate-800 dark:text-slate-100">
                      {job.progress?.best_solo != null
                        ? Number(job.progress.best_solo).toFixed(4)
                        : job.models.find((m) => m.score != null)
                          ? Number([...job.models].sort((a, b) => Number(b.score || -1) - Number(a.score || -1))[0].score).toFixed(4)
                          : "—"}
                    </div>
                  </div>
                  <div className="rounded-xl border border-violet-200 bg-violet-50/80 px-3 py-2 dark:border-violet-800 dark:bg-violet-950/40">
                    <div className="text-[9px] font-bold uppercase text-violet-500">Best merge so far</div>
                    <div className="font-mono text-lg font-black tabular-nums text-violet-800 dark:text-violet-100">
                      {job.progress?.best_so_far != null ? Number(job.progress.best_so_far).toFixed(4) : "—"}
                    </div>
                    {job.progress?.best_so_far_name && (
                      <div className="truncate text-[10px] text-violet-600/80 dark:text-violet-300/80">{job.progress.best_so_far_name}</div>
                    )}
                  </div>
                  <div className="rounded-xl border border-slate-200 bg-white/80 px-3 py-2 dark:border-slate-700 dark:bg-ink-950/60">
                    <div className="text-[9px] font-bold uppercase text-slate-400">This pair best</div>
                    <div className="font-mono text-lg font-black tabular-nums text-slate-800 dark:text-slate-100">
                      {job.progress?.pair_best != null ? Number(job.progress.pair_best).toFixed(4) : "—"}
                    </div>
                    {job.progress?.pair_scores && (
                      <div className="text-[10px] text-slate-500">
                        members {job.progress.pair_scores.map((s) => (s != null ? Number(s).toFixed(3) : "—")).join(" / ")}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {job.models.some((m) => m.score != null) && (
                <div className="rounded-2xl border border-slate-200 p-3 dark:border-slate-700">
                  <div className="mb-2 text-[10px] font-bold uppercase tracking-wider text-slate-500">Before merge · solo scores</div>
                  <ul className="space-y-1">
                    {[...job.models]
                      .sort((a, b) => Number(b.score ?? -1) - Number(a.score ?? -1))
                      .map((m, i) => (
                        <li key={m.slug} className="flex items-center justify-between gap-2 rounded-lg bg-slate-50 px-2.5 py-1.5 text-[11px] dark:bg-ink-950">
                          <span className="truncate font-mono">
                            <span className="mr-1.5 text-slate-400">#{i + 1}</span>
                            {m.repo_id}@{m.revision === "main" ? "main" : m.revision.slice(0, 7)}
                          </span>
                          <span className="font-mono font-bold tabular-nums text-emerald-600">
                            {m.score != null ? Number(m.score).toFixed(4) : "—"}
                          </span>
                        </li>
                      ))}
                  </ul>
                </div>
              )}

              <ul className="max-h-72 space-y-2 overflow-auto">
                {job.pair_results.map((p, i) => (
                  <li key={`${p.a}-${p.b}-${i}`} className="rounded-xl border border-slate-200 px-3 py-2.5 text-[11px] dark:border-slate-700">
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div className="min-w-0 font-semibold text-slate-800 dark:text-slate-100">
                        Pair {i + 1} · {p.a} × {p.b}
                      </div>
                      <div className="font-mono font-bold tabular-nums text-emerald-600">
                        {p.best_score != null ? Number(p.best_score).toFixed(4) : "—"}
                      </div>
                    </div>
                    <div className="mt-0.5 text-slate-500">
                      solo {p.a_score != null ? Number(p.a_score).toFixed(4) : "—"} / {p.b_score != null ? Number(p.b_score).toFixed(4) : "—"}
                      {" · "}tried {p.n_tried} · ok {p.n_ok}
                      {p.best_method && ` · ${p.best_method}`}
                      {p.best_weights && ` · w=[${p.best_weights.map((w) => Number(w).toFixed(2)).join(",")}]`}
                    </div>
                    {p.top_mixes && p.top_mixes.length > 0 && (
                      <div className="mt-1.5 flex flex-wrap gap-1">
                        {p.top_mixes.slice(0, 3).map((m) => (
                          <span
                            key={`${m.name}-${m.score}`}
                            className="rounded-md bg-violet-50 px-1.5 py-0.5 font-mono text-[10px] text-violet-700 dark:bg-violet-950/50 dark:text-violet-200"
                          >
                            {m.score != null ? Number(m.score).toFixed(4) : "—"} {m.method}
                          </span>
                        ))}
                      </div>
                    )}
                  </li>
                ))}
              </ul>

              {job.best && (
                <div className="rounded-2xl border border-emerald-300 bg-emerald-50/80 p-4 dark:border-emerald-800 dark:bg-emerald-950/30">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-emerald-700 dark:text-emerald-300">Winner</div>
                  <div className="mt-1 font-mono text-3xl font-black tabular-nums text-emerald-900 dark:text-emerald-100">
                    {Number(job.best.score).toFixed(4)}
                  </div>
                  <div className="mt-1 text-[12px] text-emerald-900/80 dark:text-emerald-100/80">
                    {job.best.recipe || job.best.name}
                  </div>
                  <div className="mt-2 grid gap-2 text-[11px] sm:grid-cols-3">
                    <div>
                      Method <span className="font-bold">{job.best.method || "—"}</span>
                    </div>
                    <div>
                      Best solo <span className="font-bold tabular-nums">{job.best.best_solo_score != null ? Number(job.best.best_solo_score).toFixed(4) : "—"}</span>
                    </div>
                    <div>
                      Δ vs solo{" "}
                      <span className={cn("font-bold tabular-nums", Number(job.best.delta_vs_best_solo || 0) >= 0 ? "text-emerald-700" : "text-rose-600")}>
                        {job.best.delta_vs_best_solo != null
                          ? `${Number(job.best.delta_vs_best_solo) >= 0 ? "+" : ""}${Number(job.best.delta_vs_best_solo).toFixed(4)}`
                          : "—"}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {step === "ship" && job && (
            <div className="space-y-4">
              {job.best && (
                <div className="rounded-2xl border border-emerald-300 bg-gradient-to-br from-emerald-50 to-white p-4 dark:border-emerald-800 dark:from-emerald-950/40 dark:to-ink-950">
                  <div className="flex flex-wrap items-end justify-between gap-3">
                    <div>
                      <div className="text-[10px] font-bold uppercase tracking-wider text-emerald-700">Best submission</div>
                      <div className="font-mono text-3xl font-black tabular-nums text-emerald-900 dark:text-emerald-100">
                        {Number(job.best.score).toFixed(4)}
                      </div>
                      <div className="mt-1 max-w-xl text-[12px] text-slate-600 dark:text-slate-300">
                        <span className="font-semibold">How:</span> {job.best.method || "mix"} · {job.best.recipe || job.best.name}
                        {job.best.template ? ` · template ${job.best.template}` : ""}
                      </div>
                      <div className="mt-1 text-[11px] text-slate-500">
                        Before best solo {job.best.best_solo_score != null ? Number(job.best.best_solo_score).toFixed(4) : "—"}
                        {job.best.best_solo_repo ? ` (${job.best.best_solo_repo})` : ""}
                        {job.best.delta_vs_best_solo != null && (
                          <>
                            {" "}
                            · Δ{" "}
                            <span className="font-bold text-emerald-700">
                              {Number(job.best.delta_vs_best_solo) >= 0 ? "+" : ""}
                              {Number(job.best.delta_vs_best_solo).toFixed(4)}
                            </span>
                          </>
                        )}
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <button
                        type="button"
                        disabled={!job.ship_ready || running}
                        onClick={() => setEvalOpen(true)}
                        className="inline-flex items-center gap-1 rounded-lg border border-violet-300 bg-violet-50 px-3 py-1.5 text-xs font-bold text-violet-800 dark:border-violet-800 dark:bg-violet-950/40 dark:text-violet-200"
                      >
                        <ImagePlus className="h-3.5 w-3.5" />
                        Drag-drop eval
                      </button>
                      <button
                        type="button"
                        disabled={!job.ship_ready || running}
                        onClick={() => void startDpp()}
                        className="inline-flex items-center gap-1 rounded-lg bg-fuchsia-600 px-3 py-1.5 text-xs font-bold text-white disabled:opacity-40"
                      >
                        {job.status === "dpp" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                        Deep post-process
                      </button>
                      <button
                        type="button"
                        disabled={!job.ship_ready}
                        onClick={() => void api.manualMergeDownload(job.id)}
                        className="inline-flex items-center gap-1 rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-bold text-white"
                      >
                        <Download className="h-3.5 w-3.5" />
                        ZIP
                      </button>
                    </div>
                  </div>
                  {job.best.models_before && job.best.models_before.length > 0 && (
                    <div className="mt-3 grid gap-1 sm:grid-cols-2">
                      {job.best.models_before.map((m, i) => (
                        <div key={`${m.repo_id}-${i}`} className="flex items-center justify-between rounded-lg bg-white/70 px-2 py-1 text-[10px] dark:bg-ink-950/50">
                          <span className="truncate font-mono">{m.repo_id}@{m.revision === "main" ? "main" : String(m.revision || "").slice(0, 7)}</span>
                          <span className="font-mono font-bold tabular-nums">{m.score != null ? Number(m.score).toFixed(4) : "—"}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              <div className="rounded-2xl border border-slate-200 p-4 dark:border-slate-700">
                <div className="mb-2 text-[10px] font-bold uppercase tracking-wider text-slate-500">Compare vs R1 on a Done GT image</div>
                <p className="mb-2 text-[11px] text-slate-400">Scores both merged and R1 against your Done GT labels for that image.</p>
                <div className="flex flex-wrap gap-2">
                  <select
                    value={gtStem}
                    onChange={(e) => setGtStem(e.target.value)}
                    className="min-w-[220px] flex-1 rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-[12px] dark:border-slate-700 dark:bg-ink-900"
                  >
                    <option value="">Select staged GT image…</option>
                    {gtImages.map((g) => (
                      <option key={g.stem} value={g.stem}>
                        {g.block != null ? `block ${g.block} · ` : ""}
                        {g.filename}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    disabled={!gtStem || busy || !job.ship_ready}
                    onClick={() => void runGtCompare(gtStem)}
                    className="inline-flex items-center gap-1 rounded-lg border border-sky-300 bg-sky-50 px-3 py-1.5 text-xs font-bold text-sky-800 disabled:opacity-40 dark:border-sky-800 dark:bg-sky-950/40 dark:text-sky-200"
                  >
                    {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FlaskConical className="h-3.5 w-3.5" />}
                    Compare R1
                  </button>
                </div>
                {gtCompare?.ours.score && gtCompare?.r1.score && (
                  <div className="mt-3 grid gap-2 sm:grid-cols-3">
                    <div className="rounded-xl border border-emerald-200 bg-emerald-50/70 px-3 py-2 dark:border-emerald-900 dark:bg-emerald-950/30">
                      <div className="text-[9px] font-bold uppercase text-emerald-600">Merged vs GT</div>
                      <div className="font-mono text-xl font-black tabular-nums text-emerald-900 dark:text-emerald-100">
                        {Number(gtCompare.ours.score.composite).toFixed(4)}
                      </div>
                      <div className="text-[10px] text-emerald-800/80">
                        mAP50 {Number(gtCompare.ours.score.map50).toFixed(3)} · FP {gtCompare.ours.score.fp} · FN {gtCompare.ours.score.fn}
                      </div>
                    </div>
                    <div className="rounded-xl border border-sky-200 bg-sky-50/70 px-3 py-2 dark:border-sky-900 dark:bg-sky-950/30">
                      <div className="text-[9px] font-bold uppercase text-sky-600">R1 vs GT</div>
                      <div className="font-mono text-xl font-black tabular-nums text-sky-900 dark:text-sky-100">
                        {Number(gtCompare.r1.score.composite).toFixed(4)}
                      </div>
                      <div className="text-[10px] text-sky-800/80">
                        mAP50 {Number(gtCompare.r1.score.map50).toFixed(3)} · FP {gtCompare.r1.score.fp} · FN {gtCompare.r1.score.fn}
                      </div>
                    </div>
                    <div className="rounded-xl border border-violet-200 bg-violet-50/70 px-3 py-2 dark:border-violet-900 dark:bg-violet-950/30">
                      <div className="text-[9px] font-bold uppercase text-violet-600">Δ composite</div>
                      <div
                        className={cn(
                          "font-mono text-xl font-black tabular-nums",
                          Number(gtCompare.delta_composite || 0) >= 0 ? "text-emerald-700" : "text-rose-600",
                        )}
                      >
                        {gtCompare.delta_composite != null
                          ? `${Number(gtCompare.delta_composite) >= 0 ? "+" : ""}${Number(gtCompare.delta_composite).toFixed(4)}`
                          : "—"}
                      </div>
                      <div className="text-[10px] text-violet-800/80">GT boxes {gtCompare.n_gt ?? "—"}</div>
                    </div>
                  </div>
                )}
                {gtCompare && gtUrl && (
                  <div className="mt-3">
                    <OverlayCompare
                      fileUrl={gtUrl}
                      width={gtCompare.width}
                      height={gtCompare.height}
                      left={gtCompare.ours.boxes}
                      right={gtCompare.r1.boxes}
                      classNames={gtCompare.class_names}
                      leftTitle="Merged"
                      rightTitle="R1"
                    />
                  </div>
                )}
                {gtCompare && !gtUrl && (
                  <p className="mt-2 text-[11px] text-amber-600">
                    Compared · merged {gtCompare.ours.n_boxes} boxes vs R1 {gtCompare.r1.n_boxes} boxes (preview image unavailable for this stem).
                  </p>
                )}
              </div>

              <div className="rounded-2xl border border-amber-200 bg-amber-50/50 p-4 dark:border-amber-900 dark:bg-amber-950/20">
                <div className="mb-2 text-[10px] font-bold uppercase tracking-wider text-amber-700">Push private HF repo</div>
                <HfPushTarget
                  accountId={pushAccount}
                  repoName={pushRepo}
                  onChange={(v) => {
                    setPushAccount(v.accountId);
                    setPushRepo(v.repoName);
                    setPushFull(v.repo);
                  }}
                />
                <button
                  type="button"
                  disabled={busy || !job.ship_ready || !pushFull.trim()}
                  onClick={() => void pushHf()}
                  className="mt-2 inline-flex items-center gap-1 rounded-lg border border-amber-300 bg-amber-100 px-3 py-1.5 text-xs font-bold text-amber-900 disabled:opacity-40 dark:border-amber-700 dark:bg-amber-900/40 dark:text-amber-100"
                >
                  <Upload className="h-3.5 w-3.5" />
                  {busy ? "Pushing…" : "Push private"}
                </button>
                {pushMsg && <p className="mt-2 truncate text-[11px] font-medium text-emerald-700 dark:text-emerald-300">{pushMsg}</p>}
              </div>
            </div>
          )}

          {error && (
            <div className="mt-3 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-[12px] text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200">
              {error}
            </div>
          )}
        </div>

        <div className="flex items-center justify-between gap-2 border-t border-slate-200 px-4 py-3 dark:border-slate-700">
          <div className="text-[11px] text-slate-400">{job?.id ? `job ${job.id}` : "Stage GT to begin"}</div>
          <div className="flex items-center gap-2">
            {running && (
              <button
                type="button"
                onClick={() => job?.id && void api.manualMergeStop(job.id).then((r) => setJob(r.job))}
                className="inline-flex items-center gap-1 rounded-lg border border-rose-300 px-3 py-1.5 text-xs font-bold text-rose-600"
              >
                <Square className="h-3 w-3 fill-current" />
                Stop
              </button>
            )}
            <button type="button" onClick={onClose} className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold dark:border-slate-700">
              Close
            </button>
            {step === "gt" && (
              <button
                type="button"
                disabled={busy || count < 1}
                onClick={() => void createJob()}
                className="inline-flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-bold text-white disabled:opacity-40"
              >
                {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ChevronRight className="h-3.5 w-3.5" />}
                Continue
              </button>
            )}
            {step === "models" && (
              <label
                className="mr-1 flex items-center gap-1.5 text-[11px] text-slate-600 dark:text-slate-300"
                title="Pairs run on the two merge machines (194.163.173.245, 75.119.131.0). A busy machine is waited for; this server works only if neither machine can be reached."
              >
                <input
                  type="checkbox"
                  checked={useRemote}
                  onChange={(e) => {
                    setUseRemote(e.target.checked);
                    try {
                      localStorage.setItem("sn44-merge-remote", e.target.checked ? "1" : "0");
                    } catch {
                      /* private mode */
                    }
                  }}
                />
                <span className="font-semibold">Use free machines</span>
                {useRemote && (
                  <span className="text-slate-500">
                    {machines == null
                      ? "checking…"
                      : machines.length === 0
                        ? "status unknown"
                        : machines.map((m) => `${m.host.split(".").slice(-2).join(".")} ${m.state === "free" ? `free (${m.slots})` : m.state}`).join(" · ")}
                  </span>
                )}
              </label>
            )}
            {step === "models" && (
              <button
                type="button"
                disabled={running || (job?.models.length || 0) < 2}
                onClick={() => void startMerge()}
                className="inline-flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-bold text-white disabled:opacity-40"
              >
                <ChevronRight className="h-3.5 w-3.5" />
                Start merge ({pairCount} pairs)
              </button>
            )}
            {step === "merge" && job?.ship_ready && (
              <button
                type="button"
                onClick={() => setStep("ship")}
                className="inline-flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-bold text-white"
              >
                <ChevronRight className="h-3.5 w-3.5" />
                Ship
              </button>
            )}
          </div>
        </div>
      </div>

      <EvaluateMergedModal
        open={evalOpen}
        historyId={job?.id || ""}
        title={job?.best?.name || "manual merge"}
        score={job?.best?.score}
        mode="manual"
        onClose={() => setEvalOpen(false)}
      />
    </div>
  );
}
