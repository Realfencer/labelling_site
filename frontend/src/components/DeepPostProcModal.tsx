import HfPushTarget from "./HfPushTarget";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Check,
  ChevronRight,
  Download,
  FlaskConical,
  History,
  ImagePlus,
  Loader2,
  Sparkles,
  Square,
  Swords,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import {
  api,
  type DppCompareResult,
  type DppEvalImageResult,
  type DppHistoryEntry,
  type DppJob,
} from "../api";
import EvaluateMergedModal from "./EvaluateMergedModal";
import type { ChallengeItem, DetectionBox } from "../types";
import { checksFailComplete, checksPassComplete } from "../labelChecks";
import { colorForClass } from "../utils/colors";
import { cn } from "../lib/utils";
import ImageSelectionField from "./ImageSelectionField";

const HIST_BINS = 48;
const IMAGE_RE = /\.(jpe?g|png|webp|bmp)$/i;
const WEIGHT_RE = /\.(onnx|pt)$/i;

type Tab = "process" | "compare" | "image";
type Quality = "all" | "pass" | "fail";

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

function MetricPill({
  label,
  value,
  tone = "slate",
  hint,
}: {
  label: string;
  value: string;
  tone?: "slate" | "emerald" | "rose" | "sky" | "amber";
  hint?: string;
}) {
  const tones = {
    slate: "border-slate-200 bg-white text-slate-800 dark:border-slate-700 dark:bg-ink-950 dark:text-slate-100",
    emerald: "border-emerald-300/80 bg-emerald-50 text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-200",
    rose: "border-rose-300/80 bg-rose-50 text-rose-800 dark:border-rose-800 dark:bg-rose-950/40 dark:text-rose-200",
    sky: "border-sky-300/80 bg-sky-50 text-sky-800 dark:border-sky-800 dark:bg-sky-950/40 dark:text-sky-200",
    amber: "border-amber-300/80 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-200",
  };
  return (
    <div className={cn("rounded-xl border px-3 py-2", tones[tone])}>
      <div className="text-[9px] font-bold uppercase tracking-wider opacity-70">{label}</div>
      <div className="mt-0.5 font-mono text-lg font-black tabular-nums leading-none">{value}</div>
      {hint && <div className="mt-1 text-[9px] font-semibold leading-tight opacity-70">{hint}</div>}
    </div>
  );
}

function OverlaySide({
  file,
  boxes,
  classNames,
  accent,
  title,
}: {
  file: File;
  boxes: DetectionBox[];
  classNames: string[];
  accent: string;
  title: string;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const url = useMemo(() => URL.createObjectURL(file), [file]);
  useEffect(() => () => URL.revokeObjectURL(url), [url]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const paint = () => {
      const img = new Image();
      img.onload = () => {
        const maxW = Math.max(280, wrap.clientWidth - 4);
        const maxH = Math.max(200, wrap.clientHeight - 4);
        const s = Math.min(maxW / img.width, maxH / img.height, 1);
        const w = Math.max(1, Math.round(img.width * s));
        const h = Math.max(1, Math.round(img.height * s));
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(img, 0, 0, w, h);
        const sx = w / img.width;
        const sy = h / img.height;
        for (const box of boxes) {
          const x1 = Math.min(box.x1, box.x2) * sx;
          const y1 = Math.min(box.y1, box.y2) * sy;
          const bw = Math.abs(box.x2 - box.x1) * sx;
          const bh = Math.abs(box.y2 - box.y1) * sy;
          const color = colorForClass(box.cls);
          ctx.strokeStyle = color;
          ctx.lineWidth = Math.max(2, Math.round(w / 380));
          ctx.strokeRect(x1, y1, bw, bh);
          const name = classNames[box.cls] ?? `cls ${box.cls}`;
          const label = (box.raw_conf ?? box.conf) != null ? `${name} ${Math.round(((box.raw_conf ?? box.conf) as number) * 100)}%` : name;
          const fontPx = Math.max(10, Math.round(w / 72));
          ctx.font = `600 ${fontPx}px Inter, Segoe UI, sans-serif`;
          const tw = ctx.measureText(label).width;
          ctx.fillStyle = color;
          ctx.fillRect(x1, Math.max(0, y1 - fontPx - 4), tw + 8, fontPx + 4);
          ctx.fillStyle = "#fff";
          ctx.fillText(label, x1 + 4, Math.max(fontPx, y1 - 4));
        }
      };
      img.src = url;
    };
    paint();
    const ro = new ResizeObserver(() => paint());
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [url, boxes, classNames]);

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-slate-200 bg-slate-950/90 dark:border-slate-700">
      <div className={cn("px-2.5 py-1.5 text-[10px] font-bold uppercase tracking-wider text-white", accent)}>
        {title} · {boxes.length} boxes
      </div>
      <div ref={wrapRef} className="flex min-h-[220px] flex-1 items-center justify-center p-2">
        <canvas ref={canvasRef} className="max-h-full max-w-full rounded-md" />
      </div>
    </div>
  );
}

export default function DeepPostProcModal({
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
  const fileRef = useRef<HTMLInputElement>(null);
  const imgRef = useRef<HTMLInputElement>(null);
  const [tab, setTab] = useState<Tab>("process");
  const [job, setJob] = useState<DppJob | null>(null);
  const [minGt, setMinGt] = useState(20);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [quality, setQuality] = useState<Quality>("all");
  const [picked, setPicked] = useState<string[] | null>(null);
  const [hfRepo, setHfRepo] = useState("Alexei999/carwash004");
  const [compare, setCompare] = useState<DppCompareResult | null>(null);
  const [imgFile, setImgFile] = useState<File | null>(null);
  const [imgResult, setImgResult] = useState<DppEvalImageResult | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [downloadBusy, setDownloadBusy] = useState(false);
  const [history, setHistory] = useState<DppHistoryEntry[]>([]);
  const [topicSlug, setTopicSlug] = useState("");
  const [historyDownloadId, setHistoryDownloadId] = useState("");
  const [historyLoadId, setHistoryLoadId] = useState("");
  const [clearBusy, setClearBusy] = useState(false);
  const [clearMsg, setClearMsg] = useState("");
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
  const [evalEntry, setEvalEntry] = useState<DppHistoryEntry | null>(null);

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

  useEffect(() => {
    if (!open) return;
    setTab("process");
    setError("");
    setCompare(null);
    setImgFile(null);
    setImgResult(null);
    setFrom(minBlock ? String(minBlock) : "");
    setTo(maxBlock ? String(maxBlock) : "");
    setQuality("all");
    setPicked(null);
    void api
      .dppDefaults()
      .then((d) => {
        setMinGt(d.min_gt);
        setTopicSlug(d.topic_slug || "");
      })
      .catch(() => undefined);
    void api
      .dppHistory()
      .then((res) => {
        setHistory(res.history || []);
        setTopicSlug(res.topic_slug || "");
      })
      .catch(() => setHistory([]));
  }, [open, minBlock, maxBlock]);

  // Poll while running
  useEffect(() => {
    if (!open || !job?.id) return;
    if (job.status !== "running" && job.status !== "comparing") return;
    // One status request at a time: while the server is busy with the search itself, a new request every
    // 0.8 s only piles up. A slow/timed-out check is normal then — keep trying quietly, and clear a
    // "timed out" banner as soon as a check goes through.
    let inflight = false;
    const t = window.setInterval(() => {
      if (inflight) return;
      inflight = true;
      void api
        .dppStatus(job.id)
        .then((r) => {
          setJob(r.job);
          setError((cur) => (/timed out|Lost connection/i.test(cur) ? "" : cur));
        })
        .catch((e) => {
          const msg = String(e).replace(/^Error:\s*/i, "");
          if (!/timed out|Failed to fetch|NetworkError|aborted/i.test(msg)) setError(msg || "Lost connection to job status");
        })
        .finally(() => {
          inflight = false;
        });
    }, 800);
    return () => window.clearInterval(t);
  }, [open, job?.id, job?.status]);

  useEffect(() => {
    if (!open || job?.status !== "done") return;
    void api
      .dppHistory()
      .then((res) => setHistory(res.history || []))
      .catch(() => undefined);
  }, [open, job?.status, job?.id]);

  const clearHistoryDisk = async () => {
    if (clearBusy) return;
    const ok = window.confirm(
      "Clear Deep Post-Process history on disk?\n\n" +
        "Deletes finished DPP job folders. Running jobs are kept.\n" +
        "Labels and Own images are never deleted.",
    );
    if (!ok) return;
    setClearBusy(true);
    setClearMsg("");
    try {
      const res = await api.dppClearHistory(job?.id);
      const hist = await api.dppHistory().catch(() => ({ history: [] as typeof history }));
      setHistory(hist.history || []);
      setClearMsg(`Freed ${res.freed_human || "disk"} · removed ${res.removed ?? 0} jobs`);
    } catch (e) {
      setClearMsg(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setClearBusy(false);
    }
  };

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

  const uploadWeights = async (file: File) => {
    setBusy(true);
    setError("");
    try {
      const res = await api.dppUpload(file);
      setJob(res.job);
      if (!res.job.verify?.ok) {
        setError(res.job.verify?.error || res.job.error || "Verification failed");
      }
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setBusy(false);
    }
  };

  const runDpp = async () => {
    if (!job?.id) return;
    setBusy(true);
    setError("");
    try {
      const res = await api.dppRun({
        job_id: job.id,
        block_min: lo,
        block_max: hi,
        item_keys: picked ?? undefined,
        quality: picked ? "all" : quality,
        n_images: 0,
      });
      setJob(res.job);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setBusy(false);
    }
  };

  const runCompare = async () => {
    if (!job?.id) return;
    setBusy(true);
    setError("");
    try {
      const res = await api.dppCompare({
        job_id: job.id,
        hf_repo: hfRepo.trim(),
        block_min: lo,
        block_max: hi,
        item_keys: picked ?? undefined,
        quality: picked ? "all" : quality,
      });
      setJob(res.job);
      setCompare(res.compare);
      setTab("compare");
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setBusy(false);
    }
  };

  const runImage = useCallback(
    async (file: File) => {
      if (!job?.id) return;
      setImgFile(file);
      setImgResult(null);
      setBusy(true);
      setError("");
      try {
        const res = await api.dppEvaluateImage(job.id, file, hfRepo.trim() || undefined);
        setImgResult(res);
        setTab("image");
      } catch (e) {
        setError(String(e).replace(/^Error:\s*/i, ""));
      } finally {
        setBusy(false);
      }
    },
    [job?.id, hfRepo],
  );

  if (!open) return null;

  const verified = Boolean(job?.verify?.ok);
  const shipReady = Boolean(job?.ship_ready);

  const downloadShip = async () => {
    if (!job?.id) return;
    setDownloadBusy(true);
    setError("");
    try {
      await api.dppDownload(job.id);
      await new Promise((r) => window.setTimeout(r, 400));
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setDownloadBusy(false);
    }
  };

  const downloadHistory = async (id: string) => {
    setHistoryDownloadId(id);
    setError("");
    try {
      await api.dppDownload(id);
      // Native <a download> returns immediately; brief busy state for click feedback.
      await new Promise((r) => window.setTimeout(r, 400));
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setHistoryDownloadId("");
    }
  };

  const loadHistoryJob = async (id: string) => {
    setHistoryLoadId(id);
    setError("");
    try {
      const res = await api.dppStatus(id);
      setJob(res.job);
      setCompare(res.job.compare || null);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setHistoryLoadId("");
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
      const res = await api.dppPushToHf(id, { token: "", repo });
      setPushMsg(res.created ? `Created and pushed → ${res.url}` : `Updated → ${res.url}`);
      setPushFormId(null);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setPushId(null);
    }
  };

  const running = job?.status === "running" || job?.status === "comparing";
  const delta =
    job?.baseline_score && job?.postproc?.final_live
      ? Number(job.postproc.final_live.score) - Number(job.baseline_score.score)
      : job?.baseline_score && job?.postproc
        ? Number(job.postproc.score) - Number(job.baseline_score.score)
        : null;

  const cmp = compare || job?.compare || null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-3 backdrop-blur-[3px]">
      <div className="flex max-h-[min(92vh,860px)] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-ink-900">
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-700">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 text-white shadow-lg shadow-violet-500/25">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-sm font-extrabold text-slate-900 dark:text-white">Deep post-process</h2>
              <p className="text-[11px] text-slate-500">
                Verify · tune · ship pack · compare HF · single-image duel
              </p>
            </div>
          </div>
          <button type="button" onClick={onClose} className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex gap-1 border-b border-slate-200 px-3 pt-2 dark:border-slate-700">
          {(
            [
              ["process", "Process", Sparkles],
              ["compare", "Compare", Swords],
              ["image", "One image", FlaskConical],
            ] as const
          ).map(([id, label, Icon]) => (
            <button
              key={id}
              type="button"
              onClick={() => setTab(id)}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-t-lg border border-b-0 px-3 py-1.5 text-[11px] font-bold",
                tab === id
                  ? "border-slate-200 bg-white text-violet-700 dark:border-slate-700 dark:bg-ink-900 dark:text-violet-300"
                  : "border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-200",
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              {label}
            </button>
          ))}
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          {tab === "process" && (
            <div className="space-y-4">
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOver(false);
                  const f = Array.from(e.dataTransfer.files || []).find((x) => WEIGHT_RE.test(x.name));
                  if (f) void uploadWeights(f);
                }}
                className={cn(
                  "rounded-2xl border-2 border-dashed px-4 py-6 text-center transition",
                  dragOver
                    ? "border-violet-400 bg-violet-50 dark:bg-violet-950/30"
                    : "border-slate-200 bg-slate-50 dark:border-slate-700 dark:bg-ink-950",
                )}
              >
                <Upload className="mx-auto h-7 w-7 text-violet-500" />
                <p className="mt-2 text-sm font-bold text-slate-800 dark:text-slate-100">
                  Drop .onnx or .pt weights
                </p>
                <p className="mt-0.5 text-[11px] text-slate-500">
                  We verify class count for this topic, then wrap with a SN44 miner.py template
                </p>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => fileRef.current?.click()}
                  className="mt-3 inline-flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-bold text-white hover:bg-violet-500 disabled:opacity-50"
                >
                  {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Upload className="h-3.5 w-3.5" />}
                  Choose file
                </button>
                <input
                  ref={fileRef}
                  type="file"
                  accept=".onnx,.pt"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) void uploadWeights(f);
                    e.target.value = "";
                  }}
                />
              </div>

              {job && (
                <div
                  className={cn(
                    "rounded-xl border px-3 py-2.5",
                    verified
                      ? "border-emerald-300 bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-950/30"
                      : "border-rose-300 bg-rose-50 dark:border-rose-800 dark:bg-rose-950/30",
                  )}
                >
                  <div className="flex items-center gap-2 text-xs font-bold">
                    {verified ? <Check className="h-4 w-4 text-emerald-600" /> : <X className="h-4 w-4 text-rose-600" />}
                    {verified ? "Verified for this topic" : "Not available for this topic"}
                  </div>
                  <p className="mt-1 text-[11px] text-slate-600 dark:text-slate-300">
                    Topic classes ({job.verify?.topic_n_classes}): {(job.class_names || []).join(", ") || "—"}
                    {job.verify?.model_n_classes != null && ` · model ≈ ${job.verify.model_n_classes}`}
                  </p>
                  {job.verify?.warning && (
                    <p className="mt-1 text-[11px] text-amber-700 dark:text-amber-300">{job.verify.warning}</p>
                  )}
                </div>
              )}

              <ImageSelectionField suggest={inRange.map((c) => c.item_key)} challenges={challenges} ownMode={ownMode} value={picked} onChange={setPicked} verb="use" minCount={minGt} disabled={running} />
              <div className={cn("rounded-xl border border-slate-200 bg-slate-50 px-3 pb-3 pt-3 dark:border-slate-700 dark:bg-ink-950", picked && "pointer-events-none opacity-40")}>
                <div className="mb-2 flex h-8 items-end gap-px">
                  {hist.map((h, i) => {
                    const binLo = minBlock + (i / HIST_BINS) * span;
                    const binHi = minBlock + ((i + 1) / HIST_BINS) * span;
                    const active = binHi >= lo && binLo <= hi;
                    return (
                      <div
                        key={i}
                        className={cn("min-h-[2px] flex-1 rounded-sm", active ? "bg-violet-500" : "bg-slate-300 dark:bg-slate-700")}
                        style={{ height: `${Math.max(8, h * 100)}%` }}
                      />
                    );
                  })}
                </div>
                <label className="mb-2 block">
                  <span className="mb-0.5 flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-500">
                    From <span className="font-mono tabular-nums">{lo}</span>
                  </span>
                  <input type="range" min={minBlock} max={maxBlock || 1} value={lo} disabled={!minBlock} onChange={(e) => setLo(Number(e.target.value))} className="w-full accent-violet-500" />
                </label>
                <label className="block">
                  <span className="mb-0.5 flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-500">
                    To <span className="font-mono tabular-nums">{hi}</span>
                  </span>
                  <input type="range" min={minBlock} max={maxBlock || 1} value={hi} disabled={!minBlock} onChange={(e) => setHi(Number(e.target.value))} className="w-full accent-violet-500" />
                </label>
                {!ownMode && (
                  <div className="mt-2 grid grid-cols-3 gap-1.5">
                    {(
                      [
                        ["all", "All", inRange.length],
                        ["pass", "Pass", passCount],
                        ["fail", "Fail", failCount],
                      ] as const
                    ).map(([id, label, n]) => (
                      <button
                        key={id}
                        type="button"
                        onClick={() => setQuality(id)}
                        className={cn(
                          "rounded-lg border px-2 py-1.5 text-[11px] font-bold",
                          quality === id
                            ? "border-violet-500 bg-violet-500 text-white"
                            : "border-slate-200 text-slate-600 dark:border-slate-700 dark:text-slate-300",
                        )}
                      >
                        {label} <span className="tabular-nums opacity-80">{n}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <p className="text-[11px] text-slate-500">
                {picked ? `${picked.length} hand-picked images` : `${ownMode ? inRange.length : count} Done images in range`} · need ≥{minGt} for deep post-process
              </p>
              {(picked ? picked.length : ownMode ? inRange.length : count) < 150 && (
                <p className="rounded-lg border border-amber-300/80 bg-amber-50 px-3 py-2 text-[11px] text-amber-900 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-200">
                  Few images: settings tuned on a small set rarely hold up on other images. Use the whole Done range
                  (hundreds of images) — 30% of them are held out to check the gain is real.
                </p>
              )}

              {(job?.baseline_score || job?.postproc) && (
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-6">
                  <MetricPill label="Before" value={job.baseline_score?.score != null ? Number(job.baseline_score.score).toFixed(4) : "—"} />
                  <MetricPill
                    label="After"
                    value={
                      job.postproc?.final_live?.score != null
                        ? Number(job.postproc.final_live.score).toFixed(4)
                        : job.postproc?.score != null
                          ? Number(job.postproc.score).toFixed(4)
                          : "—"
                    }
                    tone="emerald"
                  />
                  <MetricPill
                    label="Δ score"
                    value={delta != null ? `${delta >= 0 ? "+" : ""}${delta.toFixed(4)}` : "—"}
                    tone={delta != null && delta >= 0 ? "emerald" : "rose"}
                  />
                  <MetricPill
                    label="Held-out Δ"
                    value={
                      job.postproc?.holdout
                        ? `${job.postproc.holdout.delta >= 0 ? "+" : ""}${Number(job.postproc.holdout.delta).toFixed(4)}`
                        : "—"
                    }
                    hint={
                      job.postproc?.holdout
                        ? `${Number(job.postproc.holdout.before).toFixed(4)} → ${Number(job.postproc.holdout.after).toFixed(4)} on ${job.postproc.holdout.n_holdout} images the search never tuned on` +
                          (job.postproc.reverted ? ` · tuned settings lost there (${Number(job.postproc.holdout.candidate_delta).toFixed(4)}), original kept` : "")
                        : "Only shown for the robust search (array / gated miners)"
                    }
                    tone={job.postproc?.holdout ? (job.postproc.holdout.delta > 0 ? "emerald" : "amber") : "slate"}
                  />
                  <MetricPill label="mAP50" value={job.postproc?.map50 != null ? Number(job.postproc.map50).toFixed(4) : "—"} tone="sky" />
                  <MetricPill
                    label="p95 lat"
                    value={
                      job.postproc?.latency?.p95_ms != null
                        ? `${Number(job.postproc.latency.p95_ms).toFixed(0)}ms`
                        : "—"
                    }
                    hint={
                      job.postproc?.latency?.normalized
                        ? `idle-equivalent · measured ${Number(job.postproc.latency.p95_raw_ms).toFixed(0)}ms while busy (winner ${Number(job.postproc.latency.ref_ms).toFixed(0)}ms)`
                        : undefined
                    }
                    tone={
                      job.postproc?.latency?.passed === true
                        ? "emerald"
                        : job.postproc?.latency?.passed === false
                          ? "rose"
                          : "amber"
                    }
                  />
                </div>
              )}

              {job?.message && (
                <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-[12px] text-slate-600 dark:border-slate-700 dark:bg-ink-950 dark:text-slate-300">
                  {running && <Loader2 className="mr-1.5 inline h-3.5 w-3.5 animate-spin text-violet-500" />}
                  {job.message}
                </div>
              )}

              {running && job?.progress && (
                <div className="rounded-xl border border-violet-200 bg-violet-50/80 px-3 py-3 dark:border-violet-800 dark:bg-violet-950/30">
                  <div className="mb-1.5 flex items-center justify-between gap-2 text-[11px] font-semibold text-violet-800 dark:text-violet-200">
                    <span className="truncate">
                      Live search
                      {job.progress.style ? ` · ${job.progress.style}` : ""}
                      {job.progress.phase ? ` · ${job.progress.phase}` : ""}
                    </span>
                    <span className="shrink-0 font-mono tabular-nums">
                      {job.progress.trial ?? 0}/{job.progress.total ?? "?"}
                      {job.progress.pct != null ? ` · ${Number(job.progress.pct).toFixed(0)}%` : ""}
                    </span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-violet-200/80 dark:bg-violet-900/60">
                    <div
                      className="h-full rounded-full bg-violet-500 transition-[width] duration-300 ease-out dark:bg-violet-400"
                      style={{ width: `${Math.max(2, Math.min(100, Number(job.progress.pct) || 0))}%` }}
                    />
                  </div>
                  <div className="mt-2 grid grid-cols-3 gap-2">
                    <div className="rounded-lg border border-violet-200/80 bg-white/80 px-2 py-1.5 dark:border-violet-800 dark:bg-ink-950/60">
                      <div className="text-[9px] font-bold uppercase tracking-wider text-violet-500">Trial score</div>
                      <div className="font-mono text-sm font-black tabular-nums text-slate-800 dark:text-slate-100">
                        {job.progress.score != null ? Number(job.progress.score).toFixed(4) : "—"}
                      </div>
                    </div>
                    <div className="rounded-lg border border-emerald-200/80 bg-white/80 px-2 py-1.5 dark:border-emerald-800 dark:bg-ink-950/60">
                      <div className="text-[9px] font-bold uppercase tracking-wider text-emerald-600">Best so far</div>
                      <div className="font-mono text-sm font-black tabular-nums text-emerald-700 dark:text-emerald-300">
                        {job.progress.best != null ? Number(job.progress.best).toFixed(4) : "—"}
                      </div>
                    </div>
                    <div className="rounded-lg border border-sky-200/80 bg-white/80 px-2 py-1.5 dark:border-sky-800 dark:bg-ink-950/60">
                      <div className="text-[9px] font-bold uppercase tracking-wider text-sky-600">mAP50</div>
                      <div className="font-mono text-sm font-black tabular-nums text-slate-800 dark:text-slate-100">
                        {job.progress.map50 != null ? Number(job.progress.map50).toFixed(4) : "—"}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="rounded-xl border border-slate-200 bg-white px-3 py-3 dark:border-slate-700 dark:bg-ink-950">
                <div className="mb-1.5 flex items-center gap-2">
                  <History className="h-3.5 w-3.5 text-violet-500" />
                  <h3 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                    Post-process history{topicSlug ? ` · ${topicSlug}` : ""}
                  </h3>
                  <button
                    type="button"
                    disabled={clearBusy}
                    onClick={() => void clearHistoryDisk()}
                    className="ml-auto inline-flex items-center gap-1 rounded-md border border-rose-200 px-1.5 py-0.5 text-[10px] font-bold text-rose-700 hover:bg-rose-50 disabled:opacity-40 dark:border-rose-900/50 dark:text-rose-300"
                    title="Delete finished deep-pp jobs to free disk"
                  >
                    {clearBusy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
                    Clear history
                  </button>
                </div>
                <p className="mb-2 text-[10px] text-slate-400">
                  Eval, push to HF, or download ship packs from earlier deep post-process runs for this topic.
                </p>
                {clearMsg && <p className="mb-2 text-[10px] text-slate-600 dark:text-slate-300">{clearMsg}</p>}
                <ul className="max-h-52 space-y-1.5 overflow-auto">
                  {history.length === 0 && (
                    <li className="text-[11px] text-slate-400">No completed post-process runs yet for this topic.</li>
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
                          title="Load this run in the modal"
                        >
                          <div className="flex items-center gap-1.5">
                            {historyLoadId === h.id && <Loader2 className="h-3 w-3 shrink-0 animate-spin text-violet-500" />}
                            <div className="truncate font-mono font-semibold text-slate-800 dark:text-slate-100">{h.id}</div>
                          </div>
                          <div className="truncate text-slate-500">
                            {formatHistoryWhen(h.updated_at)}
                            {h.score != null && ` · ${Number(h.score).toFixed(4)}`}
                            {h.delta != null && ` · Δ ${h.delta >= 0 ? "+" : ""}${Number(h.delta).toFixed(4)}`}
                            {` · ${formatBytes(h.onnx_size)}`}
                            {h.n_gt ? ` · ${h.n_gt} imgs` : ""}
                          </div>
                        </button>
                        <div className="flex shrink-0 items-center gap-1">
                          <button
                            type="button"
                            disabled={!h.has_ship}
                            onClick={() => setEvalEntry(h)}
                            className="inline-flex items-center gap-1 rounded border border-violet-200 bg-violet-50 px-1.5 py-0.5 font-semibold text-violet-700 disabled:opacity-40 dark:border-violet-800 dark:bg-violet-950/40 dark:text-violet-300"
                            title="Evaluate on uploaded test images"
                          >
                            <FlaskConical className="h-3 w-3" />
                            Eval
                          </button>
                          <button
                            type="button"
                            disabled={!h.has_ship || historyDownloadId === h.id}
                            onClick={() => void downloadHistory(h.id)}
                            className="inline-flex items-center gap-1 rounded border border-emerald-200 bg-emerald-50 px-1.5 py-0.5 font-semibold text-emerald-700 disabled:opacity-40 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300"
                            title="Download ship zip"
                          >
                            <Download className="h-3 w-3" />
                            {historyDownloadId === h.id ? "…" : "ZIP"}
                          </button>
                          <button
                            type="button"
                            disabled={!h.has_ship || pushId === h.id}
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
          )}

          {tab === "compare" && (
            <div className="space-y-4">
              <label className="block">
                <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">
                  Hugging Face miner repo
                </span>
                <input
                  value={hfRepo}
                  onChange={(e) => setHfRepo(e.target.value)}
                  placeholder="Alexei999/carwash004"
                  className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 font-mono text-sm dark:border-slate-700 dark:bg-ink-950"
                />
              </label>
              <p className="text-[11px] text-slate-500">
                Downloads miner.py + weights.onnx + chute_config.yml and scores both packs on the same Done GT range.
              </p>
              <button
                type="button"
                disabled={busy || !shipReady || !hfRepo.trim()}
                onClick={() => void runCompare()}
                className="inline-flex items-center gap-1.5 rounded-lg bg-fuchsia-600 px-4 py-2 text-sm font-bold text-white hover:bg-fuchsia-500 disabled:opacity-40"
              >
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Swords className="h-4 w-4" />}
                Compare on selected range
              </button>

              {cmp && (
                <div className="overflow-hidden rounded-2xl border border-slate-200 bg-gradient-to-br from-slate-50 via-white to-violet-50 dark:border-slate-700 dark:from-ink-950 dark:via-ink-900 dark:to-violet-950/40">
                  <div className="border-b border-slate-200/80 px-4 py-3 dark:border-slate-700">
                    <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Duel result</div>
                    <div className="mt-1 text-lg font-black text-slate-900 dark:text-white">
                      {cmp.winner === "ours" ? "Your ship wins" : cmp.winner === "theirs" ? `${cmp.hf_repo} wins` : "Tie"}
                      <span className="ml-2 font-mono text-sm font-bold text-violet-600 dark:text-violet-300">
                        Δ {cmp.delta_score >= 0 ? "+" : ""}
                        {Number(cmp.delta_score).toFixed(4)}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-500">{cmp.n_images} images · same GT range</div>
                  </div>
                  <div className="grid gap-3 p-4 sm:grid-cols-2">
                    <div className="rounded-xl border border-emerald-200 bg-emerald-50/80 p-3 dark:border-emerald-800 dark:bg-emerald-950/30">
                      <div className="text-[10px] font-bold uppercase text-emerald-700 dark:text-emerald-300">Yours (post-processed)</div>
                      <div className="mt-1 font-mono text-2xl font-black tabular-nums text-emerald-900 dark:text-emerald-100">
                        {Number(cmp.ours.score).toFixed(4)}
                      </div>
                      <div className="mt-2 grid grid-cols-2 gap-2 text-[11px]">
                        <div>mAP50 <span className="font-bold tabular-nums">{Number(cmp.ours.map50).toFixed(4)}</span></div>
                        <div>FFPI <span className="font-bold tabular-nums">{Number(cmp.ours.ffpi).toFixed(3)}</span></div>
                      </div>
                    </div>
                    <div className="rounded-xl border border-slate-200 bg-white/80 p-3 dark:border-slate-700 dark:bg-ink-950/60">
                      <div className="text-[10px] font-bold uppercase text-slate-500">{cmp.hf_repo}</div>
                      <div className="mt-1 font-mono text-2xl font-black tabular-nums text-slate-900 dark:text-white">
                        {Number(cmp.theirs.score).toFixed(4)}
                      </div>
                      <div className="mt-2 grid grid-cols-2 gap-2 text-[11px]">
                        <div>mAP50 <span className="font-bold tabular-nums">{Number(cmp.theirs.map50).toFixed(4)}</span></div>
                        <div>FFPI <span className="font-bold tabular-nums">{Number(cmp.theirs.ffpi).toFixed(3)}</span></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {tab === "image" && (
            <div className="space-y-3">
              <p className="text-[11px] text-slate-500">
                Run your ship (and optional HF repo) on one photo — side-by-side detections.
              </p>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  disabled={busy || !job?.id}
                  onClick={() => imgRef.current?.click()}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-2 text-xs font-bold text-white disabled:opacity-40"
                >
                  <ImagePlus className="h-3.5 w-3.5" />
                  Upload image
                </button>
                <input
                  ref={imgRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f && (IMAGE_RE.test(f.name) || f.type.startsWith("image/"))) void runImage(f);
                    e.target.value = "";
                  }}
                />
              </div>
              {imgFile && imgResult && (
                <div className="grid min-h-[260px] gap-2 md:grid-cols-2">
                  <OverlaySide
                    file={imgFile}
                    boxes={(imgResult.ours.boxes || []) as DetectionBox[]}
                    classNames={imgResult.ours.class_names || imgResult.class_names || []}
                    accent="bg-emerald-600"
                    title="Yours"
                  />
                  {imgResult.theirs ? (
                    <OverlaySide
                      file={imgFile}
                      boxes={(imgResult.theirs.boxes || []) as DetectionBox[]}
                      classNames={imgResult.theirs.class_names || imgResult.class_names || []}
                      accent="bg-slate-700"
                      title={String(imgResult.theirs.hf_repo || hfRepo || "HF")}
                    />
                  ) : (
                    <div className="flex items-center justify-center rounded-xl border border-dashed border-slate-300 text-[12px] text-slate-500 dark:border-slate-700">
                      Set HF repo on Compare tab, then re-run image for a duel view
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="mt-3 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-[12px] text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200">
              {error}
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-200 px-4 py-3 dark:border-slate-700">
          <div className="text-[11px] text-slate-500">
            {job?.id ? (
              <span className="font-mono">job {job.id}</span>
            ) : (
              "Upload weights to begin"
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            {running && (
              <button
                type="button"
                onClick={() => job?.id && void api.dppStop(job.id).then((r) => setJob(r.job))}
                className="inline-flex items-center gap-1 rounded-lg border border-rose-300 px-3 py-1.5 text-xs font-bold text-rose-600"
              >
                <Square className="h-3 w-3 fill-current" />
                Stop
              </button>
            )}
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold dark:border-slate-700"
            >
              Close
            </button>
            {tab === "process" && (
              <button
                type="button"
                disabled={busy || running || !verified || count < minGt}
                onClick={() => void runDpp()}
                className="inline-flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-bold text-white disabled:opacity-40"
              >
                {busy || running ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ChevronRight className="h-3.5 w-3.5" />}
                Run deep post-process
              </button>
            )}
            {shipReady && (
              <button
                type="button"
                disabled={downloadBusy}
                onClick={() => void downloadShip()}
                className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-bold text-white disabled:opacity-50"
              >
                {downloadBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
                Download ship zip
              </button>
            )}
          </div>
        </div>
      </div>
      <EvaluateMergedModal
        open={Boolean(evalEntry)}
        historyId={evalEntry?.id || ""}
        title={evalEntry?.id || "post-processed model"}
        score={evalEntry?.score}
        mode="dpp"
        onClose={() => setEvalEntry(null)}
      />
    </div>
  );
}
