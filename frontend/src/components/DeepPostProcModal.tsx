import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Check,
  ChevronRight,
  Download,
  FlaskConical,
  ImagePlus,
  Loader2,
  Sparkles,
  Square,
  Swords,
  Upload,
  X,
} from "lucide-react";
import {
  api,
  type DppCompareResult,
  type DppEvalImageResult,
  type DppJob,
} from "../api";
import type { ChallengeItem, DetectionBox } from "../types";
import { colorForClass } from "../utils/colors";
import { cn } from "../lib/utils";

const HIST_BINS = 48;
const IMAGE_RE = /\.(jpe?g|png|webp|bmp)$/i;
const WEIGHT_RE = /\.(onnx|pt)$/i;

type Tab = "process" | "compare" | "image";
type Quality = "all" | "pass" | "fail";

function MetricPill({
  label,
  value,
  tone = "slate",
}: {
  label: string;
  value: string;
  tone?: "slate" | "emerald" | "rose" | "sky" | "amber";
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
          const label = box.conf != null ? `${name} ${Math.round(box.conf * 100)}%` : name;
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
  const [hfRepo, setHfRepo] = useState("Alexei999/carwash004");
  const [compare, setCompare] = useState<DppCompareResult | null>(null);
  const [imgFile, setImgFile] = useState<File | null>(null);
  const [imgResult, setImgResult] = useState<DppEvalImageResult | null>(null);
  const [dragOver, setDragOver] = useState(false);

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
    void api.dppDefaults().then((d) => setMinGt(d.min_gt)).catch(() => undefined);
  }, [open, minBlock, maxBlock]);

  // Poll while running
  useEffect(() => {
    if (!open || !job?.id) return;
    if (job.status !== "running" && job.status !== "comparing") return;
    const t = window.setInterval(() => {
      void api
        .dppStatus(job.id)
        .then((r) => setJob(r.job))
        .catch(() => undefined);
    }, 1500);
    return () => window.clearInterval(t);
  }, [open, job?.id, job?.status]);

  const fromN = Number(from);
  const toN = Number(to);
  const rawLo = Number.isFinite(fromN) ? fromN : minBlock;
  const rawHi = Number.isFinite(toN) ? toN : maxBlock;
  const lo = Math.min(rawLo, rawHi);
  const hi = Math.max(rawLo, rawHi);
  const span = Math.max(1, maxBlock - minBlock);
  const inRange = labeled.filter((c) => c.block >= lo && c.block <= hi);
  const passCount = inRange.filter((c) => c.checks_passed === 3).length;
  const failCount = inRange.filter((c) => c.checks_passed === 0).length;
  const count = quality === "pass" ? passCount : quality === "fail" ? failCount : inRange.length;

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
        quality,
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
        quality,
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

              <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 pb-3 pt-3 dark:border-slate-700 dark:bg-ink-950">
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
                <p className="mt-2 text-[11px] text-slate-500">
                  {ownMode ? inRange.length : count} Done images in range · need ≥{minGt} for deep post-process
                </p>
              </div>

              {(job?.baseline_score || job?.postproc) && (
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
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
                  <MetricPill label="mAP50" value={job.postproc?.map50 != null ? Number(job.postproc.map50).toFixed(4) : "—"} tone="sky" />
                </div>
              )}

              {job?.message && (
                <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-[12px] text-slate-600 dark:border-slate-700 dark:bg-ink-950 dark:text-slate-300">
                  {running && <Loader2 className="mr-1.5 inline h-3.5 w-3.5 animate-spin text-violet-500" />}
                  {job.message}
                </div>
              )}
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
                onClick={() => job?.id && void api.dppDownload(job.id)}
                className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-bold text-white"
              >
                <Download className="h-3.5 w-3.5" />
                Download ship zip
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
