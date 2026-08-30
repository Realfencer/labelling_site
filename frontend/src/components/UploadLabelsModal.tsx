import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronLeft,
  ChevronRight,
  FolderUp,
  Loader2,
  Replace,
  SkipForward,
  Upload,
  X,
} from "lucide-react";
import {
  api,
  type LabelImportConflictItem,
  type LabelImportPreviewResult,
  type LabelImportScoreTriple,
} from "../api";
import type { AppState, DetectionBox } from "../types";
import { challengeImageUrl, colorForClass } from "../utils/colors";
import { cn } from "../lib/utils";

type Phase = "pick" | "scanning" | "previewing" | "review" | "applying" | "done";

async function readAllDirectoryEntries(dir: FileSystemDirectoryEntry): Promise<FileSystemEntry[]> {
  const reader = dir.createReader();
  const all: FileSystemEntry[] = [];
  for (;;) {
    const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => {
      reader.readEntries(resolve, reject);
    });
    if (!batch.length) break;
    all.push(...batch);
  }
  return all;
}

async function collectTxtFilesFromEntry(entry: FileSystemEntry, out: File[]): Promise<void> {
  if (entry.isFile) {
    const file = await new Promise<File>((resolve, reject) => {
      (entry as FileSystemFileEntry).file(resolve, reject);
    });
    const rel = entry.fullPath.replace(/^\//, "") || file.name;
    out.push(new File([file], rel, { type: file.type || "text/plain" }));
    return;
  }
  if (entry.isDirectory) {
    const children = await readAllDirectoryEntries(entry as FileSystemDirectoryEntry);
    for (const child of children) await collectTxtFilesFromEntry(child, out);
  }
}

async function filesFromDataTransfer(dt: DataTransfer): Promise<File[]> {
  const out: File[] = [];
  const items = Array.from(dt.items || []);
  let usedEntries = false;
  for (const item of items) {
    const entry = item.webkitGetAsEntry?.();
    if (!entry) continue;
    usedEntries = true;
    await collectTxtFilesFromEntry(entry, out);
  }
  if (!usedEntries) out.push(...Array.from(dt.files || []));
  return out;
}

function pct(v: number | null | undefined): string {
  return v == null || Number.isNaN(v) ? "—" : `${Math.round(v * 100)}`;
}

function plausibilityTone(p: LabelImportScoreTriple, t: LabelImportScoreTriple): string {
  if (t.s == null || t.m == null || t.f == null) return "text-slate-500";
  if (p.s == null || p.m == null || p.f == null) return "text-slate-500";
  const mean = (Math.abs(t.s - p.s) + Math.abs(t.m - p.m) + Math.abs(t.f - p.f)) / 3;
  if (mean <= 0.03) return "text-emerald-500";
  if (mean <= 0.06) return "text-lime-500";
  if (mean <= 0.1) return "text-amber-500";
  if (mean <= 0.16) return "text-orange-500";
  return "text-rose-500";
}

function MetricRow({
  label,
  platform,
  scored,
  accent,
}: {
  label: string;
  platform: LabelImportScoreTriple;
  scored: LabelImportScoreTriple;
  accent?: "sky" | "amber";
}) {
  const tone = plausibilityTone(platform, scored);
  const border =
    accent === "amber"
      ? "border-amber-200 dark:border-amber-800/60"
      : "border-sky-200 dark:border-sky-800/60";
  const bg =
    accent === "amber"
      ? "bg-amber-50/80 dark:bg-amber-950/20"
      : "bg-sky-50/80 dark:bg-sky-950/20";
  return (
    <div className={cn("rounded-xl border p-3", border, bg)}>
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="text-xs font-extrabold tracking-tight text-slate-800 dark:text-slate-100">{label}</span>
        <span className={cn("text-[10px] font-black uppercase", tone)}>t vs p</span>
      </div>
      <div className="grid grid-cols-4 gap-1 text-[10px] font-black tabular-nums">
        <span />
        <span className="text-center text-slate-500" title="s = total_weighted">
          s
        </span>
        <span className="text-center text-slate-500" title="m = map50">
          m
        </span>
        <span className="text-center text-slate-500" title="f = false positive">
          f
        </span>
        <span className="text-slate-500">p</span>
        <span className="rounded bg-black/5 px-1 py-0.5 text-center dark:bg-white/5">{pct(platform.s)}</span>
        <span className="rounded bg-black/5 px-1 py-0.5 text-center dark:bg-white/5">{pct(platform.m)}</span>
        <span className="rounded bg-black/5 px-1 py-0.5 text-center dark:bg-white/5">{pct(platform.f)}</span>
        <span className="text-slate-500">t</span>
        <span className={cn("rounded bg-black/5 px-1 py-0.5 text-center dark:bg-white/5", tone)}>{pct(scored.s)}</span>
        <span className={cn("rounded bg-black/5 px-1 py-0.5 text-center dark:bg-white/5", tone)}>{pct(scored.m)}</span>
        <span className={cn("rounded bg-black/5 px-1 py-0.5 text-center dark:bg-white/5", tone)}>{pct(scored.f)}</span>
      </div>
    </div>
  );
}

function SummaryChip({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div className={cn("rounded-lg border px-3 py-2", tone)}>
      <div className="text-[10px] font-bold uppercase tracking-wider opacity-80">{label}</div>
      <div className="text-xl font-black tabular-nums">{value}</div>
    </div>
  );
}

function LabelPreview({
  imageUrl,
  width,
  height,
  boxes,
  classNames,
}: {
  imageUrl: string;
  width: number;
  height: number;
  boxes: DetectionBox[];
  classNames: string[];
}) {
  const w = width > 0 ? width : 1;
  const h = height > 0 ? height : 1;
  return (
    <div
      className="relative w-full overflow-hidden rounded-xl bg-black"
      style={{ aspectRatio: `${w} / ${h}`, minHeight: 260 }}
    >
      <img src={imageUrl} alt="" className="absolute inset-0 h-full w-full object-contain" />
      <svg
        viewBox={`0 0 ${w} ${h}`}
        preserveAspectRatio="xMidYMid meet"
        className="pointer-events-none absolute inset-0 h-full w-full"
      >
        {boxes.map((box, i) => {
          const x = Math.min(box.x1, box.x2);
          const y = Math.min(box.y1, box.y2);
          const bw = Math.abs(box.x2 - box.x1);
          const bh = Math.abs(box.y2 - box.y1);
          if (bw < 1 || bh < 1) return null;
          const color = colorForClass(box.cls);
          const name = classNames[box.cls] ?? `cls ${box.cls}`;
          const font = Math.max(14, Math.min(w, h) * 0.028);
          return (
            <g key={`${i}-${box.cls}-${x}-${y}`} className="upload-fade-up" style={{ animationDelay: `${i * 18}ms` }}>
              <rect
                x={x}
                y={y}
                width={bw}
                height={bh}
                fill={`${color}33`}
                stroke={color}
                strokeWidth={Math.max(2, Math.min(w, h) * 0.004)}
              />
              <rect
                x={x}
                y={Math.max(0, y - font - 6)}
                width={Math.min(bw, name.length * font * 0.62 + 10)}
                height={font + 6}
                fill={color}
                opacity={0.9}
              />
              <text
                x={x + 5}
                y={Math.max(font, y - 4)}
                fill="#fff"
                fontSize={font}
                fontWeight={700}
                fontFamily="Inter, Segoe UI, sans-serif"
              >
                {name}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function ConflictCard({
  item,
  reviewMode,
  classNames,
  replace,
  slide,
  onReplace,
  onSkip,
}: {
  item: LabelImportConflictItem;
  reviewMode: string;
  classNames: string[];
  replace: boolean;
  slide: "left" | "right";
  onReplace: () => void;
  onSkip: () => void;
}) {
  const imgUrl = challengeImageUrl(item.index, reviewMode, item.block);
  const w = item.width && item.width > 0 ? item.width : 1024;
  const h = item.height && item.height > 0 ? item.height : 1024;
  const existing = item.existing_boxes ?? [];
  const uploaded = item.upload_boxes ?? [];
  return (
    <div
      className={cn(
        "rounded-2xl border border-amber-200 bg-gradient-to-br from-amber-50/90 to-white p-4 dark:border-amber-800/50 dark:from-amber-950/30 dark:to-ink-900",
        slide === "left" ? "upload-slide-left" : "upload-slide-right",
      )}
    >
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <div className="text-sm font-extrabold text-slate-900 dark:text-white">{item.display_name}</div>
          <div className="mt-0.5 font-mono text-[10px] text-slate-500">
            block {item.block} · {item.filename}
          </div>
        </div>
        <div className="inline-flex items-center gap-1 rounded-full border border-amber-300 bg-amber-100 px-2 py-0.5 text-[10px] font-bold text-amber-900 dark:border-amber-700 dark:bg-amber-950/50 dark:text-amber-100">
          <AlertTriangle className="h-3 w-3" />
          Already labeled
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <div className="space-y-2">
          <LabelPreview imageUrl={imgUrl} width={w} height={h} boxes={existing} classNames={classNames} />
          <MetricRow
            label={`Current Done · ${item.existing_box_count} boxes`}
            platform={item.platform}
            scored={item.existing_t}
            accent="sky"
          />
        </div>
        <div className="space-y-2">
          <LabelPreview imageUrl={imgUrl} width={w} height={h} boxes={uploaded} classNames={classNames} />
          <MetricRow
            label={`Upload · ${item.upload_box_count} boxes`}
            platform={item.platform}
            scored={item.upload_t}
            accent="amber"
          />
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
        <p className="max-w-xl text-[11px] leading-relaxed text-amber-900/90 dark:text-amber-100/90">
          Compare s / m / f (platform p vs your labels as t). Choose whether to keep the current Done labels or replace
          with the upload for this image.
        </p>
        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            onClick={onSkip}
            className={cn(
              "inline-flex items-center gap-1 rounded-lg border px-3 py-1.5 text-[11px] font-bold transition",
              !replace
                ? "border-slate-400 bg-slate-100 text-slate-800 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
                : "border-slate-200 text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800",
            )}
          >
            <SkipForward className="h-3.5 w-3.5" />
            Skip
          </button>
          <button
            type="button"
            onClick={onReplace}
            className={cn(
              "inline-flex items-center gap-1 rounded-lg border px-3 py-1.5 text-[11px] font-bold transition",
              replace
                ? "border-emerald-500 bg-emerald-500 text-white"
                : "border-emerald-300 bg-emerald-50 text-emerald-800 hover:bg-emerald-100 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-200",
            )}
          >
            <Replace className="h-3.5 w-3.5" />
            Replace
          </button>
        </div>
      </div>
    </div>
  );
}

export default function UploadLabelsModal({
  open,
  onClose,
  classCount,
  reviewMode,
  onApplied,
}: {
  open: boolean;
  onClose: () => void;
  classCount: number;
  reviewMode: string;
  onApplied: (state: AppState) => void;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [phase, setPhase] = useState<Phase>("pick");
  const [error, setError] = useState("");
  const [folderName, setFolderName] = useState("");
  const [preview, setPreview] = useState<LabelImportPreviewResult | null>(null);
  const [replaceBlocks, setReplaceBlocks] = useState<Set<number>>(new Set());
  const [replaceAll, setReplaceAll] = useState(false);
  const [conflictIdx, setConflictIdx] = useState(0);
  const [applyResult, setApplyResult] = useState<{ applied: number; skipped: number; errors: number } | null>(null);
  const [scanProgress, setScanProgress] = useState({ done: 0, total: 0 });
  const [dragOver, setDragOver] = useState(false);
  const [slideDir, setSlideDir] = useState<"left" | "right">("left");
  const [applyProgress, setApplyProgress] = useState(0);

  const bindFolderInput = useCallback((el: HTMLInputElement | null) => {
    inputRef.current = el;
    if (!el) return;
    el.setAttribute("webkitdirectory", "");
    el.setAttribute("directory", "");
    el.multiple = true;
  }, []);

  const reset = useCallback(() => {
    setPhase("pick");
    setError("");
    setFolderName("");
    setPreview(null);
    setReplaceBlocks(new Set());
    setReplaceAll(false);
    setConflictIdx(0);
    setApplyResult(null);
    setScanProgress({ done: 0, total: 0 });
    setApplyProgress(0);
    setSlideDir("left");
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  useEffect(() => {
    if (!open) reset();
  }, [open, reset]);

  const conflicts = preview?.conflicts ?? [];
  const readyCount = preview?.summary.ready ?? 0;
  const conflictCount = preview?.summary.conflicts ?? 0;
  const willApply = useMemo(() => {
    if (!preview) return 0;
    const replaced = replaceAll ? conflictCount : replaceBlocks.size;
    return readyCount + replaced;
  }, [preview, readyCount, conflictCount, replaceAll, replaceBlocks.size]);

  const readFolder = async (fileList: ArrayLike<File>) => {
    setError("");
    setPhase("scanning");
    const all = Array.from(fileList);
    const txtFiles = all.filter((f) => f.name.toLowerCase().endsWith(".txt"));
    if (!txtFiles.length) {
      setError("No .txt label files found. Select the folder that contains files like block_8597100.txt.");
      setPhase("pick");
      return;
    }
    const firstRel = txtFiles[0]?.webkitRelativePath || txtFiles[0]?.name || "";
    const root = firstRel.includes("/") ? firstRel.split("/")[0] : "labels";
    setFolderName(`${root} · ${txtFiles.length} txt`);
    setScanProgress({ done: 0, total: txtFiles.length });
    const payload: { name: string; content: string }[] = [];
    for (let i = 0; i < txtFiles.length; i += 1) {
      const f = txtFiles[i];
      const content = await f.text();
      payload.push({ name: f.webkitRelativePath || f.name, content });
      setScanProgress({ done: i + 1, total: txtFiles.length });
    }
    setPhase("previewing");
    try {
      const res = await api.previewLabelImport(payload);
      setPreview(res);
      setReplaceBlocks(new Set());
      setReplaceAll(false);
      setConflictIdx(0);
      setPhase("review");
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
      setPhase("pick");
    }
  };

  const onDropFolder = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    if (phase !== "pick") return;
    try {
      const files = await filesFromDataTransfer(e.dataTransfer);
      if (!files.length) {
        setError("Could not read that drop. Use Choose folder… or drop the folder that contains the .txt labels.");
        return;
      }
      await readFolder(files);
    } catch (err) {
      setError(String(err).replace(/^Error:\s*/i, ""));
      setPhase("pick");
    }
  };

  const apply = async () => {
    if (!preview?.token || willApply <= 0) return;
    setPhase("applying");
    setError("");
    setApplyProgress(8);
    const timer = window.setInterval(() => {
      setApplyProgress((p) => (p >= 88 ? p : p + Math.max(1.4, (88 - p) * 0.08)));
    }, 80);
    try {
      const res = await api.applyLabelImport({
        token: preview.token,
        replace_blocks: replaceAll ? undefined : Array.from(replaceBlocks),
        replace_all: replaceAll,
      });
      window.clearInterval(timer);
      setApplyProgress(100);
      setApplyResult({
        applied: res.applied,
        skipped: res.skipped,
        errors: res.errors.length,
      });
      onApplied(res.state);
      window.setTimeout(() => setPhase("done"), 280);
    } catch (e) {
      window.clearInterval(timer);
      setError(String(e).replace(/^Error:\s*/i, ""));
      setPhase("review");
    }
  };

  if (!open) return null;

  const activeConflict = conflicts[conflictIdx] ?? null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/55 p-2 backdrop-blur-[2px] sm:p-4"
      onDragOver={(e) => {
        e.preventDefault();
        e.stopPropagation();
      }}
      onDrop={(e) => {
        e.preventDefault();
        e.stopPropagation();
        if (phase === "pick") void onDropFolder(e);
      }}
    >
      <div className="flex max-h-[96vh] w-full max-w-7xl flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-ink-900">
        <div className="flex items-center justify-between border-b border-slate-200 px-5 py-3.5 dark:border-slate-700">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-violet-500 text-white">
              <FolderUp className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-base font-extrabold text-slate-900 dark:text-white">Upload labels</h2>
              <p className="text-[11px] text-slate-500">
                Folder of YOLO .txt files · matched by block number · {classCount} classes in this topic
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-auto px-5 py-4">
          {(phase === "pick" || phase === "scanning" || phase === "previewing") && (
            <div className="mx-auto max-w-lg space-y-4 py-6 text-center">
              <button
                type="button"
                disabled={phase !== "pick"}
                onClick={() => inputRef.current?.click()}
                onDragEnter={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  if (phase === "pick") setDragOver(true);
                }}
                onDragOver={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  if (phase === "pick") setDragOver(true);
                }}
                onDragLeave={(e) => {
                  e.preventDefault();
                  if (e.currentTarget.contains(e.relatedTarget as Node)) return;
                  setDragOver(false);
                }}
                onDrop={(e) => void onDropFolder(e)}
                className={cn(
                  "w-full rounded-2xl border-2 border-dashed px-5 py-8 transition",
                  dragOver
                    ? "border-violet-400 bg-violet-100/80 dark:border-violet-400 dark:bg-violet-950/50"
                    : "border-violet-300 bg-violet-50 dark:border-violet-800 dark:bg-violet-950/30",
                  phase !== "pick" && "cursor-wait opacity-80",
                )}
              >
                <div className="mx-auto mb-3 flex h-16 w-16 items-center justify-center rounded-2xl bg-white/80 dark:bg-ink-900/60">
                  <Upload className="h-7 w-7 text-violet-500" />
                </div>
                <h3 className="text-lg font-extrabold text-slate-900 dark:text-white">
                  {dragOver ? "Drop the labels folder" : "Select a labels folder"}
                </h3>
                <p className="mx-auto mt-1 max-w-sm text-[12px] leading-relaxed text-slate-500">
                  Drop the folder here, or click to choose it. Do not open the folder in Chrome — that leaves the
                  labeler. Names like <code className="font-mono">block_8597100.txt</code> or{" "}
                  <code className="font-mono">8597100.txt</code> are fine. Class IDs must be 0–{Math.max(0, classCount - 1)}.
                </p>
                <span className="mt-4 inline-flex items-center gap-2 rounded-xl bg-violet-600 px-5 py-2.5 text-sm font-bold text-white">
                  {phase === "pick" ? (
                    <>
                      <FolderUp className="h-4 w-4" />
                      Choose folder…
                    </>
                  ) : (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      {phase === "scanning"
                        ? `Reading ${scanProgress.done}/${scanProgress.total}…`
                        : "Validating on server…"}
                    </>
                  )}
                </span>
              </button>
              <input
                ref={bindFolderInput}
                type="file"
                multiple
                className="hidden"
                onChange={(e) => {
                  const files = e.target.files;
                  if (files?.length) void readFolder(files);
                }}
              />
              {folderName && phase !== "pick" && (
                <div className="space-y-2">
                  <p className="font-mono text-[11px] text-slate-400">{folderName}</p>
                  <div className="mx-auto h-1.5 w-full overflow-hidden rounded-full bg-violet-200/70 dark:bg-violet-950">
                    <div
                      className={cn(
                        "h-full rounded-full bg-violet-500 transition-[width] duration-200 ease-out",
                        phase === "previewing" && "relative overflow-hidden",
                      )}
                      style={{
                        width:
                          phase === "previewing"
                            ? "100%"
                            : `${scanProgress.total ? (100 * scanProgress.done) / scanProgress.total : 8}%`,
                      }}
                    >
                      {phase === "previewing" && (
                        <div className="absolute inset-0 origin-left bg-gradient-to-r from-transparent via-white/50 to-transparent" style={{ animation: "upload-bar 1.1s ease-in-out infinite" }} />
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {phase === "review" && preview && (
            <div className="space-y-4 upload-fade-up">
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <SummaryChip
                  label="Ready"
                  value={preview.summary.ready}
                  tone="border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-900/50 dark:bg-emerald-950/30 dark:text-emerald-100"
                />
                <SummaryChip
                  label="Conflicts"
                  value={preview.summary.conflicts}
                  tone="border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-100"
                />
                <SummaryChip
                  label="Unmatched"
                  value={preview.summary.unmatched}
                  tone="border-slate-200 bg-slate-50 text-slate-700 dark:border-slate-700 dark:bg-ink-950 dark:text-slate-200"
                />
                <SummaryChip
                  label="Invalid"
                  value={preview.summary.invalid}
                  tone="border-rose-200 bg-rose-50 text-rose-900 dark:border-rose-900/50 dark:bg-rose-950/30 dark:text-rose-100"
                />
              </div>

              {preview.summary.ready > 0 && (
                <div className="rounded-xl border border-emerald-200 bg-emerald-50/70 px-3 py-2 text-[12px] text-emerald-900 dark:border-emerald-900/50 dark:bg-emerald-950/20 dark:text-emerald-100">
                  <Check className="mr-1 inline h-3.5 w-3.5" />
                  <span className="font-bold">{preview.summary.ready}</span> new labels will be imported as{" "}
                  <span className="font-bold">Done</span>.
                </div>
              )}

              {conflictCount > 0 && (
                <div className="space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <h3 className="text-[11px] font-bold uppercase tracking-wider text-amber-700 dark:text-amber-300">
                      Resolve conflicts ({conflictIdx + 1} / {conflictCount})
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      <button
                        type="button"
                        onClick={() => {
                          setReplaceAll(false);
                          setReplaceBlocks(new Set());
                        }}
                        className="rounded-lg border border-slate-200 px-2.5 py-1 text-[10px] font-bold text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
                      >
                        Skip all conflicts
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setReplaceAll(true);
                          setReplaceBlocks(new Set(conflicts.map((c) => c.block)));
                        }}
                        className="rounded-lg border border-emerald-300 bg-emerald-50 px-2.5 py-1 text-[10px] font-bold text-emerald-800 hover:bg-emerald-100 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-200"
                      >
                        Replace all conflicts
                      </button>
                    </div>
                  </div>

                  {activeConflict && (
                    <ConflictCard
                      key={activeConflict.block}
                      item={activeConflict}
                      reviewMode={reviewMode}
                      classNames={preview.class_names?.length ? preview.class_names : []}
                      slide={slideDir}
                      replace={replaceAll || replaceBlocks.has(activeConflict.block)}
                      onReplace={() => {
                        setReplaceAll(false);
                        setReplaceBlocks((prev) => new Set(prev).add(activeConflict.block));
                      }}
                      onSkip={() => {
                        if (replaceAll) {
                          setReplaceAll(false);
                          setReplaceBlocks(new Set());
                        } else {
                          setReplaceBlocks((prev) => {
                            const next = new Set(prev);
                            next.delete(activeConflict.block);
                            return next;
                          });
                        }
                      }}
                    />
                  )}

                  {conflictCount > 1 && (
                    <div className="flex items-center justify-between">
                      <button
                        type="button"
                        disabled={conflictIdx <= 0}
                        onClick={() => {
                          setSlideDir("right");
                          setConflictIdx((i) => Math.max(0, i - 1));
                        }}
                        className="inline-flex items-center gap-1 rounded-lg border border-slate-200 px-2 py-1 text-[11px] font-bold disabled:opacity-40 dark:border-slate-700"
                      >
                        <ChevronLeft className="h-3.5 w-3.5" />
                        Prev
                      </button>
                      <div className="flex gap-1">
                        {conflicts.map((c, i) => (
                          <button
                            key={c.block}
                            type="button"
                            onClick={() => {
                              setSlideDir(i > conflictIdx ? "left" : "right");
                              setConflictIdx(i);
                            }}
                            className={cn(
                              "h-2 w-2 rounded-full transition",
                              i === conflictIdx
                                ? "bg-violet-500"
                                : replaceAll || replaceBlocks.has(c.block)
                                  ? "bg-emerald-400"
                                  : "bg-slate-300 dark:bg-slate-600",
                            )}
                            aria-label={`Conflict ${i + 1}`}
                          />
                        ))}
                      </div>
                      <button
                        type="button"
                        disabled={conflictIdx >= conflictCount - 1}
                        onClick={() => {
                          setSlideDir("left");
                          setConflictIdx((i) => Math.min(conflictCount - 1, i + 1));
                        }}
                        className="inline-flex items-center gap-1 rounded-lg border border-slate-200 px-2 py-1 text-[11px] font-bold disabled:opacity-40 dark:border-slate-700"
                      >
                        Next
                        <ChevronRight className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  )}
                </div>
              )}

              {(preview.unmatched.length > 0 || preview.invalid.length > 0) && (
                <details className="rounded-xl border border-slate-200 dark:border-slate-700">
                  <summary className="cursor-pointer px-3 py-2 text-[11px] font-bold text-slate-600 dark:text-slate-300">
                    Skipped files ({preview.unmatched.length + preview.invalid.length})
                  </summary>
                  <ul className="max-h-40 space-y-1 overflow-auto border-t border-slate-200 px-3 py-2 text-[10px] dark:border-slate-700">
                    {[...preview.invalid, ...preview.unmatched].map((row) => (
                      <li key={`${row.name}-${row.reason}`} className="text-slate-600 dark:text-slate-400">
                        <span className="font-mono text-slate-800 dark:text-slate-200">{row.name}</span>
                        <span className="text-slate-400"> — </span>
                        {row.reason}
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          )}

          {phase === "applying" && (
            <div className="flex flex-col items-center justify-center gap-4 py-16 upload-fade-up">
              <div className="relative">
                <Loader2 className="h-12 w-12 animate-spin text-violet-500" />
              </div>
              <p className="text-sm font-bold text-slate-700 dark:text-slate-200">
                Applying {willApply} label{willApply === 1 ? "" : "s"}…
              </p>
              <div className="h-2 w-64 overflow-hidden rounded-full bg-violet-200/70 dark:bg-violet-950">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-400 transition-[width] duration-200 ease-out"
                  style={{ width: `${Math.min(100, applyProgress)}%` }}
                />
              </div>
              <p className="text-[11px] font-semibold tabular-nums text-slate-400">{Math.round(applyProgress)}%</p>
            </div>
          )}

          {phase === "done" && applyResult && (
            <div className="mx-auto max-w-md space-y-4 py-8 text-center upload-fade-up">
              <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-emerald-100 dark:bg-emerald-950/50">
                <Check className="h-7 w-7 text-emerald-600 dark:text-emerald-300" />
              </div>
              <h3 className="text-lg font-extrabold text-slate-900 dark:text-white">Import complete</h3>
              <div className="grid grid-cols-3 gap-2 text-left">
                <SummaryChip
                  label="Applied"
                  value={applyResult.applied}
                  tone="border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-900/50 dark:bg-emerald-950/30 dark:text-emerald-100"
                />
                <SummaryChip
                  label="Skipped"
                  value={applyResult.skipped}
                  tone="border-slate-200 bg-slate-50 text-slate-700 dark:border-slate-700 dark:bg-ink-950 dark:text-slate-200"
                />
                <SummaryChip
                  label="Errors"
                  value={applyResult.errors}
                  tone="border-rose-200 bg-rose-50 text-rose-900 dark:border-rose-900/50 dark:bg-rose-950/30 dark:text-rose-100"
                />
              </div>
            </div>
          )}

          {error && (
            <div className="mt-3 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-[12px] text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200">
              {error}
            </div>
          )}
        </div>

        <div className="flex items-center justify-between gap-2 border-t border-slate-200 px-5 py-3 dark:border-slate-700">
          <div className="text-[11px] text-slate-500">
            {phase === "review" && preview ? (
              <>
                Will apply <span className="font-bold text-slate-800 dark:text-slate-200">{willApply}</span> of{" "}
                {preview.summary.matched} matched
              </>
            ) : null}
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
            >
              {phase === "done" ? "Close" : "Cancel"}
            </button>
            {phase === "review" && (
              <button
                type="button"
                disabled={willApply <= 0}
                onClick={() => void apply()}
                className="inline-flex items-center gap-2 rounded-lg bg-violet-600 px-4 py-2 text-sm font-bold text-white hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Check className="h-4 w-4" />
                Apply {willApply > 0 ? `(${willApply})` : ""}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
