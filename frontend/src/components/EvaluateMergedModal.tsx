import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FlaskConical, ImagePlus, Loader2, X } from "lucide-react";
import { api, type FindBestEvalResult, type FindBestEvalImage, type FindBestEvalClassCount } from "../api";
import type { DetectionBox } from "../types";
import { colorForClass } from "../utils/colors";
import { cn } from "../lib/utils";

const IMAGE_RE = /\.(jpe?g|png|webp|bmp)$/i;

function boxesFor(image: FindBestEvalImage): DetectionBox[] {
  return (image.boxes || []).map((b) => ({
    cls: b.cls,
    x1: b.x1,
    y1: b.y1,
    x2: b.x2,
    y2: b.y2,
    conf: b.conf,
  }));
}

function ClassCountStrip({
  counts,
  total,
}: {
  counts: FindBestEvalClassCount[];
  total: number;
}) {
  if (!counts.length) return null;
  const max = Math.max(1, ...counts.map((c) => c.count));
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="rounded-md bg-slate-900/80 px-1.5 py-0.5 text-[11px] font-bold text-white dark:bg-white/15">
        {total} box{total === 1 ? "" : "es"}
      </span>
      {counts.map((c) => {
        const active = c.count > 0;
        const color = colorForClass(c.cls);
        return (
          <span
            key={`${c.cls}-${c.name}`}
            title={`${c.name}: ${c.count}`}
            className={cn(
              "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-semibold transition",
              active
                ? "border-transparent text-white shadow-sm"
                : "border-slate-200 bg-slate-50 text-slate-400 dark:border-slate-700 dark:bg-ink-950 dark:text-slate-500",
            )}
            style={active ? { backgroundColor: color } : undefined}
          >
            <span className="max-w-[7rem] truncate">{c.name}</span>
            <span className={cn("tabular-nums", active ? "rounded bg-black/20 px-1" : "text-slate-500")}>
              {c.count}
            </span>
            {active && (
              <span className="ml-0.5 inline-block h-1 w-8 overflow-hidden rounded-full bg-black/25" aria-hidden>
                <span
                  className="block h-full rounded-full bg-white/90"
                  style={{ width: `${Math.max(12, Math.round((c.count / max) * 100))}%` }}
                />
              </span>
            )}
          </span>
        );
      })}
    </div>
  );
}

function OverlayPreview({
  file,
  image,
  classNames,
}: {
  file: File;
  image: FindBestEvalImage;
  classNames: string[];
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
        const maxW = Math.max(320, wrap.clientWidth - 8);
        const maxH = Math.max(240, wrap.clientHeight - 8);
        const s = Math.min(maxW / img.width, maxH / img.height, 1);
        const w = Math.max(1, Math.round(img.width * s));
        const h = Math.max(1, Math.round(img.height * s));
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(img, 0, 0, w, h);
        const sx = w / image.width;
        const sy = h / image.height;
        for (const box of boxesFor(image)) {
          const x1 = Math.min(box.x1, box.x2) * sx;
          const y1 = Math.min(box.y1, box.y2) * sy;
          const bw = Math.abs(box.x2 - box.x1) * sx;
          const bh = Math.abs(box.y2 - box.y1) * sy;
          const color = colorForClass(box.cls);
          ctx.strokeStyle = color;
          ctx.lineWidth = Math.max(2, Math.round(w / 400));
          ctx.strokeRect(x1, y1, bw, bh);
          const name = classNames[box.cls] ?? `cls ${box.cls}`;
          const label = (box.raw_conf ?? box.conf) != null ? `${name} ${Math.round(((box.raw_conf ?? box.conf) as number) * 100)}%` : name;
          const fontPx = Math.max(11, Math.round(w / 70));
          ctx.font = `600 ${fontPx}px Inter, Segoe UI, sans-serif`;
          const tw = ctx.measureText(label).width;
          const lh = fontPx + 4;
          ctx.fillStyle = color;
          ctx.fillRect(x1, Math.max(0, y1 - lh), tw + 8, lh);
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
  }, [url, image, classNames]);

  return (
    <div ref={wrapRef} className="flex h-full w-full items-center justify-center">
      <canvas ref={canvasRef} className="max-h-full max-w-full rounded-md transition-opacity duration-200 ease-out" />
    </div>
  );
}

export default function EvaluateMergedModal({
  open,
  historyId,
  title,
  score,
  onClose,
  mode = "find-best",
}: {
  open: boolean;
  historyId: string;
  title: string;
  score?: number | null;
  onClose: () => void;
  mode?: "find-best" | "dpp" | "manual";
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const runIdRef = useRef(0);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<FindBestEvalResult | null>(null);
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    if (!open) return;
    runIdRef.current += 1;
    setFile(null);
    setBusy(false);
    setError("");
    setResult(null);
    setDragOver(false);
  }, [open, historyId, mode]);

  const evaluateOne = useCallback(
    async (next: File) => {
      const runId = ++runIdRef.current;
      setFile(next);
      setResult(null);
      setError("");
      setBusy(true);
      try {
        const data =
          mode === "dpp"
            ? await api.dppEvaluateShip(historyId, [next])
            : mode === "manual"
              ? await api.manualMergeEvaluate(historyId, [next])
              : await api.evaluateFindBestModel(historyId, [next]);
        if (runId !== runIdRef.current) return;
        setResult(data);
      } catch (e) {
        if (runId !== runIdRef.current) return;
        setError(String(e).replace(/^Error:\s*/i, ""));
      } finally {
        if (runId === runIdRef.current) setBusy(false);
      }
    },
    [historyId, mode],
  );

  const pickFile = useCallback(
    (incoming: FileList | File[] | null) => {
      const list = Array.from(incoming || []);
      const next = list.find((f) => IMAGE_RE.test(f.name) || f.type.startsWith("image/"));
      if (!next) {
        setError("Choose a jpg, png, webp, or bmp image.");
        return;
      }
      void evaluateOne(next);
    },
    [evaluateOne],
  );

  if (!open) return null;

  const image = result?.images[0];
  const imageCounts =
    image?.class_counts && image.class_counts.length
      ? image.class_counts
      : (result?.classes || []).map((c) => ({ ...c, count: 0 }));
  const heading =
    mode === "dpp" ? "Evaluate post-processed model" : mode === "manual" ? "Evaluate manual merge" : "Evaluate merged model";

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-slate-950/70 p-3 backdrop-blur-[3px]">
      <div className="flex h-[92vh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-ink-900">
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-2.5 dark:border-slate-700">
          <div className="flex min-w-0 items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-violet-500 text-white">
              <FlaskConical className="h-4 w-4" />
            </div>
            <div className="min-w-0">
              <h2 className="truncate text-sm font-extrabold text-slate-900 dark:text-white">{heading}</h2>
              <p className="truncate text-[11px] text-slate-500">
                {title}
                {score != null ? ` · ${Number(score).toFixed(4)}` : ""}
              </p>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <button
              type="button"
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                pickFile(e.dataTransfer.files);
              }}
              onClick={() => inputRef.current?.click()}
              disabled={busy}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-lg border border-dashed px-2.5 py-1.5 text-[11px] font-bold transition disabled:opacity-60",
                dragOver
                  ? "border-violet-400 bg-violet-50 text-violet-700 dark:border-violet-500 dark:bg-violet-950/40 dark:text-violet-200"
                  : "border-slate-300 bg-slate-50 text-slate-700 hover:border-violet-300 dark:border-slate-600 dark:bg-ink-950 dark:text-slate-200",
              )}
            >
              {busy ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin text-violet-500" />
              ) : (
                <ImagePlus className="h-3.5 w-3.5 text-violet-500" />
              )}
              {busy ? "…" : file ? "Replace" : "Upload"}
            </button>
            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp,image/bmp,.jpg,.jpeg,.png,.webp"
              className="hidden"
              onChange={(e) => {
                pickFile(e.target.files);
                e.target.value = "";
              }}
            />
            <button type="button" onClick={onClose} className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {error && (
          <div className="mx-4 mt-2 rounded-lg border border-rose-200 bg-rose-50 px-2.5 py-2 text-[11px] text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200">
            {error}
          </div>
        )}

        {result && (
          <div className="mx-4 mt-2 space-y-1.5 rounded-xl border border-slate-200 bg-slate-50/80 px-3 py-2 dark:border-slate-700 dark:bg-ink-950/50">
            <div className="text-[10px] font-bold uppercase tracking-wide text-slate-500">This image · per class</div>
            <ClassCountStrip counts={imageCounts} total={image?.n_boxes ?? result.n_boxes} />
            {result.elapsed_ms != null && (
              <div className="text-[10px] text-slate-400">
                {result.elapsed_ms} ms · {result.class_names.length} classes
              </div>
            )}
          </div>
        )}

        <div className="relative min-h-0 flex-1 p-3">
          <div className="relative h-full w-full overflow-hidden rounded-xl bg-slate-950/5 dark:bg-black/30">
            {!result && !busy && (
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOver(false);
                  pickFile(e.dataTransfer.files);
                }}
                className={cn(
                  "flex h-full w-full flex-col items-center justify-center gap-2 text-slate-400 transition",
                  dragOver && "bg-violet-500/10 text-violet-500",
                )}
              >
                <ImagePlus className="h-10 w-10" />
                <span className="text-[13px] font-semibold">Drop image here</span>
              </button>
            )}

            {busy && (
              <div className="absolute inset-0 z-10 flex items-center justify-center bg-white/50 dark:bg-ink-950/50">
                <Loader2 className="h-9 w-9 animate-spin text-violet-500" />
              </div>
            )}

            {result && image && file && (
              <div className="h-full w-full opacity-100 transition-opacity duration-200">
                <OverlayPreview file={file} image={image} classNames={result.class_names} />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
