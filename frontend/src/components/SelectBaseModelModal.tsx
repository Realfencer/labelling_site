import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Check,
  Crosshair,
  Loader2,
  RotateCcw,
  Search,
  Trash2,
  Upload,
  X,
  ZoomIn,
} from "lucide-react";
import {
  api,
  type BaseModelCandidate,
  type BaseModelGtEvalResult,
  type BaseModelPassImage,
  type BaseModelRVariant,
  type BaseModelStatus,
  type HfModelSuggestion,
} from "../api";
import type { DetectionBox } from "../types";
import { colorForClass, challengeThumbnailUrl } from "../utils/colors";
import { withTabSessionParam } from "../sessionId";
import { cn } from "../lib/utils";

const IMAGE_RE = /\.(jpe?g|png|webp|bmp)$/i;

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
  return /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(id) && !/\s/.test(id);
}

function boxesFor(raw?: { boxes?: Array<Partial<DetectionBox>> } | Array<Partial<DetectionBox>> | null): DetectionBox[] {
  const list = Array.isArray(raw) ? raw : raw?.boxes || [];
  return list.map((b) => ({
    cls: Number(b.cls ?? 0),
    x1: Number(b.x1 ?? 0),
    y1: Number(b.y1 ?? 0),
    x2: Number(b.x2 ?? 0),
    y2: Number(b.y2 ?? 0),
    conf: b.conf == null ? undefined : Number(b.conf),
    raw_conf: b.raw_conf == null ? undefined : Number(b.raw_conf),
  }));
}

function fmtScore(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return n.toFixed(4);
}

type Layer = { id: string; boxes: DetectionBox[]; dashed?: boolean; color?: string; showConf?: boolean };

function paintCanvas(
  canvas: HTMLCanvasElement,
  img: HTMLImageElement,
  imageWidth: number,
  imageHeight: number,
  layers: Layer[],
  classNames: string[],
  zoom: number,
  fitW: number,
  fitH: number,
) {
  const base = Math.min(fitW / img.width, fitH / img.height, 1) * zoom;
  const w = Math.max(1, Math.round(img.width * base));
  const h = Math.max(1, Math.round(img.height * base));
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.drawImage(img, 0, 0, w, h);
  const sx = w / (imageWidth || img.width);
  const sy = h / (imageHeight || img.height);
  for (const layer of layers) {
    for (const box of layer.boxes) {
      const x1 = Math.min(box.x1, box.x2) * sx;
      const y1 = Math.min(box.y1, box.y2) * sy;
      const bw = Math.abs(box.x2 - box.x1) * sx;
      const bh = Math.abs(box.y2 - box.y1) * sy;
      const color = layer.color || colorForClass(box.cls);
      ctx.save();
      if (layer.dashed) ctx.setLineDash([5, 4]);
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(2, Math.round(w / 380));
      ctx.strokeRect(x1, y1, bw, bh);
      ctx.restore();
      if (!layer.showConf) continue;
      const name = classNames[box.cls] ?? `cls ${box.cls}`;
      const label = (box.raw_conf ?? box.conf) != null ? `${name} ${Math.round(((box.raw_conf ?? box.conf) as number) * 100)}%` : name;
      const fontPx = Math.max(11, Math.round(w / 68));
      ctx.font = `600 ${fontPx}px ui-sans-serif, system-ui, sans-serif`;
      const tw = ctx.measureText(label).width;
      const lh = fontPx + 4;
      ctx.fillStyle = color;
      ctx.fillRect(x1, Math.max(0, y1 - lh), tw + 8, lh);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x1 + 4, Math.max(fontPx, y1 - 4));
    }
  }
}

function ResultView({
  src,
  imageWidth,
  imageHeight,
  layers,
  classNames,
  onOpenLightbox,
}: {
  src: string;
  imageWidth: number;
  imageHeight: number;
  layers: Layer[];
  classNames: string[];
  onOpenLightbox: () => void;
}) {
  const wrapRef = useRef<HTMLButtonElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap || !src) return;
    let cancelled = false;
    let ro: ResizeObserver | null = null;
    const img = new Image();
    const paint = () => {
      if (cancelled || !img.complete || img.naturalWidth < 1) return;
      paintCanvas(
        canvas,
        img,
        imageWidth || img.naturalWidth,
        imageHeight || img.naturalHeight,
        layers,
        classNames,
        1,
        Math.max(280, wrap.clientWidth - 8),
        Math.max(220, wrap.clientHeight - 8),
      );
    };
    img.onload = () => {
      if (cancelled) return;
      paint();
      ro = new ResizeObserver(paint);
      ro.observe(wrap);
    };
    img.onerror = () => {
      /* keep empty; session URL may retry on next render */
    };
    img.src = src;
    if (img.complete && img.naturalWidth > 0) {
      paint();
      ro = new ResizeObserver(paint);
      ro.observe(wrap);
    }
    return () => {
      cancelled = true;
      ro?.disconnect();
      img.onload = null;
      img.onerror = null;
    };
  }, [src, imageWidth, imageHeight, layers, classNames]);

  return (
    <button
      ref={wrapRef}
      type="button"
      onClick={onOpenLightbox}
      className="group relative flex h-full min-h-[280px] w-full items-center justify-center overflow-hidden rounded-xl bg-slate-200 dark:bg-ink-950"
      title="Click to inspect with zoom"
    >
      {/* Fallback <img> so the photo is never blank if canvas paint races */}
      <img
        src={src}
        alt=""
        className="pointer-events-none absolute inset-0 h-full w-full object-contain opacity-30"
        aria-hidden
      />
      <canvas ref={canvasRef} className="relative z-[1] max-h-full max-w-full" />
      <span className="absolute bottom-2 right-2 z-[2] inline-flex items-center gap-1 rounded bg-black/55 px-1.5 py-0.5 text-[10px] font-semibold text-white opacity-80 group-hover:opacity-100">
        <ZoomIn className="h-3 w-3" /> Inspect
      </span>
    </button>
  );
}

function Lightbox({
  src,
  imageWidth,
  imageHeight,
  layers,
  classNames,
  onClose,
}: {
  src: string;
  imageWidth: number;
  imageHeight: number;
  layers: Layer[];
  classNames: string[];
  onClose: () => void;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap || !src) return;
    const img = new Image();
    let ro: ResizeObserver | null = null;
    img.onload = () => {
      const paint = () => {
        paintCanvas(
          canvas,
          img,
          imageWidth,
          imageHeight,
          layers,
          classNames,
          zoom,
          Math.max(400, wrap.clientWidth - 24),
          Math.max(320, wrap.clientHeight - 24),
        );
      };
      paint();
      ro = new ResizeObserver(paint);
      ro.observe(wrap);
    };
    img.src = src;
    return () => ro?.disconnect();
  }, [src, imageWidth, imageHeight, layers, classNames, zoom]);

  return (
    <div className="fixed inset-0 z-[60] flex flex-col bg-slate-950/90 p-3 backdrop-blur-sm">
      <div className="mb-2 flex items-center justify-between text-white">
        <div className="text-sm font-bold">Detection inspect · scroll to zoom · Esc to close</div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs tabular-nums">{Math.round(zoom * 100)}%</span>
          <button type="button" onClick={onClose} className="rounded-lg bg-white/10 p-1.5 hover:bg-white/20">
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>
      <div
        ref={wrapRef}
        className="min-h-0 flex-1 overflow-auto"
        onWheel={(e) => {
          e.preventDefault();
          setZoom((z) => Math.min(6, Math.max(0.4, Number((z + (e.deltaY > 0 ? -0.12 : 0.12)).toFixed(2)))));
        }}
      >
        <div className="flex min-h-full items-center justify-center p-2">
          <canvas ref={canvasRef} className="rounded-md shadow-2xl" />
        </div>
      </div>
    </div>
  );
}

const R_COLOR: Record<string, string> = {
  model: "#0d9488",
  gt: "#94a3b8",
  r1: "#22c55e",
  r2: "#3b82f6",
  r3: "#a855f7",
};

export default function SelectBaseModelModal({
  open,
  onClose,
  onApplied,
}: {
  open: boolean;
  onClose: () => void;
  onApplied?: () => void | Promise<void>;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const suggestTimer = useRef<number | null>(null);
  const [status, setStatus] = useState<BaseModelStatus | null>(null);
  const [draft, setDraft] = useState("");
  const [suggestions, setSuggestions] = useState<HfModelSuggestion[]>([]);
  const [candidate, setCandidate] = useState<BaseModelCandidate | null>(null);
  const [taxonomy, setTaxonomy] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [runBusy, setRunBusy] = useState(false);
  const [applyBusy, setApplyBusy] = useState(false);
  const [resetBusy, setResetBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [clearBusy, setClearBusy] = useState(false);
  const [clearMsg, setClearMsg] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [previewSrc, setPreviewSrc] = useState("");
  const [previewMeta, setPreviewMeta] = useState<{ w: number; h: number; classNames: string[] } | null>(null);
  const [previewBoxes, setPreviewBoxes] = useState<DetectionBox[]>([]);
  const [gtBoxes, setGtBoxes] = useState<DetectionBox[]>([]);
  const [r123, setR123] = useState<BaseModelRVariant[]>([]);
  const [modelScore, setModelScore] = useState<BaseModelGtEvalResult["score"] | null>(null);
  const [modeLabel, setModeLabel] = useState("");
  const [selectedPass, setSelectedPass] = useState<BaseModelPassImage | null>(null);
  const [lightbox, setLightbox] = useState(false);
  const [show, setShow] = useState({ model: true, gt: true, r1: false, r2: false, r3: false });

  const refreshStatus = useCallback(async () => {
    try {
      const s = await api.baseModelStatus();
      setStatus(s);
      if (s.class_names?.length) setTaxonomy(s.class_names);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    setCandidate(null);
    setError("");
    setMessage("");
    setPreviewSrc("");
    setPreviewMeta(null);
    setPreviewBoxes([]);
    setGtBoxes([]);
    setR123([]);
    setModelScore(null);
    setModeLabel("");
    setSelectedPass(null);
    setDraft("");
    setLightbox(false);
    void refreshStatus();
    void api
      .hfCompareSearch("bev", 16)
      .then((res) => setSuggestions(res.models || []))
      .catch(() => undefined);
  }, [open, refreshStatus]);

  const fetchSuggestions = (query: string) => {
    if (suggestTimer.current) window.clearTimeout(suggestTimer.current);
    suggestTimer.current = window.setTimeout(() => {
      void api
        .hfCompareSearch(query.trim() || "bev", 20)
        .then((res) => setSuggestions(res.models || []))
        .catch(() => setSuggestions([]));
    }, 180);
  };

  const clearPreview = () => {
    setPreviewSrc((prev) => {
      if (prev.startsWith("blob:")) URL.revokeObjectURL(prev);
      return "";
    });
    setPreviewMeta(null);
    setPreviewBoxes([]);
    setGtBoxes([]);
    setR123([]);
    setModelScore(null);
    setModeLabel("");
  };

  const selectModel = async (raw?: string) => {
    const id = normalizeRepo(raw || draft);
    if (!validRepo(id)) {
      setError("Use owner/name (e.g. WhiteCrow323/bev2)");
      return;
    }
    setBusy(true);
    setError("");
    setMessage(`Loading miner.py + weights for ${id}…`);
    setCandidate(null);
    clearPreview();
    try {
      const res = await api.baseModelSelect({
        hf_repo: id,
      });
      setCandidate(res.candidate);
      const tax = res.taxonomy?.length ? res.taxonomy : taxonomy;
      if (res.taxonomy?.length) setTaxonomy(res.taxonomy);
      setDraft(id);
      const minerNames = res.candidate.miner_class_names || res.candidate.verify?.class_names || [];
      setMessage(`Ready · miner.py [${minerNames.join(", ")}] → topic [${tax.join(", ") || "—"}]`);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
      setMessage("");
    } finally {
      setBusy(false);
    }
  };

  const runDrop = async (file: File) => {
    if (!candidate?.id) {
      setError("Select a Hugging Face model first");
      return;
    }
    setRunBusy(true);
    setError("");
    setSelectedPass(null);
    setGtBoxes([]);
    setR123([]);
    setModelScore(null);
    setShow({ model: true, gt: false, r1: false, r2: false, r3: false });
    try {
      const url = URL.createObjectURL(file);
      setPreviewSrc((prev) => {
        if (prev.startsWith("blob:")) URL.revokeObjectURL(prev);
        return url;
      });
      const res = await api.baseModelEvaluate(candidate.id, [file]);
      const im = res.images[0];
      if (!im) throw new Error("No detections returned");
      setPreviewMeta({ w: im.width, h: im.height, classNames: res.class_names || [] });
      setPreviewBoxes(boxesFor(im));
      setModeLabel(`Drag-drop · ${im.n_boxes} boxes · ${res.elapsed_ms} ms · ${res.inference_path || "miner.py"}`);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setRunBusy(false);
    }
  };

  const runGt = async (pass: BaseModelPassImage) => {
    if (!candidate?.id) {
      setError("Select a Hugging Face model first");
      return;
    }
    setSelectedPass(pass);
    setRunBusy(true);
    setError("");
    setShow({ model: true, gt: true, r1: true, r2: false, r3: false });
    try {
      const res = await api.baseModelEvaluateGt({
        candidate_id: candidate.id,
        block: pass.block || undefined,
        item_key: pass.item_key || undefined,
        mode: pass.mode,
        index: pass.index,
      });
      const src = withTabSessionParam(res.image_url || res.image_data_url || "");
      setPreviewSrc((prev) => {
        if (prev.startsWith("blob:")) URL.revokeObjectURL(prev);
        return src;
      });
      setPreviewMeta({ w: res.width, h: res.height, classNames: res.class_names || taxonomy });
      setPreviewBoxes(boxesFor(res));
      setGtBoxes(boxesFor({ boxes: res.gt_boxes }));
      setR123(res.r123 || []);
      setModelScore(res.score || null);
      setModeLabel(
        `GT ${res.block || pass.block} · model ${fmtScore(res.score?.composite ?? res.score?.score)} · ${res.elapsed_ms} ms · ${res.inference_path || "miner.py"}`,
      );
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setRunBusy(false);
    }
  };

  const applyModel = async () => {
    if (!candidate?.id) return;
    setApplyBusy(true);
    setError("");
    try {
      const res = await api.baseModelApply(candidate.id);
      setMessage(`Applied ${res.hf_model_id} as topic base`);
      await refreshStatus();
      await onApplied?.();
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setApplyBusy(false);
    }
  };

  const resetDefault = async () => {
    setResetBusy(true);
    setError("");
    try {
      const res = await api.baseModelReset();
      setCandidate(null);
      clearPreview();
      setMessage(`Restored manak0 winner: ${res.hf_model_id}`);
      await refreshStatus();
      await onApplied?.();
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setResetBusy(false);
    }
  };

  const clearHistoryDisk = async () => {
    if (clearBusy) return;
    const ok = window.confirm(
      "Clear Select Base Model candidate cache on disk?\n\n" +
        "Deletes downloaded candidate packs used for preview/eval. The active topic base is kept.\n" +
        "Labels and Own images are never deleted.",
    );
    if (!ok) return;
    setClearBusy(true);
    setClearMsg("");
    try {
      const res = await api.baseModelClearHistory();
      setClearMsg(`Freed ${res.freed_human || "disk"} · removed ${res.removed ?? 0} candidates`);
    } catch (e) {
      setClearMsg(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setClearBusy(false);
    }
  };

  const layers = useMemo(() => {
    const out: Layer[] = [];
    if (show.gt && gtBoxes.length) out.push({ id: "gt", boxes: gtBoxes, dashed: true, color: R_COLOR.gt });
    for (const v of r123) {
      const key = `r${v.rank}` as "r1" | "r2" | "r3";
      if (!show[key]) continue;
      out.push({ id: key, boxes: boxesFor(v), dashed: true, color: R_COLOR[key] });
    }
    // Model boxes: per-class colors (do not force a single teal).
    if (show.model) out.push({ id: "model", boxes: previewBoxes, showConf: true });
    return out;
  }, [show, gtBoxes, r123, previewBoxes]);

  const browse = useMemo(
    () => suggestions.filter((s) => s.id !== draft).slice(0, 6),
    [suggestions, draft],
  );
  const passImages = status?.pass_images || [];

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/55 p-2 backdrop-blur-[2px]">
      <div className="flex max-h-[96vh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-ink-900">
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-2.5 dark:border-slate-700">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-teal-500 text-white">
              <Crosshair className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-sm font-extrabold text-slate-900 dark:text-white">Select base model</h2>
              <p className="text-[11px] text-slate-500">
                <span className="font-mono font-semibold">{status?.hf_model_id || "—"}</span>
                {status?.custom ? " · custom" : " · manak0 winner"} · inference via miner.py
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={clearBusy}
              onClick={() => void clearHistoryDisk()}
              className="inline-flex items-center gap-1 rounded-md border border-rose-200 px-2 py-1 text-[10px] font-bold text-rose-700 hover:bg-rose-50 disabled:opacity-40 dark:border-rose-900/50 dark:text-rose-300"
              title="Delete cached base-model candidates to free disk"
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

        <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-4 py-3">
          <div className="grid gap-2 md:grid-cols-[1fr_auto_auto]">
            <p className="self-end pb-1.5 text-[11px] text-slate-500 dark:text-slate-400">
              Private repos use your saved Hugging Face accounts (API keys).
            </p>
            <div className="flex items-end">
              <button
                type="button"
                disabled={resetBusy}
                onClick={() => void resetDefault()}
                className="inline-flex h-[34px] items-center gap-1.5 rounded-lg border border-slate-200 px-3 text-[12px] font-bold text-slate-700 dark:border-slate-700 dark:text-slate-200"
              >
                {resetBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RotateCcw className="h-3.5 w-3.5" />}
                Default
              </button>
            </div>
            <div className="flex items-end gap-2 md:col-span-1">
              <div className="relative min-w-0 flex-1">
                <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
                <input
                  value={draft}
                  onChange={(e) => {
                    setDraft(e.target.value);
                    fetchSuggestions(e.target.value);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      void selectModel();
                    }
                  }}
                  placeholder="WhiteCrow323/bev2"
                  className="w-full rounded-lg border border-slate-200 bg-white py-1.5 pl-7 pr-2 font-mono text-[12px] dark:border-slate-700 dark:bg-ink-950"
                />
              </div>
              <button
                type="button"
                disabled={busy || !draft.trim()}
                onClick={() => void selectModel()}
                className="inline-flex h-[34px] shrink-0 items-center gap-1.5 rounded-lg bg-teal-600 px-3 text-[12px] font-bold text-white disabled:opacity-40"
              >
                {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
                Select
              </button>
            </div>
          </div>

          {browse.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {browse.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => {
                    setDraft(s.id);
                    void selectModel(s.id);
                  }}
                  className="rounded-md border border-slate-200 px-1.5 py-0.5 font-mono text-[10px] text-slate-600 dark:border-slate-700 dark:text-slate-300"
                >
                  {s.id}
                </button>
              ))}
            </div>
          )}

          {candidate?.verify?.ok && (
            <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-2.5 py-1.5 text-[11px] text-emerald-800 dark:border-emerald-900/40 dark:bg-emerald-950/30 dark:text-emerald-200">
              <span className="font-mono font-bold">{candidate.hf_repo}</span> · miner.py OK
              {(candidate.verify.warnings || []).length > 0 && (
                <span className="text-amber-700 dark:text-amber-300"> · {candidate.verify.warnings![0]}</span>
              )}
            </div>
          )}

          {/* Single drop / result stage */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              const f = Array.from(e.dataTransfer.files || []).find(
                (x) => IMAGE_RE.test(x.name) || x.type.startsWith("image/"),
              );
              if (f) void runDrop(f);
            }}
            className={cn(
              "relative overflow-hidden rounded-xl border-2 border-dashed",
              dragOver ? "border-teal-400 bg-teal-50/50 dark:bg-teal-950/20" : "border-slate-200 dark:border-slate-700",
            )}
          >
            {previewSrc && previewMeta ? (
              <div className="space-y-1">
                <div className="h-[320px]">
                  <ResultView
                    src={previewSrc}
                    imageWidth={previewMeta.w}
                    imageHeight={previewMeta.h}
                    layers={layers}
                    classNames={previewMeta.classNames}
                    onOpenLightbox={() => setLightbox(true)}
                  />
                </div>
                {previewMeta.classNames.length > 0 && (
                  <div className="flex flex-wrap gap-2 px-1 pb-1">
                    {previewMeta.classNames.map((name, i) => (
                      <span key={`${name}-${i}`} className="inline-flex items-center gap-1 text-[10px] font-semibold text-slate-600 dark:text-slate-300">
                        <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: colorForClass(i) }} />
                        {name}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="flex h-[220px] flex-col items-center justify-center px-4 text-center">
                <Upload className="h-7 w-7 text-teal-500" />
                <p className="mt-2 text-sm font-bold text-slate-800 dark:text-slate-100">Drop image here</p>
                <p className="mt-0.5 text-[11px] text-slate-500">Runs full miner.py pipeline · or pick a pass GT below</p>
                <button
                  type="button"
                  disabled={!candidate || runBusy}
                  onClick={() => fileRef.current?.click()}
                  className="mt-3 inline-flex items-center gap-1 rounded-lg bg-teal-600 px-3 py-1.5 text-[11px] font-bold text-white disabled:opacity-40"
                >
                  {runBusy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Upload className="h-3 w-3" />}
                  Choose image
                </button>
              </div>
            )}
            {previewSrc && (
              <div className="flex items-center justify-between gap-2 border-t border-slate-200 px-2 py-1.5 dark:border-slate-700">
                <div className="truncate text-[11px] text-slate-500">{modeLabel || "Result"}</div>
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    disabled={!candidate || runBusy}
                    onClick={() => fileRef.current?.click()}
                    className="rounded-md border border-slate-200 px-2 py-0.5 text-[10px] font-bold text-slate-600 dark:border-slate-700 dark:text-slate-300"
                  >
                    Replace
                  </button>
                  <button
                    type="button"
                    onClick={clearPreview}
                    className="rounded-md border border-slate-200 px-2 py-0.5 text-[10px] font-bold text-slate-600 dark:border-slate-700 dark:text-slate-300"
                  >
                    Clear
                  </button>
                </div>
              </div>
            )}
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void runDrop(f);
                e.target.value = "";
              }}
            />
            {runBusy && (
              <div className="absolute inset-0 flex items-center justify-center bg-white/50 dark:bg-ink-900/50">
                <Loader2 className="h-6 w-6 animate-spin text-teal-600" />
              </div>
            )}
          </div>

          {/* Scores row */}
          {(modelScore || r123.length > 0) && (
            <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
              <div className="rounded-lg border border-teal-200 bg-teal-50 px-2 py-1.5 dark:border-teal-900/40 dark:bg-teal-950/30">
                <div className="text-[10px] font-bold uppercase text-teal-700 dark:text-teal-300">Model vs left GT</div>
                <div className="font-mono text-sm font-bold tabular-nums text-teal-900 dark:text-teal-100">
                  {fmtScore(modelScore?.composite ?? modelScore?.score)}
                </div>
                <div className="font-mono text-[10px] text-teal-700/80 dark:text-teal-300/80">
                  mAP {fmtScore(modelScore?.map50)} · FFPI {modelScore?.ffpi?.toFixed?.(2) ?? "—"}
                  {modelScore?.platform_score != null ? ` · plat ${fmtScore(modelScore.platform_score)}` : ""}
                </div>
              </div>
              {r123.map((v) => {
                const plat = v.platform?.score ?? v.platform?.total_weighted ?? v.platform?.composite_score;
                const local = v.score?.composite ?? v.score?.score;
                return (
                  <button
                    key={v.rank}
                    type="button"
                    onClick={() =>
                      setShow((s) => ({
                        ...s,
                        [`r${v.rank}`]: !s[`r${v.rank}` as "r1" | "r2" | "r3"],
                      }))
                    }
                    className="rounded-lg border border-slate-200 px-2 py-1.5 text-left dark:border-slate-700"
                    style={{ borderColor: show[`r${v.rank}` as "r1" | "r2" | "r3"] ? R_COLOR[`r${v.rank}`] : undefined }}
                  >
                    <div className="text-[10px] font-bold uppercase" style={{ color: R_COLOR[`r${v.rank}`] }}>
                      {v.label}
                      {v.is_winner ? " · win" : ""}
                    </div>
                    <div className="font-mono text-sm font-bold tabular-nums">
                      {fmtScore(plat ?? local)}
                    </div>
                    <div className="truncate font-mono text-[10px] text-slate-500">
                      {plat != null ? (
                        <>
                          platform {fmtScore(plat)}
                          {v.platform?.map50 != null ? ` · mAP ${fmtScore(v.platform.map50)}` : ""}
                        </>
                      ) : (
                        <>vs GT {fmtScore(local)} · {v.n_boxes} boxes</>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          )}

          {(gtBoxes.length > 0 || r123.length > 0) && (
            <div className="flex flex-wrap gap-1">
              {(
                [
                  ["model", "Model"],
                  ["gt", "GT"],
                  ["r1", "R1"],
                  ["r2", "R2"],
                  ["r3", "R3"],
                ] as const
              ).map(([id, label]) => {
                if (id !== "model" && id !== "gt" && !r123.some((v) => `r${v.rank}` === id)) return null;
                if (id === "gt" && !gtBoxes.length) return null;
                return (
                  <button
                    key={id}
                    type="button"
                    onClick={() => setShow((s) => ({ ...s, [id]: !s[id] }))}
                    className={cn(
                      "rounded-md px-2 py-0.5 text-[10px] font-bold",
                      show[id] ? "text-white" : "border border-slate-200 text-slate-500 dark:border-slate-700",
                    )}
                    style={show[id] ? { backgroundColor: R_COLOR[id] } : undefined}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
          )}

          <div>
            <div className="mb-1 text-[11px] font-bold uppercase tracking-wider text-slate-500">
              Passed GT · click to score via miner.py
            </div>
            {passImages.length === 0 ? (
              <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] text-amber-800 dark:border-amber-900/40 dark:bg-amber-950/30 dark:text-amber-200">
                No passed GT loaded. Open Platform for this topic so pass images with R1/R2/R3 appear.
              </div>
            ) : (
              <div className="grid max-h-36 grid-cols-4 gap-1 overflow-y-auto sm:grid-cols-6 md:grid-cols-8">
                {passImages.map((p) => {
                  const active =
                    selectedPass?.index === p.index && selectedPass?.mode === p.mode && selectedPass?.block === p.block;
                  return (
                    <button
                      key={`${p.mode}-${p.index}-${p.block}`}
                      type="button"
                      disabled={!candidate || runBusy}
                      onClick={() => void runGt(p)}
                      className={cn(
                        "overflow-hidden rounded-md border",
                        active ? "border-indigo-500 ring-2 ring-indigo-400/40" : "border-slate-200 dark:border-slate-700",
                      )}
                    >
                      <img
                        src={
                          p.block
                            ? challengeThumbnailUrl(p.index, p.mode, p.block, true, undefined, p.item_key || undefined)
                            : withTabSessionParam(
                                `/api/challenges/${p.index}/thumbnail?mode=${encodeURIComponent(p.mode)}&block=${p.block}`,
                              )
                        }
                        alt=""
                        className="h-14 w-full object-cover bg-slate-100 dark:bg-ink-950"
                      />
                      <div className="px-1 py-0.5 font-mono text-[9px] text-slate-600 dark:text-slate-300">
                        {p.mode === "challenge" ? p.block : `#${p.index + 1}`}
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {message && (
            <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-[12px] text-emerald-800 dark:border-emerald-900/40 dark:bg-emerald-950/30 dark:text-emerald-200">
              {message}
            </div>
          )}
          {error && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-1.5 text-[12px] text-rose-700 dark:border-rose-900/40 dark:bg-rose-950/30 dark:text-rose-200">
              {error}
            </div>
          )}
        </div>

        <div className="flex justify-between gap-2 border-t border-slate-200 px-4 py-2.5 dark:border-slate-700">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-semibold text-slate-600 dark:border-slate-700 dark:text-slate-300"
          >
            Close
          </button>
          <button
            type="button"
            disabled={!candidate?.verify?.ok || applyBusy}
            onClick={() => void applyModel()}
            className="inline-flex items-center gap-1.5 rounded-lg bg-teal-600 px-4 py-1.5 text-sm font-bold text-white disabled:opacity-40"
          >
            {applyBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
            Apply as topic base
          </button>
        </div>
      </div>

      {lightbox && previewSrc && previewMeta && (
        <Lightbox
          src={previewSrc}
          imageWidth={previewMeta.w}
          imageHeight={previewMeta.h}
          layers={layers}
          classNames={previewMeta.classNames}
          onClose={() => setLightbox(false)}
        />
      )}
    </div>
  );
}
