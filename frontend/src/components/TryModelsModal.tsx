import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Layers, Loader2, Play, Plus, Tag, X } from "lucide-react";
import { api, type CollectorSubmission, type TryBox, type TryModelResult, type TryResult } from "../api";
import { cn } from "../lib/utils";
import { colorForClass } from "../utils/colors";

const MODEL_COLORS = ["#38bdf8", "#f472b6", "#a3e635", "#fbbf24", "#a78bfa", "#2dd4bf", "#fb923c", "#f87171", "#60a5fa", "#e879f9", "#4ade80", "#facc15"];
const MAX_IMAGES = 4;
const MAX_MODELS = 12;

type Layer = { boxes: TryBox[]; color?: string; label?: string };

/** Canvas that draws the image and animates the detection boxes in (staggered, with a glow). */
export function DetectionCanvas({
  src,
  w,
  h,
  layers,
  classes,
  minConf,
  showLabels,
  replayKey,
  className,
}: {
  src: string;
  w: number;
  h: number;
  layers: Layer[];
  classes: string[];
  minConf: number;
  showLabels: boolean;
  replayKey: string | number;
  className?: string;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imgReady, setImgReady] = useState(0);
  const [cw, setCw] = useState(0);

  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      setImgReady((n) => n + 1);
    };
    img.src = src;
  }, [src]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setCw(Math.floor(el.clientWidth)));
    ro.observe(el);
    setCw(Math.floor(el.clientWidth));
    return () => ro.disconnect();
  }, []);

  const ch = cw > 0 ? Math.round((cw * h) / Math.max(1, w)) : 0;

  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || cw <= 0 || ch <= 0) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.floor(cw * dpr);
    canvas.height = Math.floor(ch * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const k = cw / w;
    const all: Array<{ b: TryBox; color: string; layer: number }> = [];
    layers.forEach((l, li) =>
      l.boxes.forEach((b) => {
        if ((b.conf ?? 1) >= minConf) all.push({ b, color: l.color ?? colorForClass(b.cls_id), layer: li });
      }),
    );
    let raf = 0;
    const start = performance.now();
    const STAGGER = 55;
    const DUR = 420;
    const draw = (now: number) => {
      const t = now - start;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cw, ch);
      ctx.drawImage(img, 0, 0, cw, ch);
      // slightly dim the photo so the boxes pop
      ctx.fillStyle = "rgba(2,6,23,0.18)";
      ctx.fillRect(0, 0, cw, ch);
      let animating = false;
      all.forEach((item, i) => {
        const p = Math.min(1, Math.max(0, (t - i * STAGGER) / DUR));
        if (p < 1) animating = true;
        if (p <= 0) return;
        const e = 1 - Math.pow(1 - p, 3); // ease-out
        const { b, color } = item;
        const x1 = Math.min(b.x1, b.x2) * k;
        const y1 = Math.min(b.y1, b.y2) * k;
        const bw = Math.abs(b.x2 - b.x1) * k;
        const bh = Math.abs(b.y2 - b.y1) * k;
        const grow = 1 + (1 - e) * 0.18;
        // overlay mode: nest each model's box a few px apart so identical detections stay visible
        const inset = layers.length > 1 ? item.layer * 2.5 : 0;
        const cx = x1 + bw / 2;
        const cy = y1 + bh / 2;
        const rw = Math.max(4, bw * grow - inset * 2);
        const rh = Math.max(4, bh * grow - inset * 2);
        ctx.save();
        ctx.globalAlpha = e;
        ctx.shadowColor = color;
        ctx.shadowBlur = 10 + (1 - e) * 16;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.2;
        ctx.strokeRect(cx - rw / 2, cy - rh / 2, rw, rh);
        ctx.shadowBlur = 0;
        ctx.fillStyle = color + "22";
        ctx.fillRect(cx - rw / 2, cy - rh / 2, rw, rh);
        if (showLabels && layers.length === 1) {
          const name = classes[b.cls_id] ?? `class ${b.cls_id}`;
          const text = `${name}${b.conf != null ? ` ${Math.round(b.conf * 100)}%` : ""}`;
          ctx.font = "600 11px Inter, system-ui, sans-serif";
          const tw = ctx.measureText(text).width + 8;
          const ty = Math.max(0, cy - rh / 2 - 16);
          const tx = Math.max(0, Math.min(cx - rw / 2, cw - tw));
          ctx.fillStyle = color;
          ctx.fillRect(tx, ty, tw, 16);
          ctx.fillStyle = "#0b1220";
          ctx.fillText(text, tx + 4, ty + 12);
        }
        ctx.restore();
      });
      if (animating) raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [layers, classes, minConf, showLabels, cw, ch, w, imgReady, replayKey]);

  return (
    <div ref={wrapRef} className={cn("w-full", className)}>
      <canvas ref={canvasRef} style={{ width: "100%", height: ch || undefined }} className="block rounded-md" />
    </div>
  );
}

export function CountChips({ boxes, classes, minConf }: { boxes: TryBox[]; classes: string[]; minConf: number }) {
  const counts = useMemo(() => {
    const m = new Map<number, number>();
    boxes.forEach((b) => {
      if ((b.conf ?? 1) >= minConf) m.set(b.cls_id, (m.get(b.cls_id) ?? 0) + 1);
    });
    return [...m.entries()].sort((a, b) => a[0] - b[0]);
  }, [boxes, minConf]);
  return (
    <div className="flex flex-wrap gap-1">
      {counts.length === 0 && <span className="text-[10px] text-slate-400">nothing detected</span>}
      {counts.map(([cls, n]) => (
        <span key={cls} className="inline-flex items-center gap-1 rounded px-1 py-px text-[10px] font-bold" style={{ backgroundColor: colorForClass(cls) + "26", color: colorForClass(cls) }}>
          <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: colorForClass(cls) }} />
          {n} {classes[cls] ?? cls}
        </span>
      ))}
    </div>
  );
}

export default function TryModelsModal({
  initialFiles,
  models,
  onClose,
}: {
  initialFiles: File[];
  models: CollectorSubmission[];
  onClose: () => void;
}) {
  const passed = useMemo(
    () => models.filter((m) => m.status === "passed").sort((a, b) => (b.mean_s ?? b.best_s ?? 0) - (a.mean_s ?? a.best_s ?? 0)),
    [models],
  );
  const [files, setFiles] = useState<File[]>(initialFiles.slice(0, MAX_IMAGES));
  const [picked, setPicked] = useState<Set<number>>(() => new Set(passed.slice(0, 5).map((m) => m.id)));
  const [phase, setPhase] = useState<"pick" | "run">("pick");
  const [error, setError] = useState("");
  const [result, setResult] = useState<TryResult | null>(null);
  const [progress, setProgress] = useState("");
  const [running, setRunning] = useState(false);
  const [imgIdx, setImgIdx] = useState(0);
  const [minConf, setMinConf] = useState(0);
  const [showLabels, setShowLabels] = useState(true);
  const [replay, setReplay] = useState(0);
  const [lightbox, setLightbox] = useState<number | null>(null);
  const [overlay, setOverlay] = useState(false);
  const [hidden, setHidden] = useState<Set<number>>(new Set());
  const fileRef = useRef<HTMLInputElement>(null);
  const timer = useRef<number | null>(null);

  const urls = useMemo(() => files.map((f) => URL.createObjectURL(f)), [files]);
  useEffect(() => () => urls.forEach((u) => URL.revokeObjectURL(u)), [urls]);
  useEffect(
    () => () => {
      if (timer.current != null) window.clearTimeout(timer.current);
    },
    [],
  );

  const toggle = (id: number) =>
    setPicked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else if (next.size < MAX_MODELS) next.add(id);
      return next;
    });

  const run = useCallback(async () => {
    setError("");
    setRunning(true);
    setResult(null);
    setProgress("uploading…");
    setPhase("run");
    try {
      const { command_id } = await api.collectorTry(files, [...picked]);
      const poll = async () => {
        try {
          const c = await api.collectorCommand(command_id);
          const r = c.result as unknown as TryResult & { error?: string };
          if (c.status === "error") {
            setError(String((c.result as { error?: string }).error ?? "failed"));
            setRunning(false);
            return;
          }
          if (r && Array.isArray(r.models)) setResult({ ...r });
          setProgress(c.progress || (c.status === "queued" ? "waiting for the collector…" : ""));
          if (c.status === "done") {
            setRunning(false);
            return;
          }
        } catch {
          /* keep polling */
        }
        timer.current = window.setTimeout(() => void poll(), 700);
      };
      void poll();
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/, ""));
      setRunning(false);
      setPhase("pick");
    }
  }, [files, picked]);

  const img = result?.images[imgIdx] ?? null;
  const ordered = useMemo(() => (result?.models ?? []).filter((m) => m.ok || m.error), [result]);
  const layersFor = (m: TryModelResult): Layer[] => [{ boxes: img ? m.boxes?.[img.name] ?? [] : [] }];
  const lbModel = lightbox != null ? ordered[lightbox] : null;
  const colorOf = (m: TryModelResult) => MODEL_COLORS[Math.max(0, ordered.findIndex((x) => x.id === m.id)) % MODEL_COLORS.length];

  useEffect(() => {
    if (lightbox == null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setLightbox(null);
      if (e.key === "ArrowRight") setLightbox((i) => (i == null ? i : (i + 1) % Math.max(1, ordered.length)));
      if (e.key === "ArrowLeft") setLightbox((i) => (i == null ? i : (i - 1 + ordered.length) % Math.max(1, ordered.length)));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lightbox, ordered.length]);

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/70 p-3" onClick={onClose}>
      <div
        className="mt-4 w-full max-w-[1280px] rounded-xl border border-slate-200 bg-white p-4 shadow-2xl dark:border-slate-700 dark:bg-ink-900"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-extrabold">Try collected models on your images</h2>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              Each model runs in the sandbox on the images you dropped. The images are not kept.
            </p>
          </div>
          <button type="button" onClick={onClose} className="rounded p-1 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* ---- pick images + models ---- */}
        <div className={cn("mt-3 grid gap-3", phase === "pick" ? "md:grid-cols-[320px_1fr]" : "")}>
          {phase === "pick" ? (
            <>
              <div>
                <div className="mb-1 text-[11px] font-bold uppercase tracking-wide text-slate-400">Images ({files.length}/{MAX_IMAGES})</div>
                <div className="grid grid-cols-2 gap-2">
                  {urls.map((u, i) => (
                    <div key={u} className="relative overflow-hidden rounded-lg border border-slate-200 dark:border-slate-700">
                      <img src={u} alt="" className="h-28 w-full object-cover" />
                      <button
                        type="button"
                        onClick={() => setFiles((f) => f.filter((_, j) => j !== i))}
                        className="absolute right-1 top-1 rounded bg-black/60 p-0.5 text-white hover:bg-rose-600"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </div>
                  ))}
                  {files.length < MAX_IMAGES && (
                    <button
                      type="button"
                      onClick={() => fileRef.current?.click()}
                      className="flex h-28 items-center justify-center rounded-lg border-2 border-dashed border-slate-300 text-slate-400 hover:border-sky-400 hover:text-sky-500 dark:border-slate-600"
                    >
                      <Plus className="h-5 w-5" />
                    </button>
                  )}
                </div>
                <input
                  ref={fileRef}
                  type="file"
                  accept="image/*"
                  multiple
                  hidden
                  onChange={(e) => {
                    const added = Array.from(e.target.files ?? []).filter((f) => f.type.startsWith("image/"));
                    setFiles((f) => [...f, ...added].slice(0, MAX_IMAGES));
                    e.target.value = "";
                  }}
                />
              </div>
              <div>
                <div className="mb-1 flex items-center gap-2 text-[11px]">
                  <span className="font-bold uppercase tracking-wide text-slate-400">
                    Models ({picked.size}/{MAX_MODELS})
                  </span>
                  <button type="button" className="text-sky-600 hover:underline dark:text-sky-300" onClick={() => setPicked(new Set(passed.slice(0, 3).map((m) => m.id)))}>top 3</button>
                  <button type="button" className="text-sky-600 hover:underline dark:text-sky-300" onClick={() => setPicked(new Set(passed.slice(0, 5).map((m) => m.id)))}>top 5</button>
                  <button type="button" className="text-sky-600 hover:underline dark:text-sky-300" onClick={() => setPicked(new Set(passed.slice(0, MAX_MODELS).map((m) => m.id)))}>top {MAX_MODELS}</button>
                  <button type="button" className="text-slate-400 hover:underline" onClick={() => setPicked(new Set())}>none</button>
                </div>
                <div className="max-h-[360px] overflow-y-auto rounded-lg border border-slate-200 dark:border-slate-700">
                  {passed.map((m) => (
                    <label key={m.id} className="flex cursor-pointer items-center gap-2 border-b border-slate-100 px-2.5 py-1.5 text-[12px] last:border-0 hover:bg-sky-50/60 dark:border-slate-800 dark:hover:bg-sky-950/20">
                      <input type="checkbox" checked={picked.has(m.id)} onChange={() => toggle(m.id)} />
                      <span className="min-w-0 flex-1 truncate font-semibold">{m.model}</span>
                      <span className="font-mono text-[10px] text-slate-400">{m.revision.slice(0, 6)}</span>
                      <span className="w-12 text-right font-bold tabular-nums text-emerald-600 dark:text-emerald-300">{(m.mean_s ?? m.best_s ?? 0).toFixed(3)}</span>
                      <span className="w-14 text-right text-[10px] tabular-nums text-slate-400">{m.eval_p95 != null ? `${m.eval_p95.toFixed(0)}ms` : ""}</span>
                    </label>
                  ))}
                </div>
                <div className="mt-2 flex items-center gap-3">
                  <button
                    type="button"
                    disabled={files.length === 0 || picked.size === 0 || running}
                    onClick={() => void run()}
                    className="inline-flex items-center gap-1.5 rounded-md bg-sky-600 px-4 py-2 text-[13px] font-bold text-white hover:bg-sky-500 disabled:opacity-40"
                  >
                    <Play className="h-4 w-4" /> Run {picked.size} model{picked.size === 1 ? "" : "s"} on {files.length} image{files.length === 1 ? "" : "s"}
                  </button>
                  <span className="text-[11px] text-slate-400">≈ {Math.max(5, picked.size * 4)} s</span>
                </div>
                {error && <div className="mt-1 text-[12px] text-rose-500">{error}</div>}
              </div>
            </>
          ) : (
            <div>
              {/* ---- controls ---- */}
              <div className="mb-2 flex flex-wrap items-center gap-3 rounded-lg bg-slate-50 px-3 py-2 text-[12px] dark:bg-slate-800/50">
                <button type="button" onClick={() => setPhase("pick")} disabled={running} className="rounded border border-slate-300 px-2 py-0.5 font-semibold hover:bg-white disabled:opacity-40 dark:border-slate-600 dark:hover:bg-slate-800">
                  ← change
                </button>
                {running ? (
                  <span className="inline-flex items-center gap-1.5 font-semibold text-sky-600 dark:text-sky-300">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" /> {progress || "running…"}
                  </span>
                ) : (
                  <span className="font-semibold text-emerald-600 dark:text-emerald-300">done · {result?.done ?? 0} models</span>
                )}
                {result && result.images.length > 1 && (
                  <div className="flex items-center gap-1">
                    {result.images.map((im, i) => (
                      <button
                        key={im.name}
                        type="button"
                        onClick={() => {
                          setImgIdx(i);
                          setReplay((n) => n + 1);
                        }}
                        className={cn("overflow-hidden rounded border-2", i === imgIdx ? "border-sky-500" : "border-transparent opacity-60 hover:opacity-100")}
                      >
                        <img src={urls[i]} alt="" className="h-9 w-12 object-cover" />
                      </button>
                    ))}
                  </div>
                )}
                <label className="ml-auto flex items-center gap-1.5">
                  min confidence
                  <input type="range" min={0} max={0.95} step={0.05} value={minConf} onChange={(e) => setMinConf(Number(e.target.value))} />
                  <span className="w-8 tabular-nums">{minConf.toFixed(2)}</span>
                </label>
                <label className="flex items-center gap-1">
                  <input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} /> <Tag className="h-3 w-3" /> labels
                </label>
                <button type="button" onClick={() => setReplay((n) => n + 1)} className="rounded border border-slate-300 px-2 py-0.5 font-semibold hover:bg-white dark:border-slate-600 dark:hover:bg-slate-800">
                  ▶ replay
                </button>
              </div>
              {error && <div className="mb-2 text-[12px] text-rose-500">{error}</div>}

              {/* ---- results grid ---- */}
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {ordered.map((m, idx) => {
                  const boxes = img ? m.boxes?.[img.name] ?? [] : [];
                  const t = img && m.ms ? m.ms[imgIdx] : undefined;
                  return (
                    <button
                      type="button"
                      key={m.id}
                      onClick={() => m.ok && setLightbox(idx)}
                      className="group overflow-hidden rounded-xl border border-slate-200 bg-slate-950 text-left shadow-sm transition hover:-translate-y-0.5 hover:border-sky-400 hover:shadow-lg dark:border-slate-700"
                    >
                      <div className="flex items-center gap-2 bg-slate-900 px-2.5 py-1.5">
                        <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ backgroundColor: colorOf(m) }} />
                        <span className="min-w-0 flex-1 truncate text-[12px] font-bold text-white" title={m.model}>{m.model}</span>
                        {m.score != null && <span className="text-[11px] font-extrabold tabular-nums text-emerald-300">{m.score.toFixed(3)}</span>}
                        {t != null && <span className="text-[10px] tabular-nums text-slate-400">{t.toFixed(0)} ms</span>}
                      </div>
                      {m.ok && img ? (
                        <DetectionCanvas src={urls[imgIdx]} w={img.w} h={img.h} layers={layersFor(m)} classes={result?.classes ?? []} minConf={minConf} showLabels={showLabels} replayKey={replay} />
                      ) : (
                        <div className="flex h-40 items-center justify-center px-3 text-center text-[11px] text-rose-300">{m.error || "no result"}</div>
                      )}
                      <div className="bg-slate-900 px-2.5 py-1.5">
                        <CountChips boxes={boxes} classes={result?.classes ?? []} minConf={minConf} />
                      </div>
                    </button>
                  );
                })}
                {running &&
                  Array.from({ length: Math.max(0, (result?.total ?? picked.size) - ordered.length) }).map((_, i) => (
                    <div key={`ph${i}`} className="flex h-56 animate-pulse items-center justify-center rounded-xl border border-dashed border-slate-300 text-[11px] text-slate-400 dark:border-slate-700">
                      waiting…
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ---- lightbox ---- */}
      {lbModel && img && result && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/90 p-4" onClick={(e) => { e.stopPropagation(); setLightbox(null); }}>
          <div className="flex max-h-full w-full max-w-[1100px] flex-col gap-2" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center gap-2 text-white">
              <button type="button" onClick={() => setLightbox((i) => (i == null ? i : (i - 1 + ordered.length) % ordered.length))} className="rounded bg-white/10 p-1.5 hover:bg-white/20"><ChevronLeft className="h-4 w-4" /></button>
              <div className="min-w-0 flex-1 text-center">
                <div className="truncate text-sm font-extrabold">{overlay ? "All models overlaid" : lbModel.model}</div>
                {!overlay && <div className="text-[11px] text-slate-300">{lightbox! + 1} / {ordered.length}{lbModel.score != null ? ` · s ${lbModel.score.toFixed(3)}` : ""}</div>}
              </div>
              <button type="button" onClick={() => setOverlay((v) => !v)} className={cn("inline-flex items-center gap-1 rounded px-2 py-1 text-[11px] font-bold", overlay ? "bg-sky-500" : "bg-white/10 hover:bg-white/20")}>
                <Layers className="h-3.5 w-3.5" /> overlay all
              </button>
              <button type="button" onClick={() => setLightbox((i) => (i == null ? i : (i + 1) % ordered.length))} className="rounded bg-white/10 p-1.5 hover:bg-white/20"><ChevronRight className="h-4 w-4" /></button>
              <button type="button" onClick={() => setLightbox(null)} className="rounded bg-white/10 p-1.5 hover:bg-rose-600"><X className="h-4 w-4" /></button>
            </div>
            <div className="min-h-0 overflow-auto">
              <DetectionCanvas
                src={urls[imgIdx]}
                w={img.w}
                h={img.h}
                classes={result.classes}
                minConf={minConf}
                showLabels={showLabels && !overlay}
                replayKey={`${lightbox}-${overlay}-${replay}-${[...hidden].join(",")}`}
                layers={
                  overlay
                    ? ordered.filter((m) => m.ok && !hidden.has(m.id)).map((m) => ({ boxes: m.boxes?.[img.name] ?? [], color: colorOf(m) }))
                    : layersFor(lbModel)
                }
              />
            </div>
            {overlay && (
              <div className="flex flex-wrap gap-2">
                {ordered.filter((m) => m.ok).map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setHidden((h) => { const n = new Set(h); if (n.has(m.id)) n.delete(m.id); else n.add(m.id); return n; })}
                    className={cn("inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-semibold text-white", hidden.has(m.id) ? "bg-white/5 opacity-40" : "bg-white/15")}
                  >
                    <span className="h-2 w-2 rounded-full" style={{ backgroundColor: colorOf(m) }} /> {m.model}
                  </button>
                ))}
              </div>
            )}
            {!overlay && <CountChips boxes={img ? lbModel.boxes?.[img.name] ?? [] : []} classes={result.classes} minConf={minConf} />}
          </div>
        </div>
      )}
    </div>
  );
}
