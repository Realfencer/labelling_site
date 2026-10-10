import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Loader2, Tag, Upload, X } from "lucide-react";
import { api, type CollectorSubmission, type TryModelResult, type TryResult } from "../api";
import { cn } from "../lib/utils";
import { CountChips, DetectionCanvas } from "./TryModelsModal";

const MAX_IMAGES = 4;
const MAX_MB = 12;

/** Test one collected model: drop an image and it runs immediately in the sandbox. */
export default function SingleModelTest({ sub, onClose }: { sub: CollectorSubmission; onClose: () => void }) {
  const [files, setFiles] = useState<File[]>([]);
  const [result, setResult] = useState<TryResult | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState("");
  const [idx, setIdx] = useState(0);
  const [minConf, setMinConf] = useState(0);
  const [showLabels, setShowLabels] = useState(true);
  const [replay, setReplay] = useState(0);
  const [over, setOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const timer = useRef<number | null>(null);
  const runId = useRef(0);

  const urls = useMemo(() => files.map((f) => URL.createObjectURL(f)), [files]);
  useEffect(() => () => urls.forEach((u) => URL.revokeObjectURL(u)), [urls]);
  useEffect(
    () => () => {
      runId.current += 1;
      if (timer.current != null) window.clearTimeout(timer.current);
    },
    [],
  );

  const start = useCallback(
    async (incoming: File[]) => {
      const imgs = incoming.filter((f) => f.type.startsWith("image/")).slice(0, MAX_IMAGES);
      if (imgs.length === 0) {
        setError("That is not an image file.");
        return;
      }
      const big = imgs.find((f) => f.size > MAX_MB * 1024 * 1024);
      if (big) {
        setError(`${big.name}: larger than ${MAX_MB} MB`);
        return;
      }
      const my = ++runId.current;
      if (timer.current != null) window.clearTimeout(timer.current);
      setError("");
      setFiles(imgs);
      setIdx(0);
      setResult(null);
      setRunning(true);
      setProgress("uploading…");
      setReplay((n) => n + 1);
      try {
        const { command_id } = await api.collectorTry(imgs, [sub.id]);
        const poll = async () => {
          if (my !== runId.current) return;
          try {
            const c = await api.collectorCommand(command_id);
            if (my !== runId.current) return;
            if (c.status === "error") {
              setError(String((c.result as { error?: string }).error ?? "failed"));
              setRunning(false);
              return;
            }
            const r = c.result as unknown as TryResult;
            if (r && Array.isArray(r.models) && r.models.length) {
              setResult({ ...r });
              setReplay((n) => n + 1);
            }
            setProgress(c.status === "queued" ? "waiting for the collector…" : "running in the sandbox…");
            if (c.status === "done") {
              setRunning(false);
              return;
            }
          } catch {
            /* keep polling */
          }
          timer.current = window.setTimeout(() => void poll(), 500);
        };
        void poll();
      } catch (e) {
        if (my !== runId.current) return;
        setError(String(e).replace(/^Error:\s*/, ""));
        setRunning(false);
      }
    },
    [sub.id],
  );

  // Drop / paste anywhere while this window is open.
  useEffect(() => {
    const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes("Files");
    const onOver = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      setOver(true);
    };
    const onLeave = (e: DragEvent) => {
      if (e.relatedTarget == null) setOver(false);
    };
    const onDrop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      setOver(false);
      void start(Array.from(e.dataTransfer?.files ?? []));
    };
    const onPaste = (e: ClipboardEvent) => {
      const f = Array.from(e.clipboardData?.files ?? []);
      if (f.length) void start(f);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("dragover", onOver);
    window.addEventListener("dragleave", onLeave);
    window.addEventListener("drop", onDrop);
    window.addEventListener("paste", onPaste);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("dragover", onOver);
      window.removeEventListener("dragleave", onLeave);
      window.removeEventListener("drop", onDrop);
      window.removeEventListener("paste", onPaste);
      window.removeEventListener("keydown", onKey);
    };
  }, [start, onClose]);

  const m: TryModelResult | undefined = result?.models[0];
  const img = result?.images[idx];
  const boxes = img && m?.boxes ? m.boxes[img.name] ?? [] : [];
  const ms = m?.ms?.[idx];

  const zone = (compact: boolean) => (
    <div
      role="button"
      tabIndex={0}
      onClick={() => fileRef.current?.click()}
      onKeyDown={(e) => e.key === "Enter" && fileRef.current?.click()}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed text-center transition-colors",
        over ? "border-sky-400 bg-sky-500/10" : "border-slate-500/60 hover:border-sky-400 hover:bg-slate-800/40",
        compact ? "gap-0.5 px-4 py-3" : "gap-2 px-6 py-14",
      )}
    >
      {!compact && (
        <span className="mb-1 flex h-16 w-16 items-center justify-center rounded-full bg-slate-700/60">
          <Upload className="h-7 w-7 text-slate-300" />
        </span>
      )}
      <span className={cn("font-bold text-white", compact ? "text-[12px]" : "text-sm")}>
        {compact ? "Drop another image" : "Drop an image here"}
      </span>
      <span className="text-[11px] text-slate-400">or click to browse{compact ? "" : " · you can also paste with Ctrl+V"}</span>
      {!compact && <span className="mt-2 text-[11px] text-slate-500">Supports JPEG, PNG, WebP, BMP, TIFF (max {MAX_MB}MB, up to {MAX_IMAGES} images)</span>}
    </div>
  );

  return (
    <div
      data-single-test
      className="fixed inset-0 z-[60] flex items-start justify-center overflow-y-auto bg-black/75 p-3"
      onClick={(e) => {
        e.stopPropagation();
        onClose();
      }}
    >
      <div
        className="mt-6 w-full max-w-[860px] rounded-xl border border-slate-700 bg-ink-900 p-4 text-slate-100 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="text-[11px] font-bold uppercase tracking-wide text-slate-400">Test model</div>
            <h2 className="truncate text-base font-extrabold" title={sub.model}>{sub.model}</h2>
            <div className="text-[11px] text-slate-400">
              rev {sub.revision.slice(0, 8)} · stored score s={(sub.mean_s ?? sub.best_s ?? 0).toFixed(3)} · runs in the sandbox, your image is not kept
            </div>
          </div>
          <button type="button" onClick={onClose} className="rounded p-1 text-slate-400 hover:bg-slate-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={(e) => {
            void start(Array.from(e.target.files ?? []));
            e.target.value = "";
          }}
        />

        <div className="mt-3">
          {files.length === 0 ? (
            zone(false)
          ) : (
            <>
              <div className="mb-2 flex flex-wrap items-center gap-3 rounded-lg bg-slate-800/50 px-3 py-2 text-[12px]">
                {running ? (
                  <span className="inline-flex items-center gap-1.5 font-semibold text-sky-300">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" /> {progress}
                  </span>
                ) : m?.ok ? (
                  <span className="font-semibold text-emerald-300">
                    done{ms != null ? ` · ${ms.toFixed(0)} ms` : ""}
                    {m.load_s != null ? ` · load ${m.load_s.toFixed(1)} s` : ""}
                  </span>
                ) : null}
                {files.length > 1 && (
                  <div className="flex items-center gap-1">
                    {urls.map((u, i) => (
                      <button
                        key={u}
                        type="button"
                        onClick={() => {
                          setIdx(i);
                          setReplay((n) => n + 1);
                        }}
                        className={cn("overflow-hidden rounded border-2", i === idx ? "border-sky-500" : "border-transparent opacity-60 hover:opacity-100")}
                      >
                        <img src={u} alt="" className="h-9 w-12 object-cover" />
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
                <button type="button" onClick={() => setReplay((n) => n + 1)} className="rounded border border-slate-600 px-2 py-0.5 font-semibold hover:bg-slate-800">
                  ▶ replay
                </button>
              </div>

              <div
                className="relative mx-auto overflow-hidden rounded-xl bg-slate-950"
                style={img ? { maxWidth: `min(100%, calc(68vh * ${img.w} / ${img.h}))` } : undefined}
              >
                {m?.ok && img && result ? (
                  <DetectionCanvas
                    src={urls[idx]}
                    w={img.w}
                    h={img.h}
                    layers={[{ boxes }]}
                    classes={result.classes}
                    minConf={minConf}
                    showLabels={showLabels}
                    replayKey={replay}
                  />
                ) : (
                  <div className="relative">
                    <img src={urls[idx]} alt="" className={cn("block w-full", running && "opacity-60")} />
                    {running && <div className="scan-line pointer-events-none absolute inset-x-0 h-1 bg-sky-400/90 shadow-[0_0_24px_6px_rgba(56,189,248,0.7)]" />}
                  </div>
                )}
              </div>

              {m && !m.ok && <div className="mt-2 text-[12px] text-rose-300">{m.error || "The model could not run on this image."}</div>}
              {m?.ok && result && (
                <div className="mt-2">
                  <CountChips boxes={boxes} classes={result.classes} minConf={minConf} />
                </div>
              )}
              <div className="mt-3">{zone(true)}</div>
            </>
          )}
          {error && <div className="mt-2 text-[12px] text-rose-300">{error}</div>}
        </div>
      </div>
      {over && (
        <div className="pointer-events-none fixed inset-0 z-[70] flex items-center justify-center bg-sky-500/10 ring-4 ring-inset ring-sky-400/70">
          <div className="rounded-xl bg-slate-900/90 px-6 py-3 text-lg font-extrabold text-white">Drop to test {sub.model.split("/").pop()}</div>
        </div>
      )}
    </div>
  );
}
