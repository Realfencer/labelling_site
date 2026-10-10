import { useKeys } from "../store/useKeys";
import { useEffect, useMemo, useRef, useState } from "react";
import { Loader2, Sparkles, Square, X } from "lucide-react";
import type { ChallengeItem } from "../types";
import { cn } from "../lib/utils";
import ImageSelectionField from "./ImageSelectionField";

const HIST_BINS = 48;

export type OwnSam3Target = {
  index: number;
  itemKey: string;
  displayName: string;
  block: number;
};

export default function OwnSam3Modal({
  open,
  onClose,
  challenges,
  day,
  onLabelOne,
  onRunComplete,
}: {
  open: boolean;
  onClose: () => void;
  challenges: ChallengeItem[];
  day?: string | null;
  onLabelOne: (opts: {
    index: number;
    itemKey: string;
  }) => Promise<void>;
  onRunComplete?: (opts: {
    labeled: number;
    failed: number;
    stopped: boolean;
    total: number;
  }) => void | Promise<void>;
}) {
  const loaded = useMemo(
    () => challenges.filter((c) => c.loaded && (c.block || 0) > 0),
    [challenges],
  );
  const blocks = useMemo(() => loaded.map((c) => c.block).sort((a, b) => a - b), [loaded]);
  const minBlock = blocks[0] ?? 0;
  const maxBlock = blocks[blocks.length - 1] ?? 0;
  const [from, setFrom] = useState("");
  const [picked, setPicked] = useState<string[] | null>(null);
  const [to, setTo] = useState("");
  const [busy, setBusy] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState<{
    current: number;
    total: number;
    name: string;
  } | null>(null);
  const stopRef = useRef(false);

  useEffect(() => {
    if (!open) return;
    setFrom(minBlock ? String(minBlock) : "");
    setTo(maxBlock ? String(maxBlock) : "");
    setBusy(false);
    setStopping(false);
    setPicked(null);
    setError("");
    setProgress(null);
    stopRef.current = false;
  }, [open, minBlock, maxBlock]);

  const fromN = Number(from);
  const toN = Number(to);
  const rawLo = Number.isFinite(fromN) ? fromN : minBlock;
  const rawHi = Number.isFinite(toN) ? toN : maxBlock;
  const lo = Math.min(rawLo, rawHi);
  const hi = Math.max(rawLo, rawHi);
  const span = Math.max(1, maxBlock - minBlock);

  const targets: OwnSam3Target[] = useMemo(() => {
    const out: OwnSam3Target[] = [];
    const pickedSet = picked ? new Set(picked) : null;
    challenges.forEach((c, index) => {
      if (!c.loaded || (c.block || 0) <= 0) return;
      if (pickedSet ? !pickedSet.has(c.item_key) : c.block < lo || c.block > hi) return;
      if (c.reviewed || c.has_label) return;
      out.push({
        index,
        itemKey: c.item_key,
        displayName: c.display_name || c.item_key,
        block: c.block,
      });
    });
    return out;
  }, [challenges, lo, hi, picked]);

  const inRange = loaded.filter((c) => c.block >= lo && c.block <= hi);
  const doneCount = inRange.filter((c) => c.reviewed).length;
  const sam3Count = inRange.filter((c) => !c.reviewed && Boolean(c.has_label)).length;

  const byBlock = useMemo(() => {
    const map = new Map<number, ChallengeItem>();
    for (const c of loaded) {
      if (!map.has(c.block)) map.set(c.block, c);
    }
    return map;
  }, [loaded]);
  const nameAt = (block: number) => {
    const c = byBlock.get(block);
    return c ? c.display_name || c.item_key : "";
  };
  const fromName = nameAt(lo);
  const toName = nameAt(hi);

  const hist = useMemo(() => {
    const bins = new Array(HIST_BINS).fill(0);
    for (const c of loaded) {
      const t = (c.block - minBlock) / span;
      const i = Math.min(HIST_BINS - 1, Math.max(0, Math.floor(t * HIST_BINS)));
      bins[i] += 1;
    }
    const peak = Math.max(1, ...bins);
    return bins.map((n) => n / peak);
  }, [loaded, minBlock, span]);

  const setLo = (n: number) => setFrom(String(Math.min(Math.max(n, minBlock), hi)));
  const setHi = (n: number) => setTo(String(Math.max(Math.min(n, maxBlock), lo)));

  const keys = useKeys((st) => st.view);
  const setKeysOpen = useKeys((st) => st.setOpen);
  const keyOk = Boolean(keys?.chutes.configured);
  const canStart = !busy && keyOk && targets.length > 0;

  const requestStop = () => {
    if (!busy || stopping) return;
    stopRef.current = true;
    setStopping(true);
  };

  const run = async () => {
    if (!canStart) return;
    const queue = [...targets];
    stopRef.current = false;
    setBusy(true);
    setStopping(false);
    setError("");
    let labeled = 0;
    let failed = 0;
    let stopped = false;
    try {
      for (let i = 0; i < queue.length; i++) {
        if (stopRef.current) {
          stopped = true;
          break;
        }
        const t = queue[i];
        setProgress({ current: i + 1, total: queue.length, name: t.displayName });
        try {
          await onLabelOne({ index: t.index, itemKey: t.itemKey });
          labeled += 1;
        } catch (e) {
          failed += 1;
          setError(String(e).replace(/^Error:\s*/i, ""));
          // Continue to next image unless user stopped.
        }
        if (stopRef.current) {
          stopped = true;
          break;
        }
      }
      if (stopRef.current) stopped = true;
      await onRunComplete?.({ labeled, failed, stopped, total: queue.length });
      if (!stopped && failed === 0) {
        onClose();
      }
    } finally {
      setBusy(false);
      setStopping(false);
      setProgress(null);
      stopRef.current = false;
    }
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/55 p-3 backdrop-blur-[2px]">
      <div className="w-full max-w-md overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-ink-900">
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-700">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-sky-500 text-white">
              <Sparkles className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-sm font-extrabold text-slate-900 dark:text-white">SAM3 label</h2>
              <p className="text-[11px] text-slate-500">
                New images in Own index range{day ? ` · ${day}` : ""}
              </p>
            </div>
          </div>
          <button
            type="button"
            disabled={busy}
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 disabled:opacity-40 dark:hover:bg-slate-800"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-4 px-4 py-4">
          <ImageSelectionField suggest={inRange.map((c) => c.item_key)} challenges={challenges} ownMode doneOnly={false} value={picked} onChange={setPicked} verb="label" disabled={busy} />
          <div className={cn("space-y-4", picked && "pointer-events-none opacity-40")}>
          <div className="grid grid-cols-2 gap-2">
            <label className="block">
              <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">From #</span>
              <input
                value={from}
                disabled={busy}
                onChange={(e) => setFrom(e.target.value.replace(/[^\d]/g, ""))}
                className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 font-mono text-sm tabular-nums disabled:opacity-60 dark:border-slate-700 dark:bg-ink-800"
              />
              {fromName ? (
                <span className="mt-1 block truncate font-mono text-[11px] text-slate-500" title={fromName}>
                  {fromName}
                </span>
              ) : null}
            </label>
            <label className="block">
              <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">To #</span>
              <input
                value={to}
                disabled={busy}
                onChange={(e) => setTo(e.target.value.replace(/[^\d]/g, ""))}
                className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 font-mono text-sm tabular-nums disabled:opacity-60 dark:border-slate-700 dark:bg-ink-800"
              />
              {toName ? (
                <span className="mt-1 block truncate font-mono text-[11px] text-slate-500" title={toName}>
                  {toName}
                </span>
              ) : null}
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
                    className={cn("min-h-[2px] flex-1 rounded-sm", active ? "bg-sky-500" : "bg-slate-300 dark:bg-slate-700")}
                    style={{ height: `${Math.max(8, h * 100)}%` }}
                  />
                );
              })}
            </div>
            <label className="mb-2 block">
              <span className="mb-0.5 flex items-baseline justify-between gap-2 text-[10px] font-bold uppercase tracking-wider text-slate-500">
                From
                <span className="min-w-0 truncate text-right font-mono normal-case tracking-normal text-slate-700 dark:text-slate-200" title={fromName || String(lo)}>
                  {fromName ? `${lo} · ${fromName}` : lo}
                </span>
              </span>
              <input
                type="range"
                min={minBlock}
                max={maxBlock || 1}
                value={lo}
                disabled={busy || !minBlock}
                onChange={(e) => setLo(Number(e.target.value))}
                className="w-full accent-sky-500"
              />
            </label>
            <label className="block">
              <span className="mb-0.5 flex items-baseline justify-between gap-2 text-[10px] font-bold uppercase tracking-wider text-slate-500">
                To
                <span className="min-w-0 truncate text-right font-mono normal-case tracking-normal text-slate-700 dark:text-slate-200" title={toName || String(hi)}>
                  {toName ? `${hi} · ${toName}` : hi}
                </span>
              </span>
              <input
                type="range"
                min={minBlock}
                max={maxBlock || 1}
                value={hi}
                disabled={busy || !minBlock}
                onChange={(e) => setHi(Number(e.target.value))}
                className="w-full accent-sky-500"
              />
            </label>
          </div>

          <button
            type="button"
            disabled={busy || !minBlock}
            onClick={() => {
              setFrom(String(minBlock));
              setTo(String(maxBlock));
            }}
            className="text-[11px] font-bold text-sky-600 hover:underline disabled:opacity-40 dark:text-sky-300"
          >
            Reset to all Own images ({minBlock}–{maxBlock})
          </button>

          </div>

          <div className="flex items-center justify-between rounded-lg border border-slate-200 px-2.5 py-2 text-[12px] dark:border-slate-700">
            {keyOk ? (
              <span className="text-emerald-600 dark:text-emerald-300">Using the saved Chutes key ({keys?.chutes.masked})</span>
            ) : (
              <span className="text-rose-600 dark:text-rose-300">No Chutes API key saved yet</span>
            )}
            <button
              type="button"
              onClick={() => setKeysOpen(true)}
              className="rounded border border-slate-300 px-2 py-0.5 text-[11px] font-bold hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-800"
            >
              {keyOk ? "Change" : "Add key"}
            </button>
          </div>

          <div className="rounded-xl border border-sky-200 bg-sky-50 px-3 py-2.5 dark:border-sky-900/50 dark:bg-sky-950/30">
            <div className="text-[10px] font-bold uppercase tracking-wider text-sky-600 dark:text-sky-400">
              Will label (New only)
            </div>
            <div className="mt-0.5 text-2xl font-black tabular-nums text-slate-900 dark:text-white">
              {targets.length}
            </div>
            <div className="text-[11px] text-slate-500">
              {picked ? `${picked.length} hand-picked · only the New ones are labeled` : `index ${lo}–${hi} · ${inRange.length} in range · ${sam3Count} SAM3 · ${doneCount} Done`}
            </div>
            {(fromName || toName) && (
              <div
                className="mt-0.5 truncate font-mono text-[11px] text-slate-600 dark:text-slate-300"
                title={`${fromName} → ${toName}`}
              >
                {fromName || "?"} → {toName || "?"}
              </div>
            )}
            <div className="mt-1 text-[11px] text-slate-500">
              One Chutes call per image (cup+can+bottle). Results stay in SAM3 until Done.
            </div>
          </div>

          {progress && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-[12px] dark:border-slate-700 dark:bg-ink-950">
              <div className="font-semibold text-slate-800 dark:text-slate-100">
                {stopping ? "Stopping after current…" : "Labeling…"} {progress.current}/{progress.total}
              </div>
              <div className="mt-0.5 truncate font-mono text-[11px] text-slate-500">{progress.name}</div>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                <div
                  className="h-full rounded-full bg-sky-500 transition-all"
                  style={{ width: `${Math.min(100, (progress.current / Math.max(1, progress.total)) * 100)}%` }}
                />
              </div>
            </div>
          )}

          {error && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-[12px] text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200">
              {error}
            </div>
          )}
        </div>

        <div className="flex justify-end gap-2 border-t border-slate-200 px-4 py-3 dark:border-slate-700">
          {busy ? (
            <button
              type="button"
              disabled={stopping}
              onClick={requestStop}
              className="inline-flex items-center gap-1.5 rounded-lg border border-amber-300 bg-amber-50 px-4 py-2 text-sm font-bold text-amber-900 hover:bg-amber-100 disabled:opacity-50 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-100"
            >
              <Square className={cn("h-3.5 w-3.5 fill-current", stopping && "animate-pulse")} />
              {stopping ? "Stopping…" : "Stop"}
            </button>
          ) : (
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-600 dark:border-slate-700 dark:text-slate-300"
            >
              Cancel
            </button>
          )}
          <button
            type="button"
            disabled={!canStart}
            onClick={() => void run()}
            className="inline-flex items-center gap-1.5 rounded-lg bg-sky-600 px-4 py-2 text-sm font-bold text-white hover:bg-sky-500 disabled:opacity-40"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
            {busy ? "Running…" : `SAM3 ${targets.length}`}
          </button>
        </div>
      </div>
    </div>
  );
}
