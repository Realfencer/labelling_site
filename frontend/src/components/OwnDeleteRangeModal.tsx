import { useEffect, useMemo, useState } from "react";
import { Loader2, Trash2, X } from "lucide-react";
import type { ChallengeItem } from "../types";
import { cn } from "../lib/utils";

const HIST_BINS = 48;

export default function OwnDeleteRangeModal({
  open,
  onClose,
  challenges,
  onDelete,
}: {
  open: boolean;
  onClose: () => void;
  challenges: ChallengeItem[];
  onDelete: (opts: { blockMin: number; blockMax: number }) => Promise<void>;
}) {
  const loaded = useMemo(
    () => challenges.filter((c) => c.loaded && (c.block || 0) > 0),
    [challenges],
  );
  const blocks = useMemo(() => loaded.map((c) => c.block).sort((a, b) => a - b), [loaded]);
  const minBlock = blocks[0] ?? 0;
  const maxBlock = blocks[blocks.length - 1] ?? 0;
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) return;
    setFrom(minBlock ? String(minBlock) : "");
    setTo(maxBlock ? String(maxBlock) : "");
    setBusy(false);
    setError("");
  }, [open, minBlock, maxBlock]);

  const fromN = Number(from);
  const toN = Number(to);
  const rawLo = Number.isFinite(fromN) ? fromN : minBlock;
  const rawHi = Number.isFinite(toN) ? toN : maxBlock;
  const lo = Math.min(rawLo, rawHi);
  const hi = Math.max(rawLo, rawHi);
  const span = Math.max(1, maxBlock - minBlock);
  const inRange = loaded.filter((c) => c.block >= lo && c.block <= hi);
  const doneCount = inRange.filter((c) => c.reviewed).length;
  const newCount = inRange.length - doneCount;

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

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/55 p-3 backdrop-blur-[2px]">
      <div className="w-full max-w-md overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-ink-900">
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-700">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-rose-500 text-white">
              <Trash2 className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-sm font-extrabold text-slate-900 dark:text-white">Delete Own images</h2>
              <p className="text-[11px] text-slate-500">Remove a selected Own index range</p>
            </div>
          </div>
          <button type="button" onClick={onClose} className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-4 px-4 py-4">
          <div className="grid grid-cols-2 gap-2">
            <label className="block">
              <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">From #</span>
              <input
                value={from}
                onChange={(e) => setFrom(e.target.value.replace(/[^\d]/g, ""))}
                className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 font-mono text-sm tabular-nums dark:border-slate-700 dark:bg-ink-800"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">To #</span>
              <input
                value={to}
                onChange={(e) => setTo(e.target.value.replace(/[^\d]/g, ""))}
                className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 font-mono text-sm tabular-nums dark:border-slate-700 dark:bg-ink-800"
              />
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
                    className={cn("min-h-[2px] flex-1 rounded-sm", active ? "bg-rose-500" : "bg-slate-300 dark:bg-slate-700")}
                    style={{ height: `${Math.max(8, h * 100)}%` }}
                  />
                );
              })}
            </div>
            <label className="mb-2 block">
              <span className="mb-0.5 flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-500">
                From <span className="font-mono tabular-nums text-slate-700 dark:text-slate-200">{lo}</span>
              </span>
              <input type="range" min={minBlock} max={maxBlock || 1} value={lo} disabled={!minBlock} onChange={(e) => setLo(Number(e.target.value))} className="w-full accent-rose-500" />
            </label>
            <label className="block">
              <span className="mb-0.5 flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-500">
                To <span className="font-mono tabular-nums text-slate-700 dark:text-slate-200">{hi}</span>
              </span>
              <input type="range" min={minBlock} max={maxBlock || 1} value={hi} disabled={!minBlock} onChange={(e) => setHi(Number(e.target.value))} className="w-full accent-rose-500" />
            </label>
          </div>

          <button
            type="button"
            onClick={() => {
              setFrom(String(minBlock));
              setTo(String(maxBlock));
            }}
            className="text-[11px] font-bold text-rose-600 hover:underline dark:text-rose-300"
          >
            Reset to all Own images ({minBlock}–{maxBlock})
          </button>

          <div className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-2.5 dark:border-rose-900/50 dark:bg-rose-950/30">
            <div className="text-[10px] font-bold uppercase tracking-wider text-rose-500">Will remove</div>
            <div className="mt-0.5 text-2xl font-black tabular-nums text-slate-900 dark:text-white">{inRange.length}</div>
            <div className="text-[11px] text-slate-500">
              index {lo}–{hi} · {newCount} New · {doneCount} Done
            </div>
          </div>

          {error && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-[12px] text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200">
              {error}
            </div>
          )}
        </div>

        <div className="flex justify-end gap-2 border-t border-slate-200 px-4 py-3 dark:border-slate-700">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-600 dark:border-slate-700 dark:text-slate-300"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={busy || inRange.length <= 0}
            onClick={() => {
              setBusy(true);
              setError("");
              void onDelete({ blockMin: lo, blockMax: hi })
                .then(() => onClose())
                .catch((e) => setError(String(e).replace(/^Error:\s*/i, "")))
                .finally(() => setBusy(false));
            }}
            className="inline-flex items-center gap-1.5 rounded-lg bg-rose-600 px-4 py-2 text-sm font-bold text-white hover:bg-rose-500 disabled:opacity-40"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
            {busy ? "Deleting…" : `Delete ${inRange.length}`}
          </button>
        </div>
      </div>
    </div>
  );
}
