import { useEffect, useMemo, useState } from "react";
import { Download, Loader2, X } from "lucide-react";
import type { ChallengeItem } from "../types";
import { checksFailComplete, checksPassComplete } from "../labelChecks";
import { cn } from "../lib/utils";
import ImageSelectionField from "./ImageSelectionField";

const HIST_BINS = 48;

export default function DownloadGtModal({
  open,
  onClose,
  challenges,
  onDownload,
  ownMode = false,
}: {
  open: boolean;
  onClose: () => void;
  challenges: ChallengeItem[];
  ownMode?: boolean;
  onDownload: (opts: {
    blockMin: number;
    blockMax: number;
    quality: "all" | "pass" | "fail";
    itemKeys?: string[];
    onProgress?: (received: number, total: number) => void;
  }) => Promise<void>;
}) {
  const labeled = useMemo(
    () => challenges.filter((c) => c.loaded && c.reviewed),
    [challenges],
  );
  const blocks = useMemo(() => labeled.map((c) => c.block).filter((b) => b > 0).sort((a, b) => a - b), [labeled]);
  const minBlock = blocks[0] ?? 0;
  const maxBlock = blocks[blocks.length - 1] ?? 0;
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [quality, setQuality] = useState<"all" | "pass" | "fail">("all");
  const [picked, setPicked] = useState<string[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<{ received: number; total: number } | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) return;
    setFrom(minBlock ? String(minBlock) : "");
    setTo(maxBlock ? String(maxBlock) : "");
    setQuality("all");
    setPicked(null);
    setError("");
    setBusy(false);
    setProgress(null);
  }, [open, minBlock, maxBlock]);

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
  const rangeCount = ownMode || quality === "all" ? inRange.length : quality === "pass" ? passCount : failCount;
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

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/55 p-3 backdrop-blur-[2px]">
      <div className="w-full max-w-md overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-ink-900">
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-700">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-sky-500 text-white">
              <Download className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-sm font-extrabold text-slate-900 dark:text-white">Download labels</h2>
              <p className="text-[11px] text-slate-500">{ownMode ? "YOLO zip · images + txt · Own index range" : "YOLO zip · images + txt · block range"}</p>
            </div>
          </div>
          <button type="button" onClick={onClose} className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-4 px-4 py-4">
          <ImageSelectionField suggest={inRange.map((c) => c.item_key)} challenges={challenges} ownMode={ownMode} value={picked} onChange={setPicked} verb="download" />
          <div className={cn("space-y-4", picked && "pointer-events-none opacity-40")}>
          <div className="grid grid-cols-2 gap-2">
            <label className="block">
              <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">{ownMode ? "From #" : "From block"}</span>
              <input
                value={from}
                onChange={(e) => setFrom(e.target.value.replace(/[^\d]/g, ""))}
                className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 font-mono text-sm tabular-nums dark:border-slate-700 dark:bg-ink-800"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">{ownMode ? "To #" : "To block"}</span>
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
                    className={cn("min-h-[2px] flex-1 rounded-sm", active ? "bg-sky-500" : "bg-slate-300 dark:bg-slate-700")}
                    style={{ height: `${Math.max(8, h * 100)}%` }}
                  />
                );
              })}
            </div>
            <label className="mb-2 block">
              <span className="mb-0.5 flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-500">
                From <span className="font-mono tabular-nums text-slate-700 dark:text-slate-200">{lo}</span>
              </span>
              <input
                type="range"
                min={minBlock}
                max={maxBlock}
                value={lo}
                onChange={(e) => setLo(Number(e.target.value))}
                className="w-full accent-sky-500"
              />
            </label>
            <label className="block">
              <span className="mb-0.5 flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-500">
                To <span className="font-mono tabular-nums text-slate-700 dark:text-slate-200">{hi}</span>
              </span>
              <input
                type="range"
                min={minBlock}
                max={maxBlock}
                value={hi}
                onChange={(e) => setHi(Number(e.target.value))}
                className="w-full accent-sky-500"
              />
            </label>
          </div>

          <button
            type="button"
            onClick={() => {
              setFrom(String(minBlock));
              setTo(String(maxBlock));
            }}
            className="text-[11px] font-bold text-sky-600 hover:underline dark:text-sky-300"
          >
            Reset to all labeled ({minBlock}–{maxBlock})
          </button>

          {!ownMode && (
          <div>
            <div className="mb-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-500">Which Done images</div>
            <div className="grid grid-cols-3 gap-1.5">
              {(
                [
                  ["all", "All labeled", inRange.length],
                  ["pass", "Pass R1–R3", passCount],
                  ["fail", "Fail 0/3", failCount],
                ] as const
              ).map(([id, label, n]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setQuality(id)}
                  className={cn(
                    "rounded-lg border px-2 py-2 text-[11px] font-bold",
                    quality === id
                      ? id === "fail"
                        ? "border-rose-400 bg-rose-500 text-white"
                        : id === "pass"
                          ? "border-emerald-400 bg-emerald-500 text-white"
                          : "border-sky-400 bg-sky-500 text-white"
                      : "border-slate-200 text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800",
                  )}
                >
                  <div>{label}</div>
                  <div className="mt-0.5 text-[10px] font-black tabular-nums opacity-80">{n}</div>
                </button>
              ))}
            </div>
          </div>

          )}

          </div>

          <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 dark:border-slate-700 dark:bg-ink-950">
            <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Will download</div>
            <div className="mt-0.5 text-2xl font-black tabular-nums text-slate-900 dark:text-white">{count}</div>
            <div className="text-[11px] text-slate-500">
              {picked ? "hand-picked images" : `${ownMode ? "index" : "blocks"} ${lo}–${hi}`}
              {picked
                ? " · exactly these"
                : ownMode
                ? " · every Done in range"
                : quality === "pass"
                  ? " · R1–R3 all close to p"
                  : quality === "fail"
                    ? " · no miner close to p"
                    : " · every Done in range"}
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
            disabled={busy || count <= 0}
            onClick={() => {
              setBusy(true);
              setError("");
              setProgress(null);
              void onDownload({
                blockMin: lo,
                blockMax: hi,
                quality: ownMode || picked ? "all" : quality,
                itemKeys: picked ?? undefined,
                onProgress: (received, total) => setProgress({ received, total }),
              })
                .then(() => onClose())
                .catch((e) => setError(String(e).replace(/^Error:\s*/i, "")))
                .finally(() => {
                  setBusy(false);
                  setProgress(null);
                });
            }}
            className="inline-flex items-center gap-1.5 rounded-lg bg-sky-600 px-4 py-2 text-sm font-bold text-white hover:bg-sky-500 disabled:opacity-40"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
            {busy
              ? progress && progress.total > 0
                ? `Receiving ${Math.round((progress.received / progress.total) * 100)}%`
                : progress && progress.received > 0
                  ? `Receiving ${(progress.received / (1024 * 1024)).toFixed(0)} MB…`
                  : "Preparing zip…"
              : `Download ${count}`}
          </button>
        </div>
      </div>
    </div>
  );
}
