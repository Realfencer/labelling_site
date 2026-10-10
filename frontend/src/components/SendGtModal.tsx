import { useEffect, useMemo, useState } from "react";
import { Loader2, Send, X } from "lucide-react";
import type { ChallengeItem } from "../types";
import { checksFailComplete, checksPassComplete } from "../labelChecks";
import { cn } from "../lib/utils";
import ImageSelectionField from "./ImageSelectionField";

const HIST_BINS = 48;

export default function SendGtModal({
  open,
  onClose,
  challenges,
  onSend,
}: {
  open: boolean;
  onClose: () => void;
  challenges: ChallengeItem[];
  onSend: (opts: {
    blockMin: number;
    blockMax: number;
    quality: "all" | "pass" | "fail";
    itemKeys?: string[];
  }) => Promise<{ n_images?: number; remote?: string } | void>;
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
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  useEffect(() => {
    if (!open) return;
    setFrom(minBlock ? String(minBlock) : "");
    setTo(maxBlock ? String(maxBlock) : "");
    setQuality("all");
    setPicked(null);
    setError("");
    setSuccess("");
    setBusy(false);
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
  const rangeCount = quality === "all" ? inRange.length : quality === "pass" ? passCount : failCount;

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
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500 text-white">
              <Send className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-sm font-extrabold text-slate-900 dark:text-white">Send GT</h2>
              <p className="text-[11px] text-slate-500">
                Platform Done → replace remote beverage_gt
              </p>
            </div>
          </div>
          <button type="button" onClick={onClose} className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-4 px-4 py-4">
          <ImageSelectionField suggest={inRange.map((c) => c.item_key)} challenges={challenges} ownMode={false} value={picked} onChange={setPicked} verb="send" />
          <div className={cn("space-y-4", picked && "pointer-events-none opacity-40")}>
          <div className="grid grid-cols-2 gap-2">
            <label className="block">
              <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">From block</span>
              <input
                value={from}
                onChange={(e) => setFrom(e.target.value.replace(/[^\d]/g, ""))}
                className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 font-mono text-sm tabular-nums dark:border-slate-700 dark:bg-ink-800"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">To block</span>
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
                    className={cn("min-h-[2px] flex-1 rounded-sm", active ? "bg-emerald-500" : "bg-slate-300 dark:bg-slate-700")}
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
                className="w-full accent-emerald-500"
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
                className="w-full accent-emerald-500"
              />
            </label>
          </div>

          <button
            type="button"
            onClick={() => {
              setFrom(String(minBlock));
              setTo(String(maxBlock));
            }}
            className="text-[11px] font-bold text-emerald-600 hover:underline dark:text-emerald-300"
          >
            Reset to all labeled ({minBlock}–{maxBlock})
          </button>

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

          </div>

          <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 dark:border-slate-700 dark:bg-ink-950">
            <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Will send (replace remote)</div>
            <div className="mt-0.5 text-2xl font-black tabular-nums text-slate-900 dark:text-white">{count}</div>
            <div className="text-[11px] text-slate-500">
              {picked ? "hand-picked images" : `blocks ${lo}–${hi}`}
              {picked
                ? " · exactly these"
                : quality === "pass"
                ? " · R1–R3 all close to p"
                : quality === "fail"
                  ? " · no miner close to p"
                  : " · every Done in range"}
            </div>
            <div className="mt-1 truncate text-[10px] text-slate-400">
              root@194.163.173.245:/home/generate/beverage-studio/beverage_gt
            </div>
          </div>

          {error && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-[12px] text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200">
              {error}
            </div>
          )}
          {success && (
            <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-[12px] text-emerald-800 dark:border-emerald-900/50 dark:bg-emerald-950/40 dark:text-emerald-200">
              {success}
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
              setSuccess("");
              void onSend({ blockMin: lo, blockMax: hi, quality: picked ? "all" : quality, itemKeys: picked ?? undefined })
                .then((res) => {
                  const n = res?.n_images ?? count;
                  const remote = res?.remote || "remote beverage_gt";
                  setSuccess(`Sent ${n} image(s) → ${remote}`);
                })
                .catch((e) => setError(String(e).replace(/^Error:\s*/i, "")))
                .finally(() => setBusy(false));
            }}
            className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-bold text-white hover:bg-emerald-500 disabled:opacity-40"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            {busy ? "Sending…" : `Send ${count}`}
          </button>
        </div>
      </div>
    </div>
  );
}
