import { Activity } from "lucide-react";
import type { ChutesStatus } from "../api";
import { formatChutesUsd } from "../chutesSettings";
import { cn } from "../lib/utils";

/** Compact live Chutes balance chip for the footer brand row. */
export default function ChutesBalanceChip({
  snap,
  error,
  onClick,
  className,
}: {
  snap: ChutesStatus | null;
  error?: boolean;
  onClick: () => void;
  className?: string;
}) {
  const configured = Boolean(snap?.configured && snap.balance != null);
  const balance = configured ? Number(snap!.balance) : null;
  const threshold = configured ? Number(snap!.balance_alert_usd || 20) : 20;
  const low = Boolean(snap?.low_balance);
  const ratio = balance == null || threshold <= 0 ? 0 : balance / threshold;
  // Fill against 2× threshold so healthy balances don't look maxed-out.
  const fillPct = Math.max(0, Math.min(100, (ratio / 2) * 100));
  const title = error
    ? "Chutes unreachable — click to open settings"
    : configured
      ? `${snap?.username || "Chutes"} · ${formatChutesUsd(balance)} · alert below ${formatChutesUsd(threshold)}`
      : "Open Chutes settings to set API key / threshold";

  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className={cn(
        "flex h-7 w-[6.5rem] shrink-0 flex-col justify-center gap-0.5 rounded border px-1.5 py-0.5 text-left transition hover:brightness-110",
        error
          ? "border-rose-300 bg-rose-50 dark:border-rose-800/60 dark:bg-rose-950/40"
          : low
            ? "border-amber-300 bg-amber-50 dark:border-amber-800/60 dark:bg-amber-950/40"
            : configured
              ? "border-emerald-300 bg-emerald-50 dark:border-emerald-800/60 dark:bg-emerald-950/40"
              : "border-slate-300 bg-slate-50 dark:border-slate-700 dark:bg-slate-900",
        className,
      )}
    >
      <div className="flex items-center gap-1">
        <Activity
          className={cn(
            "h-3 w-3 shrink-0",
            error
              ? "text-rose-500"
              : low
                ? "text-amber-600 dark:text-amber-300"
                : configured
                  ? "text-emerald-600 dark:text-emerald-300"
                  : "text-slate-500",
          )}
        />
        <span
          className={cn(
            "min-w-0 flex-1 truncate text-[10px] font-extrabold leading-none tabular-nums",
            error
              ? "text-rose-700 dark:text-rose-300"
              : low
                ? "text-amber-800 dark:text-amber-200"
                : configured
                  ? "text-emerald-800 dark:text-emerald-200"
                  : "text-slate-600 dark:text-slate-300",
          )}
        >
          {error ? "Offline" : configured ? formatChutesUsd(balance) : "No key"}
        </span>
        <span
          className={cn(
            "shrink-0 text-[8px] font-bold uppercase tracking-wide",
            error
              ? "text-rose-500"
              : low
                ? "text-amber-600 dark:text-amber-300"
                : configured
                  ? "text-emerald-600 dark:text-emerald-300"
                  : "text-slate-400",
          )}
        >
          {error ? "ERR" : low ? "LOW" : configured ? "OK" : "SET"}
        </span>
      </div>
      <div className="h-1 overflow-hidden rounded-full bg-black/10 dark:bg-white/10">
        <div
          className={cn(
            "h-full rounded-full transition-all duration-500",
            error ? "bg-rose-500" : low ? "bg-amber-500" : configured ? "bg-emerald-500" : "bg-slate-400",
          )}
          style={{ width: `${configured ? Math.max(4, fillPct) : 0}%` }}
        />
      </div>
    </button>
  );
}
