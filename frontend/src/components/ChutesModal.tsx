import { useEffect, useMemo, useState } from "react";
import { Activity, RefreshCw, X } from "lucide-react";
import { api, type ChutesStatus } from "../api";
import {
  formatChutesUsd,
  loadChutesSettings,
  saveChutesSettings,
} from "../chutesSettings";
import { cn } from "../lib/utils";

function BalanceGauge({
  balance,
  threshold,
}: {
  balance: number | null;
  threshold: number;
}) {
  const max = Math.max(threshold * 2, balance ?? 0, 1);
  const pct = balance == null ? 0 : Math.min(100, Math.max(0, (balance / max) * 100));
  const low = balance != null && balance < threshold;
  const r = 54;
  const c = 2 * Math.PI * r;
  const dash = (pct / 100) * c;

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative h-36 w-36">
        <svg viewBox="0 0 140 140" className="h-full w-full -rotate-90">
          <circle cx="70" cy="70" r={r} fill="none" stroke="currentColor" strokeWidth="12" className="text-slate-200 dark:text-slate-800" />
          <circle
            cx="70"
            cy="70"
            r={r}
            fill="none"
            stroke="currentColor"
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={`${dash} ${c - dash}`}
            className={cn(low ? "text-rose-500" : "text-emerald-500")}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div className={cn("text-xl font-extrabold tracking-tight", low ? "text-rose-600 dark:text-rose-400" : "text-slate-900 dark:text-slate-50")}>
            {formatChutesUsd(balance)}
          </div>
          <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Balance</div>
        </div>
      </div>
      <div className="text-center text-[11px] text-slate-500 dark:text-slate-400">
        Alert below <span className="font-semibold text-slate-700 dark:text-slate-200">{formatChutesUsd(threshold)}</span>
      </div>
    </div>
  );
}

export default function ChutesModal({
  open,
  onClose,
  onStatus,
}: {
  open: boolean;
  onClose: () => void;
  onStatus?: (status: ChutesStatus | null) => void;
}) {
  const initial = loadChutesSettings();
  const [apiKey, setApiKey] = useState(initial.apiKey);
  const [threshold, setThreshold] = useState(String(initial.threshold));
  const [status, setStatus] = useState<ChutesStatus | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const parsedThreshold = useMemo(() => {
    const n = Number(threshold);
    return Number.isFinite(n) && n >= 0 ? n : 20;
  }, [threshold]);

  const load = async (force = true) => {
    const saved = saveChutesSettings({ apiKey, threshold: parsedThreshold });
    // Allow empty UI key when server .env already has CHUTES_API_KEY.
    setLoading(true);
    setError("");
    try {
      await api.saveChutesMonitor({
        api_key: saved.apiKey,
        threshold: saved.threshold,
      });
      const data = await api.getChutesStatus({
        api_key: saved.apiKey,
        threshold: saved.threshold,
        force,
      });
      setStatus(data);
      onStatus?.(data);
    } catch (e) {
      setError(String(e));
      onStatus?.(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!open) return;
    const saved = loadChutesSettings();
    setApiKey(saved.apiKey);
    setThreshold(String(saved.threshold));
    let alive = true;
    setLoading(true);
    setError("");
    void (async () => {
      try {
        const data = await api.getChutesStatus({
          api_key: saved.apiKey,
          threshold: saved.threshold,
          force: true,
        });
        if (!alive) return;
        setStatus(data);
        onStatus?.(data);
      } catch (e) {
        if (!alive) return;
        setStatus(null);
        // Don't force an error on open if user still needs to enter a key.
        if (saved.apiKey) setError(String(e));
        onStatus?.(null);
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [open, onStatus]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const title = useMemo(() => {
    if (!status?.username) return "Chutes settings";
    return `Chutes · ${status.username}`;
  }, [status?.username]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center p-4">
      <button type="button" aria-label="Close overlay" className="absolute inset-0 bg-black/50 backdrop-blur-[2px]" onClick={onClose} />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="relative z-[81] flex max-h-[90vh] w-full max-w-lg flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-[#0f141d]"
      >
        <div className="flex items-center justify-between gap-3 border-b border-slate-200 px-4 py-3 dark:border-slate-800">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-sky-500" />
            <div>
              <div className="text-sm font-extrabold text-slate-900 dark:text-slate-50">{title}</div>
              <div className="text-[10px] text-slate-500">
                {status?.fetched_at ? `Updated ${status.fetched_at}` : "API key + alert threshold"}
              </div>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="overflow-y-auto px-4 py-4">
          <div className="space-y-3">
            <label className="block">
              <span className="mb-1 block text-[11px] font-bold uppercase tracking-wider text-slate-500">Chutes API key</span>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="cpk_..."
                autoComplete="off"
                spellCheck={false}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none ring-sky-500/40 focus:ring-2 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-[11px] font-bold uppercase tracking-wider text-slate-500">Threshold (USD)</span>
              <input
                type="number"
                min={0}
                step={0.01}
                value={threshold}
                onChange={(e) => setThreshold(e.target.value)}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none ring-sky-500/40 focus:ring-2 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
              />
              <span className="mt-1 block text-[10px] text-slate-500">Notify when balance falls below this amount.</span>
            </label>
            <button
              type="button"
              onClick={() => void load(true)}
              disabled={loading}
              className="inline-flex w-full items-center justify-center gap-1.5 rounded-lg bg-sky-600 px-3 py-2 text-sm font-semibold text-white hover:bg-sky-500 disabled:opacity-50"
            >
              <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
              {loading ? "Checking…" : "Save & observe"}
            </button>
          </div>

          {error && (
            <div className="mt-3 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-semibold text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-300">
              {error}
            </div>
          )}

          {status && (
            <div className="mt-5 flex flex-col items-center gap-3 border-t border-slate-200 pt-4 dark:border-slate-800">
              <BalanceGauge balance={status.balance} threshold={status.balance_alert_usd} />
              {status.low_balance && (
                <div className="w-full rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs font-semibold text-amber-800 dark:border-amber-800/60 dark:bg-amber-950/40 dark:text-amber-200">
                  Balance is below {formatChutesUsd(status.balance_alert_usd)}. Top up soon to avoid interruption.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
