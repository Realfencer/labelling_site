import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import type { ChutesBalanceAlert } from "../api";
import { formatChutesUsd, loadChutesSettings } from "../chutesSettings";

const RESHOW_MS = 30 * 60_000;
const DISMISS_KEY = "chutes_balance_toast_dismissed_at";

/**
 * Windows-style toast (IPMSG-like) for low Chutes balance.
 * Consumes the same poll as the header chip — no duplicate /api/chutes/balance-alert traffic.
 */
export default function ChutesBalanceToast() {
  const [visible, setVisible] = useState(false);
  const [balance, setBalance] = useState<number | null>(null);
  const [threshold, setThreshold] = useState(20);
  const [username, setUsername] = useState<string | null>(null);
  const [fetchedAt, setFetchedAt] = useState<string>("");
  const lastLowRef = useRef(false);

  useEffect(() => {
    const onAlert = (ev: Event) => {
      const alert = (ev as CustomEvent<ChutesBalanceAlert>).detail;
      if (!alert?.configured) {
        setVisible(false);
        lastLowRef.current = false;
        return;
      }
      setBalance(alert.balance);
      setThreshold(alert.balance_alert_usd);
      setUsername(alert.username ?? null);
      setFetchedAt(alert.fetched_at || new Date().toUTCString());

      if (!alert.low_balance) {
        lastLowRef.current = false;
        setVisible(false);
        return;
      }

      const dismissedAt = Number(window.localStorage.getItem(DISMISS_KEY) || 0);
      const recentlyDismissed = Date.now() - dismissedAt < RESHOW_MS;
      const crossedIntoLow = !lastLowRef.current;
      lastLowRef.current = true;
      if (crossedIntoLow || !recentlyDismissed) {
        setVisible(true);
      }
    };
    const onSettings = () => {
      lastLowRef.current = false;
      void loadChutesSettings();
    };
    window.addEventListener("chutes-balance-alert", onAlert as EventListener);
    window.addEventListener("chutes-settings-changed", onSettings);
    return () => {
      window.removeEventListener("chutes-balance-alert", onAlert as EventListener);
      window.removeEventListener("chutes-settings-changed", onSettings);
    };
  }, []);

  if (!visible) return null;

  const when = fetchedAt || new Date().toString();

  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-[90] w-[min(360px,calc(100vw-1.5rem))]">
      <div className="pointer-events-auto overflow-hidden rounded-xl border border-white/10 bg-[#1f1f1f]/92 shadow-[0_12px_40px_rgba(0,0,0,0.45)] backdrop-blur-md chutes-toast-enter">
        <div className="flex items-center gap-2 border-b border-white/10 px-3 py-1.5">
          <div className="flex h-4 w-4 items-center justify-center rounded-sm bg-sky-500 text-[9px] font-black text-white">C</div>
          <div className="flex-1 truncate text-[11px] font-semibold text-white/90">Chutes Balance</div>
          <button
            type="button"
            aria-label="Dismiss"
            className="rounded p-0.5 text-white/60 hover:bg-white/10 hover:text-white"
            onClick={() => {
              window.localStorage.setItem(DISMISS_KEY, String(Date.now()));
              setVisible(false);
            }}
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
        <div className="flex gap-3 px-3 py-3">
          <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-rose-500 via-amber-400 to-sky-500 text-lg font-black text-white shadow-inner">
            $
          </div>
          <div className="min-w-0 pt-0.5">
            <div className="text-sm font-bold text-white">Low balance alert</div>
            <div className="mt-0.5 text-[12px] leading-snug text-white/80">
              {username ? `${username} · ` : ""}
              {formatChutesUsd(balance)} left (threshold {formatChutesUsd(threshold)})
            </div>
            <div className="mt-1 truncate text-[11px] text-white/50">at {when}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
