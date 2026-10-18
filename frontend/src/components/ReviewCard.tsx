import { useCallback, useEffect, useState } from "react";
import { ChevronRight, Loader2, Play, ShieldAlert, Square } from "lucide-react";
import { api, type ReviewSummary } from "../api";
import { cn } from "../lib/utils";

export const KIND_LABEL: Record<string, string> = {
  huge: "big box, no drink",
  tiny: "tiny box",
  stacked: "stacked boxes",
  nested: "box in box",
  wrong: "other class",
  missed: "unlabeled drink",
};

/** Own day list: the label audit — how many labeled images look wrong, the model check, and the way into Review. */
export default function ReviewCard({ onOpen, busy = false }: { onOpen: () => void | Promise<void>; busy?: boolean }) {
  const [sum, setSum] = useState<ReviewSummary | null>(null);
  const [error, setError] = useState("");
  const [opening, setOpening] = useState(false);
  const [acting, setActing] = useState(false);

  const load = useCallback(async () => {
    try {
      setSum(await api.ownReviewSummary());
      setError("");
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/, ""));
    }
  }, []);

  const mc = sum?.model_check;
  const live = !sum?.ready || Boolean(mc?.running);
  useEffect(() => {
    void load();
    const t = window.setInterval(() => void load(), live ? 4000 : 20000);
    return () => window.clearInterval(t);
  }, [load, live]);

  const modelCheck = async (action: "start" | "stop") => {
    setActing(true);
    setError("");
    try {
      await api.ownReviewModelCheck(action);
      await load();
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/, ""));
    } finally {
      setActing(false);
    }
  };

  const pct = mc && mc.total > 0 ? Math.min(100, (mc.done / mc.total) * 100) : 0;
  const complete = mc ? mc.total > 0 && mc.done >= mc.total : false;
  const kinds = Object.entries(sum?.by_kind ?? {}).sort((a, b) => b[1] - a[1]);

  return (
    <div className="rounded-xl border border-amber-300/80 bg-amber-50/70 p-3 dark:border-amber-700/60 dark:bg-amber-950/20">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-amber-500 text-white">
          <ShieldAlert className="h-4 w-4" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-extrabold text-slate-800 dark:text-slate-100">
            Review ·{" "}
            {sum?.ready ? (
              <span className={sum.flagged ? "text-amber-700 dark:text-amber-300" : "text-emerald-600 dark:text-emerald-300"}>
                {sum.flagged} image{sum.flagged === 1 ? "" : "s"} to check
              </span>
            ) : (
              <span className="text-slate-500">checking labels…</span>
            )}
          </div>
          <div className="mt-0.5 text-[11px] text-slate-500 dark:text-slate-400">
            {sum?.ready
              ? `Found automatically in ${sum.labeled.toLocaleString()} labeled images: big boxes with no drink under them, tiny or stacked boxes, and where the model sees another class or an unlabeled drink. Riskiest first.${sum.dismissed ? ` ${sum.dismissed} marked “looks right”.` : ""}`
              : "Scoring every labeled image…"}
          </div>
          {kinds.length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {kinds.map(([k, n]) => (
                <span key={k} className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] font-bold text-amber-800 dark:text-amber-200">
                  {n} {KIND_LABEL[k] ?? k}
                </span>
              ))}
            </div>
          )}
        </div>
        <button
          type="button"
          disabled={busy || opening || !sum?.ready || !sum.flagged}
          onClick={() => {
            setOpening(true);
            void Promise.resolve(onOpen()).finally(() => setOpening(false));
          }}
          className="inline-flex items-center gap-1 rounded-lg bg-amber-500 px-3 py-1.5 text-[12px] font-extrabold text-white shadow-sm hover:bg-amber-400 disabled:opacity-40"
        >
          {opening ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null} Start review <ChevronRight className="h-3.5 w-3.5" />
        </button>
      </div>

      {mc && (
        <div className="mt-2.5 flex flex-wrap items-center gap-2 border-t border-amber-200/80 pt-2 text-[11px] dark:border-amber-800/40">
          <span className="font-semibold text-slate-600 dark:text-slate-300">Model check</span>
          <div className="h-1.5 min-w-[80px] flex-1 overflow-hidden rounded-full bg-amber-200/60 dark:bg-amber-900/40">
            <div className={cn("h-full rounded-full transition-all", complete ? "bg-emerald-500" : "bg-amber-500")} style={{ width: `${pct}%` }} />
          </div>
          <span className="tabular-nums text-slate-500 dark:text-slate-400">
            {mc.done.toLocaleString()} / {mc.total.toLocaleString()} images
            {mc.running ? " · running (low priority)" : complete ? " · done" : ""}
          </span>
          {mc.running ? (
            <button
              type="button"
              disabled={acting}
              onClick={() => void modelCheck("stop")}
              className="inline-flex items-center gap-1 rounded border border-slate-300 px-2 py-0.5 font-bold text-slate-600 hover:border-rose-400 hover:text-rose-600 disabled:opacity-50 dark:border-slate-600 dark:text-slate-300"
            >
              <Square className="h-3 w-3" /> Stop
            </button>
          ) : !complete ? (
            <button
              type="button"
              disabled={acting}
              onClick={() => void modelCheck("start")}
              title="Run the base model once on every Own image (low priority, about half an hour for 14k). It lets the review see wrong classes and unlabeled drinks, and stop flagging big or tiny boxes the model agrees with."
              className="inline-flex items-center gap-1 rounded bg-slate-800 px-2 py-0.5 font-bold text-white hover:bg-slate-700 disabled:opacity-50 dark:bg-slate-200 dark:text-slate-900"
            >
              <Play className="h-3 w-3" /> {mc.done > 0 ? "Continue" : "Run"} model check
            </button>
          ) : null}
          {mc.error ? <span className="text-rose-600 dark:text-rose-300">{mc.error}</span> : null}
        </div>
      )}
      {error ? <p className="mt-2 text-[11px] text-rose-600 dark:text-rose-300">{error}</p> : null}
    </div>
  );
}
