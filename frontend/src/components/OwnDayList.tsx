import { useCallback, useEffect, useMemo, useState } from "react";
import { CalendarDays, ChevronRight, CloudUpload, Loader2, Trash2 } from "lucide-react";
import { api, type OwnDayRow, type OwnDaysResponse, type OwnField } from "../api";
import { cn } from "../lib/utils";
import ReviewCard from "./ReviewCard";

type Props = {
  /** OpenAI / Gemini images only (null = every image). */
  field: OwnField | null;
  onFieldChange: (field: OwnField | null) => void;
  onEnterDay: (date: string) => void | Promise<void>;
  busy?: boolean;
  /** Days ticked in the list (kept by the page so the toolbar's Ultralytics button can use them too). */
  selected: string[];
  onSelectedChange: (days: string[]) => void;
  /** Open the Ultralytics upload for these days (several days = one dataset). */
  onSendDays: (days: string[]) => void;
  /** Delete every image of these days (the page deletes, clears its caches and reloads the state). */
  onDeleteDays: (days: string[]) => Promise<void>;
};

export default function OwnDayList({ field, onFieldChange, onEnterDay, busy = false, selected, onSelectedChange, onSendDays, onDeleteDays }: Props) {
  const [data, setData] = useState<OwnDaysResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [entering, setEntering] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<string[] | null>(null); // days waiting for "delete for sure"
  const [deleting, setDeleting] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setData(await api.listOwnDays());
    } catch (e) {
      setError(String(e));
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const allDays: OwnDayRow[] = data?.days ?? [];
  // a field shows only its days, with that generator's counts
  const days: OwnDayRow[] = useMemo(
    () => (field ? allDays.filter((d) => (d.fields?.[field]?.total ?? 0) > 0).map((d) => ({ ...d, ...d.fields![field] })) : allDays),
    [allDays, field],
  );
  const fieldTotals = useMemo(() => {
    const out: Record<string, { images: number; days: number; done: number }> = {};
    for (const key of ["all", "openai", "gemini"]) out[key] = { images: 0, days: 0, done: 0 };
    for (const d of allDays) {
      out.all.images += d.total;
      out.all.days += 1;
      out.all.done += d.done;
      for (const g of ["openai", "gemini"] as const) {
        const c = d.fields?.[g];
        if (c?.total) {
          out[g].images += c.total;
          out[g].days += 1;
          out[g].done += c.done;
        }
      }
    }
    return out;
  }, [allDays]);
  const byDate = useMemo(() => new Map(days.map((d) => [d.date, d])), [days]);
  const sel = useMemo(() => selected.filter((d) => byDate.has(d)), [selected, byDate]);
  const sum = (list: string[], key: "total" | "done" | "new") => list.reduce((n, d) => n + (byDate.get(d)?.[key] ?? 0), 0);
  const toggle = (date: string) => onSelectedChange(selected.includes(date) ? selected.filter((d) => d !== date) : [...selected, date]);
  const allOn = days.length > 0 && sel.length === days.length;

  const doDelete = async (list: string[]) => {
    setDeleting(true);
    setError("");
    try {
      await onDeleteDays(list);
      onSelectedChange(selected.filter((d) => !list.includes(d)));
      setConfirm(null);
      await refresh();
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/, ""));
    } finally {
      setDeleting(false);
    }
  };

  const confirming = confirm ? confirm.filter((d) => byDate.has(d)) : [];

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-ink-900/55 dark:backdrop-blur-[3px]">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-violet-500 text-white">
            <CalendarDays className="h-4 w-4" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-800 dark:text-slate-100">Own days</h2>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              Clustered by upload mtime · server {data?.timezone || "local"}
              {data?.today ? ` · today ${data.today}` : ""}
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => void refresh()}
          disabled={loading || busy}
          className="rounded-md border border-slate-200 px-2 py-1 text-[10px] font-semibold text-slate-600 hover:border-violet-300 hover:text-violet-700 disabled:opacity-50 dark:border-slate-600 dark:text-slate-300"
        >
          Refresh
        </button>
      </div>

      {allDays.length > 0 && <ReviewCard onOpen={() => onEnterDay("review")} busy={busy} />}

      {allDays.length > 0 && (
        <div className="grid gap-2 sm:grid-cols-3">
          {(
            [
              [null, "All images", "border-slate-300 dark:border-slate-600", "border-slate-800 bg-slate-800 text-white dark:border-slate-100 dark:bg-slate-100 dark:text-slate-900", fieldTotals.all],
              ["openai", "OpenAI images", "border-sky-300 dark:border-sky-800", "border-sky-500 bg-sky-500 text-white", fieldTotals.openai],
              ["gemini", "Gemini images", "border-violet-300 dark:border-violet-800", "border-violet-500 bg-violet-500 text-white", fieldTotals.gemini],
            ] as const
          ).map(([f, label, idle, active, t]) => (
            <button
              key={label}
              type="button"
              onClick={() => (onFieldChange(f), onSelectedChange([]))}
              className={cn("rounded-xl border px-3 py-2 text-left transition", field === f ? active : cn(idle, "hover:border-violet-400"))}
              title={f ? `Only the images ${f === "openai" ? "OpenAI" : "Gemini"} made, by day` : "Every Own image, by day"}
            >
              <div className="text-[13px] font-extrabold">{label}</div>
              <div className={cn("text-[11px] tabular-nums", field === f ? "opacity-90" : "text-slate-500 dark:text-slate-400")}>
                {t.images} images · {t.days} days · {t.done} done
              </div>
            </button>
          ))}
        </div>
      )}

      {days.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 rounded-lg bg-slate-50 px-3 py-2 text-[12px] dark:bg-slate-800/50">
          <label className="flex items-center gap-1.5 font-semibold">
            <input type="checkbox" checked={allOn} onChange={() => onSelectedChange(allOn ? [] : days.map((d) => d.date))} />
            {sel.length > 0 ? `${sel.length} day${sel.length === 1 ? "" : "s"} selected` : "Select days"}
          </label>
          {sel.length > 0 && (
            <span className="text-slate-500 dark:text-slate-400">
              {sum(sel, "total")} images · <b className="text-emerald-600 dark:text-emerald-300">{sum(sel, "done")} done</b> · {sum(sel, "new")} new
            </span>
          )}
          <div className="ml-auto flex items-center gap-2">
            <button
              type="button"
              disabled={sel.length === 0 || sum(sel, "done") === 0 || busy}
              onClick={() => onSendDays(sel)}
              title={field ? `Send the ${field === "openai" ? "OpenAI" : "Gemini"} Done images of the selected days to Ultralytics as one dataset` : "Send the Done images of the selected days to Ultralytics as one dataset"}
              className="inline-flex items-center gap-1 rounded-md bg-rose-600 px-2.5 py-1 font-bold text-white hover:bg-rose-500 disabled:opacity-40"
            >
              <CloudUpload className="h-3.5 w-3.5" /> Send {sel.length || ""} {sel.length === 1 ? "day" : "days"} to Ultralytics
            </button>
            {!field && <button
              type="button"
              disabled={sel.length === 0 || busy || deleting}
              onClick={() => setConfirm(sel)}
              title="Delete every image of the selected days"
              className="inline-flex items-center gap-1 rounded-md border border-rose-400 px-2.5 py-1 font-bold text-rose-600 hover:bg-rose-500/10 disabled:opacity-40 dark:text-rose-300"
            >
              <Trash2 className="h-3.5 w-3.5" /> Delete {sel.length || ""}
            </button>}
          </div>
        </div>
      )}

      {confirming.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-rose-300 bg-rose-50 px-3 py-2 text-[12px] text-rose-800 dark:border-rose-900/60 dark:bg-rose-950/30 dark:text-rose-200">
          <Trash2 className="h-4 w-4 shrink-0" />
          <span>
            Delete <b>{sum(confirming, "total")} images</b> of {confirming.length === 1 ? confirming[0] : `${confirming.length} days`} — including{" "}
            <b>{sum(confirming, "done")} Done</b> labels. This cannot be undone.
          </span>
          <button
            type="button"
            disabled={deleting}
            onClick={() => void doDelete(confirming)}
            className="ml-auto inline-flex items-center gap-1 rounded-md bg-rose-600 px-3 py-1 font-bold text-white hover:bg-rose-500 disabled:opacity-50"
          >
            {deleting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />} Delete for sure
          </button>
          <button type="button" disabled={deleting} onClick={() => setConfirm(null)} className="rounded-md border border-rose-300 px-2.5 py-1 font-semibold dark:border-rose-800">
            Cancel
          </button>
        </div>
      )}

      {error ? (
        <p className="rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:border-rose-900 dark:bg-rose-950/40 dark:text-rose-300">
          {error}
        </p>
      ) : null}

      {loading && !data ? (
        <div className="flex flex-1 items-center justify-center gap-2 py-16 text-sm text-slate-500">
          <Loader2 className="h-4 w-4 animate-spin" />
          Loading days…
        </div>
      ) : days.length === 0 ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-2 py-16 text-sm text-slate-500 dark:text-slate-400">
          <p>{field && allDays.length ? `No ${field === "openai" ? "OpenAI" : "Gemini"} images in Own.` : "No Own images yet."}</p>
          <p className="text-xs text-slate-400">Upload or HF-import images, then pick a day here.</p>
        </div>
      ) : (
        <div className="min-h-0 flex-1 overflow-y-auto">
          <ul className="divide-y divide-slate-100 dark:divide-slate-800">
            {days.map((row) => {
              const isToday = data?.today === row.date;
              const isBusy = busy || entering === row.date;
              const on = selected.includes(row.date);
              return (
                <li key={row.date} className={cn("flex items-center", on && "bg-violet-50/60 dark:bg-violet-950/20")}>
                  <label className="flex shrink-0 cursor-pointer items-center self-stretch pl-2 pr-1" title="Select this day">
                    <input type="checkbox" checked={on} onChange={() => toggle(row.date)} />
                  </label>
                  <button
                    type="button"
                    disabled={isBusy}
                    onClick={() => {
                      setEntering(row.date);
                      void Promise.resolve(onEnterDay(row.date)).finally(() => setEntering(null));
                    }}
                    className={cn(
                      "flex min-w-0 flex-1 items-center gap-3 px-2 py-2.5 text-left transition hover:bg-violet-50/80 dark:hover:bg-violet-950/30",
                      isBusy && "opacity-60",
                    )}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-bold tabular-nums text-slate-800 dark:text-slate-100">{row.date}</span>
                        {isToday ? (
                          <span className="rounded bg-violet-100 px-1.5 py-0.5 text-[9px] font-extrabold uppercase tracking-wide text-violet-700 dark:bg-violet-900/50 dark:text-violet-200">
                            Today
                          </span>
                        ) : null}
                      </div>
                      <div className="mt-0.5 flex flex-wrap gap-2 text-[10px] font-semibold tabular-nums text-slate-500 dark:text-slate-400">
                        <span>{row.total} total</span>
                        <span className="text-amber-700 dark:text-amber-300">{row.new} new</span>
                        {(row.sam3 ?? 0) > 0 ? <span className="text-sky-700 dark:text-sky-300">{row.sam3} sam3</span> : null}
                        <span className="text-emerald-700 dark:text-emerald-300">{row.done} done</span>
                      </div>
                    </div>
                    {entering === row.date ? <Loader2 className="h-4 w-4 shrink-0 animate-spin text-violet-500" /> : <ChevronRight className="h-4 w-4 shrink-0 text-slate-400" />}
                  </button>
                  <div className="flex shrink-0 items-center gap-0.5 pr-2">
                    <button
                      type="button"
                      disabled={row.done === 0 || busy}
                      onClick={() => onSendDays([row.date])}
                      title={row.done === 0 ? "No Done images on this day yet" : `Send the ${row.done} Done images of ${row.date} to Ultralytics`}
                      className="rounded p-1.5 text-rose-500 hover:bg-rose-500/10 disabled:opacity-25"
                    >
                      <CloudUpload className="h-4 w-4" />
                    </button>
                    {!field && <button
                      type="button"
                      disabled={busy || deleting}
                      onClick={() => setConfirm([row.date])}
                      title={`Delete all ${row.total} images of ${row.date}`}
                      className="rounded p-1.5 text-slate-400 hover:bg-rose-500/10 hover:text-rose-500 disabled:opacity-25"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>}
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}
