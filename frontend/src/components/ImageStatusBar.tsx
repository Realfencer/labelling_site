import { memo, useEffect, useState } from "react";
import { AlertTriangle, Check, Eye, Loader2, Lock } from "lucide-react";
import { timeAgo, useTeam } from "../store/useTeam";
import { cn } from "../lib/utils";

function SaveBadge() {
  const save = useTeam((s) => s.save);
  const [, tick] = useState(0);
  useEffect(() => {
    if (save.status !== "saved") return;
    const t = window.setTimeout(() => tick((n) => n + 1), 4000);
    return () => window.clearTimeout(t);
  }, [save]);
  if (save.status === "idle") return null;
  if (save.status === "saving")
    return (
      <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-slate-500 dark:text-slate-400">
        <Loader2 className="h-3 w-3 animate-spin" /> Saving…
      </span>
    );
  if (save.status === "saved") {
    const fresh = save.at != null && Date.now() - save.at < 4000;
    return (
      <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-emerald-600 dark:text-emerald-400">
        <Check className="h-3 w-3" /> {fresh ? "Saved" : `Saved ${new Date(save.at ?? Date.now()).toLocaleTimeString()}`}
      </span>
    );
  }
  return (
    <span
      className="inline-flex max-w-[360px] items-center gap-1 truncate rounded bg-rose-500/15 px-1.5 py-0.5 text-[10px] font-bold text-rose-600 dark:text-rose-300"
      title={save.message}
    >
      <AlertTriangle className="h-3 w-3 shrink-0" />
      {save.status === "retrying" ? "Not saved yet — retrying…" : save.message || "Not saved"}
    </span>
  );
}

/** Directly above the image: who labeled it / who is on it, read-only banners, save state. */
function ImageStatusBar({
  itemKey,
  reviewed,
  onToggleWatch,
  layout = "inline",
}: {
  itemKey: string | null;
  reviewed: boolean;
  onToggleWatch: () => void;
  /** "rail": stacked, for the narrow middle panel. "inline": one row, for a pane header. */
  layout?: "rail" | "inline";
}) {
  const rail = layout === "rail";
  const me = useTeam((s) => s.me);
  const users = useTeam((s) => s.users);
  const label = useTeam((s) => (itemKey ? s.labels[itemKey] : undefined));
  const readOnly = useTeam((s) => s.readOnly);
  const reason = useTeam((s) => s.reason);
  const owner = useTeam((s) => s.owner);
  const watching = useTeam((s) => s.watching);
  const watchers = useTeam((s) => s.watchers);
  const notice = useTeam((s) => s.notice);
  const [, tick] = useState(0);

  useEffect(() => {
    const t = window.setInterval(() => tick((n) => n + 1), 15000);
    return () => window.clearInterval(t);
  }, []);
  useEffect(() => {
    if (!notice) return;
    const t = window.setTimeout(() => useTeam.getState().setNotice(""), 5000);
    return () => window.clearTimeout(t);
  }, [notice]);

  const colorOf = (name: string | null | undefined) => users.find((u) => u.name === name)?.color ?? "#94a3b8";

  let banner: React.ReactNode = null;
  const bannerBtn = (label: string) => (
    <button
      type="button"
      onClick={onToggleWatch}
      className="shrink-0 rounded bg-white/20 px-1.5 py-0.5 text-[10px] font-bold hover:bg-white/30"
    >
      {label}
    </button>
  );
  if (watching) {
    banner = (
      <div
        className={cn(
          "flex min-w-0 gap-1.5 rounded-md bg-violet-600 px-2 py-1 text-[11px] font-bold leading-tight text-white",
          rail ? "flex-col items-stretch" : "items-center",
        )}
        title="You are watching live and cannot change labels"
      >
        <span className="flex min-w-0 items-center gap-1">
          <Eye className="h-3.5 w-3.5 shrink-0" />
          <span className={rail ? "" : "truncate"}>Watching {watching} · read-only</span>
        </span>
        {bannerBtn("Stop (G)")}
      </div>
    );
  } else if (readOnly && reason === "claimed") {
    banner = (
      <div
        className={cn(
          "flex min-w-0 gap-1.5 rounded-md bg-rose-600 px-2 py-1 text-[11px] font-bold leading-tight text-white",
          rail ? "flex-col items-stretch" : "items-center",
        )}
        title={`${owner} is labeling this image. Press G to watch live, or pick another image.`}
      >
        <span className="flex min-w-0 items-center gap-1">
          <Lock className="h-3.5 w-3.5 shrink-0" />
          <span className={rail ? "" : "truncate"}>{owner} is labeling · read-only</span>
        </span>
        {bannerBtn("Watch (G)")}
      </div>
    );
  }

  const chip = (() => {
    if (label?.u) {
      const you = label.u === me?.user;
      return (
        <span
          className={cn(
            "inline-flex min-w-0 rounded-md px-2 py-0.5 text-[11px] font-extrabold leading-tight text-slate-900",
            rail ? "flex-col" : "items-center gap-1.5 rounded-full",
          )}
          style={{ backgroundColor: colorOf(label.u) }}
          title={`Labeled by ${label.u}${label.t ? ` · ${new Date(label.t * 1000).toLocaleString()}` : ""}`}
        >
          <span className="flex min-w-0 items-center gap-1">
            <Check className="h-3 w-3 shrink-0" />
            <span className="truncate">
              {label.u}
              {you ? " (you)" : ""}
            </span>
          </span>
          {label.t ? <span className="text-[10px] font-semibold opacity-70">{timeAgo(label.t)}</span> : null}
        </span>
      );
    }
    if (reviewed) {
      return (
        <span className="inline-flex items-center gap-1 rounded-md bg-slate-300 px-2 py-0.5 text-[11px] font-bold text-slate-800 dark:bg-slate-600 dark:text-slate-100">
          <Check className="h-3 w-3" /> Labeled
        </span>
      );
    }
    return (
      <span className="inline-flex items-center rounded-md bg-amber-400 px-2 py-0.5 text-[11px] font-extrabold text-amber-950">
        NEW
      </span>
    );
  })();

  return (
    <div className={cn("flex min-w-0 gap-1.5", rail ? "flex-col items-stretch" : "flex-wrap items-center")}>
      {banner}
      {/* The labeled-by tag is always visible, also while read-only / watching. */}
      {chip}
      {!banner && watchers.length > 0 && (
        <span className="inline-flex items-center gap-1 rounded-md bg-violet-500/15 px-2 py-0.5 text-[10px] font-bold text-violet-600 dark:text-violet-300">
          <Eye className="h-3 w-3 shrink-0" /> {watchers.join(", ")} watching
        </span>
      )}
      {notice && (
        <span className="rounded bg-amber-400/90 px-2 py-0.5 text-[10px] font-bold leading-tight text-amber-950">{notice}</span>
      )}
      <SaveBadge />
    </div>
  );
}

export default memo(ImageStatusBar);
