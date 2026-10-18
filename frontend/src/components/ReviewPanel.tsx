import { useState } from "react";
import { Check, Loader2, RotateCcw, ShieldAlert, ThumbsUp } from "lucide-react";
import { api, type ReviewIssue } from "../api";
import type { DetectionBox } from "../types";
import { useReview } from "../store/useReview";
import { syncReviewNow } from "../hooks/useReviewSync";
import { cn } from "../lib/utils";

const KIND_STYLE: Record<string, string> = {
  huge: "border-rose-400 bg-rose-500/15 text-rose-700 dark:text-rose-200",
  tiny: "border-orange-400 bg-orange-500/15 text-orange-700 dark:text-orange-200",
  stacked: "border-fuchsia-400 bg-fuchsia-500/15 text-fuchsia-700 dark:text-fuchsia-200",
  nested: "border-fuchsia-300 bg-fuchsia-500/10 text-fuchsia-700 dark:text-fuchsia-200",
  wrong: "border-amber-400 bg-amber-500/15 text-amber-800 dark:text-amber-200",
  missed: "border-cyan-400 bg-cyan-500/15 text-cyan-800 dark:text-cyan-200",
};

function describe(issue: ReviewIssue, boxes: DetectionBox[], names: string[]): string {
  const nm = (c: number | undefined) => (c != null ? names[c] ?? `class ${c}` : "?");
  const own = issue.i >= 0 ? boxes[issue.i] : undefined;
  switch (issue.kind) {
    case "huge":
      return `Huge ${nm(own?.cls)} box · ${issue.pct ?? "?"}% of the image${issue.detail ? ` · ${issue.detail}` : ""}`;
    case "tiny":
      return `Tiny ${nm(own?.cls)} box · ${issue.px ? issue.px.map((v) => Math.round(v)).join("×") : "?"} px${issue.detail ? ` · ${issue.detail}` : ""}`;
    case "stacked": {
      const other = issue.other != null ? boxes[issue.other] : undefined;
      return `Two boxes on one object · ${nm(own?.cls)} + ${nm(other?.cls)}`;
    }
    case "nested":
      return `${nm(own?.cls)} box inside another box`;
    case "wrong":
      return `Labeled ${nm(own?.cls)} · model sees ${nm(issue.model_cls)} (${Math.round((issue.conf ?? 0) * 100)}%)`;
    case "missed":
      return `Model sees an unlabeled ${nm(issue.model_cls)} (${Math.round((issue.conf ?? 0) * 100)}%)`;
    default:
      return issue.text;
  }
}

/** Own Review: what looks wrong on this image, live — fixed by a save, or marked "looks right" (O). */
export default function ReviewPanel({
  itemKey,
  boxes,
  classNames,
  openCount,
  totalCount,
  onFocus,
  onHover,
}: {
  itemKey: string | null;
  boxes: DetectionBox[];
  classNames: string[];
  openCount: number;
  totalCount: number;
  onFocus: (issue: ReviewIssue) => void;
  onHover: (issue: ReviewIssue | null) => void;
}) {
  const st = useReview((s) => (itemKey ? s.items[itemKey] : undefined));
  const [busy, setBusy] = useState(false);
  const issues = (st?.issues ?? []).filter((i) => i.risk > 0);

  const setLooksRight = async (on: boolean) => {
    if (!itemKey) return;
    setBusy(true);
    try {
      const r = await api.ownReviewDismiss(itemKey, on);
      useReview.getState().patchItem(itemKey, r.status, r.version);
      syncReviewNow();
    } finally {
      setBusy(false);
    }
  };

  const fixed = st && !st.open && !st.dismissed;
  return (
    <div
      className={cn(
        "flex flex-wrap items-center gap-2 rounded-lg border px-2.5 py-1.5 text-[11px] transition-colors duration-300",
        st === undefined
          ? "border-slate-200 bg-white/80 dark:border-slate-700 dark:bg-ink-900/70"
          : st?.open
            ? "border-amber-300 bg-amber-50/90 dark:border-amber-700/60 dark:bg-amber-950/30"
            : "border-emerald-300 bg-emerald-50/90 dark:border-emerald-700/60 dark:bg-emerald-950/30",
      )}
    >
      {st === undefined ? (
        <span className="inline-flex items-center gap-1 font-bold text-slate-500">
          <Loader2 className="h-3.5 w-3.5 animate-spin" /> Checking…
        </span>
      ) : st === null ? (
        <span className="inline-flex items-center gap-1 font-bold text-emerald-600 dark:text-emerald-300">
          <Check className="h-3.5 w-3.5" /> No labels left on this image
        </span>
      ) : st.open ? (
        <span className="inline-flex items-center gap-1 font-extrabold text-amber-700 dark:text-amber-300" title={`Risk ${st.risk}`}>
          <ShieldAlert className="h-3.5 w-3.5" /> Check this
        </span>
      ) : (
        <span className="inline-flex items-center gap-1 font-extrabold text-emerald-600 dark:text-emerald-300">
          <Check className="h-3.5 w-3.5" /> {fixed ? "Fixed" : "Looks right"}
        </span>
      )}

      {issues.map((issue, k) => (
        <button
          key={`${issue.kind}:${issue.i}:${k}`}
          type="button"
          onClick={() => onFocus(issue)}
          onMouseEnter={() => onHover(issue)}
          onMouseLeave={() => onHover(null)}
          title="Zoom to it"
          className={cn(
            "rounded-md border px-1.5 py-0.5 font-bold transition hover:brightness-110",
            KIND_STYLE[issue.kind] ?? "border-slate-300",
            st && !st.open && "opacity-60",
          )}
        >
          {describe(issue, boxes, classNames)}
        </button>
      ))}

      <span className="ml-auto tabular-nums text-slate-500 dark:text-slate-400">
        {openCount} of {totalCount} still to check
      </span>
      {st?.dismissed ? (
        <button
          type="button"
          disabled={busy}
          onClick={() => void setLooksRight(false)}
          className="inline-flex items-center gap-1 rounded-md border border-slate-300 px-2 py-0.5 font-bold text-slate-600 hover:border-amber-400 disabled:opacity-50 dark:border-slate-600 dark:text-slate-300"
        >
          <RotateCcw className="h-3 w-3" /> Undo
        </button>
      ) : st?.open ? (
        <button
          type="button"
          disabled={busy}
          onClick={() => void setLooksRight(true)}
          title="The labels are right: take this image off the list (O)"
          className="inline-flex items-center gap-1 rounded-md bg-emerald-600 px-2 py-0.5 font-bold text-white hover:bg-emerald-500 disabled:opacity-50"
        >
          <ThumbsUp className="h-3 w-3" /> Looks right (O)
        </button>
      ) : null}
    </div>
  );
}
