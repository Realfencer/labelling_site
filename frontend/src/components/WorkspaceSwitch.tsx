import { Globe2, Images } from "lucide-react";
import type { WorkspaceMetrics } from "../types";

export default function WorkspaceSwitch({
  isPlatform,
  platformMetrics,
  ownMetrics,
  modeBusy,
  onSwitchPlatform,
  onSwitchOwn,
}: {
  isPlatform: boolean;
  platformMetrics: WorkspaceMetrics;
  ownMetrics: WorkspaceMetrics;
  modeBusy: boolean;
  onSwitchPlatform: () => void;
  onSwitchOwn: () => void;
}) {
  return (
    <div
      className="inline-flex h-7 shrink-0 items-center rounded-md border border-slate-200 bg-slate-100 p-0.5 dark:border-slate-700 dark:bg-ink-950"
      role="tablist"
      aria-label="Workspace"
    >
      <button
        type="button"
        role="tab"
        aria-selected={isPlatform}
        disabled={modeBusy}
        title="Platform — eval images from score subnet"
        onClick={onSwitchPlatform}
        className={`inline-flex h-6 items-center gap-0.5 rounded px-1.5 text-[10px] font-extrabold transition disabled:opacity-50 sm:gap-1 sm:px-2 sm:text-[11px] ${
          isPlatform
            ? "bg-sky-500 text-white shadow-sm"
            : "text-slate-600 hover:bg-white/80 dark:text-slate-300 dark:hover:bg-slate-800"
        }`}
      >
        <Globe2 className="h-3 w-3 shrink-0 sm:h-3.5 sm:w-3.5" />
        <span className="hidden sm:inline">Platform</span>
        <span className={`tabular-nums ${isPlatform ? "text-white/85" : "text-slate-400"}`}>
          {platformMetrics.done}/{platformMetrics.not_reviewed}
        </span>
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={!isPlatform}
        disabled={modeBusy}
        title="Own — your uploaded / generated images"
        onClick={onSwitchOwn}
        className={`inline-flex h-6 items-center gap-0.5 rounded px-1.5 text-[10px] font-extrabold transition disabled:opacity-50 sm:gap-1 sm:px-2 sm:text-[11px] ${
          !isPlatform
            ? "bg-violet-500 text-white shadow-sm"
            : "text-slate-600 hover:bg-white/80 dark:text-slate-300 dark:hover:bg-slate-800"
        }`}
      >
        <Images className="h-3 w-3 shrink-0 sm:h-3.5 sm:w-3.5" />
        <span className="hidden sm:inline">Own</span>
        <span className={`tabular-nums ${!isPlatform ? "text-white/85" : "text-slate-400"}`}>
          {ownMetrics.done}/{ownMetrics.total}
        </span>
      </button>
    </div>
  );
}
