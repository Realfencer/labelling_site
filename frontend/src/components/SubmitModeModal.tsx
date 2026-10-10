import { GitBranch, Rocket, Sparkles, X } from "lucide-react";
import { cn } from "../lib/utils";

export default function SubmitModeModal({
  open,
  onClose,
  onAuto,
  onManual,
}: {
  open: boolean;
  onClose: () => void;
  onAuto: () => void;
  onManual: () => void;
}) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4 backdrop-blur-[3px]">
      <div className="w-full max-w-xl overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-ink-900">
        <div className="flex items-center justify-between border-b border-slate-200 px-5 py-4 dark:border-slate-700">
          <div>
            <h2 className="text-base font-extrabold text-slate-900 dark:text-white">Submit</h2>
            <p className="text-[12px] text-slate-500">Choose how you want to build the ship pack</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="grid gap-3 p-5 sm:grid-cols-2">
          <button
            type="button"
            onClick={() => {
              onClose();
              onAuto();
            }}
            className={cn(
              "group relative overflow-hidden rounded-2xl border border-sky-200 bg-gradient-to-br from-sky-50 via-white to-cyan-50 p-4 text-left transition",
              "hover:-translate-y-0.5 hover:border-sky-400 hover:shadow-lg hover:shadow-sky-500/10",
              "dark:border-sky-900 dark:from-sky-950/40 dark:via-ink-900 dark:to-cyan-950/30",
            )}
          >
            <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-xl bg-sky-500 text-white shadow-lg shadow-sky-500/30">
              <Rocket className="h-5 w-5" />
            </div>
            <div className="text-sm font-extrabold text-slate-900 dark:text-white">Auto</div>
            <p className="mt-1 text-[12px] leading-relaxed text-slate-500">
              Discover miners, score, soup, and deep post-process automatically.
            </p>
            <div className="mt-3 text-[10px] font-bold uppercase tracking-wider text-sky-600 dark:text-sky-300">
              Find best · pipeline
            </div>
          </button>

          <button
            type="button"
            onClick={() => {
              onClose();
              onManual();
            }}
            className={cn(
              "group relative overflow-hidden rounded-2xl border border-violet-200 bg-gradient-to-br from-violet-50 via-white to-fuchsia-50 p-4 text-left transition",
              "hover:-translate-y-0.5 hover:border-violet-400 hover:shadow-lg hover:shadow-violet-500/10",
              "dark:border-violet-900 dark:from-violet-950/40 dark:via-ink-900 dark:to-fuchsia-950/30",
            )}
          >
            <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 text-white shadow-lg shadow-violet-500/30">
              <GitBranch className="h-5 w-5" />
            </div>
            <div className="flex items-center gap-1.5 text-sm font-extrabold text-slate-900 dark:text-white">
              Manual
              <Sparkles className="h-3.5 w-3.5 text-fuchsia-500" />
            </div>
            <p className="mt-1 text-[12px] leading-relaxed text-slate-500">
              Pick models yourself, force all-pair merges, eval vs R1, then deep-PP and push.
            </p>
            <div className="mt-3 text-[10px] font-bold uppercase tracking-wider text-violet-600 dark:text-violet-300">
              Curated · pairs
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}
