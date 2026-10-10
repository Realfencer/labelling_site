import { KeyRound } from "lucide-react";
import { cn } from "../lib/utils";
import { useKeys } from "../store/useKeys";

/** Opens the shared API-keys window; the dot shows whether everything is configured. */
export default function KeysButton({ className, label = true }: { className?: string; label?: boolean }) {
  const view = useKeys((s) => s.view);
  const setOpen = useKeys((s) => s.setOpen);
  const missing = view ? Number(view.hf.length === 0) + Number(!view.chutes.configured) + Number(!view.ultralytics.configured) : 0;
  return (
    <button
      type="button"
      onClick={() => setOpen(true)}
      title={missing ? `API keys — ${missing} not set yet` : "API keys (all set)"}
      className={cn(
        "relative flex shrink-0 items-center gap-1 rounded-md border border-slate-300 px-1.5 py-1 text-[10px] font-bold text-slate-600 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-800",
        className,
      )}
    >
      <KeyRound className="h-3 w-3" />
      {label && "Keys"}
      {view && <span className={cn("h-1.5 w-1.5 rounded-full", missing ? "bg-amber-400" : "bg-emerald-400")} />}
    </button>
  );
}
