import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Loader2, X } from "lucide-react";
import { cn } from "../lib/utils";

export function Card({ title, right, children, className }: { title?: ReactNode; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cn("rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-ink-900", className)}>
      {(title || right) && (
        <div className="mb-3 flex items-center gap-2">
          {title && <h2 className="text-[13px] font-extrabold tracking-tight text-slate-800 dark:text-slate-100">{title}</h2>}
          {right && <div className="ml-auto flex items-center gap-2">{right}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

type BtnKind = "primary" | "secondary" | "ghost" | "danger" | "success";

export function Btn({
  kind = "secondary",
  busy,
  children,
  className,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { kind?: BtnKind; busy?: boolean }) {
  const styles: Record<BtnKind, string> = {
    primary: "bg-gradient-to-r from-sky-600 to-violet-600 text-white shadow hover:brightness-110",
    success: "bg-emerald-600 text-white hover:bg-emerald-500",
    secondary: "border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-600 dark:bg-ink-800 dark:text-slate-200 dark:hover:bg-slate-700",
    ghost: "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800",
    danger: "border border-rose-400 text-rose-600 hover:bg-rose-500/10 dark:text-rose-300",
  };
  return (
    <button
      type="button"
      {...rest}
      disabled={rest.disabled || busy}
      className={cn(
        "inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-1.5 text-[12px] font-bold transition disabled:cursor-not-allowed disabled:opacity-45",
        styles[kind],
        className,
      )}
    >
      {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
      {children}
    </button>
  );
}

export function Chip({ children, tone = "slate", title }: { children: ReactNode; tone?: "slate" | "green" | "amber" | "rose" | "sky" | "violet"; title?: string }) {
  const tones = {
    slate: "bg-slate-500/10 text-slate-600 dark:text-slate-300",
    green: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300",
    amber: "bg-amber-500/15 text-amber-700 dark:text-amber-300",
    rose: "bg-rose-500/15 text-rose-700 dark:text-rose-300",
    sky: "bg-sky-500/15 text-sky-700 dark:text-sky-300",
    violet: "bg-violet-500/15 text-violet-700 dark:text-violet-300",
  };
  return (
    <span title={title} className={cn("inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2 py-0.5 text-[10px] font-bold", tones[tone])}>
      {children}
    </span>
  );
}

export function Progress({ value, tone = "sky" }: { value: number; tone?: "sky" | "violet" | "emerald" }) {
  const colors = { sky: "from-sky-500 to-cyan-400", violet: "from-violet-500 to-fuchsia-500", emerald: "from-emerald-500 to-teal-400" };
  return (
    <div className="h-2 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
      <div
        className={cn("h-full rounded-full bg-gradient-to-r transition-all duration-500", colors[tone])}
        style={{ width: `${Math.max(2, Math.min(100, value * 100))}%` }}
      />
    </div>
  );
}

export function Banner({ kind, children }: { kind: "info" | "ok" | "warn" | "error"; children: ReactNode }) {
  const k = {
    info: "border-sky-300 bg-sky-50 text-sky-800 dark:border-sky-900 dark:bg-sky-950/40 dark:text-sky-200",
    ok: "border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-200",
    warn: "border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200",
    error: "border-rose-300 bg-rose-50 text-rose-800 dark:border-rose-900 dark:bg-rose-950/40 dark:text-rose-200",
  }[kind];
  return <div className={cn("rounded-lg border px-3 py-2 text-[12px]", k)}>{children}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="rounded-xl border border-dashed border-slate-300 px-4 py-8 text-center text-[12px] text-slate-500 dark:border-slate-700">{children}</div>;
}

/**
 * Poll a job while it runs. Returns the latest job; ``kick()`` starts polling (after you started a job).
 * Polling stops on anything but "running" and calls ``onFinish`` once when a run ends.
 */
export function useJob<T extends { status: string }>(fetcher: () => Promise<T>, intervalMs: number, onFinish?: (job: T) => void) {
  const [job, setJob] = useState<T | null>(null);
  const timer = useRef<number | null>(null);
  const wasRunning = useRef(false);
  const finishRef = useRef(onFinish);
  finishRef.current = onFinish;
  const fetchRef = useRef(fetcher);
  fetchRef.current = fetcher;

  const tick = useCallback(async () => {
    let next: T | null = null;
    try {
      next = await fetchRef.current();
      setJob(next);
    } catch {
      // the Studio may be restarting: try again a bit later
      timer.current = window.setTimeout(() => void tick(), intervalMs * 3);
      return;
    }
    if (next.status === "running") {
      wasRunning.current = true;
      timer.current = window.setTimeout(() => void tick(), intervalMs);
    } else if (wasRunning.current) {
      wasRunning.current = false;
      finishRef.current?.(next);
    }
  }, [intervalMs]);

  const kick = useCallback(() => {
    if (timer.current) window.clearTimeout(timer.current);
    wasRunning.current = true;
    void tick();
  }, [tick]);

  useEffect(() => {
    void tick();
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [tick]);

  return { job, kick, setJob };
}

export function errText(e: unknown): string {
  return String(e instanceof Error ? e.message : e).replace(/^Error:\s*/, "");
}

/** Two-pane image viewer (source | generated, or a single image). Esc / backdrop / × close. */
export function Lightbox({
  panes,
  caption,
  onClose,
}: {
  panes: { label: string; src: string }[];
  caption?: ReactNode;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  // Portal: the step column is its own stacking context, so an in-place overlay sits under the results.
  return createPortal(
    <div className="fixed inset-0 z-[90] flex flex-col bg-black/85 p-4" onClick={onClose}>
      <div className="mb-2 flex items-center gap-3 text-[12px] text-slate-200" onClick={(e) => e.stopPropagation()}>
        <div className="min-w-0 flex-1 truncate">{caption}</div>
        <button type="button" onClick={onClose} className="rounded-md p-1.5 hover:bg-white/10" aria-label="Close">
          <X className="h-5 w-5" />
        </button>
      </div>
      <div className={cn("grid min-h-0 flex-1 gap-3", panes.length > 1 ? "md:grid-cols-2" : "grid-cols-1")}>
        {panes.map((p) => (
          <figure key={p.label + p.src} className="flex min-h-0 flex-col" onClick={(e) => e.stopPropagation()}>
            <figcaption className="mb-1 text-[10px] font-bold uppercase tracking-wider text-slate-400">{p.label}</figcaption>
            <img src={p.src} alt={p.label} className="min-h-0 flex-1 rounded-lg object-contain" />
          </figure>
        ))}
      </div>
    </div>,
    document.body,
  );
}

/** Confirmation dialog for money-spending or destructive actions. */
export function Confirm({
  title,
  children,
  confirmLabel,
  danger,
  typeToConfirm,
  onConfirm,
  onCancel,
}: {
  title: string;
  children: ReactNode;
  confirmLabel: string;
  danger?: boolean;
  typeToConfirm?: string;
  onConfirm: () => void | Promise<void>;
  onCancel: () => void;
}) {
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const ok = !typeToConfirm || typed.trim().toUpperCase() === typeToConfirm;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !busy) onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [busy, onCancel]);
  return createPortal(
    <div className="fixed inset-0 z-[95] flex items-center justify-center bg-black/60 p-4">
      <div className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-5 shadow-2xl dark:border-slate-700 dark:bg-ink-900">
        <h3 className={cn("text-base font-extrabold", danger ? "text-rose-600 dark:text-rose-300" : "text-slate-800 dark:text-slate-100")}>{title}</h3>
        <div className="mt-2 space-y-2 text-[13px] text-slate-600 dark:text-slate-300">{children}</div>
        {typeToConfirm && (
          <input
            autoFocus
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            placeholder={`Type ${typeToConfirm}`}
            className="mt-3 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-600 dark:bg-ink-800"
          />
        )}
        <div className="mt-4 flex justify-end gap-2">
          <Btn kind="ghost" onClick={onCancel} disabled={busy}>
            Cancel
          </Btn>
          <Btn
            kind={danger ? "danger" : "primary"}
            busy={busy}
            disabled={!ok}
            onClick={async () => {
              setBusy(true);
              try {
                await onConfirm();
              } finally {
                setBusy(false);
              }
            }}
          >
            {confirmLabel}
          </Btn>
        </div>
      </div>
    </div>,
    document.body,
  );
}

export const CLASS_COLORS: Record<string, string> = { cup: "#ef4444", can: "#22c55e", bottle: "#0ea5e9" };

export function pct(v: number | null | undefined, d = 0): string {
  return v == null || !Number.isFinite(v) ? "—" : `${(v * 100).toFixed(d)}%`;
}

/** Which generator made a frame (OpenAI / Gemini). */
export function GenBadge({ gen, long, className }: { gen: "openai" | "gemini" | null | undefined; long?: boolean; className?: string }) {
  if (!gen) return null;
  return (
    <span
      title={gen === "openai" ? "Made by OpenAI (gpt-image)" : "Made by Gemini"}
      className={cn(
        "inline-flex items-center rounded px-1 text-[9px] font-black uppercase leading-[14px] tracking-wide text-white",
        gen === "openai" ? "bg-sky-600" : "bg-violet-600",
        className,
      )}
    >
      {long ? (gen === "openai" ? "OpenAI" : "Gemini") : gen === "openai" ? "O" : "G"}
    </span>
  );
}
