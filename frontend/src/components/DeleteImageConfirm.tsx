import { useEffect, useState } from "react";
import { Loader2, Trash2 } from "lucide-react";

/** Shift+Delete: "Delete this image from the dataset?" — Enter confirms, Esc cancels. */
export default function DeleteImageConfirm({
  name,
  hasLabel,
  onConfirm,
  onCancel,
}: {
  name: string;
  hasLabel: boolean;
  onConfirm: () => Promise<void>;
  onCancel: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const confirm = async () => {
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      await onConfirm();
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/, ""));
      setBusy(false);
    }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Enter" && !e.repeat) {
        e.preventDefault();
        e.stopPropagation();
        void confirm();
      } else if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        if (!busy) onCancel();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  });

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/60 p-4">
      <div className="w-full max-w-md rounded-2xl border border-rose-400/50 bg-white p-5 shadow-2xl dark:bg-ink-900">
        <div className="flex items-center gap-2 text-rose-600 dark:text-rose-300">
          <Trash2 className="h-5 w-5" />
          <h2 className="text-base font-bold">Delete this image from the dataset?</h2>
        </div>
        <p className="mt-3 break-all rounded-lg bg-slate-100 px-3 py-2 font-mono text-xs text-slate-700 dark:bg-slate-800 dark:text-slate-200">{name}</p>
        <p className="mt-3 text-sm text-slate-600 dark:text-slate-300">
          The image file{hasLabel ? " and its label" : ""} are removed for good. This cannot be undone.
        </p>
        {error && <p className="mt-2 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950/40 dark:text-rose-300">{error}</p>}
        <div className="mt-4 flex items-center justify-end gap-2">
          <span className="mr-auto text-[11px] text-slate-400">Enter = delete · Esc = cancel</span>
          <button type="button" disabled={busy} onClick={onCancel} className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-semibold dark:border-slate-600">
            Cancel
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => void confirm()}
            className="inline-flex items-center gap-1.5 rounded-lg bg-rose-600 px-3 py-1.5 text-xs font-bold text-white hover:bg-rose-500 disabled:opacity-50"
          >
            {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />} Delete
          </button>
        </div>
      </div>
    </div>
  );
}
