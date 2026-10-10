import { useEffect, useMemo, useState } from "react";
import { CloudDownload, Loader2, SkipForward, Replace, X } from "lucide-react";
import { api, type OwnHfImportJob } from "../api";
import type { AppState } from "../types";
import { cn } from "../lib/utils";

type Phase = "url" | "working" | "conflicts" | "done" | "error";

export default function OwnHfImportModal({
  open,
  onClose,
  onDone,
}: {
  open: boolean;
  onClose: () => void;
  onDone: (state: AppState) => void | Promise<void>;
}) {
  const [url, setUrl] = useState("");
  const [phase, setPhase] = useState<Phase>("url");
  const [error, setError] = useState("");
  const [job, setJob] = useState<OwnHfImportJob | null>(null);
  const [conflictIdx, setConflictIdx] = useState(0);
  const [replaceNames, setReplaceNames] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    setUrl("");
    setPhase("url");
    setError("");
    setJob(null);
    setConflictIdx(0);
    setReplaceNames([]);
    setBusy(false);
  }, [open]);

  useEffect(() => {
    if (!job?.running) return;
    const timer = window.setInterval(() => {
      void api
        .ownHfImportStatus(job.id)
        .then((res) => {
          setJob(res.job);
          if (res.job.status === "ready") {
            if ((res.job.conflicts || []).length > 0) {
              setPhase("conflicts");
              setConflictIdx(0);
            } else {
              // No conflicts — commit immediately
              void commitImport(res.job, []);
            }
          } else if (res.job.status === "error") {
            setPhase("error");
            setError(res.job.error || res.job.message || "Import failed");
          } else if (res.job.status === "done") {
            setPhase("done");
          }
        })
        .catch((e) => {
          setPhase("error");
          setError(String(e).replace(/^Error:\s*/i, ""));
        });
    }, 1200);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.id, job?.running]);

  const conflicts = job?.conflicts || [];
  const current = conflicts[conflictIdx];
  const remaining = Math.max(0, conflicts.length - conflictIdx);

  const progressLabel = useMemo(() => {
    if (!job) return "";
    return job.message || job.status;
  }, [job]);

  const startImport = async () => {
    setError("");
    if (!url.trim()) {
      setError("Paste a Hugging Face .tar.gz resolve URL");
      return;
    }
    setBusy(true);
    setPhase("working");
    try {
      const res = await api.ownHfImportStart({
        url: url.trim(),
      });
      setJob(res.job);
      if (res.job.status === "ready") {
        if ((res.job.conflicts || []).length > 0) {
          setPhase("conflicts");
          setConflictIdx(0);
        } else {
          await commitImport(res.job, []);
        }
      } else if (res.job.status === "error") {
        setPhase("error");
        setError(res.job.error || res.job.message || "Import failed");
      }
    } catch (e) {
      setPhase("error");
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setBusy(false);
    }
  };

  const commitImport = async (active: OwnHfImportJob, replace: string[]) => {
    setBusy(true);
    setPhase("working");
    try {
      const res = await api.ownHfImportCommit({
        job_id: active.id,
        replace_names: replace,
      });
      setJob(res.job);
      setPhase("done");
      await onDone(res);
    } catch (e) {
      setPhase("error");
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setBusy(false);
    }
  };

  const decideConflict = (replace: boolean) => {
    if (!current) return;
    const nextReplace = replace ? [...replaceNames, current.dest_name] : replaceNames;
    if (!replace) {
      /* skip */
    }
    setReplaceNames(nextReplace);
    if (conflictIdx + 1 >= conflicts.length) {
      void commitImport(job!, nextReplace);
      return;
    }
    setConflictIdx((i) => i + 1);
  };

  const replaceAllRemaining = () => {
    const rest = conflicts.slice(conflictIdx).map((c) => c.dest_name);
    const merged = [...replaceNames, ...rest];
    setReplaceNames(merged);
    void commitImport(job!, merged);
  };

  const skipAllRemaining = () => {
    void commitImport(job!, replaceNames);
  };

  const cancelJob = async () => {
    if (job?.id && job.running) {
      try {
        await api.ownHfImportCancel(job.id);
      } catch {
        /* ignore */
      }
    }
    onClose();
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/55 p-3 backdrop-blur-[2px]">
      <div className="flex w-full max-w-lg flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-ink-900">
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-700">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-violet-500 text-white">
              <CloudDownload className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-sm font-extrabold text-slate-900 dark:text-white">Import from Hugging Face</h2>
              <p className="text-[11px] text-slate-500">beverage-detect Own · .tar.gz with images/ + labels/</p>
            </div>
          </div>
          <button type="button" onClick={() => void cancelJob()} className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-3 px-4 py-4">
          {phase === "url" && (
            <>
              <label className="block">
                <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">
                  Hugging Face file URL
                </span>
                <input
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://huggingface.co/WhiteCrow323/model/resolve/main/something.tar.gz"
                  className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 font-mono text-[12px] dark:border-slate-700 dark:bg-ink-800"
                  autoFocus
                />
              </label>
              <p className="text-[11px] text-slate-500">
                Archive should contain <code className="font-mono">images/</code> + <code className="font-mono">labels/</code>{" "}
                (YOLO .txt, same stem). Existing Own filenames will ask before overwrite.
              </p>
            </>
          )}

          {phase === "working" && (
            <div className="flex flex-col items-center gap-2 py-8 text-center">
              <Loader2 className="h-8 w-8 animate-spin text-violet-500" />
              <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">{progressLabel || "Working…"}</div>
              {job && (
                <div className="text-[11px] text-slate-500">
                  {job.n_images ? `${job.n_images} images · ${job.n_labels} labels` : job.archive_name}
                </div>
              )}
            </div>
          )}

          {phase === "conflicts" && current && (
            <div className="space-y-3">
              <div className="rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-[12px] text-amber-900 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-100">
                Filename already in Own pool ({conflictIdx + 1}/{conflicts.length} conflicts · {remaining} left)
              </div>
              <div className="rounded-lg border border-slate-200 px-3 py-3 dark:border-slate-700">
                <div className="font-mono text-sm font-bold text-slate-900 dark:text-white">{current.dest_name}</div>
                <div className="mt-1 text-[11px] text-slate-500">
                  Existing: {current.existing_name || "—"}
                  {current.has_label ? " · incoming has label" : " · no label in archive"}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => decideConflict(false)}
                  className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-200"
                >
                  <SkipForward className="h-4 w-4" />
                  Keep existing
                </button>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => decideConflict(true)}
                  className="inline-flex items-center justify-center gap-1.5 rounded-lg bg-violet-600 px-3 py-2 text-sm font-bold text-white hover:bg-violet-500"
                >
                  <Replace className="h-4 w-4" />
                  Replace
                </button>
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  disabled={busy}
                  onClick={skipAllRemaining}
                  className="flex-1 rounded-lg border border-slate-200 px-2 py-1.5 text-[11px] font-semibold text-slate-600 dark:border-slate-700 dark:text-slate-300"
                >
                  Skip all remaining
                </button>
                <button
                  type="button"
                  disabled={busy}
                  onClick={replaceAllRemaining}
                  className="flex-1 rounded-lg border border-violet-200 px-2 py-1.5 text-[11px] font-semibold text-violet-700 dark:border-violet-800 dark:text-violet-300"
                >
                  Replace all remaining
                </button>
              </div>
            </div>
          )}

          {phase === "done" && job && (
            <div className="space-y-2 py-4 text-center">
              <div className="text-sm font-bold text-emerald-600 dark:text-emerald-400">Import complete</div>
              <div className="text-[12px] text-slate-600 dark:text-slate-300">{job.message}</div>
              {job.import_result && (
                <div className="font-mono text-[11px] text-slate-500">
                  added {job.import_result.added} · replaced {job.import_result.replaced} · skipped{" "}
                  {job.import_result.skipped} · labels {job.import_result.labels_written} · pool{" "}
                  {job.import_result.total}
                </div>
              )}
            </div>
          )}

          {(phase === "error" || error) && phase !== "url" && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-[12px] text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200">
              {error || job?.error || "Import failed"}
            </div>
          )}
          {phase === "url" && error && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-[12px] text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200">
              {error}
            </div>
          )}
        </div>

        <div className="flex justify-end gap-2 border-t border-slate-200 px-4 py-3 dark:border-slate-700">
          <button
            type="button"
            onClick={() => void cancelJob()}
            className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-600 dark:border-slate-700 dark:text-slate-300"
          >
            {phase === "done" ? "Close" : "Cancel"}
          </button>
          {phase === "url" && (
            <button
              type="button"
              disabled={busy || !url.trim()}
              onClick={() => void startImport()}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-lg bg-violet-600 px-4 py-2 text-sm font-bold text-white hover:bg-violet-500 disabled:opacity-40",
              )}
            >
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <CloudDownload className="h-4 w-4" />}
              Download &amp; import
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
