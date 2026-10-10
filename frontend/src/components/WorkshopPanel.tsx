import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Check, ExternalLink, FlaskConical, Fingerprint, Loader2, Send, Trash2, X } from "lucide-react";
import { api, type CollectorSubmission, type CollectorVariant } from "../api";
import { cn } from "../lib/utils";
import { timeAgo } from "../store/useTeam";
import HfPushTarget from "./HfPushTarget";

const f3 = (v: number | null | undefined) => (v == null ? "—" : v.toFixed(3));

/**
 * Workshop of one passed model. Each step is its own button and produces a variant that is measured on the
 * working image set (score + the p95 latency limit); any variant can be pushed to a saved Hugging Face
 * account on its own, whenever you decide.
 */
export default function WorkshopPanel({
  sub,
  setLabel,
  minS,
  maxP95,
  onChooseImages,
  onChanged,
}: {
  sub: CollectorSubmission;
  setLabel: string;
  minS: number;
  maxP95: number;
  onChooseImages: () => void;
  onChanged: () => void;
}) {
  const [variants, setVariants] = useState<CollectorVariant[]>([]);
  const [busy, setBusy] = useState<{ what: string; id: number; status: string; progress: string } | null>(null);
  const [error, setError] = useState("");
  const [note, setNote] = useState("");
  const [pushFor, setPushFor] = useState<number | null>(null);
  const [accountId, setAccountId] = useState("");
  const [repoName, setRepoName] = useState(() => {
    try {
      return localStorage.getItem("collector_repo_name") || "";
    } catch {
      return "";
    }
  });
  const timer = useRef<number | null>(null);

  const load = useCallback(async () => {
    try {
      setVariants((await api.collectorVariants(sub.id)).variants);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/, ""));
    }
  }, [sub.id]);
  useEffect(() => {
    void load();
    return () => {
      if (timer.current != null) window.clearTimeout(timer.current);
    };
  }, [load]);

  const follow = useCallback(
    (what: string, id: number, done: (result: Record<string, unknown>) => void) => {
      setBusy({ what, id, status: "queued", progress: "waiting for the collector…" });
      const poll = async () => {
        try {
          const c = await api.collectorCommand(id);
          if (c.status === "error") {
            setError(String((c.result as { error?: string }).error ?? "failed"));
            setBusy(null);
            void load();
            return;
          }
          if (c.status === "done") {
            setBusy(null);
            done(c.result as Record<string, unknown>);
            void load();
            onChanged();
            return;
          }
          setBusy({ what, id, status: c.status, progress: c.progress || (c.status === "queued" ? "waiting for the collector…" : "working…") });
        } catch {
          /* keep polling */
        }
        timer.current = window.setTimeout(() => void poll(), 1500);
      };
      void poll();
    },
    [load, onChanged],
  );

  const stage = async (parent: CollectorVariant, st: "dpp" | "hash") => {
    setError("");
    setNote("");
    try {
      const r = await api.collectorStage(sub.id, parent.stage === "original" ? null : parent.id, st);
      follow(st === "dpp" ? "Deep post-process" : "Changing the ONNX hash", r.command_id, (res) => {
        setNote(
          `${st === "dpp" ? "Deep post-process" : "ONNX hash change"} finished: s ${f3(res.s_before as number)} → ${f3(res.s_after as number)}${
            res.ok ? "" : " — does not meet the gate, cannot be pushed"
          }.`,
        );
      });
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/, ""));
    }
  };

  const push = async (v: CollectorVariant) => {
    setError("");
    setNote("");
    try {
      localStorage.setItem("collector_repo_name", repoName.trim());
    } catch {
      /* private mode */
    }
    try {
      const r = await api.collectorPushVariant(v.id, { account_id: accountId, repo_name: repoName.trim() });
      follow("Pushing", r.command_id, (res) => {
        setPushFor(null);
        setNote(`Pushed (private): ${String(res.url)} — s ${f3(res.s as number)}, p95 ${res.p95_ms != null ? Math.round(res.p95_ms as number) : "—"} ms.`);
      });
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/, ""));
    }
  };

  const remove = async (v: CollectorVariant) => {
    try {
      await api.collectorDeleteVariant(v.id);
      await load();
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/, ""));
    }
  };

  const byId = useMemo(() => new Map(variants.map((v) => [v.id, v])), [variants]);
  const depth = (v: CollectorVariant): number => {
    let d = 0;
    let cur: CollectorVariant | undefined = v;
    while (cur?.parent_id != null && d < 6) {
      cur = byId.get(cur.parent_id);
      d += 1;
    }
    return d;
  };
  const working = busy != null;

  return (
    <div className="mt-4 rounded-lg border border-slate-200 p-3 dark:border-slate-700">
      <div className="mb-1 flex items-start justify-between gap-2">
        <div className="text-[11px] font-bold uppercase tracking-wide text-slate-400">Workshop · improve, then push when you decide</div>
      </div>
      <div className="mb-2 text-[11px] text-slate-500 dark:text-slate-400">
        Every step is measured on <b>{setLabel || "the working image set"}</b>{" "}
        <button type="button" onClick={onChooseImages} className="text-sky-600 hover:underline dark:text-sky-300">
          change images
        </button>
        . Gate: s ≥ {minS.toFixed(2)} and p95 ≤ {maxP95} ms — a result that is too slow is kept but cannot be pushed.
      </div>

      <div className="space-y-2">
        {variants.map((v) => {
          const d = depth(v);
          const dpp = v.info.dpp;
          const rolled = dpp?.latency_reverted;
          const slow = v.p95_ms != null && v.p95_ms > maxP95;
          return (
            <div key={v.id} className="rounded-md border border-slate-200 p-2.5 dark:border-slate-700" style={{ marginLeft: Math.min(d, 3) * 14 }}>
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[12px] font-extrabold">
                  {v.stage === "dpp" ? <FlaskConical className="mr-1 inline h-3.5 w-3.5 text-violet-400" /> : v.stage === "hash" ? <Fingerprint className="mr-1 inline h-3.5 w-3.5 text-amber-400" /> : null}
                  {v.label}
                </span>
                {v.parent_id != null && byId.get(v.parent_id) && (
                  <span className="text-[10px] text-slate-400">from “{byId.get(v.parent_id)?.label}”</span>
                )}
                <span className="ml-auto text-[10px] text-slate-400">{v.stage === "original" ? "" : timeAgo(v.created_at)}</span>
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px]">
                <span className={cn("text-base font-extrabold tabular-nums", v.s != null && v.s >= minS ? "text-emerald-600 dark:text-emerald-300" : "text-rose-500")}>s {f3(v.s)}</span>
                <span className={cn("tabular-nums", slow ? "font-bold text-rose-500" : "text-slate-600 dark:text-slate-300")}>p95 {v.p95_ms != null ? Math.round(v.p95_ms) : "—"} ms</span>
                {v.stage !== "original" && v.info.s_before != null && (
                  <span className="text-[11px] text-slate-400">
                    was {f3(v.info.s_before)} ({v.n_images ?? "?"} images)
                  </span>
                )}
                <span className="font-mono text-[10px] text-slate-400" title={v.onnx_sha ?? ""}>sha {v.onnx_sha?.slice(0, 10) ?? "—"}</span>
                {v.stage !== "original" && (
                  <span className={cn("rounded px-1.5 py-0.5 text-[10px] font-bold uppercase", v.ok ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-300" : "bg-rose-500/15 text-rose-500")} title={v.reason}>
                    {v.ok ? "meets the gate" : "not pushable"}
                  </span>
                )}
              </div>
              {v.stage === "dpp" && dpp && (
                <div className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">
                  {rolled
                    ? "The search found settings, but they pushed p95 over the limit — rolled back, files unchanged."
                    : dpp.postproc_applied
                      ? "Post-process settings applied to miner.py (latency limit kept)."
                      : "No better settings found — the files are unchanged."}
                  {dpp.latency_rule?.after_p95_ms != null && ` p95 ${Math.round(dpp.latency_rule.baseline_p95_ms ?? 0)} → ${Math.round(dpp.latency_rule.after_p95_ms)} ms.`}
                </div>
              )}
              {!v.ok && v.stage !== "original" && <div className="mt-1 text-[11px] text-rose-500">{v.reason}</div>}
              {v.pushed.length > 0 && (
                <div className="mt-1 space-y-0.5 text-[11px] text-emerald-600 dark:text-emerald-300">
                  {v.pushed.map((p, i) => (
                    <div key={i} className="flex items-center gap-1">
                      <Check className="h-3 w-3" /> pushed to
                      <a href={p.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-0.5 font-semibold underline">
                        {p.repo} <ExternalLink className="h-3 w-3" />
                      </a>
                      <span className="text-slate-400">· {timeAgo(p.at)}</span>
                    </div>
                  ))}
                </div>
              )}

              <div className="mt-2 flex flex-wrap items-center gap-1.5">
                <button type="button" disabled={working} onClick={() => void stage(v, "dpp")} className="inline-flex items-center gap-1 rounded-md border border-violet-500/60 px-2 py-1 text-[11px] font-bold text-violet-600 hover:bg-violet-500/10 disabled:opacity-40 dark:text-violet-300">
                  <FlaskConical className="h-3 w-3" /> Deep post-process
                </button>
                <button type="button" disabled={working} onClick={() => void stage(v, "hash")} className="inline-flex items-center gap-1 rounded-md border border-amber-500/60 px-2 py-1 text-[11px] font-bold text-amber-600 hover:bg-amber-500/10 disabled:opacity-40 dark:text-amber-300">
                  <Fingerprint className="h-3 w-3" /> Change ONNX hash
                </button>
                <button
                  type="button"
                  disabled={working || (!v.ok && v.stage !== "original")}
                  onClick={() => setPushFor(pushFor === v.id ? null : v.id)}
                  className="inline-flex items-center gap-1 rounded-md bg-emerald-600 px-2.5 py-1 text-[11px] font-bold text-white hover:bg-emerald-500 disabled:opacity-40"
                >
                  <Send className="h-3 w-3" /> Push…
                </button>
                <a className="ml-auto text-[11px] text-sky-600 hover:underline dark:text-sky-300" href={v.stage === "original" ? `/api/collector/submissions/${sub.id}/download` : `/api/collector/variants/${v.id}/download`}>
                  download files
                </a>
                {v.stage !== "original" && (
                  <button type="button" disabled={working} onClick={() => void remove(v)} title="Delete this result" className="rounded p-1 text-slate-400 hover:bg-rose-500/10 hover:text-rose-500 disabled:opacity-40">
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>

              {pushFor === v.id && (
                <div className="mt-2 grid gap-2 rounded-md bg-slate-50 p-2 text-[12px] dark:bg-slate-800/50">
                  <HfPushTarget
                    accountId={accountId}
                    repoName={repoName}
                    onChange={(x) => {
                      setAccountId(x.accountId);
                      setRepoName(x.repoName);
                    }}
                  />
                  <div className="text-[10px] text-slate-400">
                    The files are checked once more on {setLabel || "the working images"} right before the upload; nothing is pushed if they miss the gate or the p95 limit.
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => void push(v)}
                      disabled={working || !accountId || !repoName.trim()}
                      className="inline-flex items-center gap-1 rounded-md bg-emerald-600 px-3 py-1.5 text-[12px] font-bold text-white hover:bg-emerald-500 disabled:opacity-40"
                    >
                      <Send className="h-3.5 w-3.5" /> Push (private)
                    </button>
                    <button type="button" onClick={() => setPushFor(null)} className="rounded p-1 text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700">
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
        {variants.length === 0 && <div className="text-[12px] text-slate-400">Loading…</div>}
      </div>

      {busy && (
        <div className="mt-2 flex items-center gap-2 rounded-md border border-sky-500/40 px-2.5 py-1.5 text-[12px] font-semibold text-sky-600 dark:text-sky-300">
          <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
          <span className="min-w-0 truncate">
            {busy.what}: {busy.progress}
          </span>
        </div>
      )}
      {note && <div className="mt-2 rounded bg-emerald-500/10 p-2 text-[11px] text-emerald-700 dark:text-emerald-300">{note}</div>}
      {error && <div className="mt-2 text-[11px] text-rose-500">{error}</div>}
    </div>
  );
}
