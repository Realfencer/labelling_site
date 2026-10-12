import { useSiteBackdrop } from "../store/useBackdrop";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle,
  ArrowLeft,
  Check,
  Clock,
  ExternalLink,
  FlaskConical,
  GitMerge,
  Images,
  ImageUp,
  Loader2,
  Moon,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  Undo2,
  X,
} from "lucide-react";
import {
  api,
  type CollectorSchedule,
  type CollectorCommand,
  type CollectorEvaluation,
  type CollectorOverview,
  type CollectorSubmission,
} from "../api";
import { cn } from "../lib/utils";
import { timeAgo } from "../store/useTeam";
import KeysButton from "../components/KeysButton";
import TryModelsModal from "../components/TryModelsModal";
import SingleModelTest from "../components/SingleModelTest";
import ImageSetModal from "../components/ImageSetModal";
import WorkshopPanel from "../components/WorkshopPanel";
import ManualMergeModal from "../components/ManualMergeModal";

type Status = CollectorSubmission["status"];

const STATUS_META: Record<Status, { label: string; cls: string; order: number }> = {
  passed: { label: "Passed", cls: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-300", order: 0 },
  evaluating: { label: "Evaluating", cls: "bg-sky-500/15 text-sky-600 dark:text-sky-300 animate-pulse", order: 1 },
  queued_eval: { label: "Queued", cls: "bg-sky-500/10 text-sky-600 dark:text-sky-300", order: 2 },
  waiting_public: { label: "Waiting for public", cls: "bg-amber-500/15 text-amber-600 dark:text-amber-300", order: 3 },
  rejected: { label: "Rejected", cls: "bg-rose-500/15 text-rose-600 dark:text-rose-300", order: 4 },
  duplicate: { label: "Duplicate", cls: "bg-violet-500/15 text-violet-600 dark:text-violet-300", order: 5 },
  expired: { label: "Not public in time", cls: "bg-slate-500/15 text-slate-500 dark:text-slate-400", order: 6 },
  error: { label: "Error", cls: "bg-red-500/15 text-red-600 dark:text-red-300", order: 7 },
  deleted: { label: "Deleted", cls: "bg-slate-500/10 text-slate-400 line-through", order: 8 },
};

const FILTERS: Array<{ id: string; label: string; statuses: Status[] }> = [
  { id: "passed", label: "Passed", statuses: ["passed"] },
  { id: "active", label: "In progress", statuses: ["evaluating", "queued_eval", "waiting_public"] },
  { id: "rejected", label: "Rejected", statuses: ["rejected", "error"] },
  { id: "other", label: "Expired / duplicate", statuses: ["expired", "duplicate"] },
  { id: "deleted", label: "Deleted", statuses: ["deleted"] },
  { id: "all", label: "All", statuses: [] },
];

function StatusPill({ s }: { s: CollectorSubmission }) {
  const m = STATUS_META[s.status];
  return (
    <span
      className={cn("inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide", m.cls)}
      title={s.detail || undefined}
    >
      {m.label}
      {s.status === "passed" && s.eval_passed === 0 ? " · below gate now" : ""}
    </span>
  );
}

const fmtClock = (min: number) => `${String(Math.floor(min / 60) % 24).padStart(2, "0")}:${String(min % 60).padStart(2, "0")}`;
const parseClock = (v: string): number | null => {
  const m = /^(\d{1,2}):(\d{2})$/.exec(v.trim());
  if (!m) return null;
  const h = Number(m[1]);
  const mi = Number(m[2]);
  return h < 24 && mi < 60 ? h * 60 + mi : null;
};
const fmtOffset = (min: number) => `UTC${min >= 0 ? "+" : "-"}${Math.floor(Math.abs(min) / 60)}${Math.abs(min) % 60 ? `:${String(Math.abs(min) % 60).padStart(2, "0")}` : ""}`;

/** Daily automatic re-measure of every passed model on the latest validated images. */
function SchedulePanel({
  onChanged,
  basis,
  bulk,
  imageSet,
  nPassed,
  onPick,
}: {
  onChanged: () => void;
  basis: CollectorOverview["scores_basis"];
  bulk: CollectorOverview["bulk"];
  imageSet: CollectorOverview["image_set"];
  nPassed: number;
  onPick: () => void;
}) {
  const [measureErr, setMeasureErr] = useState("");
  const startMeasure = () => {
    setMeasureErr("");
    api
      .collectorRemeasureSet([])
      .then(() => onChanged())
      .catch((e) => setMeasureErr(String(e).replace(/^Error:\s*/, "")));
  };
  const [sch, setSch] = useState<CollectorSchedule | null>(null);
  const [open, setOpen] = useState(false);
  const [start, setStart] = useState("03:00");
  const [len, setLen] = useState(60);
  const [off, setOff] = useState(9);
  const [err, setErr] = useState("");
  const load = useCallback(async () => {
    try {
      const r = await api.collectorSchedule();
      setSch(r);
      setErr("");
    } catch (e) {
      setErr(String(e));
    }
  }, []);
  const running = Boolean(sch?.state.running);
  useEffect(() => {
    void load();
    const t = window.setInterval(() => {
      if (!document.hidden) void load();
    }, running ? 4000 : 30000);
    return () => window.clearInterval(t);
  }, [load, running]);
  useEffect(() => {
    if (!sch) return;
    setStart(fmtClock(sch.settings.start_min));
    setLen(sch.settings.length_min);
    setOff(sch.settings.utc_offset_min / 60);
  }, [sch?.settings.start_min, sch?.settings.length_min, sch?.settings.utc_offset_min]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!sch) return err ? <div className="mb-3 text-[11px] text-rose-500">{err}</div> : null;
  const st = sch.state;
  const done = st.done?.length ?? 0;
  const failed = st.failed?.length ?? 0;
  const total = st.total ?? sch.n_passed;
  const s = sch.settings;
  const winText = `${fmtClock(s.start_min)}–${fmtClock((s.start_min + s.length_min) % 1440)} (${fmtOffset(s.utc_offset_min)})`;
  const save = async (body: Parameters<typeof api.collectorSetSchedule>[0]) => {
    try {
      setSch(await api.collectorSetSchedule(body));
      setErr("");
    } catch (e) {
      setErr(String(e).replace(/^Error:\s*/, ""));
    }
  };
  const nextText = sch.next_start
    ? new Date(sch.next_start * 1000).toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit" })
    : "—";

  const bulkPct = bulk && bulk.total ? Math.round((bulk.done / bulk.total) * 100) : 0;
  const upToDate = basis != null && imageSet?.label != null && basis.label === imageSet.label && basis.complete;
  return (
    <div className="mb-3 space-y-2">
      <div className="rounded-lg border border-sky-500/40 bg-white px-3 py-2.5 text-[12px] dark:bg-ink-900">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
          <Images className="h-4 w-4 text-sky-500" />
          <b>Working image set</b>
          <span className="text-slate-700 dark:text-slate-200">{imageSet?.label ?? "…"}</span>
          <button
            type="button"
            onClick={onPick}
            className="inline-flex items-center gap-1 rounded-md border border-sky-500 px-2 py-1 font-bold text-sky-600 hover:bg-sky-500/10 dark:text-sky-300"
            title="Choose any of the validated images — also used for deep post-process and push"
          >
            <Images className="h-3 w-3" /> Choose images…
          </button>
          <div className="ml-auto flex items-center gap-2">
            <button
              type="button"
              disabled={Boolean(bulk)}
              onClick={startMeasure}
              className="inline-flex items-center gap-1 rounded-md bg-sky-600 px-3 py-1.5 font-bold text-white hover:bg-sky-500 disabled:opacity-50"
              title="Measure every passed model on the working image set — scores update live in the table"
            >
              {bulk ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />} Re-measure all {nPassed} models
            </button>
          </div>
        </div>
        {bulk ? (
          <div className="mt-2">
            <div className="h-2 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
              <div className="h-full bg-sky-500 transition-all" style={{ width: `${bulkPct}%` }} />
            </div>
            <div className="mt-1 flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400">
              <span className="min-w-0 truncate">{bulk.progress || "starting…"} — scores appear in the table as each model finishes</span>
              {bulk.kind === "custom" && bulk.id != null && (
                <button type="button" onClick={() => void api.collectorCancel(bulk.id as number)} className="ml-2 shrink-0 rounded border border-rose-500/60 px-2 py-0.5 font-semibold text-rose-500 hover:bg-rose-500/10">
                  stop after this model
                </button>
              )}
            </div>
          </div>
        ) : (
          <div className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">
            {basis ? (
              <>
                Scores in the table: <b className={upToDate ? "text-emerald-600 dark:text-emerald-300" : "text-amber-600 dark:text-amber-300"}>{basis.label}</b> · {timeAgo(basis.at)}
                {!basis.complete ? " · only some models" : ""}
                {upToDate ? " ✓ matches the working set" : " — differs from the working set, press Re-measure to update"}
              </>
            ) : (
              <>Every score, deep post-process and push uses this set.</>
            )}
          </div>
        )}
        {measureErr && <div className="mt-1 text-[11px] text-rose-500">{measureErr}</div>}
      </div>

      <div className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-[12px] dark:border-slate-700 dark:bg-ink-900">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <Moon className="h-4 w-4 text-indigo-400" />
          <b>Nightly re-measure</b>
          <span className="text-slate-500 dark:text-slate-400">on the working set, {winText}</span>
          <label className="flex items-center gap-1 font-semibold">
            <input type="checkbox" checked={Boolean(s.enabled)} onChange={(e) => void save({ enabled: e.target.checked })} /> on
          </label>
          <span className="text-slate-500 dark:text-slate-400">
            {running ? (
              <span className="inline-flex items-center gap-1 font-semibold text-sky-600 dark:text-sky-300">
                <Loader2 className="h-3 w-3 animate-spin" /> running {done}/{total}
                {st.current ? ` · ${st.current}` : ""}
              </span>
            ) : s.enabled ? (
              <>next run {nextText}{sch.in_window ? " (window is open now)" : ""}</>
            ) : (
              "paused"
            )}
          </span>
          {!running && st.date && (
            <span className="text-slate-400" title={st.finished_at ? new Date(st.finished_at * 1000).toLocaleString() : ""}>
              last: {st.date} · {done}/{total} done{failed ? ` · ${failed} failed` : ""}
              {st.complete ? "" : " (partial)"}
            </span>
          )}
          <button type="button" onClick={() => setOpen((v) => !v)} className="ml-auto text-sky-600 hover:underline dark:text-sky-300">
            {open ? "close" : "edit time"}
          </button>
        </div>
      {open && (
        <form
          className="mt-2 flex flex-wrap items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            const m = parseClock(start);
            if (m == null) return setErr("Start must look like 03:00");
            void save({ start_min: m, length_min: Math.max(15, Math.min(720, len)), utc_offset_min: Math.round(off * 60) });
            setOpen(false);
          }}
        >
          <label className="flex items-center gap-1">
            start
            <input value={start} onChange={(e) => setStart(e.target.value)} className="w-16 rounded border border-slate-300 bg-transparent px-1 py-0.5 dark:border-slate-600" />
          </label>
          <label className="flex items-center gap-1">
            window
            <input type="number" min={15} max={720} value={len} onChange={(e) => setLen(Number(e.target.value) || 60)} className="w-16 rounded border border-slate-300 bg-transparent px-1 py-0.5 dark:border-slate-600" />
            min
          </label>
          <label className="flex items-center gap-1">
            your UTC offset
            <input type="number" step={0.5} min={-12} max={14} value={off} onChange={(e) => setOff(Number(e.target.value))} className="w-16 rounded border border-slate-300 bg-transparent px-1 py-0.5 dark:border-slate-600" />
            h
          </label>
          <button type="submit" className="rounded-md bg-sky-600 px-3 py-1 font-bold text-white hover:bg-sky-500">Save</button>
          <span className="text-[10px] text-slate-400">
            Local time now: {sch.local_time}. If the window closes before everyone is done, the rest continues the next night (least recently measured first).
          </span>
        </form>
      )}
        {err && <div className="mt-1 text-[11px] text-rose-500">{err}</div>}
      </div>
    </div>
  );
}

function sColor(v: number | null | undefined, min: number) {
  if (v == null) return "text-slate-400";
  if (v >= 0.8) return "text-emerald-600 dark:text-emerald-300";
  if (v >= min) return "text-lime-600 dark:text-lime-300";
  return "text-rose-600 dark:text-rose-300";
}
function msColor(v: number | null | undefined, max: number) {
  if (v == null) return "text-slate-400";
  if (v <= 100) return "text-emerald-600 dark:text-emerald-300";
  if (v <= max) return "text-amber-600 dark:text-amber-300";
  return "text-rose-600 dark:text-rose-300";
}
const pct = (v: number | null | undefined) => (v == null ? "—" : v.toFixed(3));

/** Per-image score strip: one bar per validated image, oldest -> newest. */
function PerImageBars({ data, min }: { data: NonNullable<CollectorEvaluation["per_image"]>; min: number }) {
  const entries = useMemo(
    () => Object.entries(data).sort((a, b) => Number(a[0]) - Number(b[0])),
    [data],
  );
  if (!entries.length) return null;
  return (
    <div>
      <div className="flex h-14 items-end gap-px">
        {entries.map(([block, v]) => (
          <div
            key={block}
            className={cn("flex-1 rounded-t-sm", v.s >= 0.8 ? "bg-emerald-500" : v.s >= min ? "bg-lime-500" : v.s >= 0.5 ? "bg-amber-500" : "bg-rose-500")}
            style={{ height: `${Math.max(4, v.s * 100)}%` }}
            title={`block ${block} · s ${v.s.toFixed(3)} · m ${v.m.toFixed(2)} · f ${v.f.toFixed(2)} · ${v.n_pred} pred / ${v.n_gt} gt${v.ms != null ? ` · ${v.ms.toFixed(0)}ms` : ""}`}
          />
        ))}
      </div>
      <div className="mt-0.5 flex justify-between text-[9px] text-slate-400">
        <span>block {entries[0][0]}</span>
        <span>{entries.length} images · bar = per-image s</span>
        <span>block {entries[entries.length - 1][0]}</span>
      </div>
    </div>
  );
}

function useCommand() {
  const [cmd, setCmd] = useState<CollectorCommand | null>(null);
  const timer = useRef<number | null>(null);
  const stop = () => {
    if (timer.current != null) window.clearTimeout(timer.current);
    timer.current = null;
  };
  useEffect(() => stop, []);
  const run = useCallback(async (start: () => Promise<{ command_id: number }>, onDone?: (c: CollectorCommand) => void) => {
    stop();
    const { command_id } = await start();
    const poll = async () => {
      try {
        const c = await api.collectorCommand(command_id);
        setCmd(c);
        if (c.status === "done" || c.status === "error") {
          onDone?.(c);
          return;
        }
      } catch {
        /* keep polling */
      }
      timer.current = window.setTimeout(() => void poll(), 1500);
    };
    setCmd({ id: command_id, kind: "", status: "queued", progress: "", result: {} });
    void poll();
  }, []);
  const clear = useCallback(() => setCmd(null), []);
  return { cmd, run, clear };
}

function Drawer({
  sub,
  overview,
  onClose,
  onChanged,
  onChooseImages,
}: {
  sub: CollectorSubmission;
  overview: CollectorOverview;
  onClose: () => void;
  onChanged: () => void;
  onChooseImages: () => void;
}) {
  const [evals, setEvals] = useState<CollectorEvaluation[]>([]);
  const [detail, setDetail] = useState<CollectorEvaluation | null>(null);
  const [err, setErr] = useState("");
  const { cmd, run, clear } = useCommand();
  const min = overview.gate.min_s;
  const setLabel = overview.image_set?.label ?? "";

  const load = useCallback(async () => {
    const r = await api.collectorGet(sub.id);
    setEvals(r.evaluations);
    const first = r.evaluations[0];
    if (first) setDetail(await api.collectorEval(first.id));
    else setDetail(null);
  }, [sub.id]);

  useEffect(() => {
    clear();
    void load().catch((e) => setErr(String(e)));
  }, [load, clear]);

  const busy = cmd != null && cmd.status !== "done" && cmd.status !== "error";
  const [confirmDel, setConfirmDel] = useState(false);
  const [testing, setTesting] = useState(false);
  const doDelete = async () => {
    try {
      const r = await api.collectorDelete([sub.id]);
      if (r.skipped.length) return setErr(r.skipped[0].reason);
      onChanged();
      onClose();
    } catch (e) {
      setErr(String(e).replace(/^Error:\s*/, ""));
    }
  };
  const doRestore = async () => {
    try {
      await api.collectorRestore(sub.id);
      onChanged();
      onClose();
    } catch (e) {
      setErr(String(e).replace(/^Error:\s*/, ""));
    }
  };
  const trend = useMemo(
    () => evals.filter((e) => e.mean_s != null).slice(0, 30).reverse(),
    [evals],
  );
  const startRemeasure = () => {
    setErr("");
    void run(
      () => api.collectorRemeasure(sub.id, { mode: "working" }),
      (c) => {
        if (c.status === "error") setErr(String((c.result as { error?: string }).error ?? "failed"));
        void load();
        onChanged();
      },
    ).catch((e) => setErr(String(e)));
  };

  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/50" onClick={onClose}>
      <div
        className="h-full w-full max-w-[560px] overflow-y-auto border-l border-slate-200 bg-white p-4 shadow-2xl dark:border-slate-700 dark:bg-ink-900"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="truncate text-base font-extrabold text-slate-900 dark:text-white">{sub.model}</div>
            <div className="mt-0.5 flex flex-wrap items-center gap-2 text-[11px] text-slate-500 dark:text-slate-400">
              <StatusPill s={sub} />
              <span className="font-mono">{sub.revision.slice(0, 10)}</span>
              <a
                className="inline-flex items-center gap-0.5 text-sky-600 hover:underline dark:text-sky-300"
                href={`https://huggingface.co/${sub.model}/tree/${sub.revision}`}
                target="_blank"
                rel="noreferrer"
              >
                Hugging Face <ExternalLink className="h-3 w-3" />
              </a>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-1">
            {sub.status === "passed" && (
              <button
                type="button"
                onClick={() => setTesting(true)}
                title="Drop an image and see what this model detects"
                className="inline-flex items-center gap-1 rounded-md bg-gradient-to-r from-sky-600 to-violet-600 px-2.5 py-1 text-[11px] font-bold text-white shadow hover:brightness-110"
              >
                <FlaskConical className="h-3.5 w-3.5" /> Test
              </button>
            )}
            {sub.status === "deleted" ? (
              <button type="button" onClick={() => void doRestore()} className="inline-flex items-center gap-1 rounded-md border border-emerald-500 px-2 py-1 text-[11px] font-bold text-emerald-600 hover:bg-emerald-500/10 dark:text-emerald-300">
                <Undo2 className="h-3.5 w-3.5" /> Restore
              </button>
            ) : confirmDel ? (
              <>
                <button type="button" onClick={() => void doDelete()} className="rounded-md bg-rose-600 px-2 py-1 text-[11px] font-bold text-white hover:bg-rose-500">
                  Delete for sure
                </button>
                <button type="button" onClick={() => setConfirmDel(false)} className="rounded-md border border-slate-300 px-2 py-1 text-[11px] dark:border-slate-600">
                  no
                </button>
              </>
            ) : (
              <button type="button" onClick={() => setConfirmDel(true)} title="Delete this collected model (can be restored)" className="rounded p-1.5 text-slate-500 hover:bg-rose-500/10 hover:text-rose-500">
                <Trash2 className="h-4 w-4" />
              </button>
            )}
            <button type="button" onClick={onClose} className="rounded p-1 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        <p className="mt-2 text-[12px] text-slate-600 dark:text-slate-300">{sub.detail}</p>
        <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
          <dt className="text-slate-400">Committed</dt>
          <dd>{sub.first_block ? `block ${sub.first_block} · ${timeAgo(sub.commit_ts)}` : "added manually"}</dd>
          <dt className="text-slate-400">Hotkey</dt>
          <dd className="truncate font-mono" title={sub.hotkeys.join(", ")}>{sub.hotkeys[0]?.slice(0, 16) ?? "—"}{sub.hotkeys.length > 1 ? ` +${sub.hotkeys.length - 1}` : ""}</dd>
          <dt className="text-slate-400">weights.onnx</dt>
          <dd>{sub.onnx_size != null ? `${(sub.onnx_size / 1024 / 1024).toFixed(2)} MB` : "—"}</dd>
          <dt className="text-slate-400">sha256</dt>
          <dd className="truncate font-mono" title={sub.onnx_sha ?? ""}>{sub.onnx_sha?.slice(0, 16) ?? "—"}</dd>
        </dl>

        {detail && (
          <div className="mt-4 rounded-lg border border-slate-200 p-3 dark:border-slate-700">
            <div className="mb-1 flex items-center justify-between text-[11px] font-bold uppercase tracking-wide text-slate-400">
              <span>Latest measurement · {detail.kind}</span>
              <span>{timeAgo(detail.created_at)}</span>
            </div>
            <div className="grid grid-cols-4 gap-2 text-center">
              {[
                ["s (total)", pct(detail.mean_s), sColor(detail.mean_s, min)],
                ["m (map50)", pct(detail.mean_m), "text-slate-700 dark:text-slate-200"],
                ["f (fp)", pct(detail.mean_f), "text-slate-700 dark:text-slate-200"],
                ["p95 ms", detail.p95_ms != null ? detail.p95_ms.toFixed(0) : "—", msColor(detail.p95_ms, overview.gate.max_p95_ms)],
              ].map(([k, v, c]) => (
                <div key={k} className="rounded bg-slate-50 py-1.5 dark:bg-slate-800/60">
                  <div className={cn("text-base font-extrabold tabular-nums", c)}>{v}</div>
                  <div className="text-[9px] uppercase text-slate-400">{k}</div>
                </div>
              ))}
            </div>
            <div className="mt-1.5 text-[11px] text-slate-500 dark:text-slate-400">
              {detail.n_images} validated images · blocks {detail.blocks_min}–{detail.blocks_max}
              {detail.baseline_p != null && detail.mean_s != null && (
                <span className={cn("ml-1 font-semibold", detail.mean_s >= detail.baseline_p ? "text-emerald-500" : "text-slate-500")}>
                  · {detail.mean_s - detail.baseline_p >= 0 ? "+" : ""}
                  {(detail.mean_s - detail.baseline_p).toFixed(3)} vs reference winner ({detail.baseline_p.toFixed(3)})
                </span>
              )}
            </div>
            <div className="mt-2">{detail.per_image && <PerImageBars data={detail.per_image} min={min} />}</div>
            <p className="mt-1.5 text-[11px] text-slate-500 dark:text-slate-400">{detail.reason}</p>
          </div>
        )}

        <div className="mt-4 rounded-lg border border-slate-200 p-3 dark:border-slate-700">
          <div className="mb-1 text-[11px] font-bold uppercase tracking-wide text-slate-400">Measure again</div>
          <div className="text-[12px] text-slate-600 dark:text-slate-300">
            on <b>{setLabel || "the working image set"}</b>{" "}
            <button type="button" onClick={onChooseImages} className="text-[11px] text-sky-600 hover:underline dark:text-sky-300">
              change images
            </button>
          </div>
          <div className="mt-2 flex items-center gap-2">
            <button
              type="button"
              onClick={startRemeasure}
              disabled={busy}
              className="inline-flex items-center gap-1 rounded-md bg-sky-600 px-3 py-1.5 text-[12px] font-bold text-white hover:bg-sky-500 disabled:opacity-50"
            >
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />} Measure
            </button>
            {sub.status === "expired" || sub.status === "error" ? (
              <button
                type="button"
                onClick={() => void run(() => api.collectorRetry(sub.id), () => onChanged())}
                className="rounded-md border border-slate-300 px-3 py-1.5 text-[12px] font-semibold dark:border-slate-600"
              >
                Retry download
              </button>
            ) : null}
            {cmd && (
              <span className="text-[11px] text-slate-500">
                {cmd.status === "done"
                  ? `done · s=${pct((cmd.result as { mean_s?: number }).mean_s)}`
                  : cmd.status === "error"
                    ? "failed"
                    : cmd.status === "running"
                      ? "running in sandbox…"
                      : "queued…"}
              </span>
            )}
          </div>
          {err && <div className="mt-1 text-[11px] text-rose-500">{err}</div>}
        </div>

        {sub.status === "passed" && (
          <WorkshopPanel
            sub={sub}
            setLabel={setLabel}
            minS={min}
            maxP95={overview.gate.max_p95_ms}
            onChooseImages={onChooseImages}
            onChanged={onChanged}
          />
        )}

        {trend.length > 1 && (
          <div className="mt-4">
            <div className="mb-1 text-[11px] font-bold uppercase tracking-wide text-slate-400">Score over time</div>
            <div className="flex h-12 items-end gap-0.5">
              {trend.map((e) => (
                <div
                  key={e.id}
                  className={cn("flex-1 rounded-t-sm", e.kind === "daily" ? "bg-indigo-400" : "bg-sky-500", !e.passed && "opacity-50")}
                  style={{ height: `${Math.max(6, (e.mean_s ?? 0) * 100)}%` }}
                  title={`${new Date(e.created_at * 1000).toLocaleString()} · ${e.kind} · s ${pct(e.mean_s)} · ${e.n_images} img`}
                />
              ))}
            </div>
            <div className="mt-0.5 text-[9px] text-slate-400">indigo = nightly re-measure, blue = manual / first evaluation</div>
          </div>
        )}

        {evals.length > 1 && (
          <div className="mt-4">
            <div className="mb-1 text-[11px] font-bold uppercase tracking-wide text-slate-400">History</div>
            <table className="w-full text-[11px]">
              <tbody>
                {evals.map((e) => (
                  <tr key={e.id} className="border-t border-slate-100 dark:border-slate-800">
                    <td className="py-1 text-slate-400">{timeAgo(e.created_at)}</td>
                    <td className="py-1">{e.kind}</td>
                    <td className="py-1 tabular-nums">{e.n_images} img</td>
                    <td className={cn("py-1 font-bold tabular-nums", sColor(e.mean_s, min))}>{pct(e.mean_s)}</td>
                    <td className="py-1 tabular-nums">{e.p95_ms != null ? `${e.p95_ms.toFixed(0)}ms` : "—"}</td>
                    <td className="py-1">{e.passed ? <Check className="h-3 w-3 text-emerald-500" /> : <X className="h-3 w-3 text-rose-500" />}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
      {testing && <SingleModelTest sub={sub} onClose={() => setTesting(false)} />}
    </div>
  );
}

export default function CollectedPage() {
  useSiteBackdrop("vivid");
  const [overview, setOverview] = useState<CollectorOverview | null>(null);
  const [rows, setRows] = useState<CollectorSubmission[]>([]);
  const [filter, setFilter] = useState("passed");
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<"score" | "new" | "speed">("score");
  const [openId, setOpenId] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [addOpen, setAddOpen] = useState(false);
  const [addModel, setAddModel] = useState("");
  const [sel, setSel] = useState<Set<number>>(new Set());
  const [bulkErr, setBulkErr] = useState("");
  const [confirmBulk, setConfirmBulk] = useState(false);
  const [tryFiles, setTryFiles] = useState<File[] | null>(null);
  const [pickOpen, setPickOpen] = useState(false);
  const [mergeJobId, setMergeJobId] = useState<string | null>(null);
  const [mergeBusy, setMergeBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const { cmd, run } = useCommand();

  // Drop images anywhere on the page -> choose models -> see detections.
  useEffect(() => {
    let depth = 0;
    const hasFiles = (e: DragEvent) =>
      Array.from(e.dataTransfer?.types ?? []).includes("Files") && !document.querySelector("[data-single-test]");
    const enter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth += 1;
      setDragging(true);
    };
    const leave = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0) setDragging(false);
    };
    const over = (e: DragEvent) => {
      if (hasFiles(e)) e.preventDefault();
    };
    const drop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      setDragging(false);
      const imgs = Array.from(e.dataTransfer?.files ?? []).filter((f) => f.type.startsWith("image/"));
      if (imgs.length) setTryFiles(imgs.slice(0, 4));
    };
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragleave", leave);
    window.addEventListener("dragover", over);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("dragover", over);
      window.removeEventListener("drop", drop);
    };
  }, []);
  const tryInput = useRef<HTMLInputElement>(null);
  const imp = useCommand();
  const [impMsg, setImpMsg] = useState("");
  const importing = imp.cmd != null && imp.cmd.status !== "done" && imp.cmd.status !== "error";
  const startImport = () => {
    setImpMsg("");
    void imp
      .run(
        () => api.collectorImportMine(),
        (c) => {
          const r = c.result as { added?: string[]; already_known?: number; not_miner?: string[]; errors?: string[]; error?: string };
          if (c.status === "error") setImpMsg(r.error ?? "failed");
          else
            setImpMsg(
              `${r.added?.length ?? 0} of your repos added and being measured · ${r.already_known ?? 0} already here · ${r.not_miner?.length ?? 0} skipped (not a miner repo)` +
                (r.errors?.length ? ` · ${r.errors.length} could not be read (${r.errors[0]})` : ""),
            );
          void reload();
        },
      )
      .catch((e) => setImpMsg(String(e).replace(/^Error:\s*/, "")));
  };

  const lastScores = useRef<Map<number, number | null>>(new Map());
  const [flash, setFlash] = useState<Set<number>>(new Set());
  const reload = useCallback(async () => {
    try {
      const [o, l] = await Promise.all([api.collectorOverview(), api.collectorList("", "")]);
      setOverview(o);
      // models whose score just changed (live bulk re-measure) flash for a moment
      const changed = new Set<number>();
      l.submissions.forEach((r) => {
        const prev = lastScores.current.get(r.id);
        if (lastScores.current.size > 0 && prev !== undefined && prev !== (r.mean_s ?? null)) changed.add(r.id);
        lastScores.current.set(r.id, r.mean_s ?? null);
      });
      if (changed.size) {
        setFlash((f) => new Set([...f, ...changed]));
        window.setTimeout(() => setFlash((f) => new Set([...f].filter((id) => !changed.has(id)))), 2500);
      }
      setRows(l.submissions);
      setError("");
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const bulkRunning = Boolean(overview?.bulk);
  const activeCount = overview ? (overview.counts.evaluating ?? 0) + (overview.counts.queued_eval ?? 0) + (overview.counts.waiting_public ?? 0) : 0;
  useEffect(() => {
    void reload();
    let t: number | null = null;
    const tick = () => {
      if (!document.hidden) void reload();
      t = window.setTimeout(tick, bulkRunning ? 2000 : activeCount > 0 ? 5000 : 15000);
    };
    t = window.setTimeout(tick, bulkRunning ? 2000 : 5000);
    return () => {
      if (t != null) window.clearTimeout(t);
    };
  }, [reload, activeCount, bulkRunning]);

  const visible = useMemo(() => {
    const f = FILTERS.find((x) => x.id === filter) ?? FILTERS[0];
    const needle = q.trim().toLowerCase();
    const list = rows.filter(
      (r) =>
        (f.statuses.length === 0 ? r.status !== "deleted" : f.statuses.includes(r.status)) &&
        (!needle || `${r.model} ${r.revision} ${r.hotkeys.join(" ")}`.toLowerCase().includes(needle)),
    );
    const num = (v: number | null | undefined, dflt: number) => (v == null ? dflt : v);
    list.sort((a, b) => {
      if (sort === "score") return num(b.mean_s ?? b.best_s, -1) - num(a.mean_s ?? a.best_s, -1) || b.commit_ts - a.commit_ts;
      if (sort === "speed") return num(a.eval_p95, 1e9) - num(b.eval_p95, 1e9);
      return b.commit_ts - a.commit_ts;
    });
    return list;
  }, [rows, filter, q, sort]);

  const opened = rows.find((r) => r.id === openId) ?? null;
  const counts = overview?.counts ?? {};
  const countFor = (statuses: Status[]) =>
    statuses.length === 0 ? Math.max(0, (overview?.total ?? 0) - (counts.deleted ?? 0)) : statuses.reduce((n, s) => n + (counts[s] ?? 0), 0);
  const passedRows = useMemo(() => rows.filter((r) => r.status === "passed"), [rows]);
  const selectable = visible.filter((r) => r.status !== "deleted" && r.status !== "evaluating");
  const selCount = [...sel].filter((id) => selectable.some((r) => r.id === id)).length;
  const allSel = selectable.length > 0 && selCount === selectable.length;
  useEffect(() => {
    setSel(new Set());
    setConfirmBulk(false);
  }, [filter]);
  const toggleSel = (id: number) =>
    setSel((p) => {
      const n = new Set(p);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  const deleteSelected = async () => {
    setBulkErr("");
    try {
      const ids = [...sel].filter((id) => selectable.some((r) => r.id === id));
      const r = await api.collectorDelete(ids);
      if (r.skipped.length) setBulkErr(`${r.skipped.length} skipped: ${r.skipped[0].reason}`);
      setSel(new Set());
      setConfirmBulk(false);
      void reload();
    } catch (e) {
      setBulkErr(String(e).replace(/^Error:\s*/, ""));
    }
  };
  const passedSel = useMemo(() => [...sel].filter((id) => rows.find((r) => r.id === id)?.status === "passed"), [sel, rows]);
  const startMerge = async () => {
    setBulkErr("");
    setMergeBusy(true);
    try {
      const r = await api.manualMergeFromCollected(passedSel);
      setMergeJobId(r.job.id);
    } catch (e) {
      setBulkErr(String(e).replace(/^Error:\s*/, ""));
    } finally {
      setMergeBusy(false);
    }
  };
  const restoreOne = async (id: number) => {
    try {
      await api.collectorRestore(id);
      void reload();
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/, ""));
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-transparent dark:text-slate-100">
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/95 px-4 py-2.5 backdrop-blur dark:border-slate-800 dark:bg-ink-900/95">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-3">
          <Link to="/" className="inline-flex items-center gap-1 rounded-md border border-slate-300 px-2 py-1 text-[12px] font-semibold hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-800">
            <ArrowLeft className="h-3.5 w-3.5" /> Labeler
          </Link>
          <div>
            <h1 className="text-base font-extrabold leading-tight">Collected solutions · Beverage</h1>
            <div className="text-[11px] text-slate-500 dark:text-slate-400">
              Miner commitments on SN44 → public repo → sandboxed score on your validated labels
            </div>
          </div>
          <div className="ml-auto flex flex-wrap items-center gap-2 text-[11px]">
            {overview && (
              <span
                className={cn(
                  "inline-flex items-center gap-1 rounded-full px-2 py-0.5 font-bold",
                  overview.service_alive ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-300" : "bg-rose-500/15 text-rose-600 dark:text-rose-300",
                )}
                title={overview.service_alive ? "Collector service is running" : "Collector service is not running (systemctl status score-collector)"}
              >
                <span className={cn("h-1.5 w-1.5 rounded-full", overview.service_alive ? "bg-emerald-500" : "bg-rose-500")} />
                {overview.service_alive ? "collector running" : "collector stopped"}
              </span>
            )}
            {overview?.chain_polled_at && (
              <span className="inline-flex items-center gap-1 text-slate-500" title={`chain head ${overview.chain_head}`}>
                <Clock className="h-3 w-3" /> chain checked {timeAgo(overview.chain_polled_at)}
              </span>
            )}
            <button
              type="button"
              onClick={() => void run(() => api.collectorRefresh(), () => void reload())}
              className="inline-flex items-center gap-1 rounded-md border border-slate-300 px-2 py-1 font-semibold hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-800"
            >
              <RefreshCw className={cn("h-3 w-3", cmd && cmd.status !== "done" && cmd.status !== "error" && "animate-spin")} /> Check chain now
            </button>
            <KeysButton className="px-2" />
            <button
              type="button"
              onClick={startImport}
              disabled={importing}
              className="inline-flex items-center gap-1 rounded-md border border-slate-300 px-2 py-1 font-semibold hover:bg-slate-100 disabled:opacity-50 dark:border-slate-600 dark:hover:bg-slate-800"
              title="Add the miner repos (private ones too) of your saved Hugging Face accounts and measure them"
            >
              {importing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Plus className="h-3 w-3" />}
              {importing ? imp.cmd?.progress || "importing…" : "Import my models"}
            </button>
            <button
              type="button"
              onClick={() => tryInput.current?.click()}
              className="inline-flex items-center gap-1 rounded-md bg-gradient-to-r from-sky-600 to-violet-600 px-2.5 py-1 font-bold text-white shadow hover:brightness-110"
              title="Drag images anywhere onto this page, or click to choose"
            >
              <ImageUp className="h-3.5 w-3.5" /> Try images
            </button>
            <input
              ref={tryInput}
              type="file"
              accept="image/*"
              multiple
              hidden
              onChange={(e) => {
                const f = Array.from(e.target.files ?? []).filter((x) => x.type.startsWith("image/"));
                if (f.length) setTryFiles(f.slice(0, 4));
                e.target.value = "";
              }}
            />
            <button
              type="button"
              onClick={() => setAddOpen((v) => !v)}
              className="inline-flex items-center gap-1 rounded-md border border-slate-300 px-2 py-1 font-semibold hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-800"
            >
              <Plus className="h-3 w-3" /> Add repo
            </button>
          </div>
        </div>
        {addOpen && (
          <form
            className="mx-auto mt-2 flex max-w-[1400px] items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              if (!addModel.trim()) return;
              void run(() => api.collectorAdd(addModel.trim()), (c) => {
                if (c.status === "error") setError(String((c.result as { error?: string }).error));
                else {
                  const there = (c.result as { already_there?: string | null }).already_there;
                  setImpMsg(
                    there
                      ? `${addModel.trim()} is already in the list (${STATUS_META[there as Status]?.label ?? there}) at this exact commit.`
                      : `${addModel.trim()} was added and is being measured.`,
                  );
                  setAddModel("");
                  setAddOpen(false);
                  void reload();
                }
              });
            }}
          >
            <input
              value={addModel}
              onChange={(e) => setAddModel(e.target.value)}
              placeholder="owner/repo (latest revision) — a public Hugging Face model"
              className="h-8 w-full max-w-md rounded-md border border-slate-300 bg-transparent px-2 text-[12px] dark:border-slate-600"
            />
            <button type="submit" className="h-8 rounded-md bg-sky-600 px-3 text-[12px] font-bold text-white hover:bg-sky-500">
              Evaluate
            </button>
          </form>
        )}
      </header>

      <main className="mx-auto max-w-[1400px] px-4 py-4">
        {error && (
          <div className="mb-3 flex items-center gap-2 rounded-md bg-rose-500/10 px-3 py-2 text-[12px] text-rose-600 dark:text-rose-300">
            <AlertTriangle className="h-4 w-4" /> {error}
          </div>
        )}

        {overview && (
          <div className="mb-3 rounded-lg border border-slate-200 bg-white px-3 py-2 text-[12px] text-slate-600 dark:border-slate-700 dark:bg-ink-900 dark:text-slate-300">
            <b>Gate:</b> mean <b>s ≥ {overview.gate.min_s.toFixed(2)}</b> on the latest {overview.gate.n_images} validated images · p95 ≤{" "}
            {overview.gate.max_p95_ms} ms · ONNX ≤ {overview.gate.max_onnx_mb} MB · repo must be public within {overview.gate.public_wait_min} min of the commit ·
            looked back {overview.gate.backfill_days} days.{" "}
            {overview.best && (
              <span>
                Best so far: <b>{overview.best.model}</b> s={overview.best.best_s.toFixed(3)} ({overview.best.p95_ms.toFixed(0)} ms).
              </span>
            )}
          </div>
        )}

        {impMsg && (
          <div className="mb-3 flex items-center gap-2 rounded-md bg-sky-500/10 px-3 py-2 text-[12px] text-sky-700 dark:text-sky-300">
            {impMsg}
            <button type="button" onClick={() => setImpMsg("")} className="ml-auto"><X className="h-3.5 w-3.5" /></button>
          </div>
        )}

        <SchedulePanel
          onChanged={() => void reload()}
          basis={overview?.scores_basis ?? null}
          bulk={overview?.bulk ?? null}
          imageSet={overview?.image_set ?? {}}
          nPassed={counts.passed ?? 0}
          onPick={() => setPickOpen(true)}
        />

        <div className="mb-3 flex flex-wrap items-center gap-2">
          {FILTERS.map((f) => (
            <button
              key={f.id}
              type="button"
              onClick={() => setFilter(f.id)}
              className={cn(
                "rounded-full border px-3 py-1 text-[12px] font-semibold",
                filter === f.id
                  ? "border-sky-500 bg-sky-500 text-white"
                  : "border-slate-300 text-slate-600 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-800",
              )}
            >
              {f.label} <span className="ml-1 tabular-nums opacity-80">{countFor(f.statuses)}</span>
            </button>
          ))}
          <div className="relative ml-auto">
            <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search model / hotkey…"
              className="h-8 w-56 rounded-md border border-slate-300 bg-transparent pl-7 pr-2 text-[12px] dark:border-slate-600"
            />
          </div>
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value as typeof sort)}
            className="h-8 rounded-md border border-slate-300 bg-transparent px-2 text-[12px] dark:border-slate-600"
          >
            <option value="score">Sort: best score</option>
            <option value="speed">Sort: fastest</option>
            <option value="new">Sort: newest commit</option>
          </select>
        </div>

        {(selCount > 0 || filter === "rejected" || filter === "other") && (
          <div className="mb-2 flex flex-wrap items-center gap-2 rounded-md bg-slate-100 px-3 py-1.5 text-[12px] dark:bg-slate-800/60">
            {selCount > 0 ? (
              <>
                <b>{selCount} selected</b>
                {confirmBulk ? (
                  <>
                    <button type="button" onClick={() => void deleteSelected()} className="rounded-md bg-rose-600 px-2.5 py-1 font-bold text-white hover:bg-rose-500">
                      Delete {selCount} for sure
                    </button>
                    <button type="button" onClick={() => setConfirmBulk(false)} className="rounded-md border border-slate-300 px-2 py-1 dark:border-slate-600">cancel</button>
                  </>
                ) : (
                  <button type="button" onClick={() => setConfirmBulk(true)} className="inline-flex items-center gap-1 rounded-md border border-rose-400 px-2.5 py-1 font-bold text-rose-600 hover:bg-rose-500/10 dark:text-rose-300">
                    <Trash2 className="h-3.5 w-3.5" /> Delete selected
                  </button>
                )}
                <button
                  type="button"
                  disabled={passedSel.length < 2 || passedSel.length > 12 || mergeBusy}
                  onClick={() => void startMerge()}
                  title={
                    passedSel.length < 2
                      ? "Tick at least 2 passed models to merge"
                      : passedSel.length > 12
                        ? "At most 12 models at a time"
                        : `Manual merge of the ${passedSel.length} ticked passed models, scored on the working image set`
                  }
                  className="inline-flex items-center gap-1 rounded-md bg-violet-600 px-2.5 py-1 font-bold text-white hover:bg-violet-500 disabled:opacity-40"
                >
                  {mergeBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <GitMerge className="h-3.5 w-3.5" />} Merge {passedSel.length} models…
                </button>
                <button type="button" onClick={() => setSel(new Set())} className="text-slate-500 hover:underline">clear</button>
              </>
            ) : (
              <button type="button" onClick={() => setSel(new Set(selectable.map((r) => r.id)))} className="text-sky-600 hover:underline dark:text-sky-300">
                select all {selectable.length} in this list
              </button>
            )}
            <span className="text-[10px] text-slate-400">Deleted models stay in the “Deleted” tab and can be restored. The chain poller will not add them again.</span>
            {bulkErr && <span className="text-rose-500">{bulkErr}</span>}
          </div>
        )}

        <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white dark:border-slate-700 dark:bg-ink-900">
          <table className="w-full min-w-[1020px] text-left text-[12px]">
            <thead className="bg-slate-50 text-[10px] uppercase tracking-wide text-slate-400 dark:bg-slate-800/50">
              <tr>
                <th className="w-8 py-2 pl-3" onClick={(e) => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    checked={allSel}
                    onChange={() => setSel(allSel ? new Set() : new Set(selectable.map((r) => r.id)))}
                    title="Select all in this list"
                  />
                </th>
                <th className="px-3 py-2">Model</th>
                <th className="px-2 py-2">Status</th>
                <th className="px-2 py-2 text-right" title="mean total_weighted on the validated image set">s</th>
                <th className="px-2 py-2 text-right">m</th>
                <th className="px-2 py-2 text-right">f</th>
                <th className="px-2 py-2 text-right" title="s minus the reference winner, scored by the same sandbox and formula on the same images">vs winner</th>
                <th className="px-2 py-2 text-right">p95 ms</th>
                <th className="px-2 py-2 text-right">ONNX</th>
                <th className="px-2 py-2">Committed</th>
                <th className="px-2 py-2" />
              </tr>
            </thead>
            <tbody>
              {visible.map((r) => {
                const delta = r.mean_s != null && r.baseline_p != null ? r.mean_s - r.baseline_p : null;
                return (
                  <tr
                    key={r.id}
                    onClick={() => setOpenId(r.id)}
                    className={cn(
                      "cursor-pointer border-t border-slate-100 hover:bg-sky-50/60 dark:border-slate-800 dark:hover:bg-sky-950/20",
                      overview?.bulk?.current_id === r.id && "bg-sky-500/10 ring-1 ring-inset ring-sky-500/40",
                    )}
                  >
                    <td className="w-8 py-2 pl-3" onClick={(e) => e.stopPropagation()}>
                      {r.status !== "deleted" && r.status !== "evaluating" && (
                        <input type="checkbox" checked={sel.has(r.id)} onChange={() => toggleSel(r.id)} />
                      )}
                    </td>
                    <td className="max-w-[260px] px-3 py-2">
                      <div className="truncate font-bold" title={r.model}>{r.model}</div>
                      <div className="truncate font-mono text-[10px] text-slate-400">
                        {r.revision.slice(0, 8)} · {r.hotkeys[0]?.slice(0, 8) ?? "manual"}
                        {r.n_evals > 1 ? ` · ${r.n_evals} runs` : ""}
                      </div>
                    </td>
                    <td className="px-2 py-2">
                      <StatusPill s={r} />
                      {overview?.bulk?.current_id === r.id && (
                        <span className="ml-1.5 inline-flex items-center gap-1 text-[10px] font-bold text-sky-500">
                          <Loader2 className="h-3 w-3 animate-spin" /> measuring
                        </span>
                      )}
                    </td>
                    <td className={cn("px-2 py-2 text-right font-extrabold tabular-nums", sColor(r.mean_s ?? r.best_s, overview?.gate.min_s ?? 0.7), flash.has(r.id) && "score-flash")}>{pct(r.mean_s ?? r.best_s)}</td>
                    <td className="px-2 py-2 text-right tabular-nums text-slate-500">{pct(r.mean_m)}</td>
                    <td className="px-2 py-2 text-right tabular-nums text-slate-500">{pct(r.mean_f)}</td>
                    <td className={cn("px-2 py-2 text-right tabular-nums", delta == null ? "text-slate-400" : delta >= 0 ? "text-emerald-500" : "text-slate-500")}>
                      {delta == null ? "—" : `${delta >= 0 ? "+" : ""}${delta.toFixed(3)}`}
                    </td>
                    <td
                      className={cn("px-2 py-2 text-right tabular-nums", msColor(r.eval_p95, overview?.gate.max_p95_ms ?? 150))}
                      title={
                        r.p95_raw != null && r.ref_p95 != null && Math.abs(r.p95_raw - (r.eval_p95 ?? 0)) > 1
                          ? `Speed on an idle server: ${r.eval_p95?.toFixed(0)} ms. Measured ${r.p95_raw.toFixed(0)} ms while the box was busy (the reference winner took ${r.ref_p95.toFixed(0)} ms in the same run, ${(overview?.gate.ref_idle_ms ?? 89)} ms when idle).`
                          : "Typical speed of this model (median of its timed runs, load-normalised)"
                      }
                    >
                      {r.eval_p95 != null ? r.eval_p95.toFixed(0) : "—"}
                    </td>
                    <td className="px-2 py-2 text-right tabular-nums text-slate-500">{r.onnx_size != null ? `${(r.onnx_size / 1024 / 1024).toFixed(1)} MB` : "—"}</td>
                    <td className="whitespace-nowrap px-2 py-2 text-slate-500">{r.first_block ? timeAgo(r.commit_ts) : "manual"}</td>
                    <td className="whitespace-nowrap px-2 py-2 text-right text-slate-400" onClick={(e) => e.stopPropagation()}>
                      {r.status === "deleted" && (
                        <button type="button" onClick={() => void restoreOne(r.id)} title="Restore (downloads and scores it again)" className="mr-2 inline-flex items-center gap-0.5 text-[11px] font-bold text-emerald-600 hover:underline dark:text-emerald-300">
                          <Undo2 className="h-3.5 w-3.5" /> restore
                        </button>
                      )}
                      <a
                        href={`https://huggingface.co/${r.model}/tree/${r.revision}`}
                        target="_blank"
                        rel="noreferrer"
                        title="Open on Hugging Face"
                        className="inline-flex hover:text-sky-500"
                      >
                        <ExternalLink className="h-3.5 w-3.5" />
                      </a>
                    </td>
                  </tr>
                );
              })}
              {visible.length === 0 && (
                <tr>
                  <td colSpan={11} className="px-3 py-10 text-center text-slate-400">
                    {overview ? "Nothing here yet." : "Loading…"}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </main>

      {dragging && (
        <div className="pointer-events-none fixed inset-0 z-[70] flex items-center justify-center bg-slate-950/70 backdrop-blur-sm">
          <div className="drop-pulse flex flex-col items-center gap-2 rounded-3xl border-4 border-dashed border-sky-400 bg-slate-900/80 px-16 py-12 text-white">
            <ImageUp className="h-14 w-14 text-sky-300" />
            <div className="text-xl font-extrabold">Drop images to test the collected models</div>
            <div className="text-sm text-slate-300">up to 4 images · you pick the models next</div>
          </div>
        </div>
      )}
      {mergeJobId && <ManualMergeModal open attachJobId={mergeJobId} challenges={[]} onClose={() => setMergeJobId(null)} />}
      {pickOpen && (
        <ImageSetModal
          nPassed={counts.passed ?? 0}
          onClose={() => setPickOpen(false)}
          onSaved={(measureNow) => {
            setPickOpen(false);
            if (measureNow) void api.collectorRemeasureSet([]).catch((e) => setError(String(e).replace(/^Error:\s*/, "")));
            void reload();
          }}
        />
      )}
      {tryFiles && (
        <TryModelsModal initialFiles={tryFiles} models={passedRows} onClose={() => setTryFiles(null)} />
      )}
      {opened && overview && (
        <Drawer sub={opened} overview={overview} onClose={() => setOpenId(null)} onChanged={() => void reload()} onChooseImages={() => setPickOpen(true)} />
      )}
    </div>
  );
}
