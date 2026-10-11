import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import {
  AlertTriangle,
  Ban,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Copy,
  Crosshair,
  Download,
  Eye,
  EyeOff,
  ImageOff,
  LayoutGrid,
  Rows3,
  Send,
  Tag,
  Target,
  Trash2,
  Wind,
  X,
} from "lucide-react";
import { cn } from "../lib/utils";
import { img, studio } from "./api";
import { useStudio } from "./store";
import type { GenItem, ImagePlan, LabelBox, Run, RunInsight, SourceFault, SourceInsight } from "./types";
import { Btn, Chip, CLASS_COLORS, Confirm, Empty, errText, GenBadge, Progress } from "./ui";

type Filter = "all" | "near" | "drift" | "selected";
type Layout = "source" | "pairs";

const STATUS_COLOR: Record<string, string> = { missed: "#ef4444", wrong_class: "#d946ef", false_positive: "#eab308" };
const STATUS_TEXT: Record<string, string> = { missed: "Missed", wrong_class: "Wrong class", false_positive: "False alarm" };

function runTime(name: string): string {
  const m = name.match(/_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_/);
  if (!m) return "";
  return new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6])).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

const sourceOf = (it: GenItem) => it.source_image || `${it.filename.split("__")[0]}.png`;
const blockOf = (fn: string) => fn.replace(/^block_/, "").replace(/\.png$/, "");

function scoreTone(v: number | null | undefined) {
  if (v == null) return "text-slate-400";
  return v < 0.7 ? "text-rose-500" : v < 0.85 ? "text-amber-500" : "text-emerald-500";
}

/** Label boxes in normalised (0..1) coordinates over an image. */
function BoxLayer({ boxes, thin }: { boxes: LabelBox[]; thin?: boolean }) {
  return (
    <svg viewBox="0 0 1 1" preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full">
      {boxes.map((b, i) => (
        <rect
          key={i}
          x={b.x1}
          y={b.y1}
          width={Math.max(0, b.x2 - b.x1)}
          height={Math.max(0, b.y2 - b.y1)}
          fill="none"
          stroke={CLASS_COLORS[b.name] || "#fff"}
          strokeWidth={thin ? 1.5 : 2}
          vectorEffect="non-scaling-stroke"
        />
      ))}
    </svg>
  );
}

/** The base model's mistakes on a source (pixel boxes on the 1024² frame), numbered like the fault list. */
function FaultLayer({ faults, size = 1024 }: { faults: SourceFault[]; size?: number }) {
  return (
    <svg viewBox={`0 0 ${size} ${size}`} preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full">
      {faults.map((f, i) =>
        f.xyxy ? (
          <g key={i}>
            <rect
              x={f.xyxy[0] - 4}
              y={f.xyxy[1] - 4}
              width={f.xyxy[2] - f.xyxy[0] + 8}
              height={f.xyxy[3] - f.xyxy[1] + 8}
              fill="none"
              stroke={STATUS_COLOR[f.status] || "#fff"}
              strokeWidth={2.5}
              vectorEffect="non-scaling-stroke"
            />
            <text x={f.xyxy[0] - 4} y={Math.max(14, f.xyxy[1] - 8)} fill={STATUS_COLOR[f.status] || "#fff"} fontSize={22} fontWeight={800}>
              {i + 1}
            </text>
          </g>
        ) : null,
      )}
    </svg>
  );
}

function countsOf(boxes: LabelBox[] | undefined) {
  const c: Record<string, number> = {};
  for (const b of boxes || []) c[b.name] = (c[b.name] || 0) + 1;
  return c;
}

function BoxCounts({ boxes }: { boxes: LabelBox[] | undefined }) {
  const c = countsOf(boxes);
  return (
    <span className="flex gap-1 text-[10px] font-bold tabular-nums">
      {(["cup", "can", "bottle"] as const).map((k) => (c[k] ? <span key={k} style={{ color: CLASS_COLORS[k] }}>{c[k]} {k}</span> : null))}
      {!boxes?.length && <span className="text-slate-400">no boxes</span>}
    </span>
  );
}

export default function Results() {
  const st = useStudio();
  const run = st.activeRun;
  const info = st.runs.find((r) => r.name === run) || null;
  const [items, setItems] = useState<GenItem[]>([]);
  const [insight, setInsight] = useState<RunInsight | null>(null);
  const [loading, setLoading] = useState(false);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState<Filter>("all");
  const [boxes, setBoxes] = useState(() => localStorage.getItem("studio_boxes") !== "0");
  const [raw, setRaw] = useState(false);
  const [layout, setLayout] = useState<Layout>(() => (localStorage.getItem("studio_layout") as Layout) || "source");
  const [view, setView] = useState<string | null>(null);
  const [limit, setLimit] = useState(24);
  const [askDelete, setAskDelete] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!run) {
      setItems([]);
      setInsight(null);
      return;
    }
    setLoading(true);
    const [a, b] = await Promise.all([studio.runImages(run).catch(() => null), studio.insight(run).catch(() => null)]);
    setItems(a?.images || []);
    setInsight(b);
    setLoading(false);
  }, [run]);
  useEffect(() => {
    setSel(new Set());
    setFilter("all");
    setLimit(24);
    setView(null);
    void load();
  }, [load]);
  // reload when a job for this run finishes (new images / new labels)
  const genStatus = st.gen.job?.status;
  useEffect(() => {
    if (genStatus === "done") void load();
  }, [genStatus, load]);

  const near = items.filter((i) => i.near_copy);
  const drift = items.filter((i) => (i.camera_drift || []).length);
  const shown = filter === "near" ? near : filter === "drift" ? drift : filter === "selected" ? items.filter((i) => sel.has(i.filename)) : items;

  // group by source, worst base-model score first (that is where generation should help most)
  const groups = useMemo(() => {
    const m = new Map<string, GenItem[]>();
    for (const it of shown) {
      const s = sourceOf(it);
      if (!m.has(s)) m.set(s, []);
      m.get(s)!.push(it);
    }
    const score = (s: string) => insight?.sources[s]?.faults?.score ?? 2;
    return [...m.entries()].sort((a, b) => score(a[0]) - score(b[0]) || a[0].localeCompare(b[0]));
  }, [shown, insight]);
  const ordered = useMemo(() => (layout === "source" ? groups.flatMap(([, g]) => g) : shown), [layout, groups, shown]);

  const toggle = useCallback(
    (fn: string) =>
      setSel((s) => {
        const n = new Set(s);
        if (n.has(fn)) n.delete(fn);
        else n.add(fn);
        return n;
      }),
    [],
  );
  const sending = st.send.job?.status === "running";
  const genRunning = st.gen.job?.status === "running";

  const act = async (key: string, fn: () => Promise<void>) => {
    setBusy(key);
    try {
      await fn();
    } catch (e) {
      st.setToast({ kind: "error", text: errText(e) });
    } finally {
      setBusy(null);
    }
  };
  const send = (names: string[] | null) =>
    act("send", async () => {
      st.send.setJob(await studio.send(run, names));
      st.send.kick();
    });
  const pickLayout = (l: Layout) => {
    setLayout(l);
    localStorage.setItem("studio_layout", l);
  };

  const viewIdx = view ? ordered.findIndex((i) => i.filename === view) : -1;

  return (
    <div className="space-y-3">
      {!st.runs.length ? (
        <Empty>No runs yet — start with a test on one source (step 3).</Empty>
      ) : (
        <section className="rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-ink-900">
          {/* header: run picker + filters + view controls */}
          <div className="flex flex-wrap items-center gap-2 border-b border-slate-100 px-3 py-2.5 dark:border-slate-800">
            <RunPicker runs={st.runs} active={run} onPick={st.setActiveRun} />
            <div className="flex flex-wrap gap-1">
              {(
                [
                  ["all", `All ${items.length}`],
                  ["near", `Near copies ${near.length}`],
                  ["drift", `Camera drift ${drift.length}`],
                  ["selected", `Selected ${sel.size}`],
                ] as [Filter, string][]
              ).map(([k, l]) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => setFilter(k)}
                  className={cn(
                    "rounded-full px-2.5 py-1 text-[11px] font-bold transition",
                    filter === k ? "bg-slate-900 text-white dark:bg-white dark:text-slate-900" : "bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-300",
                    k === "near" && near.length > 0 && filter !== k && "text-amber-600 dark:text-amber-300",
                  )}
                >
                  {l}
                </button>
              ))}
            </div>
            <div className="ml-auto flex flex-wrap items-center gap-1.5">
              <div className="flex overflow-hidden rounded-lg border border-slate-300 text-[11px] font-bold dark:border-slate-600">
                <button type="button" onClick={() => pickLayout("source")} className={cn("flex items-center gap-1 px-2 py-1", layout === "source" ? "bg-violet-600 text-white" : "text-slate-500")} title="Group by source: faults + what each image targeted">
                  <Rows3 className="h-3 w-3" /> By source
                </button>
                <button type="button" onClick={() => pickLayout("pairs")} className={cn("flex items-center gap-1 px-2 py-1", layout === "pairs" ? "bg-violet-600 text-white" : "text-slate-500")} title="Source | generated pairs">
                  <LayoutGrid className="h-3 w-3" /> Pairs
                </button>
              </div>
              <button
                type="button"
                onClick={() => setBoxes((b) => (localStorage.setItem("studio_boxes", b ? "0" : "1"), !b))}
                className={cn("inline-flex items-center gap-1 rounded-lg border px-2 py-1 text-[11px] font-bold", boxes ? "border-emerald-500 text-emerald-600 dark:text-emerald-300" : "border-slate-300 text-slate-500 dark:border-slate-600")}
                title="Draw the labels (B)"
              >
                {boxes ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />} Boxes
              </button>
              <button
                type="button"
                onClick={() => setRaw((r) => !r)}
                className={cn("rounded-lg border px-2 py-1 text-[11px] font-bold", raw ? "border-violet-500 text-violet-600 dark:text-violet-300" : "border-slate-300 text-slate-500 dark:border-slate-600")}
                title="Show the generator's own output (before the square stretch and palette)"
              >
                {raw ? "Raw" : "Final"}
              </button>
              {info?.pending ? (
                <>
                  <Btn kind="primary" busy={busy === "collect"} disabled={genRunning} onClick={() => act("collect", async () => (st.gen.setJob(await studio.collect(run)), st.gen.kick()))}>
                    <Download className="h-3.5 w-3.5" /> Collect
                  </Btn>
                  {info.provider !== "gemini" && (
                    <Btn
                      kind="ghost"
                      busy={busy === "cancel"}
                      onClick={() =>
                        window.confirm(`Cancel the OpenAI batch of ${run}?`) &&
                        act("cancel", async () => {
                          const r = await studio.cancel(run);
                          st.setToast({ kind: "info", text: `Batch: ${r.status}` });
                          await st.loadData();
                        })
                      }
                    >
                      <Ban className="h-3.5 w-3.5" /> Cancel
                    </Btn>
                  )}
                </>
              ) : (
                <Btn kind="ghost" busy={busy === "label"} disabled={!items.length || genRunning} onClick={() => act("label", async () => (st.gen.setJob(await studio.autolabel(run)), st.gen.kick()))} title="Label this run with the base model">
                  <Tag className="h-3.5 w-3.5" /> Re-label
                </Btn>
              )}
            </div>
          </div>

          {info?.note && <div className="border-b border-slate-100 bg-violet-500/10 px-3 py-1.5 text-[12px] text-violet-800 dark:border-slate-800 dark:text-violet-200">{info.note}</div>}
          {sending && st.send.job && (
            <div className="space-y-1 border-b border-slate-100 px-3 py-2 dark:border-slate-800">
              <div className="text-[11px] text-slate-500">{st.send.job.message}</div>
              <Progress value={st.send.job.progress} tone="emerald" />
            </div>
          )}

          <div className="p-3">
            {loading ? (
              <div className="space-y-3">
                {Array.from({ length: 3 }).map((_, i) => (
                  <div key={i} className="h-56 animate-pulse rounded-xl bg-slate-100 dark:bg-slate-800" />
                ))}
              </div>
            ) : shown.length === 0 ? (
              <Empty>{info?.pending ? "The batch is still generating — press Collect to fetch finished images." : filter === "all" ? "This run has no images." : "Nothing here."}</Empty>
            ) : layout === "source" ? (
              <div className="space-y-3">
                {groups.slice(0, limit).map(([src, group]) => (
                  <SourceGroup
                    key={src}
                    run={run}
                    src={src}
                    items={group}
                    insight={insight}
                    raw={raw}
                    boxes={boxes}
                    sel={sel}
                    onToggle={toggle}
                    onOpen={setView}
                  />
                ))}
                {groups.length > limit && (
                  <div className="text-center">
                    <Btn kind="secondary" onClick={() => setLimit((l) => l + 24)}>
                      Show more sources ({groups.length - limit} left)
                    </Btn>
                  </div>
                )}
              </div>
            ) : (
              <>
                <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
                  {shown.slice(0, limit * 2).map((it) => (
                    <PairCard
                      key={it.filename}
                      run={run}
                      it={it}
                      labels={insight?.labels[it.filename]}
                      plan={insight?.plans[it.filename]}
                      raw={raw}
                      boxes={boxes}
                      selected={sel.has(it.filename)}
                      onToggle={() => toggle(it.filename)}
                      onOpen={() => setView(it.filename)}
                    />
                  ))}
                </div>
                {shown.length > limit * 2 && (
                  <div className="mt-3 text-center">
                    <Btn kind="secondary" onClick={() => setLimit((l) => l + 24)}>
                      Show more ({shown.length - limit * 2} left)
                    </Btn>
                  </div>
                )}
              </>
            )}
          </div>
        </section>
      )}

      {/* bulk bar */}
      {run && items.length > 0 && !info?.pending && (
        <div className="sticky bottom-4 z-20 mx-auto flex w-fit flex-wrap items-center gap-2 rounded-2xl border border-slate-200 bg-white/95 px-3 py-2 shadow-xl backdrop-blur dark:border-slate-700 dark:bg-ink-900/95">
          <span className="text-[12px] font-bold">{sel.size ? `${sel.size} selected` : "Nothing selected"}</span>
          {near.length > 0 && (
            <Btn kind="ghost" onClick={() => setSel(new Set(near.map((i) => i.filename)))}>
              <Copy className="h-3.5 w-3.5" /> Select near copies
            </Btn>
          )}
          <Btn kind="ghost" onClick={() => setSel(sel.size === items.length ? new Set() : new Set(items.map((i) => i.filename)))}>
            {sel.size === items.length ? "Clear" : "Select all"}
          </Btn>
          <Btn kind="danger" disabled={!sel.size} onClick={() => setAskDelete(true)}>
            <Trash2 className="h-3.5 w-3.5" /> Delete
          </Btn>
          <Btn
            kind="success"
            busy={busy === "send" || sending}
            onClick={() => {
              if (sel.size) return send([...sel]);
              const keep = items.filter((i) => !i.near_copy).map((i) => i.filename);
              return send(keep.length === items.length ? null : keep);
            }}
            title={sel.size ? undefined : "Sends every image except near copies"}
          >
            <Send className="h-3.5 w-3.5" /> {sel.size ? `Send ${sel.size} to Own` : `Send ${items.length - near.length} to Own`}
          </Btn>
        </div>
      )}

      {viewIdx >= 0 && (
        <Viewer
          run={run}
          items={ordered}
          index={viewIdx}
          insight={insight}
          raw={raw}
          boxes={boxes}
          selected={sel}
          onToggle={toggle}
          onBoxes={() => setBoxes((b) => !b)}
          onMove={(i) => setView(ordered[i]?.filename ?? null)}
          onClose={() => setView(null)}
        />
      )}
      {askDelete && (
        <Confirm
          title={`Delete ${sel.size} images?`}
          danger
          confirmLabel="Delete"
          onCancel={() => setAskDelete(false)}
          onConfirm={async () => {
            setAskDelete(false);
            await act("delete", async () => {
              const r = await studio.deleteImages(run, [...sel]);
              st.setToast({ kind: "ok", text: `Deleted ${r.deleted} · ${r.remaining} left` });
              setSel(new Set());
              await st.loadData();
              await load();
            });
          }}
        >
          <p>They move to the run's _deleted/ folder and are never sent or uploaded.</p>
        </Confirm>
      )}
    </div>
  );
}

/** Compact run selector: the current run + ‹ › + a dropdown of every run. */
function RunPicker({ runs, active, onPick }: { runs: Run[]; active: string; onPick: (r: string) => void }) {
  const st = useStudio();
  const [open, setOpen] = useState(false);
  const [del, setDel] = useState<Set<string>>(new Set()); // runs ticked for deletion
  const [ask, setAsk] = useState<string[] | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const i = runs.findIndex((r) => r.name === active);
  const cur = runs[i];
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, [open]);
  useEffect(() => setDel((d) => new Set([...d].filter((n) => runs.some((r) => r.name === n)))), [runs]);
  const tick = (n: string) =>
    setDel((d) => {
      const x = new Set(d);
      if (x.has(n)) x.delete(n);
      else x.add(n);
      return x;
    });
  const line = (r: Run) => (
    <>
      <span className={cn("h-2 w-2 shrink-0 rounded-full", r.pending ? "animate-pulse bg-amber-500" : r.mode === "final" ? "bg-emerald-500" : "bg-sky-500")} />
      <span className="font-extrabold">{r.mode === "final" ? "Final" : "Test"}</span>
      <span className="text-slate-500">{runTime(r.name)}</span>
      <span className="tabular-nums">· {r.count} img</span>
      <span className="text-slate-500">· {(r.provider || "openai") === "gemini" ? "Gemini" : "OpenAI"}</span>
    </>
  );
  const asking = (ask || []).map((n) => runs.find((r) => r.name === n)).filter(Boolean) as Run[];
  return (
    <div ref={ref} className="relative flex items-center gap-1">
      <button type="button" disabled={i <= 0} onClick={() => onPick(runs[i - 1].name)} className="rounded-md p-1 text-slate-500 hover:bg-slate-100 disabled:opacity-30 dark:hover:bg-slate-800" title="Newer run">
        <ChevronLeft className="h-4 w-4" />
      </button>
      <button type="button" onClick={() => setOpen((o) => !o)} className="flex items-center gap-1.5 rounded-xl border border-slate-300 bg-white px-2.5 py-1.5 text-[12px] hover:border-violet-400 dark:border-slate-600 dark:bg-ink-800">
        {cur ? line(cur) : "Pick a run"}
        {cur?.label && <Chip tone="violet">{cur.label}</Chip>}
        <ChevronDown className="h-3.5 w-3.5 text-slate-400" />
      </button>
      <button type="button" disabled={i < 0 || i >= runs.length - 1} onClick={() => onPick(runs[i + 1].name)} className="rounded-md p-1 text-slate-500 hover:bg-slate-100 disabled:opacity-30 dark:hover:bg-slate-800" title="Older run">
        <ChevronRight className="h-4 w-4" />
      </button>
      {open && (
        <div className="absolute left-0 top-full z-30 mt-1 flex max-h-[65vh] w-[470px] flex-col rounded-xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-ink-900">
          <div className="min-h-0 flex-1 overflow-y-auto p-1">
            {runs.map((r) => (
              <div key={r.name} className={cn("group flex items-center gap-1 rounded-lg pl-1.5 pr-1", r.name === active ? "bg-violet-500/15" : del.has(r.name) ? "bg-rose-500/10" : "hover:bg-slate-100 dark:hover:bg-slate-800")}>
                <input type="checkbox" checked={del.has(r.name)} onChange={() => tick(r.name)} disabled={r.pending} title={r.pending ? "Still generating at the provider" : "Tick to delete"} className="accent-rose-500" />
                <button type="button" onClick={() => (onPick(r.name), setOpen(false))} className="flex min-w-0 flex-1 flex-wrap items-center gap-1.5 px-1 py-1.5 text-left text-[12px]" title={r.note || r.name}>
                  {line(r)}
                  {r.label && <Chip tone="violet">{r.label}</Chip>}
                  {r.near_copies > 0 && <Chip tone="amber">{r.near_copies} near</Chip>}
                  {(r.camera_drift ?? 0) > 0 && <Chip tone="rose">{r.camera_drift} drift</Chip>}
                </button>
                {!r.pending && (
                  <button type="button" onClick={() => setAsk([r.name])} className="rounded p-1 text-slate-400 opacity-0 transition hover:bg-rose-500/10 hover:text-rose-500 group-hover:opacity-100" title="Delete this run">
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-2 border-t border-slate-200 px-2 py-1.5 text-[11px] dark:border-slate-700">
            <span className="text-slate-500">Select</span>
            <button type="button" className="font-bold text-violet-600 hover:underline dark:text-violet-300" onClick={() => setDel(new Set())}>none</button>
            <button type="button" className="font-bold text-violet-600 hover:underline dark:text-violet-300" onClick={() => setDel(new Set(runs.filter((r) => r.mode !== "final" && !r.pending).map((r) => r.name)))}>
              all tests
            </button>
            <button type="button" className="font-bold text-violet-600 hover:underline dark:text-violet-300" onClick={() => setDel(new Set(runs.filter((r) => !r.pending).map((r) => r.name)))}>all</button>
            <button
              type="button"
              disabled={!del.size}
              onClick={() => setAsk([...del])}
              className="ml-auto inline-flex items-center gap-1 rounded-md bg-rose-600 px-2.5 py-1 font-bold text-white hover:bg-rose-500 disabled:opacity-30"
            >
              <Trash2 className="h-3.5 w-3.5" /> Delete {del.size || ""} {del.size === 1 ? "run" : "runs"}
            </button>
          </div>
        </div>
      )}
      {ask && asking.length > 0 && (
        <Confirm
          title={`Delete ${asking.length} ${asking.length === 1 ? "run" : "runs"}?`}
          danger
          typeToConfirm="DELETE"
          confirmLabel={`Delete ${asking.reduce((n, r) => n + r.count, 0)} images`}
          onCancel={() => setAsk(null)}
          onConfirm={async () => {
            try {
              const res = await studio.deleteRuns(asking.map((r) => r.name));
              st.setToast({ kind: "ok", text: `Deleted ${res.deleted.length} ${res.deleted.length === 1 ? "run" : "runs"} (${res.images} images).` });
              setDel(new Set());
              setAsk(null);
              await st.loadData();
            } catch (e) {
              st.setToast({ kind: "error", text: errText(e) });
              setAsk(null);
            }
          }}
        >
          <ul className="max-h-40 space-y-0.5 overflow-y-auto text-[12px]">
            {asking.map((r) => (
              <li key={r.name} className="flex flex-wrap items-center gap-1.5">
                {line(r)}
                {r.label && <Chip tone="violet">{r.label}</Chip>}
              </li>
            ))}
          </ul>
          <p>
            Their images, labels and batch files are removed from the Studio for good. Images already <b>sent to Own</b> stay in Own
            {asking.some((r) => r.mode === "final") ? <>, and Final runs already <b>uploaded to Hugging Face</b> stay there</> : null}.
          </p>
        </Confirm>
      )}
    </div>
  );
}

function FaultList({ faults, compact }: { faults: SourceFault[]; compact?: boolean }) {
  return (
    <ol className="space-y-1">
      {faults.map((f, i) => (
        <li key={i} className="flex gap-1.5 text-[11px] leading-snug" title={[f.reason, f.what, f.where, f.why].filter(Boolean).join("\n")}>
          <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded text-[9px] font-black text-white" style={{ background: STATUS_COLOR[f.status] }}>
            {i + 1}
          </span>
          <span className="min-w-0">
            <b style={{ color: STATUS_COLOR[f.status] }}>{STATUS_TEXT[f.status]}</b>{" "}
            <b>{f.status === "false_positive" ? `→ ${f.pred_class}` : f.cls}</b>
            {f.status === "wrong_class" && f.pred_class && <> as {f.pred_class}</>}
            <span className="text-slate-500">
              {f.distance ? ` · ${f.distance}` : ""}
              {f.height_pct != null ? ` · ${f.height_pct}% h` : ""}
            </span>
            {!compact && (f.what || f.where) && <span className="block text-slate-500">{[f.what, f.where].filter(Boolean).join(" — ")}</span>}
            {!compact && f.why && <span className="block italic text-slate-400">{f.why}</span>}
          </span>
        </li>
      ))}
    </ol>
  );
}

function PlanList({ plan }: { plan: ImagePlan | undefined }) {
  if (!plan) return <div className="text-[11px] text-slate-400">No prompt record for this image.</div>;
  return (
    <div className="space-y-1 text-[11px] leading-snug">
      <div>
        <b>Recipe {plan.variant ?? "?"}{plan.of ? `/${plan.of}` : ""}</b> · <span className="font-mono">{plan.regime || "—"}</span>
        {plan.motion_blur && <span className="text-sky-500"> · motion blur</span>}
        {plan.camera_look && <span className="text-violet-500"> · camera look</span>}
      </div>
      {plan.view && <div className="text-slate-500">View: {plan.view}</div>}
      {(plan.targets || []).map((t, i) => (
        <div key={i} className="flex gap-1.5">
          <Target className="mt-0.5 h-3 w-3 shrink-0 text-rose-500" />
          <span>
            <b>{t.distance || "target"}{t.height_pct != null ? ` · ${t.height_pct}% h` : ""}</b> — {t.directive}
            {t.why && <span className="block italic text-slate-400">why hard: {t.why}</span>}
          </span>
        </div>
      ))}
      {plan.negative && (
        <div className="flex gap-1.5">
          <Crosshair className="mt-0.5 h-3 w-3 shrink-0 text-amber-500" />
          <span>
            <b>Look-alike (not a drink)</b> — {plan.negative}
          </span>
        </div>
      )}
    </div>
  );
}

/** One source: its frame with the base model's mistakes, then every image generated from it. */
function SourceGroup({
  run,
  src,
  items,
  insight,
  raw,
  boxes,
  sel,
  onToggle,
  onOpen,
}: {
  run: string;
  src: string;
  items: GenItem[];
  insight: RunInsight | null;
  raw: boolean;
  boxes: boolean;
  sel: Set<string>;
  onToggle: (fn: string) => void;
  onOpen: (fn: string) => void;
}) {
  const { genOf } = useStudio();
  const si: SourceInsight | null | undefined = insight?.sources[src];
  const faults = si?.faults?.errors || [];
  const c = si?.faults?.counts || {};
  const [showFaults, setShowFaults] = useState(true);
  const [noSource, setNoSource] = useState(false);
  const targeted = items.reduce((n, it) => n + (insight?.plans[it.filename]?.targets?.length ? 1 : 0), 0);
  const scene = si?.scene;
  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 dark:border-slate-800">
      <div className="grid gap-0 lg:grid-cols-[minmax(260px,320px)_1fr]">
        {/* source */}
        <div className="border-b border-slate-200 bg-slate-50 p-2.5 dark:border-slate-800 dark:bg-ink-800/60 lg:border-b-0 lg:border-r">
          <div className="relative cursor-zoom-in overflow-hidden rounded-lg bg-black" onClick={() => items[0] && onOpen(items[0].filename)}>
            {noSource ? (
              <div className="flex aspect-square w-full flex-col items-center justify-center gap-1 text-[10px] text-slate-500">
                <ImageOff className="h-5 w-5" /> source frame not available
              </div>
            ) : (
              <img src={img.source(run, items[0].filename, 480)} alt="source" loading="lazy" onError={() => setNoSource(true)} className="aspect-square w-full object-cover" />
            )}
            {showFaults && faults.length > 0 && !noSource && <FaultLayer faults={faults} />}
            <span className="absolute left-1.5 top-1.5 rounded bg-black/60 px-1.5 text-[9px] font-bold uppercase tracking-wide text-white">source</span>
            {faults.length > 0 && (
              <button
                type="button"
                onClick={(e) => (e.stopPropagation(), setShowFaults((v) => !v))}
                className="absolute right-1.5 top-1.5 rounded bg-black/60 px-1.5 py-0.5 text-[9px] font-bold text-white hover:bg-black/80"
              >
                {showFaults ? "hide faults" : "show faults"}
              </button>
            )}
          </div>
          <div className="mt-2 flex items-baseline justify-between gap-2">
            <span className="flex items-center gap-1.5 font-mono text-[12px] font-bold">
              {blockOf(src)} <GenBadge gen={genOf(src)} long />
            </span>
            {si?.faults?.score != null && (
              <span className="text-[11px] text-slate-500" title="The base model's validator-style score on this frame">
                base model <b className={cn("text-[13px] tabular-nums", scoreTone(si.faults.score))}>{si.faults.score.toFixed(2)}</b>
              </span>
            )}
          </div>
          {scene && (
            <div className="text-[11px] text-slate-500">
              {[scene.place, scene.time_of_day, scene.weather === "rain_or_wet" ? "wet" : null, scene.people != null ? `${scene.people} people` : null].filter(Boolean).join(" · ")}
            </div>
          )}
          {si?.faults ? (
            <>
              <div className="mt-1.5 text-[11px]">
                <b>{c.hit ?? 0}</b>/{c.gt ?? 0} drinks found
                {c.missed ? <span className="text-rose-500"> · {c.missed} missed</span> : null}
                {c.wrong_class ? <span className="text-fuchsia-500"> · {c.wrong_class} wrong class</span> : null}
                {c.false_positive ? <span className="text-amber-500"> · {c.false_positive} false alarm{c.false_positive > 1 ? "s" : ""}</span> : null}
              </div>
              <div className="mt-1.5">{faults.length ? <FaultList faults={faults} /> : <div className="text-[11px] text-emerald-600">The base model gets every drink here right.</div>}</div>
            </>
          ) : (
            <div className="mt-1.5 text-[11px] text-slate-400">No base-model analysis for this frame{si === undefined ? "" : " (not in the analysed eval set)"}.</div>
          )}
          {si?.look && (
            <details className="mt-1.5 text-[11px]">
              <summary className="cursor-pointer font-bold text-violet-600 dark:text-violet-300">Camera look</summary>
              <p className="mt-0.5 text-slate-500">{si.look}</p>
            </details>
          )}
        </div>
        {/* generated */}
        <div className="p-2.5">
          <div className="mb-2 flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
            <b className="text-slate-700 dark:text-slate-200">{items.length} generated</b>
            {faults.length > 0 && <span>· {targeted}/{items.length} were told to reproduce its faults</span>}
          </div>
          <div className="grid gap-2" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(210px, 1fr))" }}>
            {items.map((it) => (
              <GenCard
                key={it.filename}
                run={run}
                it={it}
                plan={insight?.plans[it.filename]}
                labels={insight?.labels[it.filename]}
                raw={raw}
                boxes={boxes}
                selected={sel.has(it.filename)}
                onToggle={() => onToggle(it.filename)}
                onOpen={() => onOpen(it.filename)}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function GenCard({
  run,
  it,
  plan,
  labels,
  raw,
  boxes,
  selected,
  onToggle,
  onOpen,
}: {
  run: string;
  it: GenItem;
  plan?: ImagePlan;
  labels?: LabelBox[];
  raw: boolean;
  boxes: boolean;
  selected: boolean;
  onToggle: () => void;
  onOpen: () => void;
}) {
  const [broken, setBroken] = useState(false);
  const nT = plan?.targets?.length || 0;
  return (
    <div className={cn("group overflow-hidden rounded-lg border bg-black transition", selected ? "border-sky-500 ring-2 ring-sky-500/50" : "border-slate-200 dark:border-slate-700")}>
      <div className="relative cursor-zoom-in" onClick={onOpen}>
        {broken ? (
          <div className="flex aspect-square w-full items-center justify-center text-slate-500">
            <ImageOff className="h-6 w-6" />
          </div>
        ) : (
          <img src={raw ? img.raw(run, it.filename, 480) : img.gen(run, it.filename, 480)} alt="generated" loading="lazy" onError={() => setBroken(true)} className="aspect-square w-full object-cover" />
        )}
        {boxes && labels && !raw && <BoxLayer boxes={labels} thin />}
        <span className="absolute left-1.5 top-1.5 rounded bg-violet-600/85 px-1.5 font-mono text-[9px] font-bold text-white">v{String(plan?.variant ?? it.filename.match(/__v(\d+)/)?.[1] ?? "?").padStart(2, "0")}</span>
        <label className="absolute right-1.5 top-1.5 flex h-6 w-6 items-center justify-center rounded-md bg-black/55 opacity-80 transition group-hover:opacity-100" onClick={(e) => e.stopPropagation()}>
          <input type="checkbox" checked={selected} onChange={onToggle} className="h-4 w-4 accent-sky-500" />
        </label>
        {/* what this image was told to contain — on hover */}
        {plan && (
          <div className="pointer-events-none absolute inset-x-0 bottom-0 max-h-[70%] translate-y-full overflow-hidden bg-black/85 p-2 text-white opacity-0 transition group-hover:translate-y-0 group-hover:opacity-100 [&_*]:!text-slate-100">
            <PlanList plan={plan} />
          </div>
        )}
      </div>
      <div className="space-y-1 bg-white px-2 py-1.5 dark:bg-ink-900">
        <div className="flex flex-wrap items-center gap-1">
          {plan?.regime && <span className="rounded bg-slate-100 px-1 font-mono text-[10px] text-slate-600 dark:bg-slate-800 dark:text-slate-300">{plan.regime}</span>}
          {nT > 0 && (
            <span className="inline-flex items-center gap-0.5 rounded bg-rose-500/15 px-1 text-[10px] font-bold text-rose-600 dark:text-rose-300" title="Failure targets this image was told to reproduce">
              <Target className="h-2.5 w-2.5" /> {nT}
            </span>
          )}
          {plan?.negative && (
            <span className="inline-flex items-center gap-0.5 rounded bg-amber-500/15 px-1 text-[10px] font-bold text-amber-600 dark:text-amber-300" title="A look-alike non-drink">
              <Crosshair className="h-2.5 w-2.5" /> 1
            </span>
          )}
          {plan?.motion_blur && <Wind className="h-3 w-3 text-sky-500" aria-label="motion blur" />}
          {it.near_copy && <Chip tone="amber">≈ copy {it.layout_similarity?.toFixed(2)}</Chip>}
          {(it.camera_drift || []).length > 0 && (
            <Chip tone="rose" title={(it.camera_drift || []).join("\n")}>
              <AlertTriangle className="h-3 w-3" /> drift
            </Chip>
          )}
        </div>
        {boxes && labels && <BoxCounts boxes={labels} />}
      </div>
    </div>
  );
}

function PairCard({
  run,
  it,
  labels,
  plan,
  raw,
  boxes,
  selected,
  onToggle,
  onOpen,
}: {
  run: string;
  it: GenItem;
  labels?: LabelBox[];
  plan?: ImagePlan;
  raw: boolean;
  boxes: boolean;
  selected: boolean;
  onToggle: () => void;
  onOpen: () => void;
}) {
  const [broken, setBroken] = useState(false);
  const [noSource, setNoSource] = useState(false);
  return (
    <div className={cn("group overflow-hidden rounded-xl border bg-black transition", selected ? "border-sky-500 ring-2 ring-sky-500/50" : "border-slate-200 dark:border-slate-800")}>
      <div className="relative grid cursor-zoom-in grid-cols-2 gap-px" onClick={onOpen}>
        <div className="relative">
          {noSource ? (
            <div className="flex aspect-square w-full flex-col items-center justify-center gap-1 bg-slate-900 px-3 text-center text-[10px] text-slate-500">
              <ImageOff className="h-5 w-5" /> source frame not available
            </div>
          ) : (
            <img src={img.source(run, it.filename, 480)} alt="source" loading="lazy" onError={() => setNoSource(true)} className="aspect-square w-full object-cover opacity-90" />
          )}
          <span className="absolute left-1.5 top-1.5 rounded bg-black/60 px-1.5 text-[9px] font-bold uppercase tracking-wide text-white">source</span>
        </div>
        <div className="relative">
          {broken ? (
            <div className="flex aspect-square w-full items-center justify-center text-slate-500">
              <ImageOff className="h-6 w-6" />
            </div>
          ) : (
            <img src={raw ? img.raw(run, it.filename, 480) : img.gen(run, it.filename, 480)} alt="generated" loading="lazy" onError={() => setBroken(true)} className="aspect-square w-full object-cover" />
          )}
          {boxes && labels && !raw && <BoxLayer boxes={labels} thin />}
          <span className="absolute left-1.5 top-1.5 rounded bg-violet-600/85 px-1.5 text-[9px] font-bold uppercase tracking-wide text-white">{raw ? "raw" : "new"}</span>
        </div>
        <label className="absolute right-1.5 top-1.5 flex h-6 w-6 items-center justify-center rounded-md bg-black/55 opacity-80 transition group-hover:opacity-100" onClick={(e) => e.stopPropagation()}>
          <input type="checkbox" checked={selected} onChange={onToggle} className="h-4 w-4 accent-sky-500" />
        </label>
      </div>
      <div className="flex flex-wrap items-center gap-1 bg-white px-2 py-1.5 dark:bg-ink-900">
        <span className="min-w-0 flex-1 truncate font-mono text-[10px] text-slate-500">{it.filename}</span>
        {plan?.regime && <span className="font-mono text-[10px] text-slate-500">{plan.regime}</span>}
        {boxes && labels && <BoxCounts boxes={labels} />}
        {it.near_copy && <Chip tone="amber">≈ copy {it.layout_similarity?.toFixed(2)}</Chip>}
        {(it.camera_drift || []).length > 0 && (
          <Chip tone="rose" title={(it.camera_drift || []).join("\n")}>
            <AlertTriangle className="h-3 w-3" /> drift
          </Chip>
        )}
      </div>
    </div>
  );
}

// ── viewer ─────────────────────────────────────────────────────────────────────────────────────────────────────────

type Zoom = { s: number; x: number; y: number };
const NO_ZOOM: Zoom = { s: 1, x: 0, y: 0 };

/** Sharp image over the already-loaded thumbnail (shown instantly), zoom/pan shared with the other pane. */
function ZoomPane({
  thumb,
  full,
  label,
  zoom,
  setZoom,
  children,
}: {
  thumb: string;
  full: string;
  label: ReactNode;
  zoom: Zoom;
  setZoom: (z: Zoom) => void;
  children?: ReactNode;
}) {
  const box = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; y: number; z: Zoom } | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    setLoaded(false);
    setFailed(false);
  }, [full]);

  const onWheel = (e: React.WheelEvent) => {
    const r = box.current?.getBoundingClientRect();
    if (!r) return;
    const s = Math.min(8, Math.max(1, zoom.s * (e.deltaY < 0 ? 1.25 : 0.8)));
    // keep the point under the cursor fixed
    const cx = e.clientX - r.left - r.width / 2;
    const cy = e.clientY - r.top - r.height / 2;
    const k = s / zoom.s;
    setZoom(s === 1 ? NO_ZOOM : { s, x: cx - (cx - zoom.x) * k, y: cy - (cy - zoom.y) * k });
  };
  return (
    <figure
      ref={box}
      className={cn("relative flex min-h-0 select-none items-center justify-center overflow-hidden rounded-lg bg-black", zoom.s > 1 ? "cursor-grab active:cursor-grabbing" : "cursor-zoom-in")}
      onWheel={onWheel}
      onMouseDown={(e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        drag.current = { x: e.clientX, y: e.clientY, z: zoom };
      }}
      onMouseMove={(e) => {
        const d = drag.current;
        if (d && d.z.s > 1) setZoom({ ...d.z, x: d.z.x + e.clientX - d.x, y: d.z.y + e.clientY - d.y });
      }}
      onMouseUp={() => (drag.current = null)}
      onMouseLeave={() => (drag.current = null)}
      onDoubleClick={() => setZoom(zoom.s > 1 ? NO_ZOOM : { s: 2.5, x: 0, y: 0 })}
      style={{ containerType: "size" }}
    >
      {/* square box as large as the pane allows, so overlays stay aligned with the (square) image */}
      <div
        className="relative shrink-0"
        style={{
          width: "min(100cqw, 100cqh)",
          height: "min(100cqw, 100cqh)",
          transform: `translate(${zoom.x}px, ${zoom.y}px) scale(${zoom.s})`,
          transformOrigin: "center",
        }}
      >
        <img src={thumb} alt="" className="absolute inset-0 h-full w-full object-contain" draggable={false} />
        {!failed && (
          <img
            src={full}
            alt=""
            draggable={false}
            onLoad={() => setLoaded(true)}
            onError={() => setFailed(true)}
            className={cn("absolute inset-0 h-full w-full object-contain transition-opacity duration-150", loaded ? "opacity-100" : "opacity-0")}
            style={{ imageRendering: zoom.s >= 3 ? "pixelated" : "auto" }}
          />
        )}
        {children}
      </div>
      <figcaption className="absolute left-2 top-2 rounded bg-black/60 px-1.5 text-[10px] font-bold uppercase tracking-wide text-white">{label}</figcaption>
      {!loaded && !failed && <span className="absolute bottom-2 right-2 rounded bg-black/60 px-1.5 text-[10px] text-slate-300">loading full size…</span>}
    </figure>
  );
}

function Viewer({
  run,
  items,
  index,
  insight,
  raw,
  boxes,
  selected,
  onToggle,
  onBoxes,
  onMove,
  onClose,
}: {
  run: string;
  items: GenItem[];
  index: number;
  insight: RunInsight | null;
  raw: boolean;
  boxes: boolean;
  selected: Set<string>;
  onToggle: (fn: string) => void;
  onBoxes: () => void;
  onMove: (i: number) => void;
  onClose: () => void;
}) {
  const it = items[index];
  const src = sourceOf(it);
  const si = insight?.sources[src];
  const faults = si?.faults?.errors || [];
  const plan = insight?.plans[it.filename];
  const labels = insight?.labels[it.filename];
  const [zoom, setZoom] = useState<Zoom>(NO_ZOOM);
  const [showFaults, setShowFaults] = useState(true);
  const [panel, setPanel] = useState(() => localStorage.getItem("studio_viewer_panel") !== "0");
  const genFull = raw ? img.raw(run, it.filename) : img.gen(run, it.filename);
  const ref = useRef({ index, n: items.length });
  ref.current = { index, n: items.length };

  useEffect(() => setZoom(NO_ZOOM), [it.filename]);
  // preload the neighbours so ← → shows the next pair at once
  useEffect(() => {
    for (const j of [index + 1, index - 1, index + 2]) {
      const n = items[j];
      if (!n) continue;
      for (const u of [raw ? img.raw(run, n.filename) : img.gen(run, n.filename), img.source(run, n.filename)]) {
        const im = new Image();
        im.src = u;
      }
    }
  }, [index, items, raw, run]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const { index: i, n } = ref.current;
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowRight") onMove(Math.min(n - 1, i + 1));
      else if (e.key === "ArrowLeft") onMove(Math.max(0, i - 1));
      else if (e.key === " ") onToggle(items[i].filename);
      else if (e.key.toLowerCase() === "b") onBoxes();
      else if (e.key.toLowerCase() === "f") setShowFaults((v) => !v);
      else if (e.key.toLowerCase() === "i") setPanel((v) => (localStorage.setItem("studio_viewer_panel", v ? "0" : "1"), !v));
      else if (e.key === "0") setZoom(NO_ZOOM);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [items, onBoxes, onClose, onMove, onToggle]);

  const sel = selected.has(it.filename);
  return createPortal(
    <div className="fixed inset-0 z-[90] flex flex-col bg-black/95 p-3 text-slate-200">
      <div className="mb-2 flex flex-wrap items-center gap-2 text-[12px]">
        <span className="font-mono">{blockOf(src)}</span>
        <ChevronRight className="h-3.5 w-3.5 text-slate-500" />
        <span className="font-mono font-bold">{it.filename}</span>
        <span className="tabular-nums text-slate-400">
          {index + 1} / {items.length}
        </span>
        {plan?.regime && <span className="rounded bg-white/10 px-1.5 font-mono text-[11px]">{plan.regime}</span>}
        {it.layout_similarity != null && (
          <Chip tone={it.near_copy ? "amber" : "slate"} title="Layout similarity to the source (above 0.6 = near copy)">
            similarity {it.layout_similarity.toFixed(2)}
          </Chip>
        )}
        {(it.camera_drift || []).map((d) => (
          <Chip key={d} tone="rose">
            {d}
          </Chip>
        ))}
        {zoom.s > 1 && (
          <button type="button" onClick={() => setZoom(NO_ZOOM)} className="rounded bg-white/10 px-1.5 text-[11px] hover:bg-white/20">
            {zoom.s.toFixed(1)}× · reset (0)
          </button>
        )}
        <div className="ml-auto flex items-center gap-2">
          <span className="hidden text-[11px] text-slate-400 xl:inline">wheel zoom · drag move · ← → next · space select · B boxes · F faults · I info · Esc</span>
          <button type="button" onClick={() => onToggle(it.filename)} className={cn("rounded-lg px-2.5 py-1 text-[12px] font-bold", sel ? "bg-sky-500 text-white" : "bg-white/10 text-white hover:bg-white/20")}>
            {sel ? "Selected" : "Select"}
          </button>
          <button type="button" onClick={onClose} className="rounded-md p-1.5 hover:bg-white/10" aria-label="Close">
            <X className="h-5 w-5" />
          </button>
        </div>
      </div>
      <div className={cn("grid min-h-0 flex-1 auto-rows-[minmax(0,1fr)] gap-3", panel ? "lg:grid-cols-[1fr_1fr_300px]" : "md:grid-cols-2")}>
        <ZoomPane thumb={img.source(run, it.filename, 480)} full={img.source(run, it.filename)} label="source" zoom={zoom} setZoom={setZoom}>
          {showFaults && faults.length > 0 && <FaultLayer faults={faults} />}
        </ZoomPane>
        <ZoomPane thumb={raw ? img.raw(run, it.filename, 480) : img.gen(run, it.filename, 480)} full={genFull} label={raw ? "raw" : "generated"} zoom={zoom} setZoom={setZoom}>
          {boxes && labels && !raw && <BoxLayer boxes={labels} />}
        </ZoomPane>
        {panel && (
          <aside className="min-h-0 space-y-3 overflow-y-auto rounded-lg bg-white/5 p-3 [&_.text-slate-500]:!text-slate-400">
            <div>
              <div className="mb-1 text-[10px] font-bold uppercase tracking-wider text-slate-400">This image was told to</div>
              <PlanList plan={plan} />
            </div>
            <div>
              <div className="mb-1 flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-slate-400">
                <span>Base model on the source</span>
                {si?.faults?.score != null && <span className={cn("text-[12px]", scoreTone(si.faults.score))}>{si.faults.score.toFixed(2)}</span>}
              </div>
              {si?.faults ? faults.length ? <FaultList faults={faults} /> : <div className="text-[11px] text-emerald-400">No mistakes on this frame.</div> : <div className="text-[11px] text-slate-500">Not analysed.</div>}
            </div>
            {labels && (
              <div>
                <div className="mb-1 text-[10px] font-bold uppercase tracking-wider text-slate-400">Base model on this image</div>
                <BoxCounts boxes={labels} />
                <div className="mt-0.5 text-[10px] text-slate-500">{insight?.label_source === "detector" ? "These boxes are the base model's predictions — drinks it misses have no box." : "Labels copied from the source — re-label."}</div>
              </div>
            )}
            {si?.look && (
              <div>
                <div className="mb-1 text-[10px] font-bold uppercase tracking-wider text-slate-400">Source camera look</div>
                <p className="text-[11px] text-slate-300">{si.look}</p>
              </div>
            )}
          </aside>
        )}
      </div>
      <div className="mt-2 flex items-center justify-center gap-3">
        <button type="button" onClick={() => onMove(Math.max(0, index - 1))} disabled={index === 0} className="rounded-full bg-white/10 p-2 hover:bg-white/20 disabled:opacity-20" aria-label="Previous">
          <ChevronLeft className="h-5 w-5" />
        </button>
        <span className="text-[11px] tabular-nums text-slate-400">
          {index + 1} / {items.length}
        </span>
        <button type="button" onClick={() => onMove(Math.min(items.length - 1, index + 1))} disabled={index === items.length - 1} className="rounded-full bg-white/10 p-2 hover:bg-white/20 disabled:opacity-20" aria-label="Next">
          <ChevronRight className="h-5 w-5" />
        </button>
      </div>
    </div>,
    document.body,
  );
}
