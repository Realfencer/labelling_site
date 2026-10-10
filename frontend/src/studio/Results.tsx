import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Ban, ChevronLeft, ChevronRight, Copy, Download, Eye, EyeOff, ImageOff, Send, Tag, Trash2, X } from "lucide-react";
import { cn } from "../lib/utils";
import { img, studio } from "./api";
import { useStudio } from "./store";
import type { GenItem, Run } from "./types";
import { Btn, Chip, CLASS_COLORS, Confirm, Empty, errText, Progress } from "./ui";

type Box = { cls: number; name: string; x1: number; y1: number; x2: number; y2: number };
type Filter = "all" | "near" | "drift" | "selected";

const labelCache = new Map<string, { boxes: Box[]; source: string }>();

function useLabels(run: string, fn: string, on: boolean) {
  const key = `${run}/${fn}`;
  const [data, setData] = useState(labelCache.get(key) || null);
  useEffect(() => {
    if (!on || labelCache.has(key)) {
      setData(labelCache.get(key) || null);
      return;
    }
    let alive = true;
    fetch(img.labels(run, fn), { credentials: "include" })
      .then((r) => r.json())
      .then((j) => {
        labelCache.set(key, j);
        if (alive) setData(j);
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [key, on, run, fn]);
  return data;
}

function BoxLayer({ boxes, thin }: { boxes: Box[]; thin?: boolean }) {
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
          strokeWidth={thin ? 1.5 : 2.5}
          vectorEffect="non-scaling-stroke"
        />
      ))}
    </svg>
  );
}

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

export default function Results() {
  const st = useStudio();
  const run = st.activeRun;
  const info = st.runs.find((r) => r.name === run) || null;
  const [items, setItems] = useState<GenItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState<Filter>("all");
  const [boxes, setBoxes] = useState(() => localStorage.getItem("studio_boxes") !== "0");
  const [raw, setRaw] = useState(false);
  const [view, setView] = useState<number | null>(null);
  const [limit, setLimit] = useState(60);
  const [askDelete, setAskDelete] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!run) return setItems([]);
    setLoading(true);
    try {
      setItems((await studio.runImages(run)).images || []);
    } catch {
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, [run]);
  useEffect(() => {
    setSel(new Set());
    setFilter("all");
    setLimit(60);
    void load();
  }, [load]);
  // reload the gallery when a job for this run finishes
  const genStatus = st.gen.job?.status;
  useEffect(() => {
    if (genStatus === "done") {
      labelCache.clear();
      void load();
    }
  }, [genStatus, load]);

  const near = items.filter((i) => i.near_copy);
  const drift = items.filter((i) => (i.camera_drift || []).length);
  const shown = filter === "near" ? near : filter === "drift" ? drift : filter === "selected" ? items.filter((i) => sel.has(i.filename)) : items;
  const toggle = (fn: string) =>
    setSel((s) => {
      const n = new Set(s);
      if (n.has(fn)) n.delete(fn);
      else n.add(fn);
      return n;
    });
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

  return (
    <div className="space-y-4">
      <RunStrip runs={st.runs} active={run} onPick={st.setActiveRun} />
      {st.runs.find((r) => r.name === run)?.note && (
        <div className="rounded-xl bg-violet-500/10 px-3 py-2 text-[12px] text-violet-800 dark:text-violet-200">{st.runs.find((r) => r.name === run)?.note}</div>
      )}

      {!run ? (
        <Empty>No runs yet — start with a test on one source (step 3).</Empty>
      ) : (
        <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-ink-900">
          <div className="flex flex-wrap items-center gap-2">
            <div className="mr-2">
              <div className="text-[13px] font-extrabold">
                {info?.mode === "final" ? "Final run" : "Test run"} · {runTime(run)}
              </div>
              <div className="text-[11px] text-slate-500">
                {info?.model} · {items.length} images · {info?.autolabelled ? "labelled by the base model" : "labels copied from the source — re-label before training"}
              </div>
            </div>
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
              <button
                type="button"
                onClick={() => setBoxes((b) => (localStorage.setItem("studio_boxes", b ? "0" : "1"), !b))}
                className={cn("inline-flex items-center gap-1 rounded-lg border px-2 py-1 text-[11px] font-bold", boxes ? "border-emerald-500 text-emerald-600 dark:text-emerald-300" : "border-slate-300 text-slate-500 dark:border-slate-600")}
                title="Draw the run's labels (B)"
              >
                {boxes ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />} Boxes
              </button>
              <button
                type="button"
                onClick={() => setRaw((r) => !r)}
                className={cn("rounded-lg border px-2 py-1 text-[11px] font-bold", raw ? "border-violet-500 text-violet-600 dark:text-violet-300" : "border-slate-300 text-slate-500 dark:border-slate-600")}
                title="Show the generator's output before the CCTV look"
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

          {sending && st.send.job && (
            <div className="mt-3 space-y-1">
              <div className="text-[11px] text-slate-500">{st.send.job.message}</div>
              <Progress value={st.send.job.progress} tone="emerald" />
            </div>
          )}

          <div className="mt-4">
            {loading ? (
              <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
                {Array.from({ length: 6 }).map((_, i) => (
                  <div key={i} className="aspect-[2/1] animate-pulse rounded-xl bg-slate-100 dark:bg-slate-800" />
                ))}
              </div>
            ) : shown.length === 0 ? (
              <Empty>{info?.pending ? "The batch is still generating — press Collect to fetch finished images." : filter === "all" ? "This run has no images." : "Nothing here."}</Empty>
            ) : (
              <>
                <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
                  {shown.slice(0, limit).map((it) => (
                    <PairCard
                      key={it.filename}
                      run={run}
                      it={it}
                      raw={raw}
                      boxes={boxes}
                      selected={sel.has(it.filename)}
                      onToggle={() => toggle(it.filename)}
                      onOpen={() => setView(items.indexOf(it))}
                    />
                  ))}
                </div>
                {shown.length > limit && (
                  <div className="mt-3 text-center">
                    <Btn kind="secondary" onClick={() => setLimit((l) => l + 60)}>
                      Show more ({shown.length - limit} left)
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

      {view != null && items[view] && (
        <Viewer
          run={run}
          items={items}
          index={view}
          raw={raw}
          boxes={boxes}
          selected={sel}
          onToggle={toggle}
          onBoxes={() => setBoxes((b) => !b)}
          onMove={(i) => setView(i)}
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

function RunStrip({ runs, active, onPick }: { runs: Run[]; active: string; onPick: (r: string) => void }) {
  if (!runs.length) return null;
  return (
    <div className="flex gap-2 overflow-x-auto pb-1">
      {runs.map((r) => (
        <button
          key={r.name}
          type="button"
          title={r.note || undefined}
          onClick={() => onPick(r.name)}
          className={cn(
            "min-w-[170px] shrink-0 rounded-2xl border px-3 py-2 text-left transition",
            active === r.name ? "border-violet-500 bg-violet-500/10 shadow" : "border-slate-200 bg-white hover:border-violet-300 dark:border-slate-800 dark:bg-ink-900",
          )}
        >
          <div className="flex items-center gap-1.5">
            <span className={cn("h-2 w-2 rounded-full", r.pending ? "animate-pulse bg-amber-500" : r.mode === "final" ? "bg-emerald-500" : "bg-sky-500")} />
            <span className="text-[12px] font-extrabold">{r.mode === "final" ? "Final" : "Test"}</span>
            <span className="text-[11px] text-slate-500">{runTime(r.name)}</span>
          </div>
          <div className="mt-0.5 flex items-center gap-1 text-[11px]">
            <span className="font-mono text-base font-black tabular-nums">{r.count}</span>
            <span className="text-slate-500">img · {(r.provider || "openai") === "gemini" ? "Gemini" : "OpenAI"}</span>
          </div>
          <div className="mt-1 flex flex-wrap gap-1">
            {r.label && <Chip tone="violet">{r.label}</Chip>}
            {r.pending && <Chip tone="amber">generating</Chip>}
            {r.near_copies > 0 && <Chip tone="amber">{r.near_copies} near copies</Chip>}
            {(r.camera_drift ?? 0) > 0 && <Chip tone="rose">{r.camera_drift} drift</Chip>}
            {r.count > 0 && !r.autolabelled && <Chip tone="slate">source labels</Chip>}
          </div>
        </button>
      ))}
    </div>
  );
}

function PairCard({
  run,
  it,
  raw,
  boxes,
  selected,
  onToggle,
  onOpen,
}: {
  run: string;
  it: GenItem;
  raw: boolean;
  boxes: boolean;
  selected: boolean;
  onToggle: () => void;
  onOpen: () => void;
}) {
  const labels = useLabels(run, it.filename, boxes);
  const [broken, setBroken] = useState(false);
  const [noSource, setNoSource] = useState(false);
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const b of labels?.boxes || []) c[b.name] = (c[b.name] || 0) + 1;
    return c;
  }, [labels]);
  return (
    <div className={cn("group overflow-hidden rounded-xl border bg-black transition", selected ? "border-sky-500 ring-2 ring-sky-500/50" : "border-slate-200 dark:border-slate-800")}>
      <div className="relative grid cursor-zoom-in grid-cols-2 gap-px" onClick={onOpen}>
        <div className="relative">
          {noSource ? (
            <div className="flex aspect-square w-full flex-col items-center justify-center gap-1 bg-slate-900 px-3 text-center text-[10px] text-slate-500">
              <ImageOff className="h-5 w-5" /> source frame is no longer in the eval set
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
            <img
              src={raw ? img.raw(run, it.filename, 480) : img.gen(run, it.filename, 480)}
              alt="generated"
              loading="lazy"
              onError={() => setBroken(true)}
              className="aspect-square w-full object-cover transition duration-300 group-hover:scale-[1.02]"
            />
          )}
          {boxes && labels && !raw && <BoxLayer boxes={labels.boxes} thin />}
          <span className="absolute left-1.5 top-1.5 rounded bg-violet-600/85 px-1.5 text-[9px] font-bold uppercase tracking-wide text-white">{raw ? "raw" : "new"}</span>
        </div>
        <label
          className="absolute right-1.5 top-1.5 flex h-6 w-6 items-center justify-center rounded-md bg-black/55 opacity-80 transition group-hover:opacity-100"
          onClick={(e) => e.stopPropagation()}
        >
          <input type="checkbox" checked={selected} onChange={onToggle} className="h-4 w-4 accent-sky-500" />
        </label>
      </div>
      <div className="flex flex-wrap items-center gap-1 bg-white px-2 py-1.5 dark:bg-ink-900">
        <span className="min-w-0 flex-1 truncate font-mono text-[10px] text-slate-500">{it.filename}</span>
        {boxes && labels && (
          <span className="flex gap-1 text-[10px] font-bold tabular-nums">
            {(["cup", "can", "bottle"] as const).map((c) => (counts[c] ? <span key={c} style={{ color: CLASS_COLORS[c] }}>{counts[c]} {c}</span> : null))}
          </span>
        )}
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

function Viewer({
  run,
  items,
  index,
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
  raw: boolean;
  boxes: boolean;
  selected: Set<string>;
  onToggle: (fn: string) => void;
  onBoxes: () => void;
  onMove: (i: number) => void;
  onClose: () => void;
}) {
  const it = items[index];
  const labels = useLabels(run, it.filename, boxes);
  const ref = useRef({ index, n: items.length });
  ref.current = { index, n: items.length };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const { index: i, n } = ref.current;
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowRight") onMove(Math.min(n - 1, i + 1));
      else if (e.key === "ArrowLeft") onMove(Math.max(0, i - 1));
      else if (e.key === " ") {
        e.preventDefault();
        onToggle(items[i].filename);
      } else if (e.key.toLowerCase() === "b") onBoxes();
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [items, onBoxes, onClose, onMove, onToggle]);

  const sel = selected.has(it.filename);
  return (
    <div className="fixed inset-0 z-[90] flex flex-col bg-black/90 p-4" onClick={onClose}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-[12px] text-slate-200" onClick={(e) => e.stopPropagation()}>
        <span className="font-mono">{it.source_image}</span>
        <ChevronRight className="h-3.5 w-3.5 text-slate-500" />
        <span className="font-mono font-bold">{it.filename}</span>
        <span className="text-slate-500">
          {index + 1} / {items.length}
        </span>
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
        {labels && <Chip tone={labels.source === "detector" ? "green" : "slate"}>{labels.boxes.length} boxes · {labels.source === "detector" ? "base model" : "copied from source"}</Chip>}
        <div className="ml-auto flex items-center gap-2">
          <span className="hidden text-[11px] text-slate-400 md:inline">← → move · space select · B boxes · Esc close</span>
          <button
            type="button"
            onClick={() => onToggle(it.filename)}
            className={cn("rounded-lg px-2.5 py-1 text-[12px] font-bold", sel ? "bg-sky-500 text-white" : "bg-white/10 text-white hover:bg-white/20")}
          >
            {sel ? "Selected" : "Select"}
          </button>
          <button type="button" onClick={onClose} className="rounded-md p-1.5 hover:bg-white/10" aria-label="Close">
            <X className="h-5 w-5" />
          </button>
        </div>
      </div>
      <div className="relative grid min-h-0 flex-1 gap-3 md:grid-cols-2" onClick={(e) => e.stopPropagation()}>
        <figure className="flex min-h-0 items-center justify-center">
          <img
            src={img.source(run, it.filename)}
            alt="source frame is no longer in the eval set"
            onError={(e) => ((e.target as HTMLImageElement).style.visibility = "hidden")}
            className="max-h-full max-w-full rounded-lg object-contain"
          />
        </figure>
        <figure className="relative flex min-h-0 items-center justify-center">
          <div className="relative inline-block">
            <img src={raw ? img.raw(run, it.filename) : img.gen(run, it.filename)} alt="generated" className="block max-h-[calc(100vh-120px)] max-w-full rounded-lg" />
            {boxes && labels && !raw && <BoxLayer boxes={labels.boxes} />}
          </div>
        </figure>
        <button
          type="button"
          onClick={() => onMove(Math.max(0, index - 1))}
          className="absolute left-0 top-1/2 -translate-y-1/2 rounded-full bg-white/10 p-2 text-white hover:bg-white/20 disabled:opacity-20"
          disabled={index === 0}
        >
          <ChevronLeft className="h-6 w-6" />
        </button>
        <button
          type="button"
          onClick={() => onMove(Math.min(items.length - 1, index + 1))}
          className="absolute right-0 top-1/2 -translate-y-1/2 rounded-full bg-white/10 p-2 text-white hover:bg-white/20 disabled:opacity-20"
          disabled={index === items.length - 1}
        >
          <ChevronRight className="h-6 w-6" />
        </button>
      </div>
    </div>
  );
}
