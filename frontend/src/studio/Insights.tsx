import { useEffect, useState } from "react";
import { AlertTriangle, ChevronDown, ChevronRight, Search, X } from "lucide-react";
import { cn } from "../lib/utils";
import { img, studio } from "./api";
import type { DeError, DeSource, Gallery, MiningReport, ModelErrors, RecordRow, Summary } from "./types";
import { Card, Chip, CLASS_COLORS, Empty, Lightbox, pct } from "./ui";

/** Insights tab: what the analysis found — numbers, regimes, the base model's misses, recommendations, samples. */
export default function Insights() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [galleries, setGalleries] = useState<Gallery[]>([]);
  const [overlaps, setOverlaps] = useState<Gallery[]>([]);
  const [mining, setMining] = useState<MiningReport | null>(null);
  const [errors, setErrors] = useState<ModelErrors | null>(null);
  const [regime, setRegime] = useState<string | null>(null);
  useEffect(() => {
    void Promise.all([
      studio.summary().catch(() => null),
      studio.galleries().catch(() => ({}) as Record<string, Gallery>),
      studio.overlaps().catch(() => ({}) as Record<string, Gallery>),
      studio.mining().catch(() => null),
      studio.modelErrors().catch(() => null),
    ]).then(([s, g, o, m, e]) => {
      setSummary(s);
      setGalleries(Object.values(g || {}).filter((x) => x && x.group !== "overlap"));
      setOverlaps(Object.values(o || {}).filter((x) => x && x.group === "overlap"));
      setMining(m);
      setErrors(e);
    });
  }, []);
  return (
    <div className="space-y-4">
      <Kpis summary={summary} />
      <DetectorErrors report={errors} />
      <RegimeSection title="Scene regimes" items={galleries.filter((g) => g.group === "scene")} selected={regime} onSelect={setRegime} />
      <RegimeSection title="Class-hard regimes" items={galleries.filter((g) => g.group === "class_hard")} selected={regime} onSelect={setRegime} />
      <RegimeSection title="Overlap gates" items={overlaps} selected={regime} onSelect={setRegime} />
      <Card title="Recommendations">
        {mining?.recommendations?.length ? (
          <div className="grid gap-2 md:grid-cols-2">
            {mining.recommendations.map((r, i) => (
              <div
                key={i}
                className={cn(
                  "rounded-xl border-l-4 bg-slate-50 px-3 py-2 dark:bg-slate-800/50",
                  r.priority === "high" ? "border-rose-500" : r.priority === "medium" ? "border-amber-500" : "border-slate-400",
                )}
              >
                <div className="flex items-center gap-2 text-[12px] font-bold">
                  <Chip tone={r.priority === "high" ? "rose" : r.priority === "medium" ? "amber" : "slate"}>{r.priority}</Chip>
                  {r.title}
                </div>
                <div className="mt-1 text-[11px] text-slate-600 dark:text-slate-300">{r.detail}</div>
              </div>
            ))}
          </div>
        ) : (
          <Empty>Analyze to get recommendations.</Empty>
        )}
      </Card>
      <Samples regime={regime} clearRegime={() => setRegime(null)} />
    </div>
  );
}

function Kpis({ summary }: { summary: Summary | null }) {
  if (!summary?.count) return null;
  const t = summary.class_instance_totals || {};
  const items: [string, string][] = [
    ["Images", String(summary.count)],
    ["Objects / image", (summary.mean_object_count ?? 0).toFixed(2)],
    ["Brightness", (summary.mean_brightness ?? 0).toFixed(3)],
    ["Median box area", (summary.median_bbox_area ?? 0).toFixed(4)],
    ["Cups", String(t.cup ?? 0)],
    ["Cans", String(t.can ?? 0)],
    ["Bottles", String(t.bottle ?? 0)],
  ];
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">
      {items.map(([k, v]) => (
        <div key={k} className="rounded-xl border border-slate-200 bg-white px-3 py-2 dark:border-slate-800 dark:bg-ink-900">
          <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">{k}</div>
          <div className="font-mono text-lg font-black tabular-nums">{v}</div>
        </div>
      ))}
    </div>
  );
}

function RegimeSection({ title, items, selected, onSelect }: { title: string; items: Gallery[]; selected: string | null; onSelect: (k: string | null) => void }) {
  if (!items.length) return null;
  return (
    <Card title={title} right={<span className="text-[11px] text-slate-400">click a card to filter the samples</span>}>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
        {[...items]
          .sort((a, b) => b.count - a.count)
          .map((g) => (
            <button
              key={g.key}
              type="button"
              onClick={() => onSelect(selected === g.key ? null : g.key)}
              className={cn(
                "rounded-xl border border-t-4 bg-white px-3 py-2 text-left transition hover:shadow dark:bg-ink-900",
                selected === g.key ? "ring-2 ring-sky-500" : "border-slate-200 dark:border-slate-700",
              )}
              style={{ borderTopColor: g.color }}
            >
              <div className="flex items-baseline gap-2">
                <span className="text-[13px] font-extrabold">{g.title || g.key}</span>
                <span className="ml-auto font-mono text-base font-black tabular-nums">{g.count}</span>
              </div>
              <div className="text-[10px] font-bold uppercase text-slate-400">
                {g.group.replace("_", "-")} · {g.pct.toFixed(1)}%
              </div>
              <div className="mt-1 line-clamp-2 text-[11px] text-slate-600 dark:text-slate-300">{g.why || g.short}</div>
            </button>
          ))}
      </div>
    </Card>
  );
}

// ---- detector errors ---------------------------------------------------------------------------------------

type DeType = "all" | "missed" | "wrong_class" | "false_positive";

function errClass(e: DeError): string {
  return (e.status === "false_positive" ? e.pred_class : e.gt_class) || "";
}

function ErrorTag({ e }: { e: DeError }) {
  if (e.status === "missed") return <Chip tone="rose">missed {e.gt_class}</Chip>;
  if (e.status === "wrong_class")
    return (
      <Chip tone="violet">
        {e.gt_class} → {e.pred_class}
      </Chip>
    );
  if (e.caption?.is_actually_drink) return <Chip tone="sky">unlabelled {e.pred_class}?</Chip>;
  return <Chip tone="amber">false alarm · {e.pred_class}</Chip>;
}

function DetectorErrors({ report }: { report: ModelErrors | null }) {
  const [type, setType] = useState<DeType>("all");
  const [classes, setClasses] = useState<string[]>([]);
  const [dists, setDists] = useState<string[]>([]);
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<"errors" | "recall" | "name">("errors");
  const [viewer, setViewer] = useState<string | null>(null);

  if (!report || !report.ready) {
    return (
      <Card title="What the base model misses">
        <Empty>Analyze with the base-model detector on to see which drinks it misses.</Empty>
      </Card>
    );
  }
  const s = report.summary;
  const matches = (e: DeError) =>
    (type === "all" || e.status === type) &&
    (!classes.length || classes.includes(errClass(e))) &&
    (!dists.length || dists.includes(e.caption?.distance || ""));
  const sources = report.sources
    .filter((src) => !q || src.filename.toLowerCase().includes(q.toLowerCase()))
    .map((src) => ({ src, errs: src.errors.filter(matches) }))
    .filter((x) => x.errs.length);
  sources.sort((a, b) =>
    sort === "name"
      ? a.src.filename.localeCompare(b.src.filename)
      : sort === "recall"
        ? a.src.counts.hit / Math.max(1, a.src.counts.gt) - b.src.counts.hit / Math.max(1, b.src.counts.gt)
        : b.errs.length - a.errs.length,
  );
  const typeCount = (t: DeType) => report.sources.reduce((n, src) => n + src.errors.filter((e) => t === "all" || e.status === t).length, 0);
  const toggle = (list: string[], v: string, set: (x: string[]) => void) => set(list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);
  const total = Math.max(1, s.gt);

  return (
    <Card
      title="What the base model misses"
      right={
        <span className="text-[11px] text-slate-400">
          {report.mode} mode · {report.detector.replace(/\.pt$/, "")} · conf {report.conf}
          {report.caption_model ? ` · ${report.caption_model}` : ""}
        </span>
      }
    >
      <div className="grid gap-4 lg:grid-cols-[1.2fr_1fr_1fr]">
        <div>
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-3xl font-black tabular-nums">{pct(s.hit / total)}</span>
            <span className="text-[12px] text-slate-500">recall</span>
          </div>
          <div className="text-[11px] text-slate-500">
            {s.hit} of {s.gt} GT drinks found · errors in {s.sources_with_errors}/{report.sources.length} images
          </div>
          <div className="mt-2 flex h-3 overflow-hidden rounded-full">
            <div className="bg-emerald-500" style={{ width: `${(s.hit / total) * 100}%` }} title={`found ${s.hit}`} />
            <div className="bg-rose-500" style={{ width: `${(s.missed / total) * 100}%` }} title={`missed ${s.missed}`} />
            <div className="bg-violet-500" style={{ width: `${(s.wrong_class / total) * 100}%` }} title={`wrong class ${s.wrong_class}`} />
          </div>
          <div className="mt-1 flex flex-wrap gap-2 text-[10px] text-slate-500">
            <span>● found {s.hit}</span>
            <span className="text-rose-500">● missed {s.missed}</span>
            <span className="text-violet-500">● wrong class {s.wrong_class}</span>
            <span className="text-amber-500">● false alarms {s.false_positive}</span>
            {s.hard_negatives != null && <span>hard negatives {s.hard_negatives}</span>}
          </div>
        </div>
        <div className="space-y-1.5">
          <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Error rate by class</div>
          {Object.entries(s.by_class || {}).map(([c, v]) => (
            <div key={c} className="flex items-center gap-2 text-[11px]">
              <span className="w-12 font-bold" style={{ color: CLASS_COLORS[c] }}>
                {c}
              </span>
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                <div
                  className={cn("h-full", v.error_rate >= 50 ? "bg-rose-500" : v.error_rate >= 30 ? "bg-amber-500" : "bg-emerald-500")}
                  style={{ width: `${Math.min(100, v.error_rate)}%` }}
                />
              </div>
              <span className="w-20 text-right tabular-nums text-slate-500">
                {v.error_rate.toFixed(0)}% {v.missed + v.wrong_class}/{v.gt}
              </span>
            </div>
          ))}
        </div>
        <div className="space-y-2">
          <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Confusions</div>
          <div className="flex flex-wrap gap-1">
            {Object.entries(s.confusions || {}).map(([k, n]) => (
              <Chip key={k} tone="violet">
                {k} ×{n}
              </Chip>
            ))}
            {!Object.keys(s.confusions || {}).length && <span className="text-[11px] text-slate-400">none</span>}
          </div>
          {s.false_alarm_objects_seen && Object.keys(s.false_alarm_objects_seen).length > 0 && (
            <>
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Fires on</div>
              <div className="flex flex-wrap gap-1">
                {Object.entries(s.false_alarm_objects_seen)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 4)
                  .map(([k, n]) => (
                    <Chip key={k} tone="amber">
                      {k} ×{n}
                    </Chip>
                  ))}
              </div>
            </>
          )}
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-slate-200 pt-3 dark:border-slate-800">
        {(
          [
            ["all", "All"],
            ["missed", "Missed"],
            ["wrong_class", "Wrong class"],
            ["false_positive", "False alarm"],
          ] as [DeType, string][]
        ).map(([t, l]) => (
          <button
            key={t}
            type="button"
            onClick={() => setType(t)}
            className={cn("rounded-full px-2.5 py-1 text-[11px] font-bold", type === t ? "bg-sky-600 text-white" : "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300")}
          >
            {l} {typeCount(t)}
          </button>
        ))}
        <span className="mx-1 h-4 w-px bg-slate-300 dark:bg-slate-700" />
        {["cup", "can", "bottle"].map((c) => (
          <button
            key={c}
            type="button"
            onClick={() => toggle(classes, c, setClasses)}
            className={cn("rounded-full border px-2 py-0.5 text-[11px] font-bold", classes.includes(c) ? "text-white" : "border-slate-300 dark:border-slate-600")}
            style={classes.includes(c) ? { background: CLASS_COLORS[c], borderColor: CLASS_COLORS[c] } : { color: CLASS_COLORS[c] }}
          >
            {c}
          </button>
        ))}
        <span className="mx-1 h-4 w-px bg-slate-300 dark:bg-slate-700" />
        {["near", "mid", "far", "very_far"].map((d) => (
          <button
            key={d}
            type="button"
            onClick={() => toggle(dists, d, setDists)}
            className={cn(
              "rounded-full border px-2 py-0.5 text-[11px] font-semibold",
              dists.includes(d) ? "border-sky-500 bg-sky-500/15 text-sky-700 dark:text-sky-300" : "border-slate-300 text-slate-500 dark:border-slate-600",
            )}
          >
            {d.replace("_", " ")}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-2">
          <label className="flex items-center gap-1 rounded-lg border border-slate-300 px-2 py-1 dark:border-slate-600">
            <Search className="h-3.5 w-3.5 text-slate-400" />
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="file…" className="w-28 bg-transparent text-[11px] outline-none" />
          </label>
          <select value={sort} onChange={(e) => setSort(e.target.value as typeof sort)} className="rounded-lg border border-slate-300 bg-transparent px-2 py-1 text-[11px] dark:border-slate-600">
            <option value="errors">most errors</option>
            <option value="recall">lowest recall</option>
            <option value="name">file name</option>
          </select>
        </div>
      </div>

      {sources.length === 0 ? (
        <div className="mt-3">
          <Empty>No errors match these filters.</Empty>
        </div>
      ) : (
        <div className="mt-3 grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
          {sources.map(({ src, errs }) => (
            <ErrorSourceCard key={src.filename} src={src} errs={errs} version={report.version} onView={() => setViewer(src.filename)} />
          ))}
        </div>
      )}
      {viewer && (
        <Lightbox
          panes={[{ label: viewer, src: img.overlay(viewer, report.version) }]}
          caption={<>red = missed · magenta = wrong class · yellow = false alarm · green = found</>}
          onClose={() => setViewer(null)}
        />
      )}
    </Card>
  );
}

function ErrorSourceCard({ src, errs, version, onView }: { src: DeSource; errs: DeError[]; version: string; onView: () => void }) {
  const [more, setMore] = useState(false);
  const [open, setOpen] = useState<number | null>(null);
  const recall = src.counts.hit / Math.max(1, src.counts.gt);
  const shown = more ? errs : errs.slice(0, 4);
  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 dark:border-slate-700">
      <button type="button" onClick={onView} className="relative block w-full bg-black">
        <img src={img.overlay(src.filename, version, 640)} alt={src.filename} loading="lazy" className="aspect-square w-full object-cover" />
        <span
          className={cn(
            "absolute right-2 top-2 rounded-md px-1.5 py-0.5 text-[10px] font-bold text-white",
            recall < 0.5 ? "bg-rose-600" : recall < 0.8 ? "bg-amber-600" : "bg-emerald-600",
          )}
        >
          {src.counts.hit}/{src.counts.gt} found
        </span>
      </button>
      <div className="p-2">
        <div className="flex flex-wrap items-center gap-1">
          <span className="mr-1 truncate text-[12px] font-bold">{src.filename.replace(/\.[^.]+$/, "")}</span>
          {src.counts.missed > 0 && <Chip tone="rose">{src.counts.missed} missed</Chip>}
          {src.counts.wrong_class > 0 && <Chip tone="violet">{src.counts.wrong_class} wrong</Chip>}
          {src.counts.false_alarm_objects > 0 && <Chip tone="amber">{src.counts.false_alarm_objects} false alarm</Chip>}
        </div>
        {src.caption_error && <div className="mt-1 text-[10px] text-amber-600">Gemini: {src.caption_error}</div>}
        <div className="mt-1 divide-y divide-slate-100 dark:divide-slate-800">
          {shown.map((e) => (
            <div key={e.id} className="py-1">
              <button type="button" onClick={() => setOpen(open === e.id ? null : e.id)} className="flex w-full items-center gap-1.5 text-left text-[11px]">
                {open === e.id ? <ChevronDown className="h-3 w-3 shrink-0" /> : <ChevronRight className="h-3 w-3 shrink-0" />}
                <span className="font-mono text-slate-400">#{e.id}</span>
                <ErrorTag e={e} />
                {e.caption?.distance && <Chip>{e.caption.distance.replace("_", " ")}</Chip>}
                <span className="min-w-0 flex-1 truncate text-slate-600 dark:text-slate-300">{e.caption?.object || e.reason}</span>
                <span className="shrink-0 tabular-nums text-slate-400">{e.frame_pct ? `${e.frame_pct.h.toFixed(1)}% h` : e.size_px ? `${e.size_px[0]}×${e.size_px[1]}` : ""}</span>
              </button>
              {open === e.id && (
                <dl className="ml-5 mt-1 grid grid-cols-[70px_1fr] gap-x-2 gap-y-0.5 text-[11px]">
                  {(
                    [
                      ["where", e.caption?.placement_detail],
                      ["depth", [e.caption?.depth_detail, e.caption?.relative_size].filter(Boolean).join(" — ")],
                      ["size", e.size_px ? `${e.size_px[0]}×${e.size_px[1]} px` : ""],
                      ["detector", e.reason],
                      ["why", e.caption?.likely_cause],
                      ["looks like", e.caption?.confusers],
                    ] as [string, string | null | undefined][]
                  )
                    .filter(([, v]) => v)
                    .map(([k, v]) => (
                      <div key={k} className="contents">
                        <dt className="text-slate-400">{k}</dt>
                        <dd className="text-slate-700 dark:text-slate-200">{v}</dd>
                      </div>
                    ))}
                </dl>
              )}
            </div>
          ))}
        </div>
        {errs.length > 4 && (
          <button type="button" onClick={() => setMore(!more)} className="mt-1 text-[11px] font-bold text-sky-600 hover:underline">
            {more ? "show less" : `+${errs.length - 4} more`}
          </button>
        )}
      </div>
    </div>
  );
}

function Samples({ regime, clearRegime }: { regime: string | null; clearRegime: () => void }) {
  const [q, setQ] = useState("");
  const [rows, setRows] = useState<RecordRow[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [view, setView] = useState<RecordRow | null>(null);
  useEffect(() => {
    const t = window.setTimeout(() => {
      studio
        .records(regime, q.trim())
        .then((r) => {
          setRows(r.items || []);
          setTotal(r.total ?? null);
        })
        .catch(() => setRows([]));
    }, 200);
    return () => window.clearTimeout(t);
  }, [regime, q]);
  return (
    <Card
      title="Samples"
      right={
        <>
          {regime && (
            <button type="button" onClick={clearRegime} className="inline-flex items-center gap-1 rounded-full bg-sky-500/15 px-2 py-0.5 text-[11px] font-bold text-sky-700 dark:text-sky-300">
              {regime} <X className="h-3 w-3" />
            </button>
          )}
          <span className="text-[11px] text-slate-400">{total != null ? `${rows.length} of ${total}` : rows.length}</span>
          <label className="flex items-center gap-1 rounded-lg border border-slate-300 px-2 py-1 dark:border-slate-600">
            <Search className="h-3.5 w-3.5 text-slate-400" />
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="file name…" className="w-32 bg-transparent text-[11px] outline-none" />
          </label>
        </>
      }
    >
      {rows.length === 0 ? (
        <Empty>No samples{regime ? " in this regime" : ""}.</Empty>
      ) : (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5 2xl:grid-cols-6">
          {rows.map((r) => (
            <button key={r.filename} type="button" onClick={() => setView(r)} className="overflow-hidden rounded-lg border border-slate-200 text-left dark:border-slate-700">
              <img src={img.annotated(r.filename, 480)} alt={r.filename} loading="lazy" className="aspect-square w-full bg-black object-cover" />
              <div className="px-1.5 py-1 text-[10px] leading-tight text-slate-500">
                <div className="truncate font-bold text-slate-700 dark:text-slate-200">{r.filename}</div>
                n={r.object_count} · {r.brightness.toFixed(2)} · {(r.regimes || []).slice(0, 2).join(", ")}
              </div>
            </button>
          ))}
        </div>
      )}
      {view && <Lightbox panes={[{ label: "GT boxes", src: img.annotated(view.filename) }]} caption={view.filename} onClose={() => setView(null)} />}
      {!rows.length && regime === null && q === "" && (
        <div className="mt-2 flex items-center gap-1 text-[11px] text-slate-400">
          <AlertTriangle className="h-3 w-3" /> Samples appear after the first analysis.
        </div>
      )}
    </Card>
  );
}
