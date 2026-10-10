import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { cn } from "../lib/utils";
import { img, studio } from "./api";
import { useStudio } from "./store";
import type { Distribution, ModelErrors, Package, PackageSource } from "./types";
import { Banner, Btn, Card, Empty, errText } from "./ui";

/** Plan tab: the generation package — which sources, how the eval set is mixed, what each source targets. */
export default function PlanPanel() {
  const st = useStudio();
  const [dist, setDist] = useState<Distribution | null>(null);
  const [busy, setBusy] = useState(false);
  const [per, setPer] = useState(() => Number(localStorage.getItem("studio_per_source")) || 4);
  useEffect(() => {
    studio.distribution().then(setDist).catch(() => setDist(null));
  }, [st.health?.analysis?.at]);
  const selected = st.sources.find((s) => s.source_image === st.source) || st.sources[0] || null;

  const rebuild = async () => {
    setBusy(true);
    try {
      const r = await studio.prepare(per);
      st.setToast({ kind: "ok", text: `Package rebuilt: ${r.n_sources} sources · ${r.total_targets} targets` });
      await st.loadData();
    } catch (e) {
      st.setToast({ kind: "error", text: errText(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="grid gap-4 2xl:grid-cols-[1.4fr_1fr]">
        <SourceCards sources={st.sources} selected={selected?.source_image || ""} onPick={st.pickSource} pkg={st.pkg} />
        <WrongPanel source={selected} errors={st.errors} />
      </div>
      <div className="grid gap-4 xl:grid-cols-[1fr_auto]">
        <QuotaTable dist={dist} />
        <Card title="Package">
          <div className="space-y-2 text-[12px] text-slate-600 dark:text-slate-300">
            <p className="max-w-xs">Analyze rebuilds the package automatically. Rebuild by hand only to change how many images each source plans.</p>
            <label className="flex items-center gap-2">
              per source
              <input
                type="number"
                min={1}
                max={20}
                value={per}
                onChange={(e) => setPer(Math.max(1, Math.min(20, Number(e.target.value) || 1)))}
                className="w-14 rounded-md border border-slate-300 bg-white px-1.5 py-0.5 text-right dark:border-slate-600 dark:bg-ink-800"
              />
            </label>
            <Btn kind="secondary" busy={busy} onClick={() => void rebuild()}>
              <RefreshCw className="h-3.5 w-3.5" /> Rebuild package
            </Btn>
          </div>
        </Card>
      </div>
    </div>
  );
}

function SourceCards({ sources, selected, onPick, pkg }: { sources: PackageSource[]; selected: string; onPick: (s: string) => void; pkg: Package | null }) {
  return (
    <Card
      title="Sources · click one to test it"
      right={
        pkg?.ready ? (
          <span className="text-[11px] text-slate-400">
            {pkg.n_sources} sources · {pkg.per_source}/source · {pkg.total_targets} targets
          </span>
        ) : undefined
      }
    >
      {!pkg?.ready ? (
        <Empty>No package yet — run Analyze &amp; cluster, then Rebuild package.</Empty>
      ) : (
        <div className="grid grid-cols-3 gap-2 sm:grid-cols-4 lg:grid-cols-6 2xl:grid-cols-8">
          {sources.slice(0, 64).map((s) => (
            <button
              key={s.source_image}
              type="button"
              onClick={() => onPick(s.source_image)}
              className={cn("overflow-hidden rounded-lg border text-left", selected === s.source_image ? "border-violet-500 ring-2 ring-violet-500/40" : "border-slate-200 dark:border-slate-700")}
            >
              <div className="relative">
                <img src={img.dataset(s.source_image, 320)} alt={s.source_image} loading="lazy" className="aspect-square w-full bg-black object-cover" />
                {s.detector_errors > 0 && <span className="absolute right-1 top-1 rounded bg-rose-600 px-1 text-[9px] font-bold text-white">{s.detector_errors} err</span>}
              </div>
              <div className="px-1 py-0.5 text-[9px] leading-tight">
                <div className="truncate font-mono">{s.source_image.replace(/\.png$/, "")}</div>
                <div className="truncate text-slate-400">{s.regimes.split("|").slice(0, 2).join(", ")}</div>
              </div>
            </button>
          ))}
        </div>
      )}
      {sources.length > 64 && <div className="mt-2 text-[11px] text-slate-400">+{sources.length - 64} more (use the search on the left)</div>}
    </Card>
  );
}

function WrongPanel({ source, errors }: { source: PackageSource | null; errors: ModelErrors | null }) {
  if (!errors || !errors.ready) {
    return (
      <Card title="What the base model misses here">
        <Empty>Analyze with the base-model detector on to see what it misses on each source.</Empty>
      </Card>
    );
  }
  if (!source)
    return (
      <Card title="What the base model misses here">
        <Empty>Pick a source to see its mistakes — the prompts target them.</Empty>
      </Card>
    );
  const src = errors.sources.find((s) => s.filename === source.source_image);
  return (
    <Card title={<>What the base model misses · <span className="font-mono text-[11px] font-normal">{source.source_image}</span></>}>
      {!src || !src.errors.length ? (
        <Banner kind="ok">No wrong beverages here — the detector found all GT drinks.</Banner>
      ) : (
        <div className="space-y-2">
          <a href={img.overlay(src.filename, errors.version)} target="_blank" rel="noreferrer">
            <img src={img.overlay(src.filename, errors.version, 800)} alt="" className="w-full rounded-lg" />
          </a>
          <div className="text-[11px] text-slate-500">
            {src.counts.missed} missed · {src.counts.wrong_class} wrong class · {src.counts.false_alarm_objects} false alarms · {src.counts.hit}/{src.counts.gt} found
          </div>
          <table className="w-full text-[11px]">
            <thead className="text-left text-[10px] uppercase text-slate-400">
              <tr>
                <th>#</th>
                <th>Error</th>
                <th>Distance</th>
                <th>Looks like</th>
                <th>Size</th>
              </tr>
            </thead>
            <tbody>
              {src.errors.map((e) => (
                <tr key={e.id} className="border-t border-slate-100 align-top dark:border-slate-800" title={[e.reason, e.caption?.likely_cause, e.caption?.confusers].filter(Boolean).join(" — ")}>
                  <td className="py-1 font-mono text-slate-400">{e.id}</td>
                  <td className="py-1">
                    {e.status === "missed" ? `missed ${e.gt_class}` : e.status === "wrong_class" ? `${e.gt_class} → ${e.pred_class}` : e.caption?.is_actually_drink ? `unlabelled ${e.pred_class}?` : `false alarm: ${e.pred_class}`}
                  </td>
                  <td className="py-1">{e.caption?.distance?.replace("_", " ") || "—"}</td>
                  <td className="py-1">{e.caption?.object || "—"}</td>
                  <td className="py-1 tabular-nums">{e.size_px ? `${e.size_px[0]}×${e.size_px[1]}` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}


function QuotaTable({ dist }: { dist: Distribution | null }) {
  const rows = [...(dist?.galleries || []), ...(dist?.overlaps || [])];
  return (
    <Card title="Distribution of the eval set" right={dist?.eval_total ? <span className="text-[11px] text-slate-400">{dist.eval_total} images</span> : undefined}>
      {rows.length === 0 ? (
        <Empty>Analyze first to see the regime mix.</Empty>
      ) : (
        <div className="space-y-1">
          {rows
            .sort((a, b) => b.eval_pct - a.eval_pct)
            .map((r) => (
              <div key={r.key} className="grid grid-cols-[1fr_40px_44px_100px] items-center gap-2 text-[11px]" title={r.why}>
                <span className="truncate">
                  <span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: r.color }} />
                  {r.title || r.key}
                </span>
                <span className="text-right tabular-nums text-slate-500">{r.eval_count}</span>
                <span className="text-right tabular-nums">{r.eval_pct.toFixed(1)}%</span>
                <div className="h-1.5 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                  <div className="h-full rounded-full" style={{ width: `${Math.min(100, r.eval_pct)}%`, background: r.color }} />
                </div>
              </div>
            ))}
        </div>
      )}
    </Card>
  );
}

