import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, ChevronDown, ChevronRight, FolderUp, ScanSearch, Search, Upload, X } from "lucide-react";
import { cn } from "../lib/utils";
import { img, studio } from "./api";
import type { AnalyzeJob, CkptJob, DeError, DeSource, Gallery, GenMode, Health, MiningReport, ModelErrors, RecordRow, Summary, SwitchName } from "./types";
import { Banner, Btn, Card, Chip, CLASS_COLORS, Empty, errText, Lightbox, Progress, pct, useJob } from "./ui";

const MODE_LABEL: Record<GenMode, string> = { openai: "OpenAI", gemini: "Gemini" };
const GENERIC = new Set(["best", "final", "last", "weights", "train", "exp"]);

function ckptName(weights: string): string {
  const base = (weights.split("/").pop() || weights).replace(/\.pt$/, "");
  const [proj, model] = base.includes("__") ? base.split("__", 2) : ["", base];
  return GENERIC.has(model) && proj ? `${proj}/${model}` : model;
}

type Analysis = {
  summary: Summary | null;
  galleries: Gallery[];
  overlaps: Gallery[];
  mining: MiningReport | null;
  errors: ModelErrors | null;
};

export default function AnalyzeTab({ health, reloadHealth, onAnalyzed }: { health: Health | null; reloadHealth: () => void; onAnalyzed: () => void }) {
  const [mode, setMode] = useState<GenMode>("openai");
  const [dirty, setDirty] = useState<Record<GenMode, boolean>>({ openai: false, gemini: false });
  const [data, setData] = useState<Analysis>({ summary: null, galleries: [], overlaps: [], mining: null, errors: null });
  const [error, setError] = useState("");
  const [regime, setRegime] = useState<string | null>(null);

  useEffect(() => {
    if (health?.gen_mode) setMode(health.gen_mode);
  }, [health?.gen_mode]);

  const loadAnalysis = useCallback(async () => {
    try {
      const [summary, g, o, mining, errors] = await Promise.all([
        studio.summary(),
        studio.galleries(),
        studio.overlaps(),
        studio.mining().catch(() => null),
        studio.modelErrors().catch(() => null),
      ]);
      setData({
        summary,
        galleries: Object.values(g || {}).filter((x) => x && x.group !== "overlap"),
        overlaps: Object.values(o || {}).filter((x) => x && x.group === "overlap"),
        mining,
        errors,
      });
    } catch (e) {
      setError(errText(e));
    }
  }, []);

  useEffect(() => {
    void loadAnalysis();
  }, [loadAnalysis]);

  const analyzeJob = useJob<AnalyzeJob>(studio.analyzeJob, 800, (j) => {
    if (j.status === "done") {
      setDirty({ openai: false, gemini: false });
      reloadHealth();
      void loadAnalysis();
      onAnalyzed();
    }
  });
  const running = analyzeJob.job?.status === "running";

  const startAnalyze = async () => {
    setError("");
    try {
      analyzeJob.setJob(await studio.analyze(mode));
      analyzeJob.kick();
    } catch (e) {
      setError(errText(e));
    }
  };

  const stale = Boolean(health && (mode !== health.gen_mode || dirty[mode]));

  return (
    <div className="space-y-4">
      {error && <Banner kind="error">{error}</Banner>}
      <div className="grid gap-4 xl:grid-cols-[1.1fr_1fr]">
        <DatasetCard health={health} onUploaded={() => analyzeJob.kick()} busy={running} />
        <Card
          title="Detector & analysis"
          right={
            <Btn kind="primary" busy={running} onClick={() => void startAnalyze()} className={cn(stale && !running && "animate-pulse")}>
              <ScanSearch className="h-3.5 w-3.5" /> Analyze &amp; cluster
            </Btn>
          }
        >
          <ModePicker health={health} mode={mode} setMode={setMode} />
          {stale && !running && (
            <div className="mt-2 text-[11px] font-semibold text-amber-600 dark:text-amber-300">
              {mode !== health?.gen_mode ? `Analysis was made in ${MODE_LABEL[health!.gen_mode]} mode — re-analyze to switch.` : "New checkpoint — re-analyze to use it."}
            </div>
          )}
          {analyzeJob.job && analyzeJob.job.status !== "idle" && (
            <div className="mt-3 space-y-1">
              <div className="flex items-center justify-between text-[11px]">
                <span className={cn("font-semibold", analyzeJob.job.status === "error" ? "text-rose-600" : "text-slate-600 dark:text-slate-300")}>
                  {analyzeJob.job.status === "error" ? analyzeJob.job.error || analyzeJob.job.message : analyzeJob.job.message}
                </span>
                {running && <span className="tabular-nums text-slate-400">{Math.round(analyzeJob.job.progress * 100)}%</span>}
              </div>
              {running && <Progress value={analyzeJob.job.progress} tone="violet" />}
            </div>
          )}
          <CheckpointRow
            health={health}
            mode={mode}
            onChanged={() => {
              setDirty((d) => ({ ...d, [mode]: true }));
              reloadHealth();
            }}
          />
          <Switches
            health={health}
            onChanged={(name) => {
              if (name === "person_detector") setDirty({ openai: true, gemini: true });
              else if (name === "detector_openai") setDirty((d) => ({ ...d, openai: true }));
              else if (name === "detector_gemini") setDirty((d) => ({ ...d, gemini: true }));
              reloadHealth();
            }}
          />
        </Card>
      </div>

      <Kpis summary={data.summary} />

      <RegimeSection title="Scene regimes" items={data.galleries.filter((g) => g.group === "scene")} selected={regime} onSelect={setRegime} />
      <RegimeSection title="Class-hard regimes" items={data.galleries.filter((g) => g.group === "class_hard")} selected={regime} onSelect={setRegime} />
      <RegimeSection title="Overlap gates" items={data.overlaps} selected={regime} onSelect={setRegime} />

      <DetectorErrors report={data.errors} />

      <Card title="Mining recommendations">
        {data.mining?.recommendations?.length ? (
          <div className="grid gap-2 md:grid-cols-2">
            {data.mining.recommendations.map((r, i) => (
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
          <Empty>Run Analyze &amp; cluster to get recommendations.</Empty>
        )}
      </Card>

      <Samples regime={regime} clearRegime={() => setRegime(null)} />
    </div>
  );
}

function DatasetCard({ health, onUploaded, busy }: { health: Health | null; onUploaded: () => void; busy: boolean }) {
  const input = useRef<HTMLInputElement>(null);
  const [staged, setStaged] = useState<{ file: File; rel: string }[]>([]);
  const [phase, setPhase] = useState<"idle" | "uploading" | "done" | "error">("idle");
  const [msg, setMsg] = useState("");
  const [progress, setProgress] = useState(0);

  const counts = useMemo(() => {
    const imgs = staged.filter((f) => /\.(jpe?g|png|bmp|webp)$/i.test(f.rel)).length;
    const labels = staged.filter((f) => /\.txt$/i.test(f.rel)).length;
    return { imgs, labels };
  }, [staged]);

  const pick = (files: FileList | null) => {
    if (!files) return;
    const out: { file: File; rel: string }[] = [];
    for (const f of Array.from(files)) {
      const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name;
      if (/\.(jpe?g|png|bmp|webp|txt)$/i.test(rel) || /data\.ya?ml$/i.test(rel)) out.push({ file: f, rel });
    }
    setStaged(out);
    setPhase("idle");
    setMsg("");
  };

  const upload = async () => {
    setPhase("uploading");
    setMsg("");
    try {
      const CH = 40;
      for (let i = 0; i < staged.length; i += CH) {
        await studio.importChunk(staged.slice(i, i + CH), { replace: i === 0 });
        setProgress(Math.min(0.9, ((i + CH) / staged.length) * 0.9));
      }
      const r = await studio.importFinalize();
      setProgress(1);
      setPhase("done");
      setMsg(`Uploaded ${r.images} images + ${r.labels} labels — analysis started.`);
      setStaged([]);
      onUploaded();
    } catch (e) {
      setPhase("error");
      setMsg(errText(e));
    }
  };

  return (
    <Card title="Eval dataset (beverage_gt)" right={health && <Chip tone="sky">{health.images} images</Chip>}>
      <div className="space-y-3 text-[12px] text-slate-600 dark:text-slate-300">
        <p>
          The Studio learns the look of the challenge from these images. Refresh them from the labeler: <b>Platform workspace → Send GT</b> copies your Done GT
          here (it replaces this set), then run <b>Analyze &amp; cluster</b>.
        </p>
        <div
          role="button"
          tabIndex={0}
          onClick={() => input.current?.click()}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
          className="flex cursor-pointer items-center gap-3 rounded-xl border-2 border-dashed border-slate-300 px-4 py-3 hover:border-sky-400 hover:bg-sky-50/50 dark:border-slate-700 dark:hover:bg-sky-950/20"
        >
          <FolderUp className="h-6 w-6 shrink-0 text-sky-500" />
          <div className="min-w-0">
            <div className="font-bold text-slate-700 dark:text-slate-200">…or upload a YOLO folder (images/ + labels/)</div>
            <div className="text-[11px]">{staged.length ? `${counts.imgs} images · ${counts.labels} labels selected` : "Click to choose a folder. It replaces the current set."}</div>
          </div>
          {staged.length > 0 && (
            <button type="button" onClick={(e) => (e.stopPropagation(), setStaged([]))} className="ml-auto rounded p-1 hover:bg-slate-200 dark:hover:bg-slate-700">
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
        <input
          ref={input}
          type="file"
          className="hidden"
          multiple
          // @ts-expect-error non-standard attribute for folder picking
          webkitdirectory=""
          onChange={(e) => pick(e.target.files)}
        />
        {staged.length > 0 && (
          <div className="flex items-center gap-2">
            <Btn kind="primary" busy={phase === "uploading"} disabled={busy || counts.imgs === 0} onClick={() => void upload()}>
              <Upload className="h-3.5 w-3.5" /> Replace dataset with {counts.imgs} images
            </Btn>
            {counts.labels === 0 && <span className="text-[11px] text-amber-600">No labels found in that folder.</span>}
          </div>
        )}
        {phase === "uploading" && <Progress value={progress} />}
        {msg && <Banner kind={phase === "error" ? "error" : "ok"}>{msg}</Banner>}
      </div>
    </Card>
  );
}

function ModePicker({ health, mode, setMode }: { health: Health | null; mode: GenMode; setMode: (m: GenMode) => void }) {
  return (
    <div role="radiogroup" className="grid grid-cols-2 gap-2">
      {(["openai", "gemini"] as GenMode[]).map((m) => {
        const d = health?.detectors?.[m];
        const on = mode === m;
        const bad = d && (!d.exists || !d.enabled);
        return (
          <button
            key={m}
            type="button"
            role="radio"
            aria-checked={on}
            onClick={() => setMode(m)}
            title={
              d?.sv
                ? `SV ${d.sv.composite.toFixed(3)} · mAP50 ${d.sv.map50.toFixed(3)} · FP/img ${d.sv.fp_per_image.toFixed(2)}${d.gt_images ? ` · ${d.gt_images} GT` : ""}${d.uploaded_at ? ` · uploaded ${d.uploaded_at}` : ""}`
                : undefined
            }
            className={cn(
              "rounded-xl border px-3 py-2 text-left transition",
              on ? "border-violet-500 bg-violet-500/10 ring-1 ring-violet-500" : "border-slate-200 hover:border-violet-300 dark:border-slate-700",
            )}
          >
            <div className="flex items-center gap-2">
              <span className="text-[13px] font-extrabold">{MODE_LABEL[m]}</span>
              {health?.gen_mode === m && <Chip tone="green">analyzed</Chip>}
              {d?.source === "uploaded" && <Chip tone="violet">uploaded</Chip>}
            </div>
            <div className={cn("mt-0.5 truncate text-[11px]", bad ? "text-amber-600 dark:text-amber-300" : "text-slate-500")}>
              {d ? `${ckptName(d.weights)} · conf ${d.conf}${!d.exists ? " · missing" : !d.enabled ? " · off" : ""}` : "—"}
            </div>
          </button>
        );
      })}
    </div>
  );
}

function CheckpointRow({ health, mode, onChanged }: { health: Health | null; mode: GenMode; onChanged: () => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [msg, setMsg] = useState<{ kind: "info" | "ok" | "error"; text: string } | null>(null);
  const ckpt = useJob<CkptJob>(studio.ckptJob, 1000, (j) => {
    if (j.status === "done") {
      setMsg({ kind: "ok", text: j.message || "Checkpoint ready" });
      onChanged();
    } else if (j.status === "error") setMsg({ kind: "error", text: j.error || j.message });
  });
  const tuning = ckpt.job?.status === "running";
  const d = health?.detectors?.[mode];

  const upload = async (f: File) => {
    setMsg({ kind: "info", text: `Uploading ${f.name} (${(f.size / 1e6).toFixed(1)} MB)…` });
    try {
      ckpt.setJob(await studio.uploadCkpt(mode, f));
      ckpt.kick();
    } catch (e) {
      setMsg({ kind: "error", text: errText(e) });
    }
  };

  return (
    <div className="mt-3 rounded-xl bg-slate-50 p-3 dark:bg-slate-800/40">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Checkpoint</span>
        <Btn kind="secondary" busy={tuning} onClick={() => input.current?.click()}>
          <Upload className="h-3.5 w-3.5" /> Upload .pt for {MODE_LABEL[mode]}
        </Btn>
        {d?.source === "uploaded" && (
          <Btn
            kind="ghost"
            onClick={async () => {
              if (!window.confirm(`Use the default ${MODE_LABEL[mode]} checkpoint again (drops the uploaded one)?`)) return;
              try {
                await studio.resetCkpt(mode);
                onChanged();
              } catch (e) {
                setMsg({ kind: "error", text: errText(e) });
              }
            }}
          >
            reset to default
          </Btn>
        )}
        <input
          ref={input}
          type="file"
          accept=".pt"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void upload(f);
            e.target.value = "";
          }}
        />
      </div>
      {tuning && ckpt.job && (
        <div className="mt-2 space-y-1">
          <div className="text-[11px] text-slate-500">{ckpt.job.message}</div>
          <Progress value={ckpt.job.progress} tone="emerald" />
        </div>
      )}
      {msg && !tuning && <div className={cn("mt-2 text-[11px]", msg.kind === "error" ? "text-rose-600" : msg.kind === "ok" ? "text-emerald-600" : "text-slate-500")}>{msg.text}</div>}
    </div>
  );
}

const SWITCH_GROUPS: { title: string; items: { name: SwitchName; label: string; hint: string }[] }[] = [
  {
    title: "Use checkpoints",
    items: [
      { name: "detector_openai", label: "OpenAI-mode detector", hint: "Mines its errors for the prompts and auto-labels OpenAI runs" },
      { name: "detector_gemini", label: "Gemini-mode detector", hint: "Mines its errors for the prompts and auto-labels Gemini runs" },
      { name: "person_detector", label: "Person detector", hint: "Camera height / tilt lock and camera-drift check" },
    ],
  },
  {
    title: "CCTV look",
    items: [
      { name: "post_openai", label: "OpenAI runs", hint: "Blur + saturation + 224-colour palette PNG, like the challenge frames" },
      { name: "post_gemini", label: "Gemini runs", hint: "224-colour palette PNG" },
    ],
  },
];

function Switches({ health, onChanged }: { health: Health | null; onChanged: (name: SwitchName) => void }) {
  const [busy, setBusy] = useState<SwitchName | null>(null);
  const [err, setErr] = useState("");
  return (
    <div className="mt-3 grid gap-3 sm:grid-cols-2">
      {SWITCH_GROUPS.map((g) => (
        <div key={g.title}>
          <div className="mb-1 text-[11px] font-bold uppercase tracking-wider text-slate-400">{g.title}</div>
          {g.items.map((it) => {
            const on = Boolean(health?.checkpoint_switches?.[it.name]);
            return (
              <label key={it.name} title={it.hint} className="flex cursor-pointer items-center gap-2 py-0.5 text-[12px]">
                <button
                  type="button"
                  role="switch"
                  aria-checked={on}
                  disabled={busy === it.name || !health}
                  onClick={async () => {
                    setBusy(it.name);
                    setErr("");
                    try {
                      await studio.setSwitch(it.name, !on);
                      onChanged(it.name);
                    } catch (e) {
                      setErr(errText(e));
                    } finally {
                      setBusy(null);
                    }
                  }}
                  className={cn("relative h-4 w-7 shrink-0 rounded-full transition", on ? "bg-emerald-500" : "bg-slate-300 dark:bg-slate-600")}
                >
                  <span className={cn("absolute top-0.5 h-3 w-3 rounded-full bg-white shadow transition", on ? "left-3.5" : "left-0.5")} />
                </button>
                {it.label}
              </label>
            );
          })}
        </div>
      ))}
      {err && <div className="text-[11px] text-rose-600 sm:col-span-2">{err}</div>}
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
      <Card title="Detector errors">
        <Empty>Run Analyze &amp; cluster with a detector switched on to mine which beverages it misses.</Empty>
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
      title="Detector errors"
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
