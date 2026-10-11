import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Bot, Check, ChevronDown, DatabaseZap, FlaskConical, FolderUp, Images, Loader2, Rocket, ScanSearch, Sparkles, Upload, X } from "lucide-react";
import { cn } from "../lib/utils";
import { timeAgo } from "../store/useTeam";
import { img, studio } from "./api";
import { useStudio } from "./store";
import { ImagePickerModal, type PickItem } from "../components/ImagePicker";
import type { AnalyzeNeeded, GenMode, GtCandidate, SwitchName } from "./types";
import { Btn, Confirm, errText, GenBadge, Progress } from "./ui";

/** $ per 1024² image (OpenAI Batch API: low / medium / high). */
const PRICE: Record<string, [number, number, number]> = {
  "gpt-image-1": [0.012, 0.03, 0.09],
  "gpt-image-1.5": [0.01, 0.032, 0.08],
  "gpt-image-2": [0.008, 0.038, 0.11],
  "gpt-image-1-mini": [0.002, 0.005, 0.018],
};
const GEMINI_PRICE = 0.036; // gemini-3.1-flash-image batch, measured: $0.0358/image incl. prompt input (2026-10-10)
const CHUNK = 13;

type StepState = "done" | "todo" | "running" | "idle";

export function StepCard({
  n,
  title,
  state,
  summary,
  children,
  open,
  onToggle,
}: {
  n: number;
  title: string;
  state: StepState;
  summary: ReactNode;
  children: ReactNode;
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <section
      className={cn(
        "relative overflow-hidden rounded-2xl border bg-white shadow-sm transition dark:bg-ink-900",
        state === "todo" ? "border-amber-400/70 shadow-amber-500/10" : state === "running" ? "border-sky-400/70" : "border-slate-200 dark:border-slate-800",
      )}
    >
      {state === "running" && <div className="studio-shimmer absolute inset-x-0 top-0 h-0.5" />}
      <button type="button" onClick={onToggle} className="flex w-full items-center gap-3 px-4 py-3 text-left">
        <span
          className={cn(
            "flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[12px] font-black",
            state === "done"
              ? "bg-emerald-500 text-white"
              : state === "todo"
                ? "animate-pulse bg-amber-500 text-white"
                : state === "running"
                  ? "bg-sky-500 text-white"
                  : "bg-slate-200 text-slate-500 dark:bg-slate-700 dark:text-slate-300",
          )}
        >
          {state === "done" ? <Check className="h-4 w-4" /> : state === "running" ? <Loader2 className="h-4 w-4 animate-spin" /> : n}
        </span>
        <div className="min-w-0 flex-1">
          <div className="text-[13px] font-extrabold tracking-tight">{title}</div>
          <div className="truncate text-[11px] text-slate-500 dark:text-slate-400">{summary}</div>
        </div>
        <ChevronDown className={cn("h-4 w-4 shrink-0 text-slate-400 transition", open && "rotate-180")} />
      </button>
      {open && <div className="border-t border-slate-100 px-4 pb-4 pt-3 dark:border-slate-800">{children}</div>}
    </section>
  );
}

function useOpen(key: string, dflt: boolean) {
  const [open, setOpen] = useState(() => {
    try {
      const v = localStorage.getItem(`studio_open_${key}`);
      return v == null ? dflt : v === "1";
    } catch {
      return dflt;
    }
  });
  const toggle = () =>
    setOpen((o) => {
      try {
        localStorage.setItem(`studio_open_${key}`, o ? "0" : "1");
      } catch {
        /* private mode */
      }
      return !o;
    });
  return [open, toggle] as const;
}

// ---- 1 · eval set -------------------------------------------------------------------------------------------

export function EvalSetCard({ syncRef }: { syncRef: React.MutableRefObject<(() => void) | null> }) {
  const st = useStudio();
  const [open, toggle] = useOpen("evalset", false);
  const [latest, setLatest] = useState(110);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (st.status?.gt_images) setLatest(st.status.gt_images);
  }, [st.status?.gt_images]);

  const [picking, setPicking] = useState(false);

  /** keys: exactly these images (picker); otherwise the newest `latest` Done images. */
  const sync = async (keys?: string[]) => {
    setBusy(true);
    try {
      const r = await studio.syncGt(false, keys ? { item_keys: keys } : { latest_n: latest });
      const what = keys ? `${r.n_images} chosen` : `the newest ${r.n_images}`;
      st.setToast({ kind: "ok", text: `Copied ${what} Done Platform images. They are analyzed automatically before the next generation.` });
      setPicking(false);
      void st.loadStatus();
    } catch (e) {
      st.setToast({ kind: "error", text: errText(e) });
    } finally {
      setBusy(false);
    }
  };
  syncRef.current = () => void sync();

  const s = st.status;
  const state: StepState = busy ? "running" : !s?.gt_images ? "todo" : "done";
  return (
    <StepCard
      n={1}
      title="Eval set"
      state={state}
      open={open || state === "todo"}
      onToggle={toggle}
      summary={
        s?.gt_images ? (
          <>
            {s.gt_images} challenge frames · blocks {s.gt_block_min}–{s.gt_block_max}
            {s.gt_updated_at ? ` · ${timeAgo(s.gt_updated_at)}` : ""}
          </>
        ) : (
          "No frames yet"
        )
      }
    >
      <p className="text-[12px] text-slate-600 dark:text-slate-300">
        Real challenge frames with your labels. The Studio copies their look; every frame becomes a source in a final run, so keep it to the newest ones.
      </p>
      <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px]">
        <Btn kind="success" busy={busy && !picking} disabled={busy || st.analyze.job?.status === "running"} onClick={() => void sync()}>
          <DatabaseZap className="h-3.5 w-3.5" /> Update from labeler
        </Btn>
        <label className="flex items-center gap-1">
          newest
          <input
            type="number"
            min={10}
            max={2000}
            step={10}
            value={latest}
            onChange={(e) => setLatest(Math.max(10, Math.min(2000, Number(e.target.value) || 10)))}
            className="w-16 rounded-md border border-slate-300 bg-white px-1.5 py-0.5 text-right tabular-nums dark:border-slate-600 dark:bg-ink-800"
          />
          Done
        </label>
      </div>
      <div className="mt-2 flex items-center gap-2 text-[11px] text-slate-500 dark:text-slate-400">
        <Btn className="whitespace-nowrap" disabled={busy || st.analyze.job?.status === "running"} onClick={() => setPicking(true)}>
          <Images className="h-3.5 w-3.5" /> Choose images…
        </Btn>
        or pick the exact frames
      </div>
      {picking && <EvalPicker busy={busy} onUse={(keys) => void sync(keys)} onClose={() => setPicking(false)} />}
      <FolderUpload />
    </StepCard>
  );
}

/** Pick the eval set image by image from every Done Platform frame (the site's image picker). */
function EvalPicker({ busy, onUse, onClose }: { busy: boolean; onUse: (keys: string[]) => void; onClose: () => void }) {
  const [cands, setCands] = useState<GtCandidate[] | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    studio
      .gtCandidates()
      .then((r) => setCands(r.items))
      .catch((e) => setError(errText(e)));
  }, []);
  const items = useMemo<PickItem[]>(
    () =>
      (cands || []).map((c) => ({
        id: c.item_key,
        order: c.block,
        label: String(c.block),
        thumb: img.gt(c.block, "thumb"),
        preview: img.gt(c.block, "preview"),
        tag: c.in_eval ? "IN" : undefined,
        quality: c.quality,
        gen: c.generator ?? null,
      })),
    [cands],
  );
  const initial = useMemo(() => (cands || []).filter((c) => c.in_eval).map((c) => c.item_key), [cands]);
  // Portal: the left column is its own stacking context, so the picker would sit under the result cards.
  return createPortal(
    <ImagePickerModal
      layer="z-[70]"
      title="Choose the eval set"
      hint={
        <>
          Done Platform frames, newest first. <b>IN</b> = in the eval set now. Each picked frame becomes a source in a final run, so prefer recent
          frames that look like today's challenges.
        </>
      }
      items={items}
      initial={initial}
      minCount={10}
      loading={cands === null && !error}
      busy={busy}
      error={error}
      defaultLatest={initial.length || 110}
      onClose={onClose}
      actions={(ctx) => (
        <>
          <button
            type="button"
            disabled={ctx.small || ctx.n === 0 || busy}
            onClick={() => onUse(ctx.selected)}
            className="inline-flex items-center gap-1.5 rounded-md bg-emerald-600 px-4 py-2.5 text-[13px] font-bold text-white hover:bg-emerald-500 disabled:opacity-40"
          >
            {busy && <Loader2 className="h-4 w-4 animate-spin" />}
            Use these {ctx.n} images
          </button>
          <button type="button" onClick={onClose} className="rounded-md border border-slate-600 px-3 py-1.5 text-[12px] font-semibold hover:bg-slate-800">
            Cancel
          </button>
        </>
      )}
    />,
    document.body,
  );
}

function FolderUpload() {
  const st = useStudio();
  const input = useRef<HTMLInputElement>(null);
  const [staged, setStaged] = useState<{ file: File; rel: string }[]>([]);
  const [progress, setProgress] = useState<number | null>(null);
  const nImg = staged.filter((f) => /\.(jpe?g|png|bmp|webp)$/i.test(f.rel)).length;
  const nLab = staged.filter((f) => /\.txt$/i.test(f.rel)).length;

  const upload = async () => {
    setProgress(0);
    try {
      for (let i = 0; i < staged.length; i += 40) {
        await studio.importChunk(staged.slice(i, i + 40), { replace: i === 0 });
        setProgress(Math.min(0.9, ((i + 40) / staged.length) * 0.9));
      }
      const r = await studio.importFinalize();
      st.setToast({ kind: "ok", text: `Uploaded ${r.images} images + ${r.labels} labels — analyzing now.` });
      setStaged([]);
      st.analyze.kick();
      void st.loadStatus();
    } catch (e) {
      st.setToast({ kind: "error", text: errText(e) });
    } finally {
      setProgress(null);
    }
  };

  return (
    <details className="mt-3 text-[11px]">
      <summary className="cursor-pointer select-none font-semibold text-slate-500">or upload a YOLO folder</summary>
      <div className="mt-2 space-y-2">
        <button
          type="button"
          onClick={() => input.current?.click()}
          className="flex w-full items-center gap-2 rounded-xl border-2 border-dashed border-slate-300 px-3 py-2 text-left hover:border-sky-400 dark:border-slate-700"
        >
          <FolderUp className="h-5 w-5 text-sky-500" />
          {staged.length ? `${nImg} images · ${nLab} labels` : "Choose a folder with images/ + labels/"}
          {staged.length > 0 && <X className="ml-auto h-4 w-4" onClick={(e) => (e.stopPropagation(), setStaged([]))} />}
        </button>
        <input
          ref={input}
          type="file"
          className="hidden"
          multiple
          // @ts-expect-error non-standard attribute for folder picking
          webkitdirectory=""
          onChange={(e) =>
            setStaged(
              Array.from(e.target.files || [])
                .map((f) => ({ file: f, rel: (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name }))
                .filter((f) => /\.(jpe?g|png|bmp|webp|txt)$/i.test(f.rel) || /data\.ya?ml$/i.test(f.rel)),
            )
          }
        />
        {staged.length > 0 && (
          <Btn kind="primary" busy={progress != null} disabled={!nImg} onClick={() => void upload()}>
            <Upload className="h-3.5 w-3.5" /> Replace eval set with {nImg} images
          </Btn>
        )}
        {progress != null && <Progress value={progress} />}
      </div>
    </details>
  );
}

// ---- 2 · analyze --------------------------------------------------------------------------------------------

const ADVANCED: { name: SwitchName; label: string; hint: string }[] = [
  { name: "person_detector", label: "Camera lock (person detector)", hint: "Keeps camera height / tilt like the source and flags drift" },
  { name: "post_openai", label: "Platform palette on OpenAI runs", hint: "224-colour palette PNG, exactly like every challenge frame (no smoothing)" },
  { name: "post_gemini", label: "Platform palette on Gemini runs", hint: "224-colour palette PNG, exactly like every challenge frame (no smoothing)" },
];

export function AnalyzeCard({ analyzeRef }: { analyzeRef: React.MutableRefObject<((mode?: GenMode) => void) | null> }) {
  const st = useStudio();
  const [open, toggle] = useOpen("analyze", false);
  const h = st.health;
  const job = st.analyze.job;
  const running = job?.status === "running";
  const bm = h?.base_model;
  const a = h?.analysis;
  const mode = h?.gen_mode || "openai";
  const detectorOn = Boolean(h?.checkpoint_switches?.[`detector_${mode}` as SwitchName]);
  const outdated =
    !a?.at ||
    Boolean(st.status?.gt_updated_at && a.at < st.status.gt_updated_at - 1) ||
    Boolean(detectorOn && bm?.key && a.detector_key !== bm.key);

  const run = async (m: GenMode = mode) => {
    try {
      st.analyze.setJob(await studio.analyze(m));
      st.analyze.kick();
    } catch (e) {
      st.setToast({ kind: "error", text: errText(e) });
    }
  };
  analyzeRef.current = (m) => void run(m);

  const setDetector = async (on: boolean) => {
    try {
      await studio.setSwitch("detector_openai", on);
      await studio.setSwitch("detector_gemini", on);
      await st.loadHealth();
    } catch (e) {
      st.setToast({ kind: "error", text: errText(e) });
    }
  };

  // stale analysis is no longer a to-do: Test / Final run it first (with a confirmation)
  const state: StepState = running ? "running" : outdated ? "idle" : "done";
  return (
    <StepCard
      n={2}
      title="Analyze"
      state={state}
      open={open || running}
      onToggle={toggle}
      summary={
        running ? (
          job?.message
        ) : outdated ? (
          "Runs automatically before the next Test / Final"
        ) : a?.at ? (
          <>
            {timeAgo(a.at)}
            {a.detector ? ` · errors of ${a.detector}` : ""}
          </>
        ) : (
          "Not analyzed yet"
        )
      }
    >
      <div className="flex items-center gap-3 rounded-xl bg-gradient-to-r from-violet-500/10 to-sky-500/10 p-3">
        <Bot className="h-8 w-8 shrink-0 text-violet-500" />
        <div className="min-w-0 flex-1">
          <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Detector = labeler base model</div>
          <div className="truncate text-[13px] font-extrabold">{bm?.name || "—"}</div>
          <div className="text-[10px] text-slate-500">change it in the labeler · Select base model</div>
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={detectorOn}
          title={detectorOn ? "Finds its errors for the prompts and labels every run" : "Off: prompts use the eval set only; runs keep copied source labels"}
          onClick={() => void setDetector(!detectorOn)}
          className={cn("relative h-5 w-9 shrink-0 rounded-full transition", detectorOn ? "bg-emerald-500" : "bg-slate-300 dark:bg-slate-600")}
        >
          <span className={cn("absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition", detectorOn ? "left-4" : "left-0.5")} />
        </button>
      </div>
      <ul className="mt-3 space-y-1 text-[11px] text-slate-600 dark:text-slate-300">
        <li>• measures every frame (light, crowding, drink sizes, class mix) and groups them into regimes</li>
        <li>• {detectorOn ? <>runs <b>{bm?.name}</b> on them and turns each drink it misses into a prompt target</> : "detector off: no error mining, and runs keep the source's labels"}</li>
        <li>• writes one prompt per source (the generation package)</li>
      </ul>
      {running && job && (
        <div className="mt-3 space-y-1">
          <div className="flex justify-between text-[11px]">
            <span>{job.message}</span>
            <span className="tabular-nums">{Math.round(job.progress * 100)}%</span>
          </div>
          <Progress value={job.progress} tone="sky" />
        </div>
      )}
      {job?.status === "error" && <div className="mt-2 text-[11px] text-rose-600">{job.error || job.message}</div>}
      <div className="mt-3 flex items-center gap-2">
        <Btn kind="secondary" busy={running} onClick={() => void run()}>
          <ScanSearch className="h-3.5 w-3.5" /> Analyze now
        </Btn>
        <span className="text-[10px] text-slate-400">optional — Test / Final analyze first when needed</span>
      </div>
      <details className="mt-3 text-[11px]">
        <summary className="cursor-pointer select-none font-semibold text-slate-500">Advanced</summary>
        <div className="mt-2 space-y-1">
          {ADVANCED.map((s) => {
            const on = Boolean(h?.checkpoint_switches?.[s.name]);
            return (
              <label key={s.name} title={s.hint} className="flex cursor-pointer items-center gap-2">
                <input
                  type="checkbox"
                  checked={on}
                  onChange={async () => {
                    await studio.setSwitch(s.name, !on).catch((e) => st.setToast({ kind: "error", text: errText(e) }));
                    await st.loadHealth();
                  }}
                />
                {s.label}
              </label>
            );
          })}
        </div>
      </details>
    </StepCard>
  );
}

// ---- 3 · generate -------------------------------------------------------------------------------------------

export function GenerateCard({ startRef }: { startRef: React.MutableRefObject<((kind: "test" | "final") => void) | null> }) {
  const st = useStudio();
  const [open, toggle] = useOpen("generate", true);
  const h = st.health;
  const mode: GenMode = h?.gen_mode || "openai";
  const isGemini = mode === "gemini";
  const models = useMemo(() => (h?.image_models || []).filter((m) => m.provider === mode), [h, mode]);
  const [model, setModel] = useState("");
  const [quality, setQuality] = useState("medium");
  const [fidelity, setFidelity] = useState("");
  const [per, setPer] = useState(() => Number(localStorage.getItem("studio_per_source")) || 4);
  const [confirm, setConfirm] = useState<null | "test" | "final">(null);
  const [need, setNeed] = useState<AnalyzeNeeded | null>(null);
  const pending = useRef<StartArgs | null>(null);
  const canSpend = Boolean(st.status?.can_spend);
  const genRunning = st.gen.job?.status === "running";
  const analyzing = st.analyze.job?.status === "running";

  useEffect(() => {
    if (!models.length) return;
    const saved = localStorage.getItem("studio_image_model") || "";
    setModel((cur) => (models.some((m) => m.id === cur) ? cur : (models.find((m) => m.id === saved) || models.find((m) => m.id === h?.default_model) || models[0]).id));
  }, [models, h?.default_model]);

  const unit = isGemini ? GEMINI_PRICE : (PRICE[model] || PRICE["gpt-image-2"])[quality === "low" ? 0 : quality === "high" ? 2 : 1];
  const nFinal = st.nFinal;
  const nAll = Math.max(st.sources.length, st.nAllSources);
  const sameGen = st.sources.filter((s) => st.genOf(s.source_image) === mode).map((s) => s.source_image);
  const source = st.sources.find((s) => s.source_image === st.source) || st.sources[0] || null;
  // Test on several sources at once: the sources ticked on the Plan tab (2–20), like a small Final without HF upload.
  const [testTicked, setTestTicked] = useState(false);
  const ticked = st.finalSel && st.finalSel.length >= 2 && st.finalSel.length <= 20 ? st.finalSel : null;
  const multiTest = Boolean(testTicked && ticked);
  const nTest = multiTest && ticked ? ticked.length : 1;

  const refreshNeed = useCallback(() => studio.analyzeNeeded().then(setNeed).catch(() => setNeed(null)), []);
  useEffect(() => {
    void refreshNeed();
  }, [refreshNeed, st.status?.gt_updated_at, h?.analysis?.at, h?.base_model?.key, mode]);

  const argsFor = (kind: "test" | "final"): StartArgs => ({
    mode: kind,
    model,
    per_source: per,
    source_image: kind === "test" ? source?.source_image || null : null,
    source_images: kind === "final" ? st.finalSel : multiTest ? ticked : null,
    quality: isGemini ? null : quality,
    input_fidelity: isGemini ? null : fidelity,
  });
  const startWith = async (args: StartArgs) => {
    try {
      st.gen.setJob(await studio.start(args));
      st.gen.kick();
      const what = args.mode === "final" ? "Final run submitted — results arrive within minutes to a few hours." : `Test started on ${args.source_images?.length ? `${args.source_images.length} sources` : args.source_image}.`;
      st.setToast({ kind: "info", text: what });
    } catch (e) {
      st.setToast({ kind: "error", text: errText(e) });
    }
  };
  /** Confirmed in the dialog: analyse first when needed (generation then starts by itself), else generate now. */
  const launch = async (kind: "test" | "final") => {
    setConfirm(null);
    const args = argsFor(kind);
    if (!need?.needed) return startWith(args);
    try {
      pending.current = args;
      st.analyze.setJob(await studio.analyze(mode));
      st.analyze.kick();
      st.setToast({ kind: "info", text: "Analyzing first — generation starts by itself when it finishes." });
    } catch (e) {
      pending.current = null;
      st.setToast({ kind: "error", text: errText(e) });
    }
  };
  // the analysis started by launch() finished → start the generation that was waiting for it
  const aStatus = st.analyze.job?.status;
  useEffect(() => {
    const args = pending.current;
    if (!args || (aStatus !== "done" && aStatus !== "error")) return;
    pending.current = null;
    if (aStatus === "error") {
      st.setToast({ kind: "error", text: `Analysis failed — nothing was generated. ${st.analyze.job?.error || ""}` });
      return;
    }
    void (async () => {
      // the eval set may have changed: keep only sources that are still in the new package
      const pkg = await studio.pkg().catch(() => null);
      const have = new Set((pkg?.sources || []).map((x) => x.source_image));
      const fixed = { ...args };
      if (fixed.source_images) fixed.source_images = fixed.source_images.filter((x) => have.has(x));
      if (fixed.mode === "final" && fixed.source_images && !fixed.source_images.length) fixed.source_images = null;
      if (fixed.mode === "test" && !fixed.source_images?.length) {
        fixed.source_images = null;
        if (!fixed.source_image || !have.has(fixed.source_image)) fixed.source_image = pkg?.sources?.[0]?.source_image || null;
      }
      await st.loadData();
      void refreshNeed();
      await startWith(fixed);
    })();
  }, [aStatus]); // eslint-disable-line react-hooks/exhaustive-deps

  const ask = (kind: "test" | "final") => {
    void refreshNeed();
    setConfirm(kind);
  };
  startRef.current = ask;
  const analysisNote = need?.needed ? <AnalyzeFirst need={need} /> : null;
  const plus = need?.needed ? ` + analysis ≤ $${need.est_max_usd.toFixed(2)}` : "";

  const job = st.gen.job;
  const state: StepState = genRunning ? "running" : st.runs.length ? "done" : "idle";
  return (
    <StepCard
      n={3}
      title="Generate"
      state={state}
      open={open || genRunning}
      onToggle={toggle}
      summary={genRunning ? job?.message : `${isGemini ? "Gemini" : "OpenAI"} · ${model || "—"} · ${per}/source · ≈ $${unit.toFixed(3)}/image`}
    >
      <div className="space-y-3 text-[12px]">
        <div className="grid grid-cols-2 gap-1 rounded-xl bg-slate-100 p-1 dark:bg-slate-800">
          {(["openai", "gemini"] as GenMode[]).map((p) => (
            <button
              key={p}
              type="button"
              disabled={analyzing || genRunning}
              onClick={async () => {
                if (p === mode) return;
                try {
                  await studio.setMode(p);
                  await st.loadHealth();
                } catch (e) {
                  st.setToast({ kind: "error", text: errText(e) });
                }
              }}
              className={cn(
                "flex items-center justify-center gap-1.5 rounded-lg py-1.5 text-[12px] font-bold transition disabled:opacity-50",
                p === mode ? (p === "gemini" ? "bg-violet-600 text-white shadow" : "bg-sky-600 text-white shadow") : "text-slate-500 hover:text-slate-800 dark:hover:text-slate-200",
              )}
            >
              <Sparkles className="h-3.5 w-3.5" /> {p === "openai" ? "OpenAI" : "Gemini"}
            </button>
          ))}
        </div>
        <div className="grid grid-cols-2 gap-2">
          <label className="col-span-2">
            <span className="mb-0.5 block text-[10px] font-bold uppercase tracking-wider text-slate-400">Model</span>
            <select
              value={model}
              onChange={(e) => (setModel(e.target.value), localStorage.setItem("studio_image_model", e.target.value))}
              className="w-full rounded-lg border border-slate-300 bg-white px-2 py-1.5 dark:border-slate-600 dark:bg-ink-800"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          {!isGemini && (
            <>
              <label>
                <span className="mb-0.5 block text-[10px] font-bold uppercase tracking-wider text-slate-400">Quality</span>
                <select value={quality} onChange={(e) => setQuality(e.target.value)} className="w-full rounded-lg border border-slate-300 bg-white px-2 py-1.5 dark:border-slate-600 dark:bg-ink-800">
                  <option value="low">low</option>
                  <option value="medium">medium</option>
                  <option value="high">high · ~3×</option>
                </select>
              </label>
              <label>
                <span className="mb-0.5 block text-[10px] font-bold uppercase tracking-wider text-slate-400">Fidelity</span>
                <select value={fidelity} onChange={(e) => setFidelity(e.target.value)} className="w-full rounded-lg border border-slate-300 bg-white px-2 py-1.5 dark:border-slate-600 dark:bg-ink-800">
                  <option value="">default</option>
                  <option value="low">low</option>
                  <option value="high">high · closer</option>
                </select>
              </label>
            </>
          )}
        </div>
        <label className="block">
          <span className="mb-0.5 flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-400">
            <span>Images per source</span>
            <span className="text-slate-700 dark:text-slate-200">{per}</span>
          </span>
          <input
            type="range"
            min={1}
            max={20}
            value={per}
            onChange={(e) => (setPer(Number(e.target.value)), localStorage.setItem("studio_per_source", e.target.value))}
            className="w-full accent-violet-600"
          />
        </label>

        <div className="flex items-center gap-2 rounded-xl border border-slate-200 p-2 dark:border-slate-700">
          {source ? (
            <span className="relative shrink-0">
              <img src={img.dataset(source.source_image, 320)} alt="" className="h-12 w-12 rounded-lg object-cover" />
              <GenBadge gen={st.genOf(source.source_image)} className="absolute -bottom-1 -right-1" />
            </span>
          ) : (
            <div className="h-12 w-12 rounded-lg bg-slate-200 dark:bg-slate-700" />
          )}
          <div className="min-w-0 flex-1 text-[11px]">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Test on</span>
              {ticked && (
                <span className="flex overflow-hidden rounded-md border border-slate-300 text-[10px] font-bold dark:border-slate-600">
                  {[false, true].map((multi) => (
                    <button
                      key={String(multi)}
                      type="button"
                      onClick={() => setTestTicked(multi)}
                      className={cn("px-1.5 py-0.5", testTicked === multi ? "bg-violet-600 text-white" : "text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-700")}
                    >
                      {multi ? `${ticked.length} ticked` : "this one"}
                    </button>
                  ))}
                </span>
              )}
            </div>
            {multiTest ? (
              <>
                <div className="truncate font-mono">{ticked?.length} sources ticked on the Plan tab</div>
                <div className="truncate text-slate-500">{ticked?.map((x) => x.replace(/^block_|\.png$/g, "")).join(", ")}</div>
              </>
            ) : (
              <>
                <div className="truncate font-mono">{source?.source_image || "—"}</div>
                <div className="truncate text-slate-500">
                  {source ? `${source.regimes.split("|").slice(0, 2).join(", ")}${source.detector_errors ? ` · ${source.detector_errors} misses targeted` : ""}` : "pick one in the Plan tab"}
                </div>
              </>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 rounded-xl border border-slate-200 p-2 text-[11px] dark:border-slate-700">
          <div className="min-w-0 flex-1">
            <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Final sources</div>
            <div className="truncate">
              {st.finalSel ? (
                <>
                  <b>{nFinal}</b> of {nAll} ticked
                </>
              ) : (
                <>
                  all <b>{nAll}</b> eval-set frames
                </>
              )}
            </div>
          </div>
          <button type="button" onClick={() => window.dispatchEvent(new CustomEvent("studio-open-tab", { detail: "plan" }))} className="font-bold text-violet-600 hover:underline dark:text-violet-300">
            choose
          </button>
          {st.finalSel && (
            <button type="button" onClick={() => st.setFinalSel(null)} className="font-semibold text-slate-500 hover:underline">
              use all
            </button>
          )}
          {sameGen.length > 0 && sameGen.length < st.sources.length && (
            <button
              type="button"
              onClick={() => st.setFinalSel(sameGen)}
              className="font-semibold text-slate-500 hover:underline"
              title={`Tick only the eval frames that ${isGemini ? "Gemini" : "OpenAI"} made`}
            >
              only {isGemini ? "Gemini" : "OpenAI"} frames ({sameGen.length})
            </button>
          )}
        </div>

        <div className="grid grid-cols-2 gap-2">
          <Btn kind="secondary" disabled={!canSpend || genRunning || analyzing || (!source && !multiTest)} onClick={() => ask("test")} className="flex-col gap-0 py-2">
            <span className="flex items-center gap-1">
              <FlaskConical className="h-3.5 w-3.5" /> Test
            </span>
            <span className="text-[10px] font-semibold opacity-70">
              {nTest * per} img · ≈ ${(unit * nTest * per).toFixed(2)}
            </span>
          </Btn>
          <Btn kind="primary" disabled={!canSpend || genRunning || analyzing || nFinal === 0} onClick={() => ask("final")} className="flex-col gap-0 py-2">
            <span className="flex items-center gap-1">
              <Rocket className="h-3.5 w-3.5" /> Final
            </span>
            <span className="text-[10px] font-semibold opacity-80">
              {nFinal * per} img · ≈ ${(unit * nFinal * per).toFixed(2)}
            </span>
          </Btn>
        </div>
        {need?.needed && !analyzing && (
          <div className="rounded-lg bg-amber-500/10 px-2 py-1.5 text-[11px] text-amber-700 dark:text-amber-300">
            Analysis runs first when you generate — {need.reasons.join("; ")}.
          </div>
        )}
        {analyzing && pending.current && <div className="text-[11px] text-sky-600 dark:text-sky-300">Analyzing… the {pending.current.mode} run starts by itself afterwards.</div>}
        {!canSpend && <div className="text-[11px] text-amber-600 dark:text-amber-300">Only {st.status?.allowed_users.join(", ")} can start paid runs.</div>}
        {genRunning && job && (
          <div className="space-y-1">
            <div className="flex justify-between text-[11px]">
              <span className="truncate">{job.message}</span>
              <span className="tabular-nums">
                {job.done}/{job.total || "?"}
              </span>
            </div>
            <Progress value={job.progress} tone="violet" />
          </div>
        )}
      </div>

      {confirm === "test" && multiTest && (
        <Confirm
          title={`Test on ${nTest} sources?`}
          confirmLabel={`${need?.needed ? "Analyze + generate" : "Generate"} ${nTest * per} images · ≈ $${(unit * nTest * per).toFixed(2)}${plus}`}
          onCancel={() => setConfirm(null)}
          onConfirm={() => launch("test")}
        >
          {analysisNote}
          <p>
            <b>{model}</b> makes {per} new scenes for each of the {nTest} sources ticked on the Plan tab ({nTest} × {per} = <b>{nTest * per}</b> images).
          </p>
          <p>A test run: nothing is uploaded to Hugging Face or sent to Own.</p>
        </Confirm>
      )}
      {confirm === "test" && !multiTest && source && (
        <Confirm
          title="Test on one source?"
          confirmLabel={`${need?.needed ? "Analyze + generate" : "Generate"} ${per} images · ≈ $${(unit * per).toFixed(2)}${plus}`}
          onCancel={() => setConfirm(null)}
          onConfirm={() => launch("test")}
        >
          {analysisNote}
          <div className="flex items-center gap-3">
            <img src={img.dataset(source.source_image, 320)} alt="" className="h-20 w-20 rounded-lg object-cover" />
            <p>
              <b>{model}</b> makes {per} new scenes in the style of <code>{source.source_image}</code>.
            </p>
          </div>
        </Confirm>
      )}
      {confirm === "final" && (
        <Confirm
          title={st.finalSel ? `Generate on ${nFinal} chosen sources?` : "Generate the full set?"}
          confirmLabel={`${need?.needed ? "Analyze + spend" : "Spend"} ≈ $${(unit * nFinal * per).toFixed(2)}${plus}`}
          onCancel={() => setConfirm(null)}
          onConfirm={() => launch("final")}
        >
          {analysisNote}
          <p>
            <b>{model}</b>
            {!isGemini && <> · {quality}</>} makes <b>{nFinal * per}</b> images ({nFinal} {st.finalSel ? `ticked of ${nAll} ` : ""}sources × {per}).
          </p>
          <p>
            Estimated <b className="text-amber-600 dark:text-amber-300">≈ ${(unit * nFinal * per).toFixed(2)}</b>
            {!isGemini && <> in {Math.ceil(nFinal / CHUNK)} batches</>}. Results arrive within minutes to a few hours — you can close this page.
          </p>
        </Confirm>
      )}
    </StepCard>
  );
}

type StartArgs = Parameters<typeof studio.start>[0];

/** The "analysis runs first" note inside the generate confirmations. */
function AnalyzeFirst({ need }: { need: AnalyzeNeeded }) {
  return (
    <div className="rounded-lg border border-amber-400/60 bg-amber-500/10 p-2.5 text-[12px]">
      <b>The analysis runs first</b> — {need.reasons.join("; ")}. It measures the {need.frames} eval frames, finds what the base model
      misses and writes the prompts; Gemini describes {need.new_frames ? `the ${need.new_frames} new frames` : "the frames"} (cached
      afterwards) for <b>≈ up to ${need.est_max_usd.toFixed(2)}</b>. Generation then starts by itself.
    </div>
  );
}
