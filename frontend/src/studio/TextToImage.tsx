import { useCallback, useEffect, useMemo, useState } from "react";
import { Clock, FlaskConical, Rocket, ScanSearch, Wand2, Zap } from "lucide-react";
import { cn } from "../lib/utils";
import { img, studio } from "./api";
import { useStudio } from "./store";
import type { T2iJob, T2iPlan, T2iRecipe, T2iStatus } from "./types";
import { Banner, Btn, Card, Chip, Confirm, Empty, errText, GenBadge, Progress, useJob } from "./ui";

/** $ per text → image request through the Batch API (low / medium / high): the image-to-image price minus the
 *  reference image and the longer prompt (≈ $0.0086 less), measured token counts 2026-10-14. Instant = 2×. */
const T2I_PRICE: Record<string, [number, number, number]> = {
  "gpt-image-2.5-sunburst": [0.0044, 0.0094, 0.0294],
  "gpt-image-2.5-flare": [0.0044, 0.0094, 0.0294],
  "gpt-image-2": [0.0064, 0.029, 0.108],
};
const T2I_GEMINI = 0.034; // gemini-3.1-flash-image batch, 1K image out; the text-only prompt costs < $0.001
const INSTANT_X = 2;
const PAGE = 24;

/** Text → Image tab: deep analysis of the selected eval frames → exactly N recipes → generate them. */
export default function TextToImage() {
  const st = useStudio();
  const [status, setStatus] = useState<T2iStatus | null>(null);
  const [plan, setPlan] = useState<T2iPlan | null>(null);
  const refresh = useCallback(async () => {
    const [s, p] = await Promise.all([studio.t2iStatus().catch(() => null), studio.t2iPlanGet().catch(() => null)]);
    setStatus(s);
    setPlan(p && "recipes" in p ? (p as T2iPlan) : null);
    return s;
  }, []);
  const job = useJob<T2iJob>(studio.t2iJob, 1500, (j) => {
    st.setToast(j.status === "done" ? { kind: "ok", text: j.message } : { kind: "error", text: j.error || j.message });
    void refresh();
    void st.loadData();
  });
  useEffect(() => {
    void refresh().then((s) => {
      if (s?.job.status === "running") job.kick();
    });
  }, [refresh, st.status?.gt_updated_at, st.health?.analysis?.at]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!status) return <Empty>Loading Text → Image…</Empty>;
  return (
    <div className="space-y-4">
      <AnalyzeStep status={status} job={job.job} onStart={async () => {
        try {
          job.setJob(await studio.t2iAnalyze());
          job.kick();
        } catch (e) {
          st.setToast({ kind: "error", text: errText(e) });
        }
      }} />
      <PlanStep status={status} plan={plan} onPlan={setPlan} />
      {plan && <GenerateStep plan={plan} />}
    </div>
  );
}

function StepHead({ n, title, sub }: { n: number; title: string; sub: string }) {
  return (
    <div className="flex items-start gap-3">
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-violet-600 text-[12px] font-black text-white">{n}</span>
      <div>
        <div className="text-[13px] font-extrabold text-slate-800 dark:text-slate-100">{title}</div>
        <div className="text-[11px] text-slate-500">{sub}</div>
      </div>
    </div>
  );
}

// ---- 1 · deep analysis --------------------------------------------------------------------------------------

function AnalyzeStep({ status, job, onStart }: { status: T2iStatus; job: T2iJob | null; onStart: () => Promise<void> }) {
  const st = useStudio();
  const [ask, setAsk] = useState(false);
  const running = (job || status.job).status === "running";
  const j = running ? job || status.job : null;
  const todo = status.missing_cards > 0 || status.base.needed;
  const canSpend = Boolean(st.status?.can_spend);
  return (
    <Card>
      <StepHead
        n={1}
        title="Deep analysis of your selected eval frames"
        sub="Gemini writes a scene card (place, layout, light, people, camera) and a card for every labelled drink of each frame. Cached: each frame is described once."
      />
      <div className="mt-3 flex flex-wrap items-center gap-3 text-[12px]">
        <Chip tone={status.with_cards === status.frames && status.frames ? "green" : "amber"}>
          {status.with_cards}/{status.frames} frames have cards
        </Chip>
        {status.base.needed && <Chip tone="amber">eval-set analysis first: {status.base.reasons.join("; ")}</Chip>}
        <div className="ml-auto">
          <Btn kind={todo ? "primary" : "secondary"} disabled={!canSpend || running || !todo} busy={running} onClick={() => setAsk(true)}>
            <ScanSearch className="h-3.5 w-3.5" /> {todo ? `Analyze · ≈ $${status.est_usd.toFixed(2)}` : "All frames analysed"}
          </Btn>
        </div>
      </div>
      {j && (
        <div className="mt-3 space-y-1">
          <div className="flex justify-between text-[11px]">
            <span className="truncate">{j.message}</span>
            <span className="tabular-nums">
              {j.done}/{j.total || "?"}
            </span>
          </div>
          <Progress value={j.progress} tone="violet" />
        </div>
      )}
      {!canSpend && todo && <div className="mt-2 text-[11px] text-amber-600 dark:text-amber-300">Only {st.status?.allowed_users.join(", ")} can start paid steps.</div>}
      {ask && (
        <Confirm title="Analyze for Text → Image?" confirmLabel={`Spend ≈ $${status.est_usd.toFixed(2)}`} onCancel={() => setAsk(false)} onConfirm={async () => (setAsk(false), await onStart())}>
          {status.base.needed && (
            <p>
              First the normal eval-set analysis ({status.base.reasons.join("; ")}): ≈ ${status.base.est_usd.toFixed(2)}.
            </p>
          )}
          {status.cards_est && (
            <p>
              Then scene + drink cards for <b>{status.missing_cards}</b> frame{status.missing_cards === 1 ? "" : "s"}: ≈ ${status.cards_est.usd.toFixed(2)}
              {status.cards_est.measured ? " (from measured calls)" : " (from token counts)"}.
            </p>
          )}
          <p>
            Estimated <b className="text-amber-600 dark:text-amber-300">≈ ${status.est_usd.toFixed(2)}</b>, at most ${status.est_max_usd.toFixed(2)} (Gemini 3.6 Flash).
          </p>
        </Confirm>
      )}
    </Card>
  );
}

// ---- 2 · plan -----------------------------------------------------------------------------------------------

function PlanStep({ status, plan, onPlan }: { status: T2iStatus; plan: T2iPlan | null; onPlan: (p: T2iPlan) => void }) {
  const st = useStudio();
  const mode = st.health?.gen_mode || "openai";
  const [n, setN] = useState(() => Number(localStorage.getItem("t2i_n")) || 100);
  const [hard, setHard] = useState(() => Number(localStorage.getItem("t2i_hard") ?? 50));
  const [las, setLas] = useState(() => Number(localStorage.getItem("t2i_la") ?? 1));
  const [ticked, setTicked] = useState(false);
  const [busy, setBusy] = useState(false);
  const [page, setPage] = useState(0);
  const [open, setOpen] = useState<string | null>(null);
  const [seed, setSeed] = useState(0);
  const save = (k: string, v: number) => localStorage.setItem(k, String(v));

  const make = async (nextSeed = seed) => {
    setBusy(true);
    try {
      const p = await studio.t2iPlan({ n, provider: mode, source_images: ticked && st.finalSel ? st.finalSel : null, hard_share: hard / 100, look_alikes: las, seed: nextSeed });
      onPlan(p);
      setSeed(nextSeed);
      setPage(0);
      st.setToast({ kind: "ok", text: `Plan ready: ${p.n} recipes from ${p.frames} frames (free).` });
    } catch (e) {
      st.setToast({ kind: "error", text: errText(e) });
    } finally {
      setBusy(false);
    }
  };
  const recipes = plan?.recipes || [];
  const shown = recipes.slice(page * PAGE, (page + 1) * PAGE);
  const cov = plan?.coverage;
  const stale = plan && plan.provider !== mode;

  return (
    <Card>
      <StepHead n={2} title="Plan exactly N images" sub="Free. Each recipe starts from one of your frames' scenes, mixes in real drinks (hard ones first) and keeps that frame's camera, look, exposure and colour." />
      <div className="mt-3 grid gap-3 text-[12px] sm:grid-cols-2 lg:grid-cols-4">
        <label>
          <span className="mb-0.5 block text-[10px] font-bold uppercase tracking-wider text-slate-400">Images (N)</span>
          <input type="number" min={1} max={5000} value={n} onChange={(e) => (setN(Math.max(1, Math.min(5000, Number(e.target.value) || 1))), save("t2i_n", Number(e.target.value) || 1))} className="w-full rounded-lg border border-slate-300 bg-white px-2 py-1.5 dark:border-slate-600 dark:bg-ink-800" />
        </label>
        <label>
          <span className="mb-0.5 flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-400">
            <span>Hard drinks</span>
            <span className="text-slate-700 dark:text-slate-200">{hard}%</span>
          </span>
          <input type="range" min={0} max={100} step={10} value={hard} onChange={(e) => (setHard(Number(e.target.value)), save("t2i_hard", Number(e.target.value)))} className="w-full accent-violet-600" />
        </label>
        <label>
          <span className="mb-0.5 block text-[10px] font-bold uppercase tracking-wider text-slate-400">Look-alikes per image</span>
          <select value={las} onChange={(e) => (setLas(Number(e.target.value)), save("t2i_la", Number(e.target.value)))} className="w-full rounded-lg border border-slate-300 bg-white px-2 py-1.5 dark:border-slate-600 dark:bg-ink-800">
            {[0, 1, 2, 3].map((v) => (
              <option key={v} value={v}>
                {v === 0 ? "none" : v === 1 ? "1 (non-drink that looks like one)" : `0–${v}`}
              </option>
            ))}
          </select>
        </label>
        <div>
          <span className="mb-0.5 block text-[10px] font-bold uppercase tracking-wider text-slate-400">Frames</span>
          <div className="flex overflow-hidden rounded-lg border border-slate-300 text-[11px] font-bold dark:border-slate-600">
            {[false, true].map((t) => (
              <button
                key={String(t)}
                type="button"
                disabled={t && !st.finalSel}
                onClick={() => setTicked(t)}
                className={cn("flex-1 px-2 py-1.5 disabled:opacity-40", ticked === t ? "bg-violet-600 text-white" : "text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-700")}
              >
                {t ? `ticked (${st.finalSel?.length ?? 0})` : `all ${status.frames}`}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Btn kind="primary" disabled={busy || !status.with_cards} busy={busy} onClick={() => void make()}>
          <Wand2 className="h-3.5 w-3.5" /> Make plan for {mode === "gemini" ? "Gemini" : "OpenAI"}
        </Btn>
        {plan && (
          <Btn kind="ghost" disabled={busy} onClick={() => void make(seed + 1)}>
            Shuffle
          </Btn>
        )}
        {!status.with_cards && <span className="text-[11px] text-amber-600 dark:text-amber-300">Run step 1 first.</span>}
        {stale && <span className="text-[11px] text-amber-600 dark:text-amber-300">This plan's prompts were written for {plan?.provider} — make it again for {mode}.</span>}
      </div>

      {plan && cov && (
        <div className="mt-4 space-y-3">
          <div className="flex flex-wrap gap-1.5 text-[11px]">
            <Chip tone="violet">{cov.images} images</Chip>
            <Chip>{cov.drinks} drinks · {cov.drinks_per_image}/image</Chip>
            <Chip tone="rose">
              {Math.round(cov.hard_share * 100)}% hard · {cov.hard_cases_used}/{cov.hard_cases_total} hard cases used
            </Chip>
            <Chip>
              classes {Object.entries(cov.class_mix).map(([k, v]) => `${k} ${Math.round((v / Math.max(1, cov.drinks)) * 100)}%`).join(" · ")}
            </Chip>
            <Chip>{cov.look_alikes} look-alikes</Chip>
            <Chip>{Object.keys(cov.anchors).length} anchor frames</Chip>
            {plan.frames_missing_cards.length > 0 && <Chip tone="amber">{plan.frames_missing_cards.length} frames without cards left out</Chip>}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(cov.anchors).map(([fn, k]) => (
              <span key={fn} className="relative" title={`${fn}: ${k} recipes`}>
                <img src={img.dataset(fn, 160)} alt="" className="h-12 w-12 rounded-md object-cover" />
                <span className="absolute -right-1 -top-1 rounded-full bg-violet-600 px-1 text-[9px] font-black text-white">{k}</span>
                <GenBadge gen={st.genOf(fn)} className="absolute -bottom-1 -right-1" />
              </span>
            ))}
          </div>
          <div className="space-y-1.5">
            {shown.map((r) => (
              <RecipeRow key={r.id} r={r} open={open === r.id} onToggle={() => setOpen(open === r.id ? null : r.id)} />
            ))}
          </div>
          {recipes.length > PAGE && (
            <div className="flex items-center justify-center gap-2 text-[11px]">
              <Btn kind="ghost" disabled={page === 0} onClick={() => setPage(page - 1)}>
                ‹ prev
              </Btn>
              <span>
                {page * PAGE + 1}–{Math.min(recipes.length, (page + 1) * PAGE)} of {recipes.length}
              </span>
              <Btn kind="ghost" disabled={(page + 1) * PAGE >= recipes.length} onClick={() => setPage(page + 1)}>
                next ›
              </Btn>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

function RecipeRow({ r, open, onToggle }: { r: T2iRecipe; open: boolean; onToggle: () => void }) {
  return (
    <div className="rounded-xl border border-slate-200 p-2 dark:border-slate-700">
      <button type="button" onClick={onToggle} className="flex w-full items-center gap-2 text-left">
        <img src={img.dataset(r.anchor, 160)} alt="" className="h-10 w-10 shrink-0 rounded-md object-cover" />
        <div className="min-w-0 flex-1 text-[11px]">
          <div className="flex items-center gap-1.5">
            <b className="font-mono">{r.id}</b>
            <span className="truncate text-slate-500">
              {r.anchor.replace(/^block_|\.png$/g, "")}
              {r.variation ? ` · ${r.variation}` : ""}
            </span>
          </div>
          <div className="mt-0.5 flex flex-wrap gap-1">
            {r.drinks.map((d, i) => (
              <span
                key={i}
                title={d.directive}
                className={cn("rounded px-1 text-[10px] font-bold", d.hard ? "bg-rose-500/15 text-rose-600 dark:text-rose-300" : "bg-slate-500/10 text-slate-600 dark:text-slate-300")}
              >
                {d.cls} {d.h_pct}%{d.status && d.status !== "hit" ? ` · ${d.status === "missed" ? "missed" : "wrong class"}` : ""}
              </span>
            ))}
            {r.look_alikes.map((x, i) => (
              <span key={`la${i}`} className="rounded bg-amber-500/15 px-1 text-[10px] font-bold text-amber-700 dark:text-amber-300" title={x.object}>
                not-a-drink
              </span>
            ))}
          </div>
        </div>
        <span className="shrink-0 text-[10px] text-slate-400">{open ? "hide prompt" : "prompt"}</span>
      </button>
      {open && <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap rounded-lg bg-slate-50 p-2 text-[10.5px] leading-snug text-slate-700 dark:bg-ink-800 dark:text-slate-300">{r.prompt}</pre>}
    </div>
  );
}

// ---- 3 · generate -------------------------------------------------------------------------------------------

function GenerateStep({ plan }: { plan: T2iPlan }) {
  const st = useStudio();
  const h = st.health;
  const mode = h?.gen_mode || "openai";
  const isGemini = mode === "gemini";
  const models = useMemo(() => (h?.image_models || []).filter((m) => m.provider === mode), [h, mode]);
  const [model, setModel] = useState("");
  const [quality, setQuality] = useState("medium");
  const [instant, setInstant] = useState(() => localStorage.getItem("studio_gen_speed") === "instant");
  const [nTest, setNTest] = useState(4);
  const [ask, setAsk] = useState<null | "test" | "final">(null);
  useEffect(() => {
    if (!models.length) return;
    const saved = localStorage.getItem("studio_image_model_v2") || "";
    setModel((cur) => (models.some((m) => m.id === cur) ? cur : (models.find((m) => m.id === saved) || models.find((m) => m.id === h?.default_model) || models[0]).id));
  }, [models, h?.default_model]);
  const batchUnit = isGemini ? T2I_GEMINI : (T2I_PRICE[model] || T2I_PRICE["gpt-image-2.5-sunburst"])[quality === "low" ? 0 : quality === "high" ? 2 : 1];
  const unit = batchUnit * (instant ? INSTANT_X : 1);
  const genRunning = st.gen.job?.status === "running";
  const canSpend = Boolean(st.status?.can_spend);
  const stale = plan.provider !== mode;
  const count = (kind: "test" | "final") => (kind === "test" ? Math.min(nTest, plan.n) : plan.n);

  const start = async (kind: "test" | "final") => {
    setAsk(null);
    try {
      st.gen.setJob(
        await studio.start({ mode: kind, model, per_source: 1, source_image: null, source_images: null, quality: isGemini ? null : quality, input_fidelity: null, instant, t2i: true, t2i_count: kind === "test" ? count("test") : null }),
      );
      st.gen.kick();
      st.setToast({ kind: "info", text: `Text → Image ${kind} started: ${count(kind)} images${instant ? " (instant)" : " (batch)"}. They appear in Results.` });
    } catch (e) {
      st.setToast({ kind: "error", text: errText(e) });
    }
  };

  return (
    <Card>
      <StepHead n={3} title="Generate" sub="One image per recipe, no reference image. Test runs the first recipes of the plan; Final runs all of them." />
      <div className="mt-3 grid gap-3 text-[12px] sm:grid-cols-2 lg:grid-cols-4">
        <label>
          <span className="mb-0.5 block text-[10px] font-bold uppercase tracking-wider text-slate-400">Model ({isGemini ? "Gemini" : "OpenAI"} — switch on the left)</span>
          <select value={model} onChange={(e) => (setModel(e.target.value), localStorage.setItem("studio_image_model_v2", e.target.value))} className="w-full rounded-lg border border-slate-300 bg-white px-2 py-1.5 dark:border-slate-600 dark:bg-ink-800">
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
        </label>
        {!isGemini && (
          <label>
            <span className="mb-0.5 block text-[10px] font-bold uppercase tracking-wider text-slate-400">Quality</span>
            <select value={quality} onChange={(e) => setQuality(e.target.value)} className="w-full rounded-lg border border-slate-300 bg-white px-2 py-1.5 dark:border-slate-600 dark:bg-ink-800">
              <option value="low">low</option>
              <option value="medium">medium</option>
              <option value="high">high · ~3×</option>
            </select>
          </label>
        )}
        <div>
          <span className="mb-0.5 block text-[10px] font-bold uppercase tracking-wider text-slate-400">Speed</span>
          <div className="grid grid-cols-2 gap-1 rounded-xl bg-slate-100 p-1 dark:bg-slate-800">
            {[false, true].map((fast) => (
              <button
                key={String(fast)}
                type="button"
                onClick={() => (setInstant(fast), localStorage.setItem("studio_gen_speed", fast ? "instant" : "batch"))}
                className={cn("flex items-center justify-center gap-1 rounded-lg py-1 text-[11px] font-bold", fast === instant ? "bg-white shadow dark:bg-ink-800" : "text-slate-500")}
              >
                {fast ? <Zap className="h-3 w-3 text-amber-500" /> : <Clock className="h-3 w-3" />}
                {fast ? "Instant" : "Batch"} ≈ ${(batchUnit * (fast ? INSTANT_X : 1)).toFixed(3)}
              </button>
            ))}
          </div>
        </div>
        <label>
          <span className="mb-0.5 block text-[10px] font-bold uppercase tracking-wider text-slate-400">Test images</span>
          <input type="number" min={1} max={plan.n} value={nTest} onChange={(e) => setNTest(Math.max(1, Math.min(plan.n, Number(e.target.value) || 1)))} className="w-full rounded-lg border border-slate-300 bg-white px-2 py-1.5 dark:border-slate-600 dark:bg-ink-800" />
        </label>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2">
        <Btn kind="secondary" disabled={!canSpend || genRunning || stale} onClick={() => setAsk("test")} className="flex-col gap-0 py-2">
          <span className="flex items-center gap-1">
            <FlaskConical className="h-3.5 w-3.5" /> Test
          </span>
          <span className="text-[10px] font-semibold opacity-70">
            {count("test")} img · ≈ ${(unit * count("test")).toFixed(2)}
          </span>
        </Btn>
        <Btn kind="primary" disabled={!canSpend || genRunning || stale} onClick={() => setAsk("final")} className="flex-col gap-0 py-2">
          <span className="flex items-center gap-1">
            <Rocket className="h-3.5 w-3.5" /> Final
          </span>
          <span className="text-[10px] font-semibold opacity-80">
            {plan.n} img · ≈ ${(unit * plan.n).toFixed(2)}
          </span>
        </Btn>
      </div>
      {stale && <div className="mt-2 text-[11px] text-amber-600 dark:text-amber-300">Make the plan again for {isGemini ? "Gemini" : "OpenAI"} first.</div>}
      {genRunning && st.gen.job && (
        <div className="mt-3 space-y-1">
          <div className="flex justify-between text-[11px]">
            <span className="truncate">{st.gen.job.message}</span>
            <span className="tabular-nums">
              {st.gen.job.done}/{st.gen.job.total || "?"}
            </span>
          </div>
          <Progress value={st.gen.job.progress} tone="violet" />
        </div>
      )}
      <div className="mt-3">
        <Banner kind="warn">
          Text → Image images have no source labels. The base model auto-labels them, but it misses exactly the hard drinks these recipes ask for — send the
          images to Own and let the team label them before training.
        </Banner>
      </div>
      {ask && (
        <Confirm
          title={ask === "test" ? `Text → Image test: ${count("test")} images?` : `Text → Image final: ${plan.n} images?`}
          confirmLabel={`Spend${instant ? " (instant)" : ""} ≈ $${(unit * count(ask)).toFixed(2)}`}
          onCancel={() => setAsk(null)}
          onConfirm={() => start(ask)}
        >
          <p>
            <b>{model}</b>
            {!isGemini && <> · {quality}</>} makes {ask === "test" ? `the first ${count("test")}` : `all ${plan.n}`} recipes of the plan, one image each, from text only.
          </p>
          <p>
            {instant ? (
              <>
                <b>Instant</b>: standard price ≈ ${unit.toFixed(3)}/image (batch would be ≈ ${(batchUnit * count(ask)).toFixed(2)}).
              </>
            ) : (
              <>
                <b>Batch</b>: ≈ ${unit.toFixed(3)}/image; results within minutes to a few hours.
              </>
            )}
          </p>
          <p>
            Estimated <b className="text-amber-600 dark:text-amber-300">≈ ${(unit * count(ask)).toFixed(2)}</b>.
            {ask === "test" ? " A test run is not uploaded to Hugging Face or sent to Own." : ""}
          </p>
        </Confirm>
      )}
    </Card>
  );
}
