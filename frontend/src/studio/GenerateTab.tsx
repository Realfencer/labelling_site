import { useCallback, useEffect, useMemo, useState } from "react";
import { Ban, CloudUpload, Download, Eraser, FlaskConical, RefreshCw, Rocket, Send, Tag, Trash2, Wand2 } from "lucide-react";
import { cn } from "../lib/utils";
import { img, studio } from "./api";
import type { Distribution, GenItem, GenJob, GenMode, Health, ModelErrors, Package, PackageSource, Run, SendJob, StudioStatus } from "./types";
import { Banner, Btn, Card, Chip, Confirm, Empty, errText, Lightbox, Progress, useJob } from "./ui";

/** $ per 1024² image on the OpenAI Batch API (low / medium / high). */
const PRICE: Record<string, [number, number, number]> = {
  "gpt-image-1": [0.012, 0.03, 0.09],
  "gpt-image-1.5": [0.01, 0.032, 0.08],
  "gpt-image-2": [0.008, 0.038, 0.11],
  "gpt-image-1-mini": [0.002, 0.005, 0.018],
};
const GEMINI_PRICE = 0.039; // per image, batch
const CHUNK = 13; // OpenAI final runs are submitted in chunks of ≤13 sources

const LS = { model: "studio_image_model", source: "studio_source_image", view: "studio_pair_view", per: "studio_per_source" };
const ls = {
  get: (k: string) => {
    try {
      return localStorage.getItem(k) || "";
    } catch {
      return "";
    }
  },
  set: (k: string, v: string) => {
    try {
      localStorage.setItem(k, v);
    } catch {
      /* private mode */
    }
  },
};

function runTime(name: string): string {
  const m = name.match(/_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_/);
  if (!m) return "";
  const d = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]));
  return d.toLocaleString(undefined, { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

export default function GenerateTab({ health, status, reloadStatus }: { health: Health | null; status: StudioStatus | null; reloadStatus: () => void }) {
  const mode: GenMode = health?.gen_mode || "openai";
  const models = useMemo(() => (health?.image_models || []).filter((m) => m.provider === mode), [health, mode]);
  const [model, setModel] = useState("");
  const [quality, setQuality] = useState("medium");
  const [fidelity, setFidelity] = useState("");
  const [per, setPer] = useState(() => Number(ls.get(LS.per)) || 4);
  const [pkg, setPkg] = useState<Package | null>(null);
  const [dist, setDist] = useState<Distribution | null>(null);
  const [errors, setErrors] = useState<ModelErrors | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [activeRun, setActiveRun] = useState<string>("");
  const [items, setItems] = useState<GenItem[]>([]);
  const [source, setSource] = useState(() => ls.get(LS.source));
  const [srcFilter, setSrcFilter] = useState("");
  const [msg, setMsg] = useState<{ kind: "info" | "ok" | "error"; text: string; link?: string } | null>(null);
  const [confirm, setConfirm] = useState<null | "final" | "test" | "clear">(null);
  const [busy, setBusy] = useState<string | null>(null);
  const canSpend = Boolean(status?.can_spend);

  // model choice follows the analyzed provider
  useEffect(() => {
    if (!models.length) return;
    const saved = ls.get(LS.model);
    const pick = models.find((m) => m.id === saved) || models.find((m) => m.id === health?.default_model) || models[0];
    setModel((cur) => (models.some((m) => m.id === cur) ? cur : pick.id));
  }, [models, health?.default_model]);
  useEffect(() => {
    if (health?.openai_defaults) {
      setQuality(health.openai_defaults.quality || "medium");
      setFidelity(health.openai_defaults.input_fidelity || "");
    }
  }, [health?.openai_defaults]);

  const refresh = useCallback(async () => {
    const [p, d, r, e] = await Promise.all([
      studio.pkg().catch(() => null),
      studio.distribution().catch(() => null),
      studio.runs().catch(() => ({ runs: [] as Run[] })),
      studio.modelErrors().catch(() => null),
    ]);
    setPkg(p);
    setDist(d);
    setRuns(r.runs);
    setErrors(e);
    setActiveRun((cur) => (cur && r.runs.some((x) => x.name === cur) ? cur : r.runs[0]?.name || ""));
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);

  const loadItems = useCallback(async (run: string) => {
    if (!run) return setItems([]);
    try {
      setItems((await studio.runImages(run)).images || []);
    } catch {
      setItems([]);
    }
  }, []);
  useEffect(() => {
    void loadItems(activeRun);
  }, [activeRun, loadItems]);

  const gen = useJob<GenJob>(studio.genJob, 1000, (j) => {
    if (j.status === "done") {
      const run = (j.run_dir || "").split("/").filter(Boolean).pop() || "";
      setMsg({
        kind: "ok",
        text: j.message || "Done",
        link: j.hf_upload?.ok && j.hf_upload.url ? j.hf_upload.url : undefined,
      });
      void refresh().then(() => {
        if (run) setActiveRun(run);
        void loadItems(run || activeRun);
      });
    } else if (j.status === "error") {
      setMsg({ kind: "error", text: j.error || j.message });
      void refresh();
    }
  });
  const genRunning = gen.job?.status === "running";

  const sendJob = useJob<SendJob>(studio.sendJob, 1000, (j) => {
    if (j.status === "done" && j.result) setMsg({ kind: "ok", text: `Sent ${j.result.images} images (1024×1024) to the Own pool of the labeler.` });
    else if (j.status === "error") setMsg({ kind: "error", text: j.error || j.message });
  });
  const sending = sendJob.job?.status === "running";

  const sources = useMemo(() => (pkg?.sources || []).filter((s) => s.source_image.includes("block_")), [pkg]);
  const nSrc = sources.length;
  const selectedSource = sources.find((s) => s.source_image === source) || null;
  const isGemini = mode === "gemini";

  const unit = isGemini ? GEMINI_PRICE : (PRICE[model] || PRICE["gpt-image-2"])[quality === "low" ? 0 : quality === "high" ? 2 : 1];
  const testCost = unit * per;
  const finalCount = nSrc * per;
  const finalCost = unit * finalCount;

  const run = async (key: string, fn: () => Promise<void>) => {
    setBusy(key);
    setMsg(null);
    try {
      await fn();
    } catch (e) {
      setMsg({ kind: "error", text: errText(e) });
    } finally {
      setBusy(null);
    }
  };

  const startGen = async (kind: "test" | "final") => {
    const job = await studio.start({
      mode: kind,
      model,
      per_source: per,
      source_image: kind === "test" ? source : null,
      quality: isGemini ? null : quality,
      input_fidelity: isGemini ? null : fidelity,
    });
    gen.setJob(job);
    gen.kick();
    setMsg({ kind: "info", text: kind === "test" ? `Test run started on ${source}.` : "Final run submitted — results usually arrive within minutes to a few hours." });
  };

  const filteredSources = sources.filter((s) => !srcFilter || s.source_image.toLowerCase().includes(srcFilter.toLowerCase()));

  return (
    <div className="grid gap-4 xl:grid-cols-[360px_1fr]">
      {/* ---- left rail ------------------------------------------------------------------------------------ */}
      <div className="space-y-4 xl:sticky xl:top-[72px] xl:max-h-[calc(100vh-88px)] xl:overflow-y-auto xl:pr-1">
        <Card title="Generate" right={<Chip tone={isGemini ? "violet" : "sky"}>{isGemini ? "Gemini" : "OpenAI"} mode</Chip>}>
          <div className="space-y-3 text-[12px]">
            <label className="block">
              <span className="mb-1 block text-[11px] font-bold uppercase tracking-wider text-slate-400">Model</span>
              <select
                value={model}
                onChange={(e) => {
                  setModel(e.target.value);
                  ls.set(LS.model, e.target.value);
                }}
                className="w-full rounded-lg border border-slate-300 bg-white px-2 py-1.5 dark:border-slate-600 dark:bg-ink-800"
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))}
              </select>
              <span className="mt-1 block text-[10px] text-slate-500">
                {models.find((m) => m.id === model)?.hint}
                {health?.detectors?.[mode] &&
                  (health.detectors[mode]!.enabled && health.detectors[mode]!.exists
                    ? ` · auto-labels with ${health.detectors[mode]!.weights.split("/").pop()} @ ${health.detectors[mode]!.conf}`
                    : " · detector off: source labels are copied")}
              </span>
              <span className="mt-0.5 block text-[10px] text-slate-400">To use the other provider, re-analyze in that mode (Analyze tab).</span>
            </label>
            {!isGemini && (
              <div className="grid grid-cols-2 gap-2">
                <label>
                  <span className="mb-1 block text-[11px] font-bold uppercase tracking-wider text-slate-400">Quality</span>
                  <select value={quality} onChange={(e) => setQuality(e.target.value)} className="w-full rounded-lg border border-slate-300 bg-white px-2 py-1.5 dark:border-slate-600 dark:bg-ink-800">
                    <option value="low">low</option>
                    <option value="medium">medium</option>
                    <option value="high">high (~3× cost)</option>
                  </select>
                </label>
                <label>
                  <span className="mb-1 block text-[11px] font-bold uppercase tracking-wider text-slate-400">Fidelity</span>
                  <select value={fidelity} onChange={(e) => setFidelity(e.target.value)} className="w-full rounded-lg border border-slate-300 bg-white px-2 py-1.5 dark:border-slate-600 dark:bg-ink-800">
                    <option value="">model default</option>
                    <option value="low">low</option>
                    <option value="high">high (closer to source)</option>
                  </select>
                </label>
              </div>
            )}
            <label className="block">
              <span className="mb-1 block text-[11px] font-bold uppercase tracking-wider text-slate-400">Images per source · {per}</span>
              <input
                type="range"
                min={1}
                max={20}
                value={per}
                onChange={(e) => {
                  setPer(Number(e.target.value));
                  ls.set(LS.per, e.target.value);
                }}
                className="w-full accent-violet-600"
              />
            </label>
            <div className="rounded-xl bg-slate-50 p-2.5 text-[11px] dark:bg-slate-800/50">
              <div className="flex justify-between">
                <span className="text-slate-500">Price</span>
                <b>≈ ${unit.toFixed(3)} / image</b>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Test (1 source × {per})</span>
                <b>≈ ${testCost.toFixed(2)}</b>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">
                  Final ({nSrc} sources × {per} = {finalCount})
                </span>
                <b className={cn(finalCost > 20 && "text-amber-600 dark:text-amber-300")}>≈ ${finalCost.toFixed(2)}</b>
              </div>
              {!isGemini && nSrc > 0 && <div className="mt-0.5 text-[10px] text-slate-400">submitted as {Math.ceil(nSrc / CHUNK)} batch(es) of ≤{CHUNK} sources · batch API is ~50% cheaper</div>}
            </div>

            <div>
              <span className="mb-1 block text-[11px] font-bold uppercase tracking-wider text-slate-400">Test source</span>
              <input
                value={srcFilter}
                onChange={(e) => setSrcFilter(e.target.value)}
                placeholder="search block_…"
                className="mb-1 w-full rounded-lg border border-slate-300 bg-white px-2 py-1 dark:border-slate-600 dark:bg-ink-800"
              />
              <select
                value={source}
                onChange={(e) => {
                  setSource(e.target.value);
                  ls.set(LS.source, e.target.value);
                }}
                size={5}
                className="w-full rounded-lg border border-slate-300 bg-white px-1 py-1 font-mono text-[11px] dark:border-slate-600 dark:bg-ink-800"
              >
                {filteredSources.map((s) => (
                  <option key={s.source_image} value={s.source_image}>
                    {s.source_image}
                    {s.detector_errors ? `  · ${s.detector_errors} err` : ""}
                    {!s.resolved ? " (missing)" : ""}
                  </option>
                ))}
              </select>
              {selectedSource && (
                <div className="mt-1 text-[10px] text-slate-500">
                  regimes {selectedSource.regimes.split("|").join(", ")} · ×{selectedSource.target_outputs}
                  {selectedSource.detector_errors ? ` · ${selectedSource.detector_errors} detector errors targeted` : ""}
                </div>
              )}
            </div>

            {!canSpend && (
              <Banner kind="warn">
                Only {status?.allowed_users.join(", ") || "the listed people"} can start paid runs. You can still browse, collect, label and send runs.
              </Banner>
            )}
            <div className="grid gap-2">
              <Btn kind="secondary" busy={busy === "prepare"} onClick={() => void run("prepare", async () => {
                const r = await studio.prepare(per);
                setMsg({ kind: "ok", text: `Package rebuilt: ${r.n_sources} sources · ${r.total_targets} targets · est. $${(r.estimated_cost_usd ?? 0).toFixed(2)}` });
                await refresh();
              })}>
                <RefreshCw className="h-3.5 w-3.5" /> Rebuild package from analysis
              </Btn>
              <Btn kind="secondary" disabled={!canSpend || genRunning || !selectedSource} onClick={() => setConfirm("test")} title={!selectedSource ? "Pick a source above" : undefined}>
                <FlaskConical className="h-3.5 w-3.5" /> Test generate · 1 source · ≈ ${testCost.toFixed(2)}
              </Btn>
              <Btn kind="primary" disabled={!canSpend || genRunning || nSrc === 0} onClick={() => setConfirm("final")}>
                <Rocket className="h-3.5 w-3.5" /> Final generate · {finalCount} images · ≈ ${finalCost.toFixed(2)}
              </Btn>
            </div>

            {gen.job && gen.job.status === "running" && (
              <div className="space-y-1 rounded-xl border border-violet-300 bg-violet-50 p-2.5 dark:border-violet-900 dark:bg-violet-950/30">
                <div className="flex justify-between text-[11px] font-semibold">
                  <span>{gen.job.mode === "autolabel" ? "Auto-labelling" : `${gen.job.mode} run`}</span>
                  <span className="tabular-nums">
                    {gen.job.done}/{gen.job.total || "?"}
                  </span>
                </div>
                <Progress value={gen.job.progress} tone="violet" />
                <div className="text-[11px] text-slate-600 dark:text-slate-300">{gen.job.message}</div>
              </div>
            )}
            {msg && (
              <Banner kind={msg.kind === "error" ? "error" : msg.kind === "ok" ? "ok" : "info"}>
                {msg.text}
                {msg.link && (
                  <a href={msg.link} target="_blank" rel="noreferrer" className="ml-1 font-bold underline">
                    open
                  </a>
                )}
              </Banner>
            )}
          </div>
        </Card>

        <RunsCard
          runs={runs}
          active={activeRun}
          onPick={setActiveRun}
          busy={busy}
          sending={sending}
          genRunning={genRunning}
          onCollect={(r) => run(`collect:${r}`, async () => {
            gen.setJob(await studio.collect(r));
            gen.kick();
          })}
          onCancel={(r) => run(`cancel:${r}`, async () => {
            if (!window.confirm(`Cancel the OpenAI batch of ${r}?`)) return;
            const res = await studio.cancel(r);
            setMsg({ kind: "info", text: `Batch ${r}: ${res.status}` });
            await refresh();
          })}
          onRelabel={(r) => run(`label:${r}`, async () => {
            gen.setJob(await studio.autolabel(r));
            gen.kick();
          })}
        />

        <Card title="Hugging Face">
          <div className="space-y-2 text-[12px]">
            <div className="text-slate-600 dark:text-slate-300">
              {health?.hf_repo ? (
                <>
                  Final runs are uploaded to <b>{health.hf_repo}</b> ({health.hf_private ? "private" : "public"}) as <code>runs/&lt;date&gt;/&lt;run&gt;.tar.gz</code>.
                </>
              ) : (
                <>No dataset repo set — choose one in Studio settings (top right).</>
              )}
            </div>
            <Btn
              kind="secondary"
              disabled={!health?.hf_repo || !health?.has_hf_token || !activeRun}
              busy={busy === "hf"}
              onClick={() => void run("hf", async () => {
                const r = await studio.uploadHf(activeRun || null);
                setMsg({ kind: "ok", text: `Uploaded ${r.n_images} images → ${r.repo_id}/${r.path_in_repo}`, link: r.url });
              })}
            >
              <CloudUpload className="h-3.5 w-3.5" /> Upload {activeRun ? "this run" : ""} to Hugging Face
            </Btn>
          </div>
        </Card>

        <Card title="Danger zone">
          <Btn kind="danger" onClick={() => setConfirm("clear")} disabled={genRunning}>
            <Eraser className="h-3.5 w-3.5" /> Clear dataset, runs and analysis…
          </Btn>
        </Card>
      </div>

      {/* ---- workspace ------------------------------------------------------------------------------------- */}
      <div className="min-w-0 space-y-4">
        <PairGrid
          run={activeRun}
          runInfo={runs.find((r) => r.name === activeRun) || null}
          items={items}
          sending={sending}
          sendProgress={sendJob.job}
          onSend={(names) => run("send", async () => {
            sendJob.setJob(await studio.send(activeRun, names));
            sendJob.kick();
          })}
          onDelete={(names) => run("delete", async () => {
            const r = await studio.deleteImages(activeRun, names);
            setMsg({ kind: "ok", text: `Deleted ${r.deleted} · ${r.remaining} left (kept in the run's _deleted/ folder)` });
            await refresh();
            await loadItems(activeRun);
          })}
        />
        <div className="grid gap-4 2xl:grid-cols-2">
          <QuotaTable dist={dist} />
          <WrongPanel source={selectedSource} errors={errors} />
        </div>
        <SourceCards sources={sources} selected={source} onPick={(s) => (setSource(s), ls.set(LS.source, s))} pkg={pkg} />
      </div>

      {confirm === "test" && selectedSource && (
        <Confirm
          title="Start a test run?"
          confirmLabel={`Generate ${per} images`}
          onCancel={() => setConfirm(null)}
          onConfirm={async () => {
            setConfirm(null);
            await run("start", () => startGen("test"));
          }}
        >
          <p>
            <b>{model}</b> makes {per} new images from <code>{selectedSource.source_image}</code>.
          </p>
          <p>
            Estimated cost <b>≈ ${testCost.toFixed(2)}</b>.
          </p>
        </Confirm>
      )}
      {confirm === "final" && (
        <Confirm
          title="Submit the final run?"
          confirmLabel={`Spend ≈ $${finalCost.toFixed(2)}`}
          onCancel={() => setConfirm(null)}
          onConfirm={async () => {
            setConfirm(null);
            await run("start", () => startGen("final"));
          }}
        >
          <p>
            <b>{model}</b>
            {!isGemini && <> · {quality} quality</>} makes <b>{finalCount}</b> images ({nSrc} sources × {per}).
          </p>
          <p>
            Estimated cost <b className="text-amber-600 dark:text-amber-300">≈ ${finalCost.toFixed(2)}</b>
            {!isGemini && <> in {Math.ceil(nSrc / CHUNK)} batch(es)</>}. Results usually arrive within minutes to a few hours; you can leave this page.
          </p>
        </Confirm>
      )}
      {confirm === "clear" && (
        <ClearConfirm
          onDone={async () => {
            setConfirm(null);
            setActiveRun("");
            await refresh();
            reloadStatus();
          }}
          onCancel={() => setConfirm(null)}
        />
      )}
    </div>
  );
}

function ClearConfirm({ onDone, onCancel }: { onDone: () => void; onCancel: () => void }) {
  const [preview, setPreview] = useState<Awaited<ReturnType<typeof studio.clearPreview>> | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    studio.clearPreview().then(setPreview).catch((e) => setErr(errText(e)));
  }, []);
  const busy = preview && (preview.gen_running || preview.analyze_running);
  return (
    <Confirm title="Clear the Studio?" danger typeToConfirm={busy ? "—" : "CLEAR"} confirmLabel="Delete everything" onCancel={onCancel} onConfirm={async () => {
      await studio.clear();
      onDone();
    }}>
      {err && <p className="text-rose-600">{err}</p>}
      {!preview ? (
        <p>Loading…</p>
      ) : busy ? (
        <p className="text-amber-600">A job is running — wait for it to finish first.</p>
      ) : (
        <ul className="list-disc pl-5 text-[12px]">
          <li>
            Eval dataset: {preview.beverage_gt_images} images + {preview.beverage_gt_labels} labels
          </li>
          <li>
            {preview.history_runs} generated runs{preview.history_run_names.length ? ` (${preview.history_run_names.slice(0, 4).join(", ")}…)` : ""}
          </li>
          <li>{preview.analysis_files} analysis files</li>
        </ul>
      )}
      <p>This cannot be undone. Type CLEAR to confirm.</p>
    </Confirm>
  );
}

function RunsCard({
  runs,
  active,
  onPick,
  busy,
  sending,
  genRunning,
  onCollect,
  onCancel,
  onRelabel,
}: {
  runs: Run[];
  active: string;
  onPick: (r: string) => void;
  busy: string | null;
  sending: boolean;
  genRunning: boolean;
  onCollect: (r: string) => void;
  onCancel: (r: string) => void;
  onRelabel: (r: string) => void;
}) {
  return (
    <Card title="Runs" right={<span className="text-[11px] text-slate-400">{runs.length}</span>}>
      {runs.length === 0 ? (
        <Empty>No runs yet.</Empty>
      ) : (
        <div className="space-y-1.5">
          {runs.map((r) => {
            const provider = r.provider || (r.model?.startsWith("gemini") ? "gemini" : "openai");
            const hex = r.name.split("_").pop();
            return (
              <div
                key={r.name}
                role="button"
                tabIndex={0}
                onClick={() => onPick(r.name)}
                onKeyDown={(e) => e.key === "Enter" && onPick(r.name)}
                className={cn(
                  "cursor-pointer rounded-xl border px-2.5 py-2 transition",
                  active === r.name ? "border-violet-500 bg-violet-500/10" : "border-slate-200 hover:border-violet-300 dark:border-slate-700",
                )}
              >
                <div className="flex flex-wrap items-center gap-1">
                  <Chip tone={provider === "gemini" ? "violet" : "sky"}>{provider}</Chip>
                  <Chip tone={r.mode === "final" ? "green" : "slate"}>{r.mode || "run"}</Chip>
                  <span className="text-[11px] font-bold">{runTime(r.name)}</span>
                  <span className="font-mono text-[10px] text-slate-400">#{hex}</span>
                  <span className="ml-auto text-[11px] font-bold tabular-nums">{r.count} img</span>
                </div>
                <div className="mt-1 flex flex-wrap items-center gap-1 text-[10px]">
                  <span className="text-slate-500">{(r.model || "").replace("gpt-image-", "gpt-image ").replace("gemini-", "")}</span>
                  {r.pending && <Chip tone="amber">● batch pending{r.chunks > 1 ? ` · ${r.chunks} chunks` : ""}</Chip>}
                  {r.count > 0 && (r.autolabelled ? <Chip tone="green">detector labels</Chip> : <Chip tone="amber">source labels</Chip>)}
                  {(r.camera_drift ?? 0) > 0 && <Chip tone="rose">{r.camera_drift} camera drift</Chip>}
                  {r.near_copies > 0 && <Chip tone="amber">≈ {r.near_copies} near copies</Chip>}
                </div>
                <div className="mt-1.5 flex flex-wrap gap-1" onClick={(e) => e.stopPropagation()}>
                  {r.pending ? (
                    <>
                      <Btn kind="secondary" className="px-2 py-1 text-[11px]" busy={busy === `collect:${r.name}`} disabled={genRunning} onClick={() => onCollect(r.name)}>
                        <Download className="h-3 w-3" /> Collect
                      </Btn>
                      {provider === "openai" && (
                        <Btn kind="ghost" className="px-2 py-1 text-[11px]" busy={busy === `cancel:${r.name}`} onClick={() => onCancel(r.name)}>
                          <Ban className="h-3 w-3" /> Cancel
                        </Btn>
                      )}
                    </>
                  ) : (
                    <Btn kind="ghost" className="px-2 py-1 text-[11px]" busy={busy === `label:${r.name}`} disabled={r.count <= 0 || genRunning} onClick={() => onRelabel(r.name)}>
                      <Tag className="h-3 w-3" /> Re-label
                    </Btn>
                  )}
                  {sending && active === r.name && <span className="self-center text-[10px] text-slate-500">sending…</span>}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}

function PairGrid({
  run,
  runInfo,
  items,
  sending,
  sendProgress,
  onSend,
  onDelete,
}: {
  run: string;
  runInfo: Run | null;
  items: GenItem[];
  sending: boolean;
  sendProgress: SendJob | null;
  onSend: (names: string[] | null) => void;
  onDelete: (names: string[]) => void;
}) {
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [view, setView] = useState<"final" | "raw">(() => (ls.get(LS.view) === "raw" ? "raw" : "final"));
  const [box, setBox] = useState<GenItem | null>(null);
  const [askDelete, setAskDelete] = useState(false);
  useEffect(() => setSel(new Set()), [run]);
  const nearCopies = items.filter((i) => i.near_copy);
  const toggle = (fn: string) =>
    setSel((s) => {
      const n = new Set(s);
      if (n.has(fn)) n.delete(fn);
      else n.add(fn);
      return n;
    });
  const genSrc = (it: GenItem, w?: number) => (view === "raw" ? img.raw(run, it.filename, w) : img.gen(run, it.filename, w));
  const srcSrc = (it: GenItem, w?: number) => img.source(run, it.filename, w);

  return (
    <Card
      title={
        <span className="flex items-center gap-2">
          <Wand2 className="h-4 w-4 text-violet-500" /> Generated · source | output
          {run && <span className="font-mono text-[11px] font-normal text-slate-400">{run}</span>}
        </span>
      }
      right={
        run && (
          <div className="flex flex-wrap items-center gap-1.5">
            <div className="flex overflow-hidden rounded-lg border border-slate-300 text-[11px] font-bold dark:border-slate-600">
              {(["final", "raw"] as const).map((v) => (
                <button
                  key={v}
                  type="button"
                  onClick={() => (setView(v), ls.set(LS.view, v))}
                  className={cn("px-2 py-1", view === v ? "bg-slate-800 text-white dark:bg-slate-200 dark:text-slate-900" : "text-slate-500")}
                  title={v === "raw" ? "Generator output before the CCTV look / resize" : "What is trained on"}
                >
                  {v === "final" ? "Final" : "Raw"}
                </button>
              ))}
            </div>
            {nearCopies.length > 0 && (
              <Btn kind="ghost" className="px-2 py-1 text-[11px]" onClick={() => setSel(new Set(nearCopies.map((i) => i.filename)))}>
                Select near copies ({nearCopies.length})
              </Btn>
            )}
            <Btn kind="ghost" className="px-2 py-1 text-[11px]" onClick={() => setSel(new Set(items.map((i) => i.filename)))}>
              Select all
            </Btn>
            {sel.size > 0 && (
              <Btn kind="ghost" className="px-2 py-1 text-[11px]" onClick={() => setSel(new Set())}>
                Clear
              </Btn>
            )}
            <Btn kind="danger" className="px-2 py-1 text-[11px]" disabled={sel.size === 0} onClick={() => setAskDelete(true)}>
              <Trash2 className="h-3 w-3" /> Delete {sel.size || ""}
            </Btn>
            <Btn kind="success" className="px-2 py-1 text-[11px]" busy={sending} disabled={!items.length || runInfo?.pending} onClick={() => onSend(sel.size ? [...sel] : null)}>
              <Send className="h-3 w-3" /> Send {sel.size ? sel.size : "all"} to Own
            </Btn>
          </div>
        )
      }
    >
      {sending && sendProgress && (
        <div className="mb-3 space-y-1">
          <div className="text-[11px] text-slate-500">{sendProgress.message}</div>
          <Progress value={sendProgress.progress} tone="emerald" />
        </div>
      )}
      {!run ? (
        <Empty>Pick a run on the left.</Empty>
      ) : items.length === 0 ? (
        <Empty>{runInfo?.pending ? "The batch is still being generated — press Collect on the run to fetch finished images." : "This run has no images."}</Empty>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
          {items.map((it) => {
            const on = sel.has(it.filename);
            return (
              <div
                key={it.filename}
                className={cn("overflow-hidden rounded-xl border transition", on ? "border-sky-500 ring-2 ring-sky-500/40" : "border-slate-200 dark:border-slate-700")}
              >
                <div className="relative grid grid-cols-2 gap-px bg-slate-200 dark:bg-slate-800" onClick={() => setBox(it)}>
                  <figure className="relative cursor-zoom-in bg-black">
                    <img src={srcSrc(it, 480)} alt="source" loading="lazy" className="aspect-square w-full object-cover" />
                    <figcaption className="absolute left-1 top-1 rounded bg-black/60 px-1 text-[9px] font-bold uppercase text-white">source</figcaption>
                  </figure>
                  <figure className="relative cursor-zoom-in bg-black">
                    <img
                      src={genSrc(it, 480)}
                      alt="generated"
                      loading="lazy"
                      className="aspect-square w-full object-cover"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.opacity = "0.15";
                      }}
                    />
                    <figcaption className="absolute left-1 top-1 rounded bg-violet-600/80 px-1 text-[9px] font-bold uppercase text-white">{view === "raw" ? "raw" : "generated"}</figcaption>
                  </figure>
                  <label className="absolute right-1.5 top-1.5 flex h-6 w-6 items-center justify-center rounded-md bg-black/50" onClick={(e) => e.stopPropagation()}>
                    <input type="checkbox" checked={on} onChange={() => toggle(it.filename)} className="h-4 w-4 accent-sky-500" />
                  </label>
                </div>
                <div className="flex flex-wrap items-center gap-1 px-2 py-1.5">
                  <span className="min-w-0 flex-1 truncate font-mono text-[10px] text-slate-500" title={`${it.source_image} → ${it.filename}`}>
                    {it.filename}
                  </span>
                  {it.near_copy && <Chip tone="amber">≈ near copy{it.layout_similarity != null ? ` ${it.layout_similarity.toFixed(2)}` : ""}</Chip>}
                  {it.camera_drift && it.camera_drift.length > 0 && (
                    <Chip tone="rose" title={it.camera_drift.join("\n")}>
                      camera drift · {it.camera_drift.map((d) => d.split(":")[0]).join(" / ")}
                    </Chip>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
      {box && (
        <Lightbox
          panes={[
            { label: "source", src: srcSrc(box) },
            { label: view === "raw" ? "generated · raw" : "generated", src: genSrc(box) },
          ]}
          caption={`${box.source_image || ""} → ${box.filename}`}
          onClose={() => setBox(null)}
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
            onDelete([...sel]);
            setSel(new Set());
          }}
        >
          <p>They are removed from the run (moved to its _deleted/ folder), so they are not sent or uploaded.</p>
        </Confirm>
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

function WrongPanel({ source, errors }: { source: PackageSource | null; errors: ModelErrors | null }) {
  if (!errors || !errors.ready) {
    return (
      <Card title="What the detector gets wrong">
        <Empty>Detector errors appear after an analysis with a detector switched on.</Empty>
      </Card>
    );
  }
  if (!source)
    return (
      <Card title="What the detector gets wrong">
        <Empty>Pick a source to see its mistakes — the prompts target them.</Empty>
      </Card>
    );
  const src = errors.sources.find((s) => s.filename === source.source_image);
  return (
    <Card title={<>What the detector gets wrong · <span className="font-mono text-[11px] font-normal">{source.source_image}</span></>}>
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

function SourceCards({ sources, selected, onPick, pkg }: { sources: PackageSource[]; selected: string; onPick: (s: string) => void; pkg: Package | null }) {
  return (
    <Card
      title="Package sources"
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
