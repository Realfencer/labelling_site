import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, ArrowRight, Image as ImageIcon, Loader2, RotateCw, Settings, Sparkles, Type, Wand2, X } from "lucide-react";
import { cn } from "../lib/utils";
import KeysButton from "../components/KeysButton";
import { studio } from "../studio/api";
import { nextStep, StudioProvider, useStudio } from "../studio/store";
import type { GenMode, StudioStatus } from "../studio/types";
import { Banner, Btn, Chip, errText } from "../studio/ui";
import ClearConfirm from "../studio/ClearConfirm";
import { AnalyzeCard, EvalSetCard, GenerateCard } from "../studio/Workbench";

const Results = lazy(() => import("../studio/Results"));
const PlanPanel = lazy(() => import("../studio/PlanPanel"));
const Insights = lazy(() => import("../studio/Insights"));

type Tab = "results" | "plan" | "insights";

export default function StudioPage() {
  return (
    <StudioProvider>
      <Studio />
    </StudioProvider>
  );
}

function Studio() {
  const st = useStudio();
  const [tab, setTab] = useState<Tab>(() => (localStorage.getItem("studio_tab2") as Tab) || "results");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const syncRef = useRef<(() => void) | null>(null);
  const analyzeRef = useRef<((mode?: GenMode) => void) | null>(null);
  const startRef = useRef<((kind: "test" | "final") => void) | null>(null);
  const pick = (t: Tab) => {
    setTab(t);
    localStorage.setItem("studio_tab2", t);
  };
  useEffect(() => {
    const open = (e: Event) => pick((e as CustomEvent<Tab>).detail);
    window.addEventListener("studio-open-tab", open);
    return () => window.removeEventListener("studio-open-tab", open);
  }, []);
  const next = nextStep(st);
  const keys = st.status?.keys;

  const doNext = async () => {
    switch (next.key) {
      case "update":
        return syncRef.current?.();
      case "analyze":
        return analyzeRef.current?.();
      case "test":
        return startRef.current?.("test");
      case "final":
        return startRef.current?.("final");
      case "collect":
        try {
          st.gen.setJob(await studio.collect(st.runs[0].name));
          st.gen.kick();
        } catch (e) {
          st.setToast({ kind: "error", text: errText(e) });
        }
        return;
      case "review":
        st.setActiveRun(st.runs[0].name);
        return pick("results");
    }
  };

  const tones = {
    violet: "from-violet-600 to-fuchsia-600",
    sky: "from-sky-600 to-cyan-500",
    amber: "from-amber-500 to-orange-500",
    emerald: "from-emerald-600 to-teal-500",
    slate: "from-slate-600 to-slate-500",
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-ink-950 dark:text-slate-100">
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/90 px-4 py-2 backdrop-blur dark:border-slate-800 dark:bg-ink-900/90">
        <div className="mx-auto flex max-w-[1700px] flex-wrap items-center gap-3">
          <Link to="/" className="inline-flex items-center gap-1 rounded-md border border-slate-300 px-2 py-1 text-[12px] font-semibold hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-800">
            <ArrowLeft className="h-3.5 w-3.5" /> Labeler
          </Link>
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 via-fuchsia-500 to-sky-500 text-white shadow-lg shadow-violet-500/30">
              <Sparkles className="h-4 w-4" />
            </div>
            <div className="text-base font-black tracking-tight">Studio</div>
          </div>
          <div className="flex rounded-xl bg-slate-100 p-1 text-[12px] font-bold dark:bg-slate-800">
            <span className="flex items-center gap-1.5 rounded-lg bg-white px-3 py-1 text-slate-900 shadow dark:bg-ink-900 dark:text-white">
              <ImageIcon className="h-3.5 w-3.5" /> Image → Image
            </span>
            <span className="flex cursor-not-allowed items-center gap-1.5 px-3 py-1 text-slate-400" title="Coming next: generate from text only, no source frame">
              <Type className="h-3.5 w-3.5" /> Text → Image <span className="rounded bg-slate-200 px-1 text-[9px] uppercase dark:bg-slate-700">soon</span>
            </span>
          </div>
          <div className="ml-auto flex flex-wrap items-center gap-2 text-[11px]">
            <Chip tone={st.status?.up ? "green" : "rose"}>
              <span className={cn("h-1.5 w-1.5 rounded-full", st.status?.up ? "animate-pulse bg-emerald-500" : "bg-rose-500")} />
              {st.status?.up ? "online" : "offline"}
            </Chip>
            {keys && (
              <>
                <Chip tone={keys.openai.configured ? "green" : "amber"}>OpenAI {keys.openai.configured ? "✓" : "—"}</Chip>
                <Chip tone={keys.gemini.configured ? "green" : "amber"}>Gemini {keys.gemini.configured ? "✓" : "—"}</Chip>
              </>
            )}
            <KeysButton className="px-2" />
            <button
              type="button"
              onClick={() => setSettingsOpen(true)}
              className="inline-flex items-center gap-1 rounded-md border border-slate-300 px-2 py-1 font-semibold hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-800"
            >
              <Settings className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1700px] space-y-4 p-4">
        {/* next step */}
        <div className={cn("relative overflow-hidden rounded-2xl bg-gradient-to-r p-[1px] shadow-lg", tones[next.tone])}>
          <div className="flex flex-wrap items-center gap-4 rounded-2xl bg-white/95 px-5 py-3 dark:bg-ink-900/95">
            <div className={cn("flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br text-white", tones[next.tone])}>
              {next.key === "running" ? <Loader2 className="h-5 w-5 animate-spin" /> : <Wand2 className="h-5 w-5" />}
            </div>
            <div className="min-w-0 flex-1">
              <div className="text-[10px] font-bold uppercase tracking-widest text-slate-400">{next.key === "running" ? "Working" : "Next step"}</div>
              <div className="text-[15px] font-extrabold tracking-tight">{next.title}</div>
              <div className="truncate text-[12px] text-slate-500 dark:text-slate-400">{next.detail}</div>
            </div>
            {next.cta && (
              <Btn kind="primary" className={cn("bg-gradient-to-r px-4 py-2 text-[13px]", tones[next.tone])} onClick={() => void doNext()}>
                {next.cta} <ArrowRight className="h-4 w-4" />
              </Btn>
            )}
          </div>
        </div>
        {st.toast && (
          <div className="flex items-start gap-2">
            <div className="flex-1">
              <Banner kind={st.toast.kind === "error" ? "error" : st.toast.kind === "ok" ? "ok" : "info"}>
                {st.toast.text}
                {st.toast.link && (
                  <a href={st.toast.link} target="_blank" rel="noreferrer" className="ml-1 font-bold underline">
                    open
                  </a>
                )}
              </Banner>
            </div>
            <button type="button" onClick={() => st.setToast(null)} className="rounded p-1 text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-800">
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        {!st.status ? (
          <div className="flex items-center justify-center gap-2 py-24 text-sm text-slate-500">
            <Loader2 className="h-4 w-4 animate-spin" /> Connecting to the Studio…
          </div>
        ) : (
          <div className="grid gap-4 xl:grid-cols-[370px_1fr]">
            <aside className="space-y-3 xl:sticky xl:top-[64px] xl:max-h-[calc(100vh-80px)] xl:self-start xl:overflow-y-auto xl:pr-1">
              <EvalSetCard syncRef={syncRef} />
              <AnalyzeCard analyzeRef={analyzeRef} />
              <GenerateCard startRef={startRef} />
            </aside>
            <section className="min-w-0 space-y-3">
              <nav className="flex gap-1 rounded-xl bg-slate-100 p-1 text-[12px] font-bold dark:bg-slate-800">
                {(
                  [
                    ["results", `Results${st.runs.length ? ` · ${st.runs.length} runs` : ""}`],
                    ["plan", `Plan${st.sources.length ? ` · ${st.sources.length} sources` : ""}`],
                    ["insights", "Insights"],
                  ] as [Tab, string][]
                ).map(([t, l]) => (
                  <button
                    key={t}
                    type="button"
                    onClick={() => pick(t)}
                    className={cn(
                      "flex-1 rounded-lg px-3 py-1.5 transition",
                      tab === t ? "bg-white text-slate-900 shadow dark:bg-ink-900 dark:text-white" : "text-slate-500 hover:text-slate-800 dark:hover:text-slate-200",
                    )}
                  >
                    {l}
                  </button>
                ))}
              </nav>
              <Suspense
                fallback={
                  <div className="flex justify-center py-16">
                    <Loader2 className="h-5 w-5 animate-spin text-slate-400" />
                  </div>
                }
              >
                {tab === "results" ? <Results /> : tab === "plan" ? <PlanPanel /> : <Insights />}
              </Suspense>
            </section>
          </div>
        )}
      </main>

      {settingsOpen && st.status && <SettingsDialog status={st.status} onClose={() => setSettingsOpen(false)} onSaved={() => void st.loadStatus()} />}
    </div>
  );
}

function SettingsDialog({ status, onClose, onSaved }: { status: StudioStatus; onClose: () => void; onSaved: (s: StudioStatus) => void }) {
  const [users, setUsers] = useState(status.allowed_users.join(", "));
  const [repo, setRepo] = useState(status.hf_repo);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [clearing, setClearing] = useState(false);
  const editable = status.can_spend;
  const suggestions = status.hf_accounts.map((a) => `${a}/beverage-gen`);

  const save = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const s = await studio.saveSettings({
        allowed_users: users
          .split(/[,\s]+/)
          .map((u) => u.trim())
          .filter(Boolean),
        hf_repo: repo.trim(),
      });
      onSaved(s);
      setMsg({ ok: true, text: repo !== status.hf_repo ? "Saved — the Studio restarts in a moment to use the new repo." : "Saved." });
    } catch (e) {
      setMsg({ ok: false, text: errText(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div className="w-full max-w-lg rounded-2xl border border-slate-200 bg-white p-5 shadow-2xl dark:border-slate-700 dark:bg-ink-900" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center">
          <h2 className="text-base font-extrabold">Studio settings</h2>
          <button type="button" onClick={onClose} className="ml-auto rounded p-1 hover:bg-slate-100 dark:hover:bg-slate-800">
            <X className="h-4 w-4" />
          </button>
        </div>
        {!editable && <div className="mt-2"><Banner kind="info">Only {status.allowed_users.join(", ")} can change these.</Banner></div>}
        <div className="mt-4 space-y-4 text-[12px]">
          <label className="block">
            <span className="mb-1 block font-bold">Who can start paid runs</span>
            <input
              value={users}
              disabled={!editable}
              onChange={(e) => setUsers(e.target.value)}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 dark:border-slate-600 dark:bg-ink-800"
            />
            <span className="mt-1 block text-[11px] text-slate-500">Labeler user names, comma separated. Everyone else can browse, collect, label and send runs.</span>
          </label>
          <label className="block">
            <span className="mb-1 block font-bold">Hugging Face dataset repo for uploads</span>
            <input
              value={repo}
              disabled={!editable}
              onChange={(e) => setRepo(e.target.value)}
              placeholder="account/beverage-gen"
              list="studio-hf-repos"
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 font-mono dark:border-slate-600 dark:bg-ink-800"
            />
            <datalist id="studio-hf-repos">
              {suggestions.map((s) => (
                <option key={s} value={s} />
              ))}
            </datalist>
            <span className="mt-1 block text-[11px] text-slate-500">
              Private dataset; created if it doesn't exist. Uses the Hugging Face token saved in Keys
              {status.hf_accounts.length ? ` (${status.hf_accounts.join(", ")})` : " — none saved yet"}.
            </span>
          </label>
          <div className="rounded-xl bg-slate-50 p-3 text-[11px] text-slate-600 dark:bg-slate-800/50 dark:text-slate-300">
            Generator keys (OpenAI, Gemini, OpenRouter) live in <b>Keys</b>. Saving a key restarts the Studio automatically once no generation is running.
          </div>
          <details className="rounded-xl border border-rose-300/60 px-3 py-2 dark:border-rose-900/60">
            <summary className="cursor-pointer select-none text-[11px] font-bold text-rose-600 dark:text-rose-300">Danger zone</summary>
            <div className="mt-2 flex items-center gap-2">
              <Btn kind="danger" disabled={!editable} onClick={() => setClearing(true)}>
                Clear eval set, runs and analysis…
              </Btn>
              <span className="text-[11px] text-slate-500">Starts the Studio from zero.</span>
            </div>
          </details>
          {msg && <Banner kind={msg.ok ? "ok" : "error"}>{msg.text}</Banner>}
        </div>
        {clearing && (
          <ClearConfirm
            onCancel={() => setClearing(false)}
            onDone={() => {
              setClearing(false);
              setMsg({ ok: true, text: "Cleared. Reload the page to start fresh." });
            }}
          />
        )}
        <div className="mt-5 flex items-center gap-2">
          <Btn
            kind="ghost"
            disabled={!editable}
            onClick={async () => {
              await studio.restart().catch(() => undefined);
              setMsg({ ok: true, text: "The Studio restarts as soon as no generation is running." });
            }}
          >
            <RotateCw className="h-3.5 w-3.5" /> Restart Studio
          </Btn>
          <div className="ml-auto flex gap-2">
            <Btn kind="ghost" onClick={onClose}>
              Close
            </Btn>
            <Btn kind="primary" busy={busy} disabled={!editable} onClick={() => void save()}>
              Save
            </Btn>
          </div>
        </div>
      </div>
    </div>
  );
}
