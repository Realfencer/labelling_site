import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, Loader2, RotateCw, Settings, Sparkles, X } from "lucide-react";
import { cn } from "../lib/utils";
import KeysButton from "../components/KeysButton";
import { studio } from "../studio/api";
import type { Health, StudioStatus } from "../studio/types";
import { Banner, Btn, Chip, errText } from "../studio/ui";
import ClearConfirm from "../studio/ClearConfirm";

const AnalyzeTab = lazy(() => import("../studio/AnalyzeTab"));
const GenerateTab = lazy(() => import("../studio/GenerateTab"));

type Tab = "analyze" | "generate";

function readTab(): Tab {
  try {
    return localStorage.getItem("studio_tab") === "analyze" ? "analyze" : "generate";
  } catch {
    return "generate";
  }
}

export default function StudioPage() {
  const [tab, setTab] = useState<Tab>(readTab);
  const [status, setStatus] = useState<StudioStatus | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);

  const reloadStatus = useCallback(async () => {
    try {
      const s = await studio.status();
      setStatus(s);
      if (s.up) setHealth(s.health as Health);
      setError(s.up ? "" : "The Studio service is starting or stopped (score-studio). It restarts by itself — this page retries.");
    } catch (e) {
      setError(errText(e));
    }
  }, []);
  const reloadHealth = useCallback(async () => {
    try {
      setHealth(await studio.health());
    } catch (e) {
      setError(errText(e));
    }
  }, []);

  useEffect(() => {
    void reloadStatus();
  }, [reloadStatus]);
  useEffect(() => {
    if (status?.up) return;
    const t = window.setInterval(() => void reloadStatus(), 4000);
    return () => window.clearInterval(t);
  }, [status?.up, reloadStatus]);

  const pick = (t: Tab) => {
    setTab(t);
    try {
      localStorage.setItem("studio_tab", t);
    } catch {
      /* private mode */
    }
  };

  const keys = status?.keys;
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-ink-950 dark:text-slate-100">
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/95 px-4 py-2.5 backdrop-blur dark:border-slate-800 dark:bg-ink-900/95">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-3">
          <Link to="/" className="inline-flex items-center gap-1 rounded-md border border-slate-300 px-2 py-1 text-[12px] font-semibold hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-800">
            <ArrowLeft className="h-3.5 w-3.5" /> Labeler
          </Link>
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500 to-sky-500 text-white shadow">
              <Sparkles className="h-4 w-4" />
            </div>
            <div>
              <h1 className="text-base font-extrabold leading-tight">Studio · Beverage image generation</h1>
              <div className="text-[11px] text-slate-500 dark:text-slate-400">Analyze the challenge look → generate look-alike training images → label them in Own</div>
            </div>
          </div>
          <nav className="ml-2 flex rounded-xl bg-slate-100 p-1 dark:bg-slate-800">
            {(["analyze", "generate"] as Tab[]).map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => pick(t)}
                className={cn(
                  "rounded-lg px-3 py-1 text-[12px] font-bold capitalize transition",
                  tab === t ? "bg-white text-slate-900 shadow dark:bg-ink-900 dark:text-white" : "text-slate-500 hover:text-slate-800 dark:hover:text-slate-200",
                )}
              >
                {t}
              </button>
            ))}
          </nav>
          <div className="ml-auto flex flex-wrap items-center gap-2 text-[11px]">
            <Chip tone={status?.up ? "green" : "rose"} title={`service ${status?.service || "?"}`}>
              <span className={cn("h-1.5 w-1.5 rounded-full", status?.up ? "bg-emerald-500" : "bg-rose-500")} />
              {status?.up ? "studio running" : "studio offline"}
            </Chip>
            {keys && (
              <>
                <Chip tone={keys.openai.configured ? "green" : "amber"} title={keys.openai.masked || "no key"}>
                  OpenAI {keys.openai.configured ? "✓" : "—"}
                </Chip>
                <Chip tone={keys.gemini.configured ? "green" : "amber"} title={keys.gemini.masked || "no key"}>
                  Gemini {keys.gemini.configured ? "✓" : "—"}
                </Chip>
              </>
            )}
            <KeysButton className="px-2" />
            <button
              type="button"
              onClick={() => setSettingsOpen(true)}
              className="inline-flex items-center gap-1 rounded-md border border-slate-300 px-2 py-1 font-semibold hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-800"
            >
              <Settings className="h-3.5 w-3.5" /> Settings
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1600px] space-y-4 p-4">
        {error && <Banner kind="warn">{error}</Banner>}
        {!status ? (
          <div className="flex items-center justify-center gap-2 py-24 text-sm text-slate-500">
            <Loader2 className="h-4 w-4 animate-spin" /> Connecting to the Studio…
          </div>
        ) : (
          <Suspense
            fallback={
              <div className="flex justify-center py-16">
                <Loader2 className="h-5 w-5 animate-spin text-slate-400" />
              </div>
            }
          >
            {tab === "analyze" ? (
              <AnalyzeTab health={health} status={status} reloadHealth={() => void reloadHealth()} onAnalyzed={() => void reloadStatus()} />
            ) : (
              <GenerateTab health={health} status={status} reloadHealth={() => void reloadHealth()} />
            )}
          </Suspense>
        )}
      </main>

      {settingsOpen && status && <SettingsDialog status={status} onClose={() => setSettingsOpen(false)} onSaved={setStatus} />}
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
