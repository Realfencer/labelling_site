import { useEffect, useMemo, useState } from "react";
import { Check, ChevronDown, KeyRound, Loader2, Lock, RefreshCw, Trash2, X } from "lucide-react";
import { api, type KeyService } from "../api";
import { cn } from "../lib/utils";
import { timeAgo, useTeam } from "../store/useTeam";
import { useKeys } from "../store/useKeys";

const LS_CHUTES = "chutes_api_key";
const LS_ULTRA = "ultralytics_platform_api_key";
// Hugging Face tokens that the older per-dialog inputs left in this browser.
const LS_HF = ["labeler_hf_compare_token", "labeler_base_model_token"];
const SS_HF = ["find_best_hf_token"];

function readHfTokens(): string[] {
  const out = new Set<string>();
  try {
    LS_HF.forEach((k) => {
      const v = (localStorage.getItem(k) || "").trim();
      if (v.startsWith("hf_")) out.add(v);
    });
    SS_HF.forEach((k) => {
      const v = (sessionStorage.getItem(k) || "").trim();
      if (v.startsWith("hf_")) out.add(v);
    });
  } catch {
    /* ignore */
  }
  return [...out];
}
function clearHfTokens() {
  try {
    LS_HF.forEach((k) => localStorage.removeItem(k));
    SS_HF.forEach((k) => sessionStorage.removeItem(k));
  } catch {
    /* ignore */
  }
}

function readLs(key: string): string {
  try {
    return (localStorage.getItem(key) || "").trim();
  } catch {
    return "";
  }
}
function clearLs(...keys: string[]) {
  try {
    keys.forEach((k) => localStorage.removeItem(k));
  } catch {
    /* ignore */
  }
}

function ServiceRow({
  service,
  title,
  hint,
  status,
}: {
  service: KeyService;
  title: string;
  hint: string;
  status: { configured: boolean; masked: string };
}) {
  const setView = useKeys((s) => s.setView);
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    setMsg(null);
    try {
      await fn();
    } catch (e) {
      setMsg({ ok: false, text: String(e).replace(/^Error:\s*/, "") });
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="rounded-lg border border-slate-200 p-3 dark:border-slate-700">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-[13px] font-bold">{title}</div>
          <div className="text-[11px] text-slate-500 dark:text-slate-400">{hint}</div>
        </div>
        {status.configured ? (
          <span className="inline-flex items-center gap-1 rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-bold text-emerald-600 dark:text-emerald-300">
            <Check className="h-3 w-3" /> saved · {status.masked}
          </span>
        ) : (
          <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] font-bold text-amber-600 dark:text-amber-300">not set</span>
        )}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <input
          type="password"
          autoComplete="off"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={status.configured ? "Paste a new key to replace it" : "Paste the API key"}
          className="h-8 min-w-[200px] flex-1 rounded border border-slate-300 bg-transparent px-2 text-[12px] dark:border-slate-600"
        />
        <button
          type="button"
          disabled={busy || value.trim().length < 8}
          onClick={() =>
            void run(async () => {
              const r = await api.keysPut(service, value.trim());
              setView(r.keys);
              setValue("");
              setMsg({ ok: true, text: "Saved" });
            })
          }
          className="h-8 rounded bg-sky-600 px-3 text-[12px] font-bold text-white hover:bg-sky-500 disabled:opacity-40"
        >
          Save
        </button>
        {status.configured && (
          <>
            <button
              type="button"
              disabled={busy}
              onClick={() =>
                void run(async () => {
                  const r = await api.keysTest(service);
                  setMsg({ ok: r.ok, text: r.detail });
                })
              }
              className="h-8 rounded border border-slate-300 px-2.5 text-[12px] font-semibold hover:bg-slate-100 disabled:opacity-40 dark:border-slate-600 dark:hover:bg-slate-800"
            >
              Test
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() =>
                void run(async () => {
                  const r = await api.keysDelete(service);
                  setView(r.keys);
                })
              }
              title="Remove the saved key"
              className="h-8 rounded border border-slate-300 px-2 text-slate-500 hover:text-rose-500 disabled:opacity-40 dark:border-slate-600"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </>
        )}
        {busy && <Loader2 className="h-4 w-4 animate-spin text-slate-400" />}
      </div>
      {msg && <div className={cn("mt-1 text-[11px]", msg.ok ? "text-emerald-600 dark:text-emerald-300" : "text-rose-500")}>{msg.text}</div>}
    </section>
  );
}

function HfSection() {
  const view = useKeys((s) => s.view);
  const models = useKeys((s) => s.models);
  const setView = useKeys((s) => s.setView);
  const refresh = useKeys((s) => s.refresh);
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);
  const [scanning, setScanning] = useState<string | null>(null);

  // While a freshly saved account is still being scanned, keep refreshing.
  const anyUnscanned = (view?.hf ?? []).some((a) => !a.scan?.at);
  useEffect(() => {
    if (!anyUnscanned) return;
    const t = window.setInterval(() => void refresh(), 2000);
    return () => window.clearInterval(t);
  }, [anyUnscanned, refresh]);

  const add = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const r = await api.keysPutHf(token.trim());
      setView(r.keys);
      setToken("");
      setMsg({ ok: true, text: `Saved — account “${r.account}”${r.orgs.length ? ` (+ ${r.orgs.join(", ")})` : ""}. Scanning its models…` });
      void refresh();
    } catch (e) {
      setMsg({ ok: false, text: String(e).replace(/^Error:\s*/, "") });
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="rounded-lg border border-slate-200 p-3 dark:border-slate-700">
      <div className="text-[13px] font-bold">Hugging Face accounts</div>
      <div className="text-[11px] text-slate-500 dark:text-slate-400">
        One token per account. Pushing picks the right token from the repo owner, and your models are scanned automatically.
      </div>
      <div className="mt-2 space-y-2">
        {(view?.hf ?? []).map((a) => {
          const mine = models.filter((m) => m.account_id === a.id);
          return (
            <div key={a.id} className="rounded-md border border-slate-200 dark:border-slate-700">
              <div className="flex flex-wrap items-center gap-2 px-2.5 py-1.5">
                <button type="button" onClick={() => setOpenId(openId === a.id ? null : a.id)} className="flex min-w-0 items-center gap-1.5 text-left">
                  <ChevronDown className={cn("h-3.5 w-3.5 shrink-0 transition", openId === a.id && "rotate-180")} />
                  <span className="truncate text-[13px] font-extrabold">{a.account}</span>
                  {a.orgs.length > 0 && <span className="truncate text-[10px] text-slate-400">+ {a.orgs.join(", ")}</span>}
                </button>
                <span className="font-mono text-[10px] text-slate-400">{a.masked}</span>
                <span className="ml-auto text-[11px] text-slate-500 dark:text-slate-400">
                  {scanning === a.id || !a.scan?.at ? (
                    <span className="inline-flex items-center gap-1">
                      <Loader2 className="h-3 w-3 animate-spin" /> scanning…
                    </span>
                  ) : a.scan.error ? (
                    <span className="text-rose-500" title={a.scan.error}>
                      scan failed
                    </span>
                  ) : (
                    `${a.n_models} models · scanned ${timeAgo(a.scan.at)}`
                  )}
                </span>
                <button
                  type="button"
                  title="Rescan this account's models"
                  disabled={scanning === a.id}
                  onClick={async () => {
                    setScanning(a.id);
                    try {
                      const r = await api.keysScanHf(a.id);
                      setView(r.keys);
                      await refresh();
                    } catch {
                      /* shown via scan.error */
                    } finally {
                      setScanning(null);
                    }
                  }}
                  className="rounded border border-slate-300 p-1 text-slate-500 hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-800"
                >
                  <RefreshCw className={cn("h-3 w-3", scanning === a.id && "animate-spin")} />
                </button>
                <button
                  type="button"
                  title="Remove this account's token"
                  onClick={async () => setView((await api.keysDeleteHf(a.id)).keys)}
                  className="rounded border border-slate-300 p-1 text-slate-500 hover:text-rose-500 dark:border-slate-600"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>
              {openId === a.id && (
                <div className="max-h-40 overflow-y-auto border-t border-slate-100 px-2.5 py-1.5 dark:border-slate-800">
                  {mine.length === 0 ? (
                    <div className="text-[11px] text-slate-400">No models found yet.</div>
                  ) : (
                    mine.map((m) => (
                      <div key={m.repo_id} className="flex items-center justify-between gap-2 py-0.5 text-[11px]">
                        <span className="truncate font-mono">{m.repo_id}</span>
                        <span className={cn("shrink-0 rounded px-1 text-[9px] font-bold uppercase", m.private ? "bg-violet-500/15 text-violet-500" : "bg-slate-500/10 text-slate-400")}>
                          {m.private ? "private" : "public"}
                        </span>
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          );
        })}
        {(view?.hf.length ?? 0) === 0 && <div className="text-[11px] text-slate-400">No account yet — add a token below.</div>}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <input
          type="password"
          autoComplete="off"
          value={token}
          onChange={(e) => setToken(e.target.value)}
          placeholder="Add a Hugging Face token (hf_…) — a write token for pushing"
          className="h-8 min-w-[220px] flex-1 rounded border border-slate-300 bg-transparent px-2 text-[12px] dark:border-slate-600"
        />
        <button
          type="button"
          disabled={busy || token.trim().length < 20}
          onClick={() => void add()}
          className="inline-flex h-8 items-center gap-1 rounded bg-sky-600 px-3 text-[12px] font-bold text-white hover:bg-sky-500 disabled:opacity-40"
        >
          {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />} Add &amp; scan
        </button>
      </div>
      {msg && <div className={cn("mt-1 text-[11px]", msg.ok ? "text-emerald-600 dark:text-emerald-300" : "text-rose-500")}>{msg.text}</div>}
    </section>
  );
}

export default function KeysModal() {
  const open = useKeys((s) => s.open);
  const setOpen = useKeys((s) => s.setOpen);
  const view = useKeys((s) => s.view);
  const setView = useKeys((s) => s.setView);
  const refresh = useKeys((s) => s.refresh);
  const me = useTeam((s) => s.me?.user);
  const [found, setFound] = useState({ chutes: "", ultra: "" });
  const [foundHf, setFoundHf] = useState<string[]>([]);
  const [importMsg, setImportMsg] = useState("");

  useEffect(() => {
    if (open) {
      void refresh();
      setFound({ chutes: readLs(LS_CHUTES), ultra: readLs(LS_ULTRA) });
      setFoundHf(readHfTokens());
      setImportMsg("");
    }
  }, [open, refresh]);

  const importable = useMemo(
    () => ({
      chutes: Boolean(found.chutes) && !view?.chutes.configured,
      ultra: Boolean(found.ultra) && !view?.ultralytics.configured,
    }),
    [found, view],
  );

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/60 p-4" onClick={() => setOpen(false)}>
      <div
        className="mt-8 w-full max-w-[640px] rounded-xl border border-slate-200 bg-white p-4 shadow-2xl dark:border-slate-700 dark:bg-ink-900"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2">
            <KeyRound className="h-5 w-5 text-sky-500" />
            <div>
              <h2 className="text-base font-extrabold">API keys</h2>
              <p className="text-[11px] text-slate-500 dark:text-slate-400">
                Set once, used everywhere on this site — shared by you and your partner{me ? ` (you are ${me})` : ""}.
              </p>
            </div>
          </div>
          <button type="button" onClick={() => setOpen(false)} className="rounded p-1 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        {(importable.chutes || importable.ultra) && (
          <div className="mt-3 flex flex-wrap items-center gap-2 rounded-md bg-sky-500/10 px-3 py-2 text-[12px] text-sky-700 dark:text-sky-300">
            Found {[importable.chutes && "a Chutes key", importable.ultra && "an Ultralytics key"].filter(Boolean).join(" and ")} saved in this browser.
            <button
              type="button"
              onClick={async () => {
                const r = await api.keysImport({
                  chutes_api_key: importable.chutes ? found.chutes : "",
                  ultralytics_api_key: importable.ultra ? found.ultra : "",
                });
                setView(r.keys);
                clearLs(...(r.imported.includes("chutes") ? [LS_CHUTES] : []), ...(r.imported.includes("ultralytics") ? [LS_ULTRA] : []));
                setFound({ chutes: "", ultra: "" });
              }}
              className="rounded bg-sky-600 px-2 py-0.5 text-[11px] font-bold text-white hover:bg-sky-500"
            >
              Import here
            </button>
          </div>
        )}

        {foundHf.length > 0 && (
          <div className="mt-3 flex flex-wrap items-center gap-2 rounded-md bg-sky-500/10 px-3 py-2 text-[12px] text-sky-700 dark:text-sky-300">
            Found {foundHf.length} Hugging Face token{foundHf.length > 1 ? "s" : ""} saved in this browser.
            <button
              type="button"
              onClick={async () => {
                let added = 0;
                for (const t of foundHf) {
                  try {
                    await api.keysPutHf(t);
                    added += 1;
                  } catch {
                    /* expired token: skip */
                  }
                }
                clearHfTokens();
                setFoundHf([]);
                setImportMsg(added ? `Added ${added} account${added > 1 ? "s" : ""} — scanning their models…` : "None of them were still valid.");
                await refresh();
              }}
              className="rounded bg-sky-600 px-2 py-0.5 text-[11px] font-bold text-white hover:bg-sky-500"
            >
              Add them (verified)
            </button>
          </div>
        )}
        {importMsg && <div className="mt-2 text-[11px] text-emerald-600 dark:text-emerald-300">{importMsg}</div>}

        <div className="mt-3 space-y-3">
          <HfSection />
          <ServiceRow
            service="chutes"
            title="Chutes API key"
            hint="Used for SAM3 auto-labeling of Own images."
            status={view?.chutes ?? { configured: false, masked: "" }}
          />
          <ServiceRow
            service="ultralytics"
            title="Ultralytics API key"
            hint="Used to upload labeled datasets to Ultralytics Platform."
            status={view?.ultralytics ?? { configured: false, masked: "" }}
          />
          <div className="pt-1 text-[10px] font-bold uppercase tracking-wider text-slate-400">Studio · image generation</div>
          <ServiceRow
            service="openai"
            title="OpenAI API key"
            hint="GPT Image generation in the Studio (Batch API). Paid per image."
            status={view?.openai ?? { configured: false, masked: "" }}
          />
          <ServiceRow
            service="gemini"
            title="Gemini API key"
            hint="Gemini image generation in the Studio, and its scene captions."
            status={view?.gemini ?? { configured: false, masked: "" }}
          />
          <ServiceRow
            service="openrouter"
            title="OpenRouter API key"
            hint="Optional route to the image models through OpenRouter."
            status={view?.openrouter ?? { configured: false, masked: "" }}
          />
        </div>

        <div className="mt-3 flex items-start gap-1.5 text-[10px] text-slate-400">
          <Lock className="mt-px h-3 w-3 shrink-0" />
          <span>
            Keys are stored encrypted on the server and are never shown again after saving. Anyone signed in to this site can use them
            {view?.updated_by ? ` · last changed by ${view.updated_by}` : ""}.
          </span>
        </div>
      </div>
    </div>
  );
}
