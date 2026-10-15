import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { studio } from "./api";
import type { AnalyzeJob, GenJob, Generator, Health, ModelErrors, Package, Run, SendJob, StudioStatus } from "./types";
import { errText, useJob } from "./ui";

export type Toast = { kind: "info" | "ok" | "error"; text: string; link?: string } | null;

/** Everything the Studio page shares: server state, the three background jobs, and one toast line. */
function useStudioState() {
  const [status, setStatus] = useState<StudioStatus | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [pkg, setPkg] = useState<Package | null>(null);
  const [nAllSources, setNAllSources] = useState(0);
  const [runs, setRuns] = useState<Run[]>([]);
  const [errors, setErrors] = useState<ModelErrors | null>(null);
  const [activeRun, setActiveRun] = useState("");
  const [source, setSource] = useState(() => {
    try {
      return localStorage.getItem("studio_source_image") || "";
    } catch {
      return "";
    }
  });
  const [toast, setToast] = useState<Toast>(null);
  const [generators, setGenerators] = useState<Record<string, Generator>>({});
  useEffect(() => {
    studio
      .generators()
      .then((r) => setGenerators(r.blocks || {}))
      .catch(() => undefined);
  }, []);
  /** openai | gemini | null for a source frame name like "block_9246600.png". */
  const genOf = useCallback((fn: string) => generators[fn.replace(/^block_/, "").replace(/\.[a-z]+$/i, "")] ?? null, [generators]);
  const [offline, setOffline] = useState("");

  const loadStatus = useCallback(async () => {
    try {
      const s = await studio.status();
      setStatus(s);
      if (s.up) setHealth(s.health as Health);
      setOffline(s.up ? "" : "The Studio service is starting or stopped. It restarts by itself — this page keeps trying.");
    } catch (e) {
      setOffline(errText(e));
    }
  }, []);

  const loadHealth = useCallback(async () => {
    try {
      setHealth(await studio.health());
    } catch {
      /* status poll reports it */
    }
  }, []);

  const loadData = useCallback(async () => {
    const [p, all, r, e] = await Promise.all([
      studio.pkg().catch(() => null),
      studio.pkg(false).catch(() => null),
      studio.runs().catch(() => ({ runs: [] as Run[] })),
      studio.modelErrors().catch(() => null),
    ]);
    setPkg(p);
    setNAllSources(all?.sources?.length ?? p?.sources?.length ?? 0);
    setRuns(r.runs);
    setErrors(e);
    setActiveRun((cur) => (cur && r.runs.some((x) => x.name === cur) ? cur : r.runs[0]?.name || ""));
  }, []);

  const refreshAll = useCallback(async () => {
    await Promise.all([loadStatus(), loadData()]);
  }, [loadStatus, loadData]);

  useEffect(() => {
    void refreshAll();
  }, [refreshAll]);
  useEffect(() => {
    if (status?.up) return;
    const t = window.setInterval(() => void loadStatus(), 4000);
    return () => window.clearInterval(t);
  }, [status?.up, loadStatus]);

  const analyze = useJob<AnalyzeJob>(studio.analyzeJob, 900, (j) => {
    if (j.status === "done") setToast({ kind: "ok", text: `${j.message || "Analysis complete"} — the generation package was rebuilt.` });
    else if (j.status === "error") setToast({ kind: "error", text: j.error || j.message });
    void loadStatus();
    void loadData();
  });
  const gen = useJob<GenJob>(studio.genJob, 1000, (j) => {
    const run = (j.run_dir || "").split("/").filter(Boolean).pop() || "";
    if (j.status === "done") {
      setToast({ kind: "ok", text: j.message || "Done", link: j.hf_upload?.ok && j.hf_upload.url ? j.hf_upload.url : undefined });
      if (run) setActiveRun(run);
    } else if (j.status === "error") setToast({ kind: "error", text: j.error || j.message });
    void loadData();
  });
  const send = useJob<SendJob>(studio.sendJob, 1000, (j) => {
    if (j.status === "done" && j.result)
      setToast({
        kind: "ok",
        text: j.result.images
          ? `Sent ${j.result.images} images to the Own workspace — they are ready to label.${j.result.skipped ? ` ${j.result.skipped} were already there and were skipped.` : ""}`
          : `Nothing new to send: all ${j.result.skipped ?? 0} images are already in the Own workspace.`,
      });
    else if (j.status === "error") setToast({ kind: "error", text: j.error || j.message });
  });

  const pickSource = useCallback((s: string) => {
    setSource(s);
    try {
      localStorage.setItem("studio_source_image", s);
    } catch {
      /* private mode */
    }
  }, []);

  const sources = useMemo(() => (pkg?.sources || []).filter((s) => s.source_image.includes("block_")), [pkg]);

  // Which sources a Final run uses: null = every source; a list = only those (ticked on the Plan tab).
  const [finalPick, setFinalPick] = useState<string[] | null>(() => {
    try {
      const v = JSON.parse(localStorage.getItem("studio_final_sel") || "null");
      return Array.isArray(v) ? v.map(String) : null;
    } catch {
      return null;
    }
  });
  const setFinalSel = useCallback((v: string[] | null) => {
    setFinalPick(v);
    try {
      if (v == null) localStorage.removeItem("studio_final_sel");
      else localStorage.setItem("studio_final_sel", JSON.stringify(v));
    } catch {
      /* private mode */
    }
  }, []);
  /** The ticked sources that still exist in the package (null = all of them). */
  const finalSel = useMemo(() => {
    if (finalPick == null) return null;
    const have = new Set(sources.map((s) => s.source_image));
    const keep = finalPick.filter((x) => have.has(x));
    // A new eval set replaced every ticked source: fall back to "all" instead of an empty run.
    return keep.length === 0 && finalPick.length > 0 && sources.length > 0 ? null : keep;
  }, [finalPick, sources]);
  const nFinal = finalSel ? finalSel.length : Math.max(sources.length, nAllSources);

  return {
    status,
    health,
    pkg,
    sources,
    nAllSources,
    runs,
    errors,
    activeRun,
    setActiveRun,
    source,
    pickSource,
    finalSel,
    setFinalSel,
    genOf,
    nFinal,
    toast,
    setToast,
    offline,
    loadStatus,
    loadHealth,
    loadData,
    refreshAll,
    analyze,
    gen,
    send,
  };
}

export type StudioState = ReturnType<typeof useStudioState>;
const Ctx = createContext<StudioState | null>(null);

export function StudioProvider({ children }: { children: ReactNode }) {
  const value = useStudioState();
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useStudio(): StudioState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useStudio outside StudioProvider");
  return v;
}

/** What the user should do next, worked out from the state (the banner at the top of the page). */
export type NextStep = {
  key: "update" | "analyze" | "test" | "collect" | "review" | "running" | "final" | "offline";
  title: string;
  detail: string;
  cta?: string;
  tone: "violet" | "sky" | "amber" | "emerald" | "slate";
};

export function nextStep(st: StudioState): NextStep {
  const { status, health, runs, analyze, gen, send, nAllSources } = st;
  if (!status?.up) return { key: "offline", title: "Connecting to the Studio…", detail: st.offline, tone: "slate" };
  if (analyze.job?.status === "running") return { key: "running", title: "Analyzing the eval set", detail: analyze.job.message, tone: "sky" };
  if (gen.job?.status === "running")
    return { key: "running", title: gen.job.mode === "autolabel" ? "Labelling a run" : `Generating · ${gen.job.mode} run`, detail: gen.job.message, tone: "violet" };
  if (send.job?.status === "running") return { key: "running", title: "Sending to Own", detail: send.job.message, tone: "emerald" };
  if (!status.gt_images) return { key: "update", title: "Start with the eval set", detail: "Copy your newest Done Platform images into the Studio.", cta: "Update from labeler", tone: "amber" };
  const latest = runs[0];
  if (latest?.pending) return { key: "collect", title: "A batch is generating", detail: `${latest.name} — collect the finished images.`, cta: "Collect", tone: "sky" };
  if (!runs.some((r) => r.mode === "test" && r.provider === health?.gen_mode))
    return { key: "test", title: "Try one source first", detail: "A test run shows the look before you pay for the full set.", cta: "Test generate", tone: "violet" };
  if (latest && latest.count > 0)
    return {
      key: "review",
      title: `Review ${latest.count} images from the last run`,
      detail: latest.near_copies ? `${latest.near_copies} look too close to their source — leave them out, then send the rest to Own.` : "Check them, then send them to Own for labelling.",
      cta: "Review",
      tone: "emerald",
    };
  const detail = st.finalSel ? `${st.nFinal} of ${nAllSources} sources ticked on the Plan tab.` : `${nAllSources} sources are ready.`;
  return { key: "final", title: "Generate the full set", detail, cta: "Final generate", tone: "violet" };
}
