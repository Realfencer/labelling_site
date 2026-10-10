import { useKeys } from "../store/useKeys";
import { useEffect, useMemo, useState } from "react";
import { CloudUpload, Loader2, X } from "lucide-react";
import { api, type OwnDayRow, type UltralyticsDataset } from "../api";
import type { ChallengeItem } from "../types";
import { checksFailComplete, checksPassComplete } from "../labelChecks";
import { cn } from "../lib/utils";
import ImageSelectionField from "./ImageSelectionField";

const HIST_BINS = 48;

type ConflictPolicy = "skip" | "replace" | "keep_both";
type Quality = "all" | "pass" | "fail";
type ForceSplit = "" | "train" | "val" | "test";

export default function UploadUltralyticsModal({
  open,
  onClose,
  challenges,
  ownMode = false,
  dayMode,
}: {
  open: boolean;
  onClose: () => void;
  challenges: ChallengeItem[];
  ownMode?: boolean;
  /** Own day list page: choose whole days (several days = one dataset) instead of a range of one day. */
  dayMode?: { preselect: string[] };
}) {
  const inDays = Boolean(dayMode);
  const [dayRows, setDayRows] = useState<OwnDayRow[]>([]);
  const [selDays, setSelDays] = useState<string[]>([]);
  const labeled = useMemo(
    () => challenges.filter((c) => c.loaded && c.reviewed),
    [challenges],
  );
  const blocks = useMemo(() => labeled.map((c) => c.block).filter((b) => b > 0).sort((a, b) => a - b), [labeled]);
  const minBlock = blocks[0] ?? 0;
  const maxBlock = blocks[blocks.length - 1] ?? 0;

  // The Ultralytics key lives in the shared key vault on the server.
  const keys = useKeys((st) => st.view);
  const setKeysOpen = useKeys((st) => st.setOpen);
  const keyOk = Boolean(keys?.ultralytics.configured);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [quality, setQuality] = useState<Quality>("all");
  const [picked, setPicked] = useState<string[] | null>(null);
  // Cumulative boundaries: train = [0, trainEnd), val = [trainEnd, valEnd), test = [valEnd, 100]
  const [trainEnd, setTrainEnd] = useState(80);
  const [valEnd, setValEnd] = useState(90);
  const [forceSplit, setForceSplit] = useState<ForceSplit>("");
  const [conflict, setConflict] = useState<ConflictPolicy>("skip");
  const [owner, setOwner] = useState("");
  const [datasets, setDatasets] = useState<UltralyticsDataset[]>([]);
  const [datasetSlug, setDatasetSlug] = useState("");
  const [createNew, setCreateNew] = useState(false);
  const [newSlug, setNewSlug] = useState("");
  const [loadingList, setLoadingList] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [resultUrl, setResultUrl] = useState("");
  const [confirmOpen, setConfirmOpen] = useState(false);

  const trainPct = trainEnd;
  const valPct = Math.max(0, valEnd - trainEnd);
  const testPct = Math.max(0, 100 - valEnd);

  useEffect(() => {
    if (!open) return;
    setFrom(minBlock ? String(minBlock) : "");
    setTo(maxBlock ? String(maxBlock) : "");
    setQuality("all");
    setPicked(null);
    setTrainEnd(80);
    setValEnd(90);
    setForceSplit("");
    setConflict("skip");
    setError("");
    setResultUrl("");
    setBusy(false);
    setConfirmOpen(false);
    setCreateNew(false);
    setNewSlug("");
  }, [open, minBlock, maxBlock]);

  useEffect(() => {
    if (!open || !inDays) return;
    let alive = true;
    api
      .listOwnDays()
      .then((r) => {
        if (!alive) return;
        setDayRows(r.days);
        const withDone = r.days.filter((d) => d.done > 0).map((d) => d.date);
        const pre = (dayMode?.preselect ?? []).filter((d) => withDone.includes(d));
        setSelDays(pre.length ? pre : withDone);
      })
      .catch((e) => alive && setError(String(e).replace(/^Error:\s*/i, "")));
    return () => {
      alive = false;
    };
  }, [open, inDays]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!open) return;
    if (!keyOk) {
      setDatasets([]);
      setOwner("");
      return;
    }
    let alive = true;
    const timer = window.setTimeout(() => {
      setLoadingList(true);
      setError("");
      api
        .ultralyticsListDatasets("")
        .then((res) => {
          if (!alive) return;
          setOwner(res.owner);
          setDatasets(res.datasets);
          setDatasetSlug((prev) => prev || res.datasets[0]?.dataset || "");
        })
        .catch((e) => {
          if (!alive) return;
          setDatasets([]);
          setOwner("");
          setError(String(e).replace(/^Error:\s*/i, ""));
        })
        .finally(() => {
          if (alive) setLoadingList(false);
        });
    }, 400);
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [open, keyOk]);

  const fromN = Number(from);
  const toN = Number(to);
  const rawLo = Number.isFinite(fromN) ? fromN : minBlock;
  const rawHi = Number.isFinite(toN) ? toN : maxBlock;
  const lo = Math.min(rawLo, rawHi);
  const hi = Math.max(rawLo, rawHi);
  const span = Math.max(1, maxBlock - minBlock);
  const inRange = labeled.filter((c) => c.block >= lo && c.block <= hi);
  const passCount = inRange.filter((c) => checksPassComplete(c)).length;
  const failCount = inRange.filter((c) => checksFailComplete(c)).length;
  const rangeCount = quality === "pass" ? passCount : quality === "fail" ? failCount : inRange.length;
  const daysDone = dayRows.filter((d) => selDays.includes(d.date)).reduce((n, d) => n + d.done, 0);
  const count = inDays ? daysDone : picked ? picked.length : rangeCount;
  const selected = datasets.find((d) => d.dataset === datasetSlug);
  const targetSlug = createNew ? newSlug.trim().toLowerCase().replace(/\s+/g, "-") : datasetSlug;

  const hist = useMemo(() => {
    const bins = new Array(HIST_BINS).fill(0);
    for (const c of labeled) {
      const t = (c.block - minBlock) / span;
      const i = Math.min(HIST_BINS - 1, Math.max(0, Math.floor(t * HIST_BINS)));
      bins[i] += 1;
    }
    const peak = Math.max(1, ...bins);
    return bins.map((n) => n / peak);
  }, [labeled, minBlock, span]);

  const setLo = (n: number) => setFrom(String(Math.min(Math.max(n, minBlock), hi)));
  const setHi = (n: number) => setTo(String(Math.max(Math.min(n, maxBlock), lo)));

  const onTrainEnd = (n: number) => {
    const next = Math.max(0, Math.min(98, n));
    setTrainEnd(next);
    if (valEnd < next + 1) setValEnd(Math.min(100, next + 1));
  };
  const onValEnd = (n: number) => {
    const next = Math.max(trainEnd + 1, Math.min(100, n));
    setValEnd(next);
  };

  const conflictLabel =
    conflict === "skip"
      ? "Skip all same-name images"
      : conflict === "replace"
        ? "Replace all same-name images"
        : "Keep both (store under another name)";

  const canUpload =
    keyOk &&
    count > 0 &&
    Boolean(targetSlug) &&
    (forceSplit !== "" || trainPct + valPct + testPct === 100) &&
    !busy;

  const estTrain = forceSplit === "train" ? count : forceSplit ? 0 : Math.round((count * trainPct) / 100);
  const estVal =
    forceSplit === "val" ? count : forceSplit ? 0 : Math.round((count * valPct) / 100);
  const estTest =
    forceSplit === "test" ? count : forceSplit ? 0 : Math.max(0, count - estTrain - estVal);

  const runUpload = async () => {
    setBusy(true);
    setError("");
    setResultUrl("");
    try {
      const res = await api.ultralyticsUpload({
        api_key: "",
        owner: owner || undefined,
        dataset: targetSlug,
        dataset_id: createNew ? undefined : selected?.id || undefined,
        create_if_missing: createNew,
        dataset_name: createNew ? targetSlug : undefined,
        block_min: inDays ? undefined : lo,
        block_max: inDays ? undefined : hi,
        item_keys: inDays ? undefined : picked ?? undefined,
        own_days: inDays ? selDays : undefined,
        quality: ownMode || picked || inDays ? "all" : quality,
        train_pct: trainPct,
        val_pct: valPct,
        test_pct: testPct,
        force_split: forceSplit || null,
        conflict_policy: conflict,
      });
      setResultUrl(res.url || "");
      setConfirmOpen(false);
      if (createNew) {
        setCreateNew(false);
        setDatasetSlug(res.dataset);
        setDatasets((prev) => {
          if (prev.some((d) => d.dataset === res.dataset)) return prev;
          return [
            { id: res.datasetId, dataset: res.dataset, name: res.dataset, imageCount: res.packed },
            ...prev,
          ];
        });
      }
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
    } finally {
      setBusy(false);
    }
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/55 p-3 backdrop-blur-[2px]">
      <div className="w-full max-w-lg overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-ink-900">
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-700">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-rose-500 text-white">
              <CloudUpload className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-sm font-extrabold text-slate-900 dark:text-white">Upload to Ultralytics</h2>
              <p className="text-[11px] text-slate-500">
                YOLO zip →{" "}
                <a
                  href="https://platform.ultralytics.com/"
                  target="_blank"
                  rel="noreferrer"
                  className="font-semibold text-sky-600 hover:underline"
                >
                  Platform
                </a>
                {owner ? ` · ${owner}` : ""}
              </p>
            </div>
          </div>
          <button type="button" onClick={onClose} className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="max-h-[min(78vh,680px)] space-y-3.5 overflow-y-auto px-4 py-4">
          <div className="flex items-center justify-between rounded-lg border border-slate-200 px-2.5 py-2 text-[12px] dark:border-slate-700">
            {keyOk ? (
              <span className="text-emerald-600 dark:text-emerald-300">Using the saved Ultralytics key ({keys?.ultralytics.masked})</span>
            ) : (
              <span className="text-rose-600 dark:text-rose-300">No Ultralytics API key saved yet</span>
            )}
            <button
              type="button"
              onClick={() => setKeysOpen(true)}
              className="rounded border border-slate-300 px-2 py-0.5 text-[11px] font-bold hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-800"
            >
              {keyOk ? "Change" : "Add key"}
            </button>
          </div>

          {inDays && (
            <div className="rounded-lg border border-slate-200 dark:border-slate-700">
              <div className="flex items-center justify-between border-b border-slate-200 px-2.5 py-1.5 text-[11px] dark:border-slate-700">
                <span className="font-bold uppercase tracking-wider text-slate-500">Days to send</span>
                <span className="flex gap-2">
                  <button type="button" className="font-bold text-sky-600 hover:underline dark:text-sky-300" onClick={() => setSelDays(dayRows.filter((d) => d.done > 0).map((d) => d.date))}>all</button>
                  <button type="button" className="font-bold text-slate-500 hover:underline" onClick={() => setSelDays([])}>none</button>
                </span>
              </div>
              <div className="max-h-52 overflow-y-auto">
                {dayRows.map((d) => (
                  <label key={d.date} className={cn("flex cursor-pointer items-center gap-2 border-b border-slate-100 px-2.5 py-1.5 text-[12px] last:border-0 dark:border-slate-800", d.done === 0 && "cursor-not-allowed opacity-40")}>
                    <input
                      type="checkbox"
                      disabled={d.done === 0}
                      checked={selDays.includes(d.date)}
                      onChange={() => setSelDays((cur) => (cur.includes(d.date) ? cur.filter((x) => x !== d.date) : [...cur, d.date]))}
                    />
                    <span className="font-mono font-bold tabular-nums">{d.date}</span>
                    <span className="ml-auto text-[11px] text-slate-500">
                      <b className="text-emerald-600 dark:text-emerald-300">{d.done} done</b> · {d.total - d.done} not done (not sent)
                    </span>
                  </label>
                ))}
                {dayRows.length === 0 && <div className="px-3 py-4 text-center text-[12px] text-slate-400">Loading days…</div>}
              </div>
            </div>
          )}
          {!inDays && (
          <>
          <ImageSelectionField suggest={inRange.map((c) => c.item_key)} challenges={challenges} ownMode={ownMode} value={picked} onChange={setPicked} verb="upload" />
          <div className={cn("space-y-3.5", picked && "pointer-events-none opacity-40")}>
          <div className="grid grid-cols-2 gap-2">
            <label className="block">
              <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">{ownMode ? "From #" : "From block"}</span>
              <input
                value={from}
                onChange={(e) => setFrom(e.target.value.replace(/[^\d]/g, ""))}
                className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 font-mono text-sm tabular-nums dark:border-slate-700 dark:bg-ink-800"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">{ownMode ? "To #" : "To block"}</span>
              <input
                value={to}
                onChange={(e) => setTo(e.target.value.replace(/[^\d]/g, ""))}
                className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 font-mono text-sm tabular-nums dark:border-slate-700 dark:bg-ink-800"
              />
            </label>
          </div>

          <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 pb-3 pt-3 dark:border-slate-700 dark:bg-ink-950">
            <div className="mb-2 flex h-8 items-end gap-px">
              {hist.map((h, i) => {
                const binLo = minBlock + (i / HIST_BINS) * span;
                const binHi = minBlock + ((i + 1) / HIST_BINS) * span;
                const active = binHi >= lo && binLo <= hi;
                return (
                  <div
                    key={i}
                    className={cn("min-h-[2px] flex-1 rounded-sm", active ? "bg-rose-500" : "bg-slate-300 dark:bg-slate-700")}
                    style={{ height: `${Math.max(8, h * 100)}%` }}
                  />
                );
              })}
            </div>
            <label className="mb-2 block">
              <span className="mb-0.5 flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-500">
                From <span className="font-mono tabular-nums text-slate-700 dark:text-slate-200">{lo}</span>
              </span>
              <input
                type="range"
                min={minBlock}
                max={maxBlock || 1}
                value={lo}
                disabled={!minBlock}
                onChange={(e) => setLo(Number(e.target.value))}
                className="w-full accent-rose-500"
              />
            </label>
            <label className="block">
              <span className="mb-0.5 flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-500">
                To <span className="font-mono tabular-nums text-slate-700 dark:text-slate-200">{hi}</span>
              </span>
              <input
                type="range"
                min={minBlock}
                max={maxBlock || 1}
                value={hi}
                disabled={!minBlock}
                onChange={(e) => setHi(Number(e.target.value))}
                className="w-full accent-rose-500"
              />
            </label>
            <button
              type="button"
              onClick={() => {
                setFrom(String(minBlock));
                setTo(String(maxBlock));
              }}
              className="mt-2 text-[11px] font-bold text-rose-600 hover:underline dark:text-rose-300"
            >
              Reset to all labeled ({minBlock}–{maxBlock})
            </button>
          </div>

          {!ownMode && (
          <div>
            <div className="mb-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-500">Which Done images</div>
            <div className="grid grid-cols-3 gap-1.5">
              {(
                [
                  ["all", "All labeled", inRange.length],
                  ["pass", "Pass R1–R3", passCount],
                  ["fail", "Fail 0/3", failCount],
                ] as const
              ).map(([id, label, n]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setQuality(id)}
                  className={cn(
                    "rounded-lg border px-2 py-2 text-[11px] font-bold",
                    quality === id
                      ? id === "fail"
                        ? "border-rose-400 bg-rose-500 text-white"
                        : id === "pass"
                          ? "border-emerald-400 bg-emerald-500 text-white"
                          : "border-sky-400 bg-sky-500 text-white"
                      : "border-slate-200 text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800",
                  )}
                >
                  <div>{label}</div>
                  <div className="mt-0.5 text-[10px] font-black tabular-nums opacity-80">{n}</div>
                </button>
              ))}
            </div>
          </div>
          )}

          </div>
          </>
          )}

          <div>
            <div className="mb-1 flex items-center justify-between">
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Dataset</span>
              <button
                type="button"
                className="text-[10px] font-semibold text-sky-600 hover:underline"
                onClick={() => setCreateNew((v) => !v)}
              >
                {createNew ? "Pick existing" : "Create new"}
              </button>
            </div>
            {createNew ? (
              <input
                value={newSlug}
                onChange={(e) => setNewSlug(e.target.value)}
                placeholder="my-dataset-slug"
                className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-xs dark:border-slate-700 dark:bg-ink-800"
              />
            ) : (
              <select
                value={datasetSlug}
                onChange={(e) => setDatasetSlug(e.target.value)}
                disabled={loadingList || !datasets.length}
                className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-xs dark:border-slate-700 dark:bg-ink-800"
              >
                {!datasets.length && <option value="">{loadingList ? "Loading…" : "Enter a valid API key"}</option>}
                {datasets.map((d) => (
                  <option key={d.id || d.dataset} value={d.dataset}>
                    {d.name || d.dataset}
                    {d.imageCount != null ? ` · ${d.imageCount} imgs` : ""}
                  </option>
                ))}
              </select>
            )}
          </div>

          <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-3 dark:border-slate-700 dark:bg-ink-950">
            <div className="mb-2 flex items-center justify-between gap-2">
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Train / Val / Test</span>
              <div className="flex flex-wrap gap-1">
                {(["", "train", "val", "test"] as const).map((s) => (
                  <button
                    key={s || "pct"}
                    type="button"
                    onClick={() => setForceSplit(s)}
                    className={cn(
                      "rounded-full border px-2 py-0.5 text-[9px] font-bold uppercase",
                      forceSplit === s
                        ? "border-violet-500 bg-violet-500 text-white"
                        : "border-slate-200 text-slate-500 dark:border-slate-700 dark:text-slate-400",
                    )}
                  >
                    {s === "" ? "% split" : `all ${s}`}
                  </button>
                ))}
              </div>
            </div>

            {forceSplit === "" ? (
              <>
                <div className="mb-2 flex h-4 overflow-hidden rounded-full">
                  <div className="bg-sky-500 transition-all" style={{ width: `${trainPct}%` }} title={`Train ${trainPct}%`} />
                  <div className="bg-violet-500 transition-all" style={{ width: `${valPct}%` }} title={`Val ${valPct}%`} />
                  <div className="bg-amber-400 transition-all" style={{ width: `${testPct}%` }} title={`Test ${testPct}%`} />
                </div>
                <div className="mb-3 grid grid-cols-3 gap-2 text-center text-[11px] font-bold">
                  <div>
                    <div className="text-sky-600 dark:text-sky-300">Train {trainPct}%</div>
                    <div className="tabular-nums text-slate-500">~{estTrain}</div>
                  </div>
                  <div>
                    <div className="text-violet-600 dark:text-violet-300">Val {valPct}%</div>
                    <div className="tabular-nums text-slate-500">~{estVal}</div>
                  </div>
                  <div>
                    <div className="text-amber-600 dark:text-amber-300">Test {testPct}%</div>
                    <div className="tabular-nums text-slate-500">~{estTest}</div>
                  </div>
                </div>
                <label className="mb-2 block">
                  <span className="mb-0.5 flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-500">
                    Train ends <span className="tabular-nums text-sky-600">{trainEnd}%</span>
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={98}
                    value={trainEnd}
                    onChange={(e) => onTrainEnd(Number(e.target.value))}
                    className="w-full accent-sky-500"
                  />
                </label>
                <label className="block">
                  <span className="mb-0.5 flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-500">
                    Val ends <span className="tabular-nums text-violet-600">{valEnd}%</span>
                  </span>
                  <input
                    type="range"
                    min={Math.min(99, trainEnd + 1)}
                    max={100}
                    value={valEnd}
                    onChange={(e) => onValEnd(Number(e.target.value))}
                    className="w-full accent-violet-500"
                  />
                </label>
              </>
            ) : (
              <p className="text-[12px] text-slate-600 dark:text-slate-300">
                All <strong>{count}</strong> images go into <strong>{forceSplit}</strong>.
              </p>
            )}
          </div>

          <div>
            <span className="mb-1.5 block text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Same-name images
            </span>
            <div className="grid grid-cols-3 gap-1.5">
              {(
                [
                  ["skip", "Skip all"],
                  ["replace", "Replace all"],
                  ["keep_both", "New names"],
                ] as const
              ).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setConflict(id)}
                  className={cn(
                    "rounded-lg border px-2 py-2 text-[11px] font-bold",
                    conflict === id
                      ? "border-amber-400 bg-amber-500 text-white"
                      : "border-slate-200 text-slate-600 dark:border-slate-700 dark:text-slate-300",
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 dark:border-slate-700 dark:bg-ink-950">
            <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Will upload</div>
            <div className="mt-0.5 text-2xl font-black tabular-nums text-slate-900 dark:text-white">{count}</div>
            <div className="text-[11px] text-slate-500">
              {inDays ? `${selDays.length} day${selDays.length === 1 ? "" : "s"} · Done images only` : picked ? "hand-picked images" : `blocks ${lo}–${hi}`}
              {forceSplit
                ? ` · all → ${forceSplit}`
                : ` · ~${estTrain}/${estVal}/${estTest} train/val/test`}
            </div>
          </div>

          {error && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-[12px] text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200">
              {error}
            </div>
          )}
          {resultUrl && (
            <p className="text-[11px] font-semibold text-emerald-600">
              Uploaded ·{" "}
              <a href={resultUrl} target="_blank" rel="noreferrer" className="underline">
                open dataset
              </a>
            </p>
          )}
        </div>

        <div className="flex items-center justify-between gap-2 border-t border-slate-200 px-4 py-3 dark:border-slate-700">
          <span className="text-[11px] font-semibold text-slate-500">{count} image{count === 1 ? "" : "s"}</span>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-600 dark:border-slate-700 dark:text-slate-300"
            >
              Close
            </button>
            <button
              type="button"
              disabled={!canUpload}
              onClick={() => setConfirmOpen(true)}
              className="inline-flex items-center gap-1.5 rounded-lg bg-rose-500 px-4 py-2 text-sm font-bold text-white hover:bg-rose-600 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <CloudUpload className="h-4 w-4" />}
              Upload
            </button>
          </div>
        </div>
      </div>

      {confirmOpen && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-slate-950/50 p-4">
          <div className="w-full max-w-sm rounded-xl border border-slate-200 bg-white p-4 shadow-xl dark:border-slate-700 dark:bg-ink-900">
            <h3 className="text-sm font-extrabold text-slate-900 dark:text-white">Confirm upload</h3>
            <p className="mt-2 text-[12px] leading-relaxed text-slate-600 dark:text-slate-300">
              Upload <strong>{count}</strong> labeled images to{" "}
              <strong>
                {owner || "…"}/{targetSlug}
              </strong>
              .
              <br />
              Split: {forceSplit ? `all → ${forceSplit}` : `${trainPct}/${valPct}/${testPct}`}.
              <br />
              Conflicts: <strong>{conflictLabel}</strong>.
            </p>
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                disabled={busy}
                onClick={() => setConfirmOpen(false)}
                className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold dark:border-slate-700"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={busy}
                onClick={() => void runUpload()}
                className="inline-flex items-center gap-1.5 rounded-lg bg-rose-500 px-3 py-1.5 text-xs font-bold text-white disabled:opacity-50"
              >
                {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <CloudUpload className="h-3.5 w-3.5" />}
                {busy ? "Uploading…" : "Confirm"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
