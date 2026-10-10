import { useEffect, useMemo, useState } from "react";
import { Check, Loader2, Play } from "lucide-react";
import { api } from "../api";
import { ImagePickerModal, type PickItem } from "./ImagePicker";

const imgUrl = (block: number, kind: "thumb" | "preview") => `/api/collector/images/${block}/${kind}`;

/**
 * Choose the WORKING IMAGE SET: any validated images. Every measurement, deep post-process and push on
 * the Collected page then uses exactly these images.
 */
export default function ImageSetModal({
  nPassed,
  onClose,
  onSaved,
}: {
  nPassed: number;
  onClose: () => void;
  /** Called after the set is saved; ``measureNow`` = the user also asked to re-measure every model. */
  onSaved: (measureNow: boolean) => void;
}) {
  const [blocks, setBlocks] = useState<number[]>([]); // newest first
  const [minImages, setMinImages] = useState(10);
  const [defaultN, setDefaultN] = useState(50);
  const [initial, setInitial] = useState<string[]>([]);
  const [curLabel, setCurLabel] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    void Promise.all([api.collectorImages(), api.collectorImageSet()])
      .then(([r, cur]) => {
        setBlocks(r.blocks);
        setMinImages(r.min_images);
        setDefaultN(r.default_n);
        setCurLabel(cur.label);
        setInitial(cur.blocks.map(String)); // start from what is in use now
      })
      .catch((e) => setError(String(e).replace(/^Error:\s*/, "")));
  }, []);

  const items = useMemo<PickItem[]>(
    () => blocks.map((b) => ({ id: String(b), order: b, label: String(b), thumb: imgUrl(b, "thumb"), preview: imgUrl(b, "preview") })),
    [blocks],
  );

  const save = async (selected: string[], measureNow: boolean) => {
    setError("");
    setSaving(true);
    try {
      await api.collectorSetImageSet({ mode: "blocks", blocks: selected.map(Number) });
      onSaved(measureNow);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/, ""));
      setSaving(false);
    }
  };
  const resetLatest = async () => {
    setError("");
    setSaving(true);
    try {
      await api.collectorSetImageSet({ mode: "latest", n: defaultN });
      onSaved(false);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/, ""));
      setSaving(false);
    }
  };

  return (
    <ImagePickerModal
      title="Choose the working image set"
      hint={
        <>
          The {blocks.length || "…"} validated images (you labeled them and all 3 miners agree), newest first. Pick any images — they do not have to be in one range. Every score, deep post-process and push on
          the page uses exactly these.
        </>
      }
      items={items}
      initial={initial}
      minCount={minImages}
      loading={blocks.length === 0 && !error}
      busy={saving}
      error={error}
      defaultLatest={defaultN}
      onClose={onClose}
      summary={(ctx) => (
        <div className="mt-2 text-[11px] text-slate-400">
          Re-measuring {nPassed} passed models on these takes about{" "}
          <b className="text-slate-200">{Math.max(1, Math.round(((6 + 0.12 * ctx.n) * nPassed) / 60))} min</b>. Deep post-process and push use the same images.
          {ctx.n === defaultN && <span> (the usual latest {defaultN})</span>}
        </div>
      )}
      actions={(ctx) => (
        <>
          <button
            type="button"
            onClick={() => void save(ctx.selected, false)}
            disabled={ctx.small || ctx.n === 0 || saving}
            className="inline-flex items-center justify-center gap-1.5 rounded-md bg-sky-600 px-4 py-2.5 text-[13px] font-bold text-white hover:bg-sky-500 disabled:opacity-40"
          >
            {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />} Use these {ctx.n} images
          </button>
          <button
            type="button"
            onClick={() => void save(ctx.selected, true)}
            disabled={ctx.small || ctx.n === 0 || saving}
            className="inline-flex items-center justify-center gap-1.5 rounded-md border border-sky-500 px-4 py-2 text-[12px] font-bold text-sky-300 hover:bg-sky-500/10 disabled:opacity-40"
          >
            <Play className="h-4 w-4" /> Use them and re-measure all {nPassed} models now
          </button>
          <button type="button" onClick={() => void resetLatest()} disabled={saving} className="rounded-md border border-slate-600 px-3 py-1.5 text-[12px] font-semibold hover:bg-slate-800 disabled:opacity-40">
            Back to the latest {defaultN} (follows new labels)
          </button>
          {curLabel && <div className="text-[10px] text-slate-500">In use now: {curLabel}</div>}
        </>
      )}
    />
  );
}
