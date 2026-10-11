import { useMemo, useState } from "react";
import { Images, X } from "lucide-react";
import type { ChallengeItem } from "../types";
import { checksFailComplete, checksPassComplete } from "../labelChecks";
import { challengePreviewUrl, challengeThumbnailUrl } from "../store/imageLoader";
import { useLabeler } from "../store/useLabeler";
import { useTeam } from "../store/useTeam";
import { cn } from "../lib/utils";
import { ImagePickerModal, type PickItem } from "./ImagePicker";

/** Same cache token the labeler uses for image URLs (session + project + skill). */
function useCacheToken(): string {
  const s = useLabeler((st) => st.appState);
  if (!s) return "";
  const skill = s.active_skill?.element_path || s.hf_model_id || "skill";
  return `${s.session_id || "sid"}:${s.project_id}:${skill}`;
}

/** Done images of the open pool, newest first, with thumbnails — what the site's modals offer to pick from. */
export function usePickItems(challenges: ChallengeItem[], ownMode: boolean, doneOnly = true): PickItem[] {
  const mode = useLabeler((st) => st.appState?.review_mode) ?? (ownMode ? "local" : "challenge");
  const token = useCacheToken();
  const labels = useTeam((s) => s.labels);
  return useMemo(() => {
    const done = challenges
      .map((c, i) => ({ c, i }))
      .filter(({ c }) => c.loaded && (!doneOnly || c.reviewed) && c.block > 0)
      .sort((a, b) => b.c.block - a.c.block);
    return done.map(({ c, i }) => {
      let quality: PickItem["quality"] = null;
      if (!ownMode && c.checks_total) {
        quality = checksPassComplete(c) ? "pass" : checksFailComplete(c) ? "fail" : "mid";
      }
      return {
        id: c.item_key,
        order: c.block,
        label: c.display_name || String(c.block),
        thumb: challengeThumbnailUrl(i, mode, c.block, false, token, c.item_key, c.media_rev),
        preview: challengePreviewUrl(i, mode, c.block, token, c.item_key, c.media_rev),
        tag: c.reviewed ? labels[c.item_key]?.u ?? undefined : doneOnly ? undefined : "NEW",
        quality,
        gen: c.generator ?? null,
      };
    });
  }, [challenges, ownMode, doneOnly, mode, token, labels]);
}

/**
 * "Choose images…" for the site's modals. Without a pick the modal keeps using its from–to range; with a
 * pick the modal uses exactly these images (``value`` = their item keys). The modal should grey out its
 * range controls while ``value`` is set.
 */
export default function ImageSelectionField({
  challenges,
  ownMode,
  value,
  onChange,
  minCount = 1,
  verb = "use",
  doneOnly = true,
  suggest,
  disabled = false,
  className,
}: {
  challenges: ChallengeItem[];
  ownMode: boolean;
  value: string[] | null;
  onChange: (keys: string[] | null) => void;
  minCount?: number;
  /** Button text in the picker: "Use these N images" / "Delete these N images" … */
  verb?: string;
  /** false: the picker also lists images that are not Done yet (e.g. to delete them). */
  doneOnly?: boolean;
  /** Item keys the picker starts from when nothing is chosen yet (the modal's current range). */
  suggest?: string[];
  disabled?: boolean;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const items = usePickItems(challenges, ownMode, doneOnly);
  const byId = useMemo(() => new Map(items.map((i) => [i.id, i])), [items]);
  const have = value ? value.filter((k) => byId.has(k)) : [];
  const start = value ? have : (suggest ?? []).filter((k) => byId.has(k));
  const orders = have.map((k) => byId.get(k)?.order ?? 0);

  return (
    <div className={cn("flex flex-wrap items-center gap-2", className)}>
      {value == null ? (
        <button
          type="button"
          disabled={disabled || items.length === 0}
          onClick={() => setOpen(true)}
          className="inline-flex items-center gap-1.5 rounded-lg border border-sky-400 px-3 py-1.5 text-[12px] font-bold text-sky-600 hover:bg-sky-50 disabled:opacity-40 dark:text-sky-300 dark:hover:bg-sky-950/30"
          title="Pick any images with previews — drag, shift-click, ranges"
        >
          <Images className="h-3.5 w-3.5" /> Choose images…
        </button>
      ) : (
        <>
          <span className="inline-flex items-center gap-1.5 rounded-lg bg-sky-500/15 px-2.5 py-1.5 text-[12px] font-bold text-sky-700 dark:text-sky-200">
            <Images className="h-3.5 w-3.5" /> {have.length} images chosen
            {have.length > 0 && <span className="font-normal opacity-80">· {Math.min(...orders)}–{Math.max(...orders)}</span>}
          </span>
          <button type="button" disabled={disabled} onClick={() => setOpen(true)} className="text-[11px] font-bold text-sky-600 hover:underline dark:text-sky-300">
            change
          </button>
          <button type="button" disabled={disabled} onClick={() => onChange(null)} className="inline-flex items-center gap-0.5 text-[11px] font-semibold text-slate-500 hover:underline">
            <X className="h-3 w-3" /> use the range instead
          </button>
        </>
      )}
      {open && (
        <ImagePickerModal
          layer="z-[70]"
          title="Choose images"
          hint={<>{doneOnly ? "Your Done images" : "The images of this pool"}, newest first. Pick any of them — they do not have to be in one range.</>}
          items={items}
          initial={start}
          minCount={minCount}
          rangeLabel={ownMode ? "index" : "blocks"}
          onClose={() => setOpen(false)}
          actions={(ctx) => (
            <>
              <button
                type="button"
                disabled={ctx.small || ctx.n === 0}
                onClick={() => {
                  onChange(ctx.selected);
                  setOpen(false);
                }}
                className="rounded-md bg-sky-600 px-4 py-2.5 text-[13px] font-bold text-white hover:bg-sky-500 disabled:opacity-40"
              >
                {verb[0].toUpperCase() + verb.slice(1)} these {ctx.n} images
              </button>
              <button type="button" onClick={() => setOpen(false)} className="rounded-md border border-slate-600 px-3 py-1.5 text-[12px] font-semibold hover:bg-slate-800">
                Cancel
              </button>
            </>
          )}
        />
      )}
    </div>
  );
}
