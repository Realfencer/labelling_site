import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Check, X } from "lucide-react";
import { cn } from "../lib/utils";

/** One selectable image. */
export type PickItem = {
  /** Stable key (Platform/Own: the item key; collector: the block number as text). */
  id: string;
  /** Number used by "from–to" quick picks (the block, or the position in the Own pool). */
  order: number;
  label: string;
  thumb: string;
  preview: string;
  /** Tiny tag in the corner, e.g. who labeled it. */
  tag?: string;
  /** Platform label checks: all 3 miners close / some / none. */
  quality?: "pass" | "mid" | "fail" | null;
};

export type PickCtx = { selected: string[]; n: number; bMin: number | null; bMax: number | null };

const TILE_SIZES = [120, 170, 230];
const QUALITY_DOT: Record<string, string> = { pass: "bg-emerald-400", mid: "bg-lime-400", fail: "bg-rose-400" };

/**
 * The site's image picker: click to pick, drag across images to paint, Shift+click for everything between
 * two clicks, quick picks (latest N, block range, quality), and a big preview of the hovered image.
 * Pick ANY images — they do not have to form one range.
 */
export function ImagePickerModal({
  title,
  hint,
  items,
  initial,
  minCount = 1,
  loading = false,
  busy = false,
  error = "",
  rangeLabel = "blocks",
  defaultLatest = 50,
  onClose,
  summary,
  actions,
  layer = "z-50",
}: {
  title: string;
  hint?: ReactNode;
  /** Newest first. */
  items: PickItem[];
  initial: string[];
  minCount?: number;
  loading?: boolean;
  busy?: boolean;
  error?: string;
  rangeLabel?: string;
  defaultLatest?: number;
  onClose: () => void;
  summary?: (ctx: PickCtx) => ReactNode;
  actions: (ctx: PickCtx & { small: boolean }) => ReactNode;
  layer?: string;
}) {
  const [sel, setSel] = useState<Set<string>>(() => new Set(initial));
  const [hover, setHover] = useState<string | null>(null);
  const [tile, setTile] = useState(1);
  const [latestN, setLatestN] = useState(Math.min(defaultLatest, Math.max(1, items.length || defaultLatest)));
  const [lo, setLo] = useState("");
  const [hi, setHi] = useState("");
  const [off, setOff] = useState<Set<string>>(() => new Set()); // quality classes switched off
  const painting = useRef<null | boolean>(null);
  const lastIdx = useRef<number | null>(null);
  const seeded = useRef(initial.length > 0);

  // The list may arrive after the window opened (collector): seed the selection from `initial` once.
  useEffect(() => {
    if (seeded.current || initial.length === 0) return;
    seeded.current = true;
    setSel(new Set(initial));
  }, [initial]);
  useEffect(() => {
    if (items.length) setLatestN((n) => Math.min(Math.max(1, n), items.length));
  }, [items.length]);

  useEffect(() => {
    const up = () => {
      painting.current = null;
    };
    window.addEventListener("pointerup", up);
    return () => window.removeEventListener("pointerup", up);
  }, []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !busy) {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [busy, onClose]);

  const ids = useMemo(() => items.map((i) => i.id), [items]);
  const byId = useMemo(() => new Map(items.map((i, k) => [i.id, { item: i, idx: k }])), [items]);
  // An image whose quality class is switched off is disabled: it cannot be picked and is not used. Switching
  // the class back on re-enables it with its earlier pick state, because `sel` itself is left untouched.
  const isOff = useCallback((it: PickItem) => Boolean(it.quality && off.has(it.quality)), [off]);
  const enabled = useMemo(() => items.filter((i) => !(i.quality && off.has(i.quality))), [items, off]);
  const enabledIds = useMemo(() => enabled.map((i) => i.id), [enabled]);
  const setMany = useCallback(
    (list: string[], on: boolean) => {
      setSel((prev) => {
        const next = new Set(prev);
        list.forEach((b) => {
          const it = byId.get(b)?.item;
          if (on && it && it.quality && off.has(it.quality)) return; // disabled: cannot be picked
          if (on) next.add(b);
          else next.delete(b);
        });
        return next;
      });
    },
    [byId, off],
  );
  const toggleQuality = (q: string) =>
    setOff((prev) => {
      const next = new Set(prev);
      if (next.has(q)) next.delete(q);
      else next.add(q);
      return next;
    });

  const onTileDown = (e: React.PointerEvent, id: string, i: number) => {
    if (busy || e.button !== 0) return;
    e.preventDefault();
    if (isOff(items[i])) return; // disabled by a quality filter
    const on = !sel.has(id);
    if (e.shiftKey && lastIdx.current != null) {
      const [a, b] = [Math.min(lastIdx.current, i), Math.max(lastIdx.current, i)];
      setMany(ids.slice(a, b + 1), on);
    } else {
      painting.current = on;
      setMany([id], on);
    }
    lastIdx.current = i;
  };
  const onTileEnter = (id: string) => {
    setHover(id);
    if (painting.current != null) setMany([id], painting.current);
  };

  const selected = useMemo(() => enabledIds.filter((id) => sel.has(id)), [enabledIds, sel]);
  const n = selected.length;
  const orders = selected.map((id) => byId.get(id)?.item.order ?? 0);
  const bMin = n ? Math.min(...orders) : null;
  const bMax = n ? Math.max(...orders) : null;
  const previewId = hover ?? (lastIdx.current != null ? ids[lastIdx.current] : (selected[0] ?? null));
  const prev = previewId != null ? byId.get(previewId) : undefined;
  const hasQuality = items.some((i) => i.quality);
  const tilePx = TILE_SIZES[tile];
  const useBig = tilePx > 170;
  const ctx: PickCtx = { selected, n, bMin, bMax };

  const btn = "rounded border border-slate-600 px-2 py-0.5 font-semibold hover:bg-slate-800 disabled:opacity-40";
  return (
    <div className={cn("fixed inset-0 flex bg-black/75 p-3", layer)} onClick={() => !busy && onClose()}>
      <div className="flex min-h-0 w-full flex-col rounded-xl border border-slate-700 bg-ink-900 text-slate-100 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between gap-3 border-b border-slate-800 px-4 py-2.5">
          <div>
            <h2 className="text-base font-extrabold">{title}</h2>
            {hint && <p className="text-[11px] text-slate-400">{hint}</p>}
          </div>
          <button type="button" onClick={onClose} disabled={busy} className="rounded p-1 text-slate-400 hover:bg-slate-800 disabled:opacity-30" title="Close without changing">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex min-h-0 flex-1">
          <div className="flex min-w-0 flex-1 flex-col">
            <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 px-3 py-2 text-[12px]">
              <span className="flex items-center gap-1">
                latest
                <input
                  type="number"
                  min={1}
                  max={items.length}
                  value={latestN}
                  onChange={(e) => setLatestN(Math.max(1, Number(e.target.value) || 1))}
                  className="w-16 rounded border border-slate-600 bg-transparent px-1 py-0.5"
                  disabled={busy}
                />
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => {
                    setSel(new Set(enabledIds.slice(0, latestN)));
                    lastIdx.current = null;
                  }}
                  className={btn}
                >
                  select
                </button>
              </span>
              <span className="mx-1 h-4 w-px bg-slate-700" />
              <span className="flex items-center gap-1">
                {rangeLabel}
                <input value={lo} onChange={(e) => setLo(e.target.value.replace(/[^\d]/g, ""))} placeholder="from" className="w-24 rounded border border-slate-600 bg-transparent px-1 py-0.5" disabled={busy} />
                –
                <input value={hi} onChange={(e) => setHi(e.target.value.replace(/[^\d]/g, ""))} placeholder="to" className="w-24 rounded border border-slate-600 bg-transparent px-1 py-0.5" disabled={busy} />
                {(["add", "remove"] as const).map((m) => (
                  <button
                    key={m}
                    type="button"
                    disabled={busy || !lo || !hi}
                    onClick={() => {
                      const a = Math.min(Number(lo), Number(hi));
                      const b = Math.max(Number(lo), Number(hi));
                      setMany(enabled.filter((x) => x.order >= a && x.order <= b).map((x) => x.id), m === "add");
                    }}
                    className={btn}
                  >
                    {m}
                  </button>
                ))}
              </span>
              <span className="mx-1 h-4 w-px bg-slate-700" />
              <button type="button" disabled={busy} onClick={() => setSel(new Set(enabledIds))} className={btn}>all</button>
              <button type="button" disabled={busy} onClick={() => setSel(new Set())} className={btn}>none</button>
              <button type="button" disabled={busy} onClick={() => setSel(new Set(enabledIds.filter((b) => !sel.has(b))))} className={btn}>invert</button>
              {hasQuality && (
                <>
                  <span className="mx-1 h-4 w-px bg-slate-700" />
                  <span className="text-slate-400" title="Switch a class off to disable its images and keep only the rest">only keep</span>
                  {([["pass", "3/3", "emerald"], ["mid", "1–2/3", "lime"], ["fail", "0/3", "rose"]] as const).map(([q, label, tone]) => {
                    const total = items.filter((x) => x.quality === q).length;
                    const picked = items.filter((x) => x.quality === q && sel.has(x.id)).length;
                    const isOffQ = off.has(q);
                    const on = tone === "emerald" ? "border-emerald-400 bg-emerald-500/20 text-emerald-200" : tone === "lime" ? "border-lime-400 bg-lime-500/20 text-lime-200" : "border-rose-400 bg-rose-500/20 text-rose-200";
                    return (
                      <button
                        key={q}
                        type="button"
                        disabled={busy || total === 0}
                        onClick={() => toggleQuality(q)}
                        title={isOffQ ? `${label} images are disabled — click to use them again` : `Click to disable the ${label} images (they stay out of the selection)`}
                        className={cn("rounded border px-2 py-0.5 font-semibold tabular-nums disabled:opacity-30", isOffQ ? "border-slate-600 text-slate-500 line-through" : on)}
                      >
                        {label} <span className="font-normal opacity-80">{isOffQ ? total : picked}</span>
                      </button>
                    );
                  })}
                </>
              )}
              <label className="ml-auto flex items-center gap-1.5 text-slate-400">
                size
                <input type="range" min={0} max={2} step={1} value={tile} onChange={(e) => setTile(Number(e.target.value))} />
              </label>
            </div>
            <div className="px-3 py-1 text-[10px] text-slate-500">Click to pick / unpick · drag across images to paint · Shift+click picks everything between two clicks</div>
            <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3" onPointerLeave={() => setHover(null)}>
              <div className="grid select-none gap-1.5" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${tilePx}px, 1fr))` }}>
                {items.map((it, i) => {
                  const disabled = isOff(it);
                  const on = !disabled && sel.has(it.id);
                  return (
                    <div
                      key={it.id}
                      onPointerDown={(e) => onTileDown(e, it.id, i)}
                      onPointerEnter={() => (disabled ? setHover(it.id) : onTileEnter(it.id))}
                      className={cn(
                        "relative overflow-hidden rounded-md border-2 transition",
                        disabled
                          ? "cursor-not-allowed border-transparent opacity-20 grayscale"
                          : on
                            ? "cursor-pointer border-sky-400 shadow-[0_0_0_1px_rgba(56,189,248,0.5)]"
                            : "cursor-pointer border-transparent opacity-45 hover:opacity-90",
                      )}
                      style={{ aspectRatio: "4 / 3", background: "#0b1220" }}
                    >
                      <img src={useBig ? it.preview : it.thumb} loading="lazy" decoding="async" draggable={false} alt="" className="pointer-events-none h-full w-full object-cover" />
                      <span className="absolute bottom-0 left-0 max-w-full truncate rounded-tr bg-black/70 px-1 text-[9px] font-semibold tabular-nums text-slate-200">{it.label}</span>
                      {it.quality && <span className={cn("absolute bottom-1 right-1 h-2 w-2 rounded-full ring-1 ring-black/50", QUALITY_DOT[it.quality])} />}
                      {it.tag && <span className="absolute left-1 top-1 max-w-[70%] truncate rounded bg-black/65 px-1 text-[8px] font-bold text-slate-100">{it.tag}</span>}
                      {on && (
                        <span className="absolute right-1 top-1 flex h-4 w-4 items-center justify-center rounded-full bg-sky-500 text-white">
                          <Check className="h-3 w-3" />
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
              {items.length === 0 && <div className="py-16 text-center text-slate-500">{loading ? "Loading the images…" : "No images to choose from."}</div>}
            </div>
          </div>

          <div className="flex w-[440px] shrink-0 flex-col gap-3 overflow-y-auto border-l border-slate-800 p-3">
            <div className="flex aspect-[4/3] items-center justify-center overflow-hidden rounded-lg bg-slate-950">
              {prev ? <img key={prev.item.id} src={prev.item.preview} alt="" className="h-full w-full object-contain" /> : <span className="text-[12px] text-slate-500">hover an image to preview it here</span>}
            </div>
            {prev && (
              <div className="-mt-1 flex items-center justify-between gap-2 text-[11px] text-slate-400">
                <span className="min-w-0 truncate">
                  <b className="text-slate-200">{prev.item.label}</b> · #{prev.idx + 1} of {items.length}
                </span>
                <span className={!isOff(prev.item) && sel.has(prev.item.id) ? "shrink-0 font-bold text-sky-300" : "shrink-0"}>{isOff(prev.item) ? "disabled (filtered out)" : sel.has(prev.item.id) ? "selected" : "not selected"}</span>
              </div>
            )}
            <div className="rounded-lg bg-slate-800/50 p-3 text-[12px]">
              <div className="flex items-baseline justify-between">
                <span className="text-2xl font-extrabold tabular-nums text-white">{n}</span>
                <span className="text-slate-400">images selected</span>
              </div>
              {n > 0 && (
                <div className="mt-0.5 text-[11px] text-slate-400">
                  {rangeLabel} {bMin}–{bMax}
                </div>
              )}
              {summary?.(ctx)}
              {n > 0 && n < minCount && <div className="mt-1 text-[11px] text-amber-300">Pick at least {minCount} images.</div>}
            </div>
            <div className="flex flex-col gap-2">{actions({ ...ctx, small: n < minCount })}</div>
            {error && <div className="text-[12px] text-rose-300">{error}</div>}
          </div>
        </div>
      </div>
    </div>
  );
}
