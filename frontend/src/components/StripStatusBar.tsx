import { memo, useEffect, useMemo, useRef, useState } from "react";
import { useTeam } from "../store/useTeam";

/** Per-image status codes (also the legend). */
export const STRIP_CODE = { NEW: 0, SAM3: 1, FAIL: 2, MID: 3, PASS: 4 } as const;
const CODE_RGB: Array<[number, number, number]> = [
  [245, 158, 11], // NEW  amber
  [56, 189, 248], // SAM3 sky
  [244, 63, 94], // FAIL rose   (labeled, 0/3 miners close)
  [163, 230, 53], // MID  lime
  [16, 185, 129], // PASS / done emerald
];
const CODE_NAME = ["NEW", "SAM3 draft", "Done · 0/3", "Done · 1-2/3", "Done"];

export type StripCodes = { codes: Int8Array; idxs: Int32Array };

function hexToRgb(hex: string): [number, number, number] {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex);
  if (!m) return [148, 163, 184];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/**
 * Very thin overview of every image in the strip: amber = still NEW, green = done, blue = SAM3
 * draft. Dense pools are averaged per pixel column, so unlabeled stretches stay visibly amber.
 * Lower half = who labeled it; violet = a teammate has it open; white tick = you are here.
 * Click / drag to jump.
 */
function StripStatusBar({
  challenges,
  compute,
  version,
  currentIdx,
  visible,
  onJump,
}: {
  challenges: Array<{ item_key: string; display_name?: string }>;
  compute: () => StripCodes;
  /** Changes whenever codes must be recomputed (filter / checks tick / cache token). */
  version: string;
  currentIdx: number;
  visible: { first: number; last: number };
  onJump: (challengeIndex: number) => void;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState<{ x: number; text: string } | null>(null);
  const claims = useTeam((s) => s.claims);
  const labels = useTeam((s) => s.labels);
  const colors = useTeam((s) => s.colors);

  const computeRef = useRef(compute);
  computeRef.current = compute;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const { codes, idxs } = useMemo(() => computeRef.current(), [challenges, version]);
  const n = codes.length;

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(Math.floor(el.clientWidth)));
    ro.observe(el);
    setWidth(Math.floor(el.clientWidth));
    return () => ro.disconnect();
  }, []);

  const H = 10;
  const posOfCurrent = useMemo(() => {
    for (let p = 0; p < idxs.length; p += 1) if (idxs[p] === currentIdx) return p;
    return -1;
  }, [idxs, currentIdx]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width <= 0) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(H * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, H);
    ctx.fillStyle = "rgba(100,116,139,0.18)";
    ctx.fillRect(0, 0, width, H);
    if (n === 0) return;

    // Aggregate per pixel column: average colour of the images that fall in it.
    const cols = width;
    const r = new Float32Array(cols);
    const g = new Float32Array(cols);
    const b = new Float32Array(cols);
    const cnt = new Uint16Array(cols);
    const ur = new Float32Array(cols);
    const ug = new Float32Array(cols);
    const ub = new Float32Array(cols);
    const ucnt = new Uint16Array(cols);
    const claimed = new Uint8Array(cols);
    for (let p = 0; p < n; p += 1) {
      // Sparse (n < width): an image spans several columns. Dense: several images share one.
      const x0 = Math.min(cols - 1, Math.floor((p / n) * cols));
      const x1 = Math.min(cols, Math.max(x0 + 1, Math.floor(((p + 1) / n) * cols)));
      const c = CODE_RGB[codes[p]] ?? CODE_RGB[0];
      const key = challenges[idxs[p]]?.item_key;
      const isClaimed = key ? Boolean(claims[key]) : false;
      const who = key ? labels[key]?.u : undefined;
      const uc = who ? hexToRgb(colors[who] ?? "#94a3b8") : null;
      for (let x = x0; x < x1; x += 1) {
        r[x] += c[0];
        g[x] += c[1];
        b[x] += c[2];
        cnt[x] += 1;
        if (isClaimed) claimed[x] = 1;
        if (uc) {
          ur[x] += uc[0];
          ug[x] += uc[1];
          ub[x] += uc[2];
          ucnt[x] += 1;
        }
      }
    }
    const topH = 6;
    for (let x = 0; x < cols; x += 1) {
      if (!cnt[x]) continue;
      ctx.fillStyle = `rgb(${Math.round(r[x] / cnt[x])},${Math.round(g[x] / cnt[x])},${Math.round(b[x] / cnt[x])})`;
      ctx.fillRect(x, 0, 1, topH);
      if (ucnt[x]) {
        ctx.fillStyle = `rgb(${Math.round(ur[x] / ucnt[x])},${Math.round(ug[x] / ucnt[x])},${Math.round(ub[x] / ucnt[x])})`;
        ctx.fillRect(x, topH, 1, H - topH);
      }
      if (claimed[x]) {
        ctx.fillStyle = "#a78bfa";
        ctx.fillRect(x, 0, 1, H);
      }
    }
    // Part of the strip currently on screen.
    if (visible.last >= visible.first && n > 0) {
      const x1 = (Math.max(0, visible.first) / n) * cols;
      const x2 = (Math.min(n, visible.last + 1) / n) * cols;
      ctx.strokeStyle = "rgba(226,232,240,0.85)";
      ctx.lineWidth = 1;
      ctx.strokeRect(Math.floor(x1) + 0.5, 0.5, Math.max(2, Math.ceil(x2 - x1)) - 1, H - 1);
    }
    // You are here.
    if (posOfCurrent >= 0) {
      const x = Math.min(cols - 2, Math.floor((posOfCurrent / n) * cols));
      ctx.fillStyle = "#0b1220";
      ctx.fillRect(x - 1, 0, 4, H);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(x, 0, 2, H);
    }
  }, [width, codes, idxs, n, challenges, claims, labels, colors, visible.first, visible.last, posOfCurrent]);

  const posFromEvent = (clientX: number): number => {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect || n === 0) return -1;
    const f = Math.min(0.999999, Math.max(0, (clientX - rect.left) / rect.width));
    return Math.floor(f * n);
  };
  const draggingRef = useRef(false);
  const jump = (clientX: number) => {
    const p = posFromEvent(clientX);
    if (p >= 0) onJump(idxs[p]);
  };

  return (
    <div
      ref={wrapRef}
      className="relative h-[10px] w-full shrink-0 cursor-pointer select-none overflow-visible rounded-sm"
      onPointerDown={(e) => {
        if (e.button !== 0) return;
        draggingRef.current = true;
        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        jump(e.clientX);
      }}
      onPointerMove={(e) => {
        const p = posFromEvent(e.clientX);
        if (p >= 0) {
          const c = challenges[idxs[p]];
          const key = c?.item_key ?? "";
          const who = labels[key]?.u;
          const claim = claims[key];
          setHover({
            x: e.clientX - (wrapRef.current?.getBoundingClientRect().left ?? 0),
            text: `${c?.display_name ?? key} · ${CODE_NAME[codes[p]] ?? ""}${who ? ` · ${who}` : ""}${claim ? ` · ✎ ${claim}` : ""} · ${p + 1}/${n}`,
          });
        }
        if (draggingRef.current) jump(e.clientX);
      }}
      onPointerUp={() => {
        draggingRef.current = false;
      }}
      onPointerLeave={() => setHover(null)}
      title="Image status: amber = NEW, green = done, blue = SAM3 draft · lower row = who labeled · violet = teammate is on it · click to jump"
    >
      <canvas ref={canvasRef} style={{ width: "100%", height: H }} className="block rounded-sm" />
      {hover && (
        <div
          className="pointer-events-none absolute bottom-[14px] z-20 whitespace-nowrap rounded bg-slate-900/95 px-1.5 py-0.5 text-[10px] font-semibold text-white shadow"
          style={{ left: Math.max(0, Math.min(hover.x - 40, Math.max(0, width - 260))) }}
        >
          {hover.text}
        </div>
      )}
    </div>
  );
}

export default memo(StripStatusBar);
