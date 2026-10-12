import { useEffect, useRef, useState } from "react";
import { effectiveMode, useBackdrop } from "../store/useBackdrop";

type Box = { x: number; y: number; d: number; vx: number; vy: number; cls: number; conf: number; w: number; h: number };
type Dot = { x: number; y: number; r: number; v: number; ph: number };
type Quality = "hero" | "vivid" | "calm" | "soft" | "still";

const CLASSES = [
  { name: "cup", rgb: "251,191,36" },
  { name: "can", rgb: "167,139,250" },
  { name: "bottle", rgb: "56,189,248" },
];

/** Per-quality scene settings. `budget` = average draw ms allowed before the scene lightens itself. */
const SCENE = {
  hero: { boxes: 15, dust: 46, alpha: 1, dpr: 1.25, parallax: true, frameMs: 0, budget: 8 },
  vivid: { boxes: 15, dust: 46, alpha: 0.95, dpr: 1, parallax: true, frameMs: 33, budget: 6 },
  calm: { boxes: 8, dust: 26, alpha: 0.85, dpr: 0.75, parallax: false, frameMs: 50, budget: 4 },
  soft: { boxes: 8, dust: 22, alpha: 0.5, dpr: 0.7, parallax: false, frameMs: 66, budget: 5 },
  still: { boxes: 8, dust: 22, alpha: 0.5, dpr: 1, parallax: false, frameMs: 0, budget: 1e9 },
} as const;

const rand = (a: number, b: number) => a + Math.random() * (b - a);

function lowEndDevice(): boolean {
  const nav = navigator as Navigator & { deviceMemory?: number; connection?: { saveData?: boolean } };
  return (nav.hardwareConcurrency ?? 8) <= 2 || (nav.deviceMemory ?? 8) <= 2 || Boolean(nav.connection?.saveData);
}

/** ?fx=off turns the animation off for good on this browser (one still frame is drawn); ?fx=on turns it back on. */
function effectsOff(): boolean {
  try {
    const q = new URLSearchParams(window.location.search).get("fx");
    if (q === "off" || q === "on") localStorage.setItem("sn44-fx", q);
    return localStorage.getItem("sn44-fx") === "off";
  } catch {
    return false;
  }
}

function isDark(): boolean {
  return document.documentElement.classList.contains("dark");
}

/**
 * One persistent canvas behind the whole app (perspective grid floor + cup / can / bottle detection boxes at
 * different depths). Pages opt in with useSiteBackdrop(); the labeler never does, so it costs nothing there.
 * It protects the page it sits behind: it measures its own draw cost and lightens itself ("hero" → "soft" →
 * a still frame), pauses while the tab is hidden, and stays still for reduced-motion / save-data / low-end devices.
 */
export default function SiteBackdrop() {
  const requested = useBackdrop((s) => effectiveMode(s.requests));
  const [dark, setDark] = useState(isDark);
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const obs = new MutationObserver(() => setDark(isDark()));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => obs.disconnect();
  }, []);

  // "soft" sits behind pages that follow the light/dark theme: only in dark. "hero" screens are always dark.
  const mode = requested === "hero" || (requested !== null && dark) ? requested : null;

  useEffect(() => {
    const cv = ref.current;
    const ctx = cv?.getContext("2d", { alpha: true });
    if (!mode || !cv || !ctx) return;

    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let quality: Quality = reduce || lowEndDevice() || effectsOff() ? "still" : mode === "hero" ? "hero" : mode === "vivid" ? "vivid" : mode === "calm" ? "calm" : "soft";
    const publish = () => {
      document.documentElement.dataset.bdQuality = quality; // current scene quality, visible in the DOM (debugging / support)
    };
    let lastActivity = -1e9; // calm: the scene freezes while the user is working and resumes when they pause
    let w = 0;
    let h = 0;
    let dpr = 1;
    let raf = 0;
    let alive = true;
    let lastFrame = 0;
    let ema = 0;
    let over = 0;
    let prevTick = 0; // frame pacing: raster / composite cost never shows up in our own draw() timing
    let paceEma = 16.7;
    let slow = 0;
    let warm = 0;
    const t0 = performance.now();
    const m = { tx: 0, ty: 0, x: 0, y: 0 };

    const boxes: Box[] = Array.from({ length: 15 }, (_, i) => ({
      x: rand(-1, 1),
      y: rand(-0.95, 0.55),
      d: rand(0.05, 1),
      vx: rand(-0.012, 0.012),
      vy: rand(-0.008, 0.008),
      cls: i % 3,
      conf: rand(0.78, 0.99),
      w: rand(0.7, 1.3),
      h: rand(0.9, 1.9),
    })).sort((a, b) => b.d - a.d); // far first
    const dots: Dot[] = Array.from({ length: 46 }, () => ({ x: Math.random(), y: Math.random(), r: rand(0.6, 1.8), v: rand(0.004, 0.016), ph: rand(0, 6.28) }));

    const resize = () => {
      dpr = Math.min(SCENE[quality].dpr, window.devicePixelRatio || 1);
      w = cv.clientWidth;
      h = cv.clientHeight;
      cv.width = Math.max(1, Math.floor(w * dpr));
      cv.height = Math.max(1, Math.floor(h * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (quality === "still") draw(0);
    };

    const roundRect = (x: number, y: number, bw: number, bh: number, r: number) => {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + bw, y, x + bw, y + bh, r);
      ctx.arcTo(x + bw, y + bh, x, y + bh, r);
      ctx.arcTo(x, y + bh, x, y, r);
      ctx.arcTo(x, y, x + bw, y, r);
      ctx.closePath();
    };

    const draw = (t: number) => {
      const S = SCENE[quality];
      const A = S.alpha;
      ctx.clearRect(0, 0, w, h);
      if (mode === "hero") {
        // hero screens are always dark: keep their base even after the scene has lightened itself
        ctx.fillStyle = "#070b16";
        ctx.fillRect(0, 0, w, h);
      }
      m.x += (m.tx - m.x) * 0.06;
      m.y += (m.ty - m.y) * 0.06;
      const vx = w / 2 + m.x * 60;
      const y0 = h * 0.6 + m.y * 14;

      const hg = ctx.createRadialGradient(vx, y0, 0, vx, y0, Math.max(w, h) * 0.6);
      hg.addColorStop(0, `rgba(99,102,241,${(0.3 * A).toFixed(3)})`);
      hg.addColorStop(0.35, `rgba(56,189,248,${(0.1 * A).toFixed(3)})`);
      hg.addColorStop(1, "rgba(7,11,22,0)");
      ctx.fillStyle = hg;
      ctx.fillRect(0, 0, w, h);

      const lg = ctx.createLinearGradient(0, y0, 0, h);
      lg.addColorStop(0, "rgba(129,140,248,0)");
      lg.addColorStop(1, `rgba(129,140,248,${(0.42 * A).toFixed(3)})`);
      ctx.strokeStyle = lg;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = -16; i <= 16; i++) {
        ctx.moveTo(vx, y0);
        ctx.lineTo(vx + i * w * 0.115, h);
      }
      ctx.stroke();
      const rows = 12;
      const off = (t * 0.22) % 1;
      for (let k = 0; k < rows; k++) {
        const f = (k + off) / rows;
        const y = y0 + (h - y0) * Math.pow(f, 2.3);
        ctx.strokeStyle = `rgba(129,140,248,${((0.04 + f * 0.34) * A).toFixed(3)})`;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      for (let i = 0; i < S.dust; i++) {
        const p = dots[i];
        const y = (((p.y - t * p.v) % 1) + 1) % 1;
        const a = (0.25 + 0.35 * Math.sin(t * 1.4 + p.ph)) * A;
        ctx.fillStyle = `rgba(186,230,253,${Math.max(0.04, a).toFixed(2)})`;
        ctx.beginPath();
        ctx.arc(p.x * w, y * h * 0.95, p.r, 0, 6.2832);
        ctx.fill();
      }

      // boxes: spread over the depth range, far → near; the soft scene keeps every other one
      for (let bi = 0; bi < boxes.length; bi++) {
        if ((quality === "soft" || quality === "still" || quality === "calm") && bi % 2 === 1) continue;
        const b = boxes[bi];
        b.x += b.vx * 0.016;
        b.y += b.vy * 0.016;
        if (b.x > 1.15) b.x = -1.15;
        if (b.x < -1.15) b.x = 1.15;
        if (b.y > 0.7) b.y = -1;
        if (b.y < -1) b.y = 0.7;
        const near = 1 - b.d;
        const s = 0.5 + near * 0.95;
        const par = S.parallax ? (0.4 + near) * 46 : 0;
        const cx = w / 2 + b.x * w * 0.52 + m.x * par;
        const cy = h * 0.42 + b.y * h * 0.5 + m.y * par * 0.6 + Math.sin(t * 0.8 + b.conf * 9) * 5 * s;
        const bw = 44 * s * b.w;
        const bh = 44 * s * b.h;
        // keep the middle column (hero text + cards) clean: boxes fade out as they drift behind it
        const mid = mode === "hero" ? Math.min(1, Math.abs(cx - w / 2) / Math.min(w * 0.4, 560)) : 1;
        const alpha = (0.22 + near * 0.62) * (0.06 + 0.94 * mid * mid) * A;
        const c = CLASSES[b.cls].rgb;
        const x = cx - bw / 2;
        const y = cy - bh / 2;

        ctx.fillStyle = `rgba(${c},${(alpha * 0.07).toFixed(3)})`;
        roundRect(x, y, bw, bh, 5 * s);
        ctx.fill();

        const L = Math.min(bw, bh) * 0.3;
        ctx.strokeStyle = `rgba(${c},${alpha.toFixed(2)})`;
        ctx.lineWidth = 1.2 + near * 1.6;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(x, y + L); ctx.lineTo(x, y); ctx.lineTo(x + L, y);
        ctx.moveTo(x + bw - L, y); ctx.lineTo(x + bw, y); ctx.lineTo(x + bw, y + L);
        ctx.moveTo(x + bw, y + bh - L); ctx.lineTo(x + bw, y + bh); ctx.lineTo(x + bw - L, y + bh);
        ctx.moveTo(x + L, y + bh); ctx.lineTo(x, y + bh); ctx.lineTo(x, y + bh - L);
        ctx.stroke();

        if (near > 0.2 && mid > 0.55 && (quality === "hero" || quality === "vivid")) {
          const fs = 8.5 + near * 4;
          ctx.font = `700 ${fs.toFixed(1)}px Inter, system-ui, sans-serif`;
          const label = `${CLASSES[b.cls].name} ${Math.round(b.conf * 100)}%`;
          const tw = ctx.measureText(label).width + 10;
          const th = fs + 6;
          // content pages: translucent chips, so a label that drifts behind text never fights with it
          ctx.fillStyle = `rgba(${c},${Math.min(quality === "vivid" ? 0.55 : 0.92, alpha + 0.1).toFixed(2)})`;
          roundRect(x, y - th - 2, tw, th, 4);
          ctx.fill();
          ctx.fillStyle = "rgba(7,11,22,0.92)";
          ctx.textBaseline = "middle";
          ctx.fillText(label, x + 5, y - th / 2 - 2);
        }
      }
    };

    const lighten = () => {
      quality = quality === "hero" || quality === "vivid" ? "soft" : "still";  // calm / soft step down to a still frame
      publish();
      ema = 0;
      over = 0;
      resize();
      if (quality === "still") cancelAnimationFrame(raf);
    };

    const frame = (now: number) => {
      if (!alive || quality === "still") return;
      raf = requestAnimationFrame(frame);
      if (prevTick) {
        const gap = now - prevTick;
        if (++warm > 90 && gap < 500) {
          // a free page ticks every ~16.7ms; sustained gaps mean the page (labeling canvas, tables, this scene) cannot hold ~35 fps
          paceEma = paceEma * 0.93 + gap * 0.07;
          slow = paceEma > 27 ? slow + 1 : Math.max(0, slow - 3);
          if (slow > 60) {
            prevTick = 0;
            slow = 0;
            paceEma = 16.7;
            lighten();
            return;
          }
        }
      }
      prevTick = now;
      const S = SCENE[quality];
      if (S.frameMs && now - lastFrame < S.frameMs) return; // frame cap for the soft scene
      if (quality === "calm" && now - lastActivity < 800) return; // working: keep the last frame, spend nothing
      lastFrame = now;
      const a = performance.now();
      draw((now - t0) / 1000);
      const cost = performance.now() - a;
      ema = ema ? ema * 0.9 + cost * 0.1 : cost;
      over = ema > S.budget ? over + 1 : Math.max(0, over - 2);
      if (over > 45) lighten(); // sustained: this machine cannot afford it next to the page — step down
    };

    const onMove = (e: PointerEvent) => {
      m.tx = e.clientX / Math.max(1, window.innerWidth) - 0.5;
      m.ty = e.clientY / Math.max(1, window.innerHeight) - 0.5;
    };
    const run = () => {
      cancelAnimationFrame(raf);
      if (alive && !document.hidden && quality !== "still") raf = requestAnimationFrame(frame);
    };

    publish();
    resize();
    window.addEventListener("resize", resize);
    document.addEventListener("visibilitychange", run);
    if (SCENE[quality].parallax) window.addEventListener("pointermove", onMove, { passive: true });
    const onActivity = () => {
      lastActivity = performance.now();
    };
    const activityEvents = ["pointermove", "pointerdown", "keydown", "wheel"] as const;
    if (mode === "calm") activityEvents.forEach((ev) => window.addEventListener(ev, onActivity, { passive: true }));
    run();
    return () => {
      alive = false;
      delete document.documentElement.dataset.bdQuality;
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      window.removeEventListener("pointermove", onMove);
      activityEvents.forEach((ev) => window.removeEventListener(ev, onActivity));
      document.removeEventListener("visibilitychange", run);
    };
  }, [mode]);

  if (!mode) return null;
  return <canvas ref={ref} className="pointer-events-none fixed inset-0 h-full w-full" style={{ zIndex: -1 }} aria-hidden />;
}
