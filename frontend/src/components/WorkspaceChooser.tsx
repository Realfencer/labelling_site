import { useEffect, useRef, type CSSProperties } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, Images, Sparkles, Trophy, Upload } from "lucide-react";
import KeysButton from "./KeysButton";
import TiltCard from "./TiltCard";
import { useSiteBackdrop } from "../store/useBackdrop";
import "../landing.css";

type Props = {
  slug: string;
  title: string;
  platformImages: number;
  user?: string | null;
};

const STEPS = ["Open an image", "Fix the boxes", "Watch the score climb"];

/** 3D hero mark: tilts toward the pointer, floats, and three class-coloured dots orbit it. */
function HeroMark() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const el = ref.current;
    if (!el) return;
    const onMove = (e: PointerEvent) => {
      const px = e.clientX / Math.max(1, window.innerWidth) - 0.5;
      const py = e.clientY / Math.max(1, window.innerHeight) - 0.5;
      el.style.setProperty("--hy", `${(px * 26).toFixed(1)}deg`);
      el.style.setProperty("--hx", `${(-py * 22).toFixed(1)}deg`);
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    return () => window.removeEventListener("pointermove", onMove);
  }, []);
  return (
    <div className="hero-stage">
      <div ref={ref} className="hero-tilt">
        <div className="hero-glow" aria-hidden />
        <div className="hero-orbit2" aria-hidden />
        <div className="hero-orbit" aria-hidden>
          <i className="hero-dot d1" />
          <i className="hero-dot d2" />
          <i className="hero-dot d3" />
        </div>
        <img className="hero-icon" src="/favicon.svg" alt="Score Labeler" draggable={false} />
      </div>
    </div>
  );
}

/** Home screen for a single-topic deployment: pick Platform or Own, then go straight in. */
export default function WorkspaceChooser({ slug, title, platformImages, user }: Props) {
  const navigate = useNavigate();
  useSiteBackdrop("hero");

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (e.key === "p" || e.key === "P") navigate(`/t/${slug}?workspace=platform`);
      if (e.key === "o" || e.key === "O") navigate(`/t/${slug}?workspace=own`);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navigate, slug]);

  const rise = (ms: number) => ({ "--d": `${ms}ms` }) as CSSProperties;

  return (
    <div className="lp-root">
      <div className="lp-vignette" aria-hidden />

      <div className="absolute right-4 top-4 z-20 flex items-center gap-2 sm:right-6 sm:top-5">
        {user && <span className="lp-chip">{user}</span>}
        <KeysButton className="!rounded-full !border-slate-600/60 !bg-slate-900/60 !px-3 !py-1.5 !text-xs !text-slate-300 backdrop-blur" />
      </div>

      <main className="lp-main relative z-10 mx-auto w-full max-w-5xl px-4 sm:px-6">
        <div className="lp-rise lp-hero-col" style={rise(0)}>
          <HeroMark />
          <span className="lp-chip lp-chip-top">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
            </span>
            Score Labeler · Bittensor SN44
          </span>
          <h1 className="lp-title lp-h1">{title}</h1>
          <p className="lp-p">
            {user ? `Welcome back, ${user}. ` : "Welcome. "}
            Spot cups, cans and bottles — every box you fix teaches the model. Pick a workspace and jump in.
          </p>
          <div className="lp-steps">
            {STEPS.map((s, i) => (
              <span key={s} className="lp-chip">
                <b>{i + 1}</b>
                {s}
              </span>
            ))}
          </div>
        </div>

        <div className="lp-grid2">
          <div className="lp-rise" style={rise(140)}>
            <TiltCard to={`/t/${slug}?workspace=platform`} glow="56 189 248" autoFocus className="lp-card-pad">
              <div className="lp-ico tz-2 fx-icon flex items-center justify-center rounded-2xl bg-gradient-to-br from-sky-400 to-cyan-500 text-white shadow-lg shadow-sky-500/40">
                <Images />
              </div>
              <div className="tz-1 lp-card-gap">
                <div className="lp-card-title">Platform</div>
                <p className="lp-card-desc">
                  Label the live challenge images and see how your boxes score against the miner references.
                </p>
              </div>
              <div className="tz-1 lp-card-foot flex items-center justify-between">
                <span className="inline-flex items-center gap-1.5 text-sm font-bold text-sky-300">
                  {platformImages > 0 ? `${platformImages.toLocaleString()} images` : "Start labeling"}
                  <ArrowRight className="fx-arrow h-4 w-4" />
                </span>
                <kbd className="rounded-md border border-slate-600/80 bg-slate-900/60 px-2 py-0.5 text-[11px] font-semibold text-slate-400">P</kbd>
              </div>
            </TiltCard>
          </div>

          <div className="lp-rise" style={rise(240)}>
            <TiltCard to={`/t/${slug}?workspace=own`} glow="167 139 250" className="lp-card-pad">
              <div className="lp-ico tz-2 fx-icon flex items-center justify-center rounded-2xl bg-gradient-to-br from-violet-400 to-fuchsia-500 text-white shadow-lg shadow-violet-500/40">
                <Upload />
              </div>
              <div className="tz-1 lp-card-gap">
                <div className="lp-card-title">Own</div>
                <p className="lp-card-desc">
                  Your own images, organised by day. Upload, label and improve at your own pace.
                </p>
              </div>
              <div className="tz-1 lp-card-foot flex items-center justify-between">
                <span className="inline-flex items-center gap-1.5 text-sm font-bold text-violet-300">
                  Pick a day
                  <ArrowRight className="fx-arrow h-4 w-4" />
                </span>
                <kbd className="rounded-md border border-slate-600/80 bg-slate-900/60 px-2 py-0.5 text-[11px] font-semibold text-slate-400">O</kbd>
              </div>
            </TiltCard>
          </div>
        </div>

        <div className="lp-grid2">
          <div className="lp-rise" style={rise(340)}>
            <TiltCard to="/collected" glow="251 191 36" max={5} border className="lp-tile-pad">
              <div className="flex items-center gap-4">
                <div className="fx-icon flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-amber-300 to-orange-500 text-white shadow-lg shadow-amber-500/30">
                  <Trophy className="h-6 w-6" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="text-[15px] font-extrabold text-white">Collected miner solutions</div>
                  <p className="mt-0.5 text-[12.5px] leading-snug text-slate-400">
                    Every submitted miner, scored and latency-checked — find the best pack.
                  </p>
                </div>
                <ArrowRight className="fx-arrow h-5 w-5 shrink-0 text-amber-300" />
              </div>
            </TiltCard>
          </div>
          <div className="lp-rise" style={rise(420)}>
            <TiltCard to="/studio" glow="232 121 249" max={5} border className="lp-tile-pad">
              <div className="flex items-center gap-4">
                <div className="fx-icon flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-fuchsia-400 to-pink-500 text-white shadow-lg shadow-fuchsia-500/30">
                  <Sparkles className="h-6 w-6" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="text-[15px] font-extrabold text-white">Studio · generate images</div>
                  <p className="mt-0.5 text-[12.5px] leading-snug text-slate-400">
                    Create fresh training images with AI and send them straight to your workspace.
                  </p>
                </div>
                <ArrowRight className="fx-arrow h-5 w-5 shrink-0 text-fuchsia-300" />
              </div>
            </TiltCard>
          </div>
        </div>

        <p className="lp-rise lp-tip" style={rise(520)}>
          Tip: press <kbd className="rounded border border-slate-700 px-1 text-slate-400">P</kbd> for Platform or{" "}
          <kbd className="rounded border border-slate-700 px-1 text-slate-400">O</kbd> for Own. Hover the cards to feel the 3D.
        </p>
      </main>
    </div>
  );
}
