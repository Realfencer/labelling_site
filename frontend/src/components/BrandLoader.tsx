import type { CSSProperties } from "react";
import { useSiteBackdrop } from "../store/useBackdrop";

export type LoaderAccent = "sky" | "amber" | "fuchsia" | "emerald" | "violet";

// "r g b" pairs: ring / glow / progress bar use a (from) → b (to)
const ACCENTS: Record<LoaderAccent, [string, string]> = {
  sky: ["56 189 248", "167 139 250"],
  amber: ["251 191 36", "249 115 22"],
  fuchsia: ["232 121 249", "167 139 250"],
  emerald: ["52 211 153", "56 189 248"],
  violet: ["167 139 250", "232 121 249"],
};

/**
 * Full-screen loading screen. It sits on the same site-wide 3D backdrop as the pages (hero scene) and takes the colour of
 * the page that is loading, so a navigation reads as one continuous scene. The splash in index.html uses the same classes
 * (without .bl-live: CSS-only grid) so there is no flash between the two.
 */
export default function BrandLoader({
  label = "Loading",
  hint,
  accent = "sky",
}: {
  label?: string;
  hint?: string;
  accent?: LoaderAccent;
}) {
  useSiteBackdrop("hero");
  const [a, b] = ACCENTS[accent];
  return (
    <div className="bl-screen bl-live" role="status" aria-live="polite" style={{ "--bl-a": a, "--bl-b": b } as CSSProperties}>
      <div className="bl-stack">
        <div className="bl-glow" aria-hidden />
        <div className="bl-ring" aria-hidden />
        <img className="bl-icon" src="/favicon.svg" width={76} height={76} alt="" draggable={false} />
      </div>
      <div className="bl-title">Score Labeler</div>
      <div className="bl-label">{label}</div>
      {hint && <div className="bl-hint">{hint}</div>}
      <div className="bl-chips" aria-hidden>
        <span style={{ "--c": "#fbbf24", "--d": "0s" } as CSSProperties}>
          <i />
          cup
        </span>
        <span style={{ "--c": "#a78bfa", "--d": "0.3s" } as CSSProperties}>
          <i />
          can
        </span>
        <span style={{ "--c": "#38bdf8", "--d": "0.6s" } as CSSProperties}>
          <i />
          bottle
        </span>
      </div>
      <div className="bl-bar" aria-hidden>
        <i />
      </div>
    </div>
  );
}
