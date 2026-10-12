import { useRef, type CSSProperties, type PointerEvent, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { cn } from "../lib/utils";
import "../landing.css";

type Props = {
  to: string;
  children: ReactNode;
  className?: string;
  /** "r g b" of the accent glow, e.g. "56 189 248". */
  glow?: string;
  /** Max tilt in degrees. */
  max?: number;
  /** Animated gradient border + shine sweep (feature tiles). */
  border?: boolean;
  autoFocus?: boolean;
};

/** A link that tilts in 3D toward the pointer, with a moving glare. Children can use .tz-1 / .tz-2 to float above the card. */
export default function TiltCard({ to, children, className, glow = "56 189 248", max = 8, border = false, autoFocus }: Props) {
  const ref = useRef<HTMLAnchorElement>(null);
  const raf = useRef(0);

  const move = (e: PointerEvent<HTMLAnchorElement>) => {
    if (e.pointerType === "touch" || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width;
    const py = (e.clientY - r.top) / r.height;
    cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(() => {
      el.style.setProperty("--rx", `${((0.5 - py) * max * 2).toFixed(2)}deg`);
      el.style.setProperty("--ry", `${((px - 0.5) * max * 2).toFixed(2)}deg`);
      el.style.setProperty("--gx", `${(px * 100).toFixed(1)}%`);
      el.style.setProperty("--gy", `${(py * 100).toFixed(1)}%`);
      el.dataset.hot = "1";
    });
  };

  const leave = () => {
    cancelAnimationFrame(raf.current);
    const el = ref.current;
    if (!el) return;
    el.style.setProperty("--rx", "0deg");
    el.style.setProperty("--ry", "0deg");
    delete el.dataset.hot;
  };

  return (
    <div className="tilt-persp">
      <Link
        ref={ref}
        to={to}
        autoFocus={autoFocus}
        onPointerMove={move}
        onPointerLeave={leave}
        onBlur={leave}
        className={cn("tilt-card", border && "fx-border", className)}
        style={{ "--glow": glow } as CSSProperties}
      >
        {border ? <span className="fx-fill" aria-hidden /> : <span className="tilt-glass" aria-hidden />}
        <span className="tilt-glare" aria-hidden />
        {border && <span className="fx-shine" aria-hidden />}
        {children}
      </Link>
    </div>
  );
}
