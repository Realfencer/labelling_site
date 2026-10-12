import { cn } from "../lib/utils";

/** The site icon (cup · bottle · can in a detection frame) — same file as the favicon. */
export default function BrandMark({ size = 32, className }: { size?: number; className?: string }) {
  return (
    <img
      src="/favicon.svg"
      width={size}
      height={size}
      alt="Score Labeler"
      draggable={false}
      className={cn("shrink-0 select-none", className)}
      style={{ width: size, height: size }}
    />
  );
}
