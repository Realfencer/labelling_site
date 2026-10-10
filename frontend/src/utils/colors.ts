const CLASS_COLORS = [
  "#ff3b30",
  "#34c759",
  "#00c7ff",
  "#ffcc00",
  "#ff2d92",
  "#ff9500",
  "#5856d6",
  "#30b0c7",
  "#af52de",
  "#ff6b6b",
  "#64d2ff",
  "#aaff00",
];

export function colorForClass(classId: number): string {
  if (classId >= 0 && classId < CLASS_COLORS.length) {
    return CLASS_COLORS[classId];
  }
  const hue = (classId * 137) % 360;
  return `hsl(${hue}, 85%, 55%)`;
}

export function formatPercent(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export function formatScore(value: unknown): string {
  if (value == null) return "—";
  const n = Number(value);
  if (Number.isNaN(n)) return "—";
  return `${(n * 100).toFixed(1)}%`;
}

export function boxFromDict(raw: Record<string, unknown>): {
  cls: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  conf?: number;
  raw_conf?: number;
} {
  return {
    cls: Number(raw.cls_id ?? raw.cls ?? 0),
    x1: Number(raw.x1),
    y1: Number(raw.y1),
    x2: Number(raw.x2),
    y2: Number(raw.y2),
    conf: raw.conf != null ? Number(raw.conf) : undefined,
    raw_conf: raw.raw_conf != null ? Number(raw.raw_conf) : undefined,
  };
}

/** Box label with the network's own confidence (not the miner's gated / calibrated value). */
export function boxConfLabel(name: string, box: { conf?: number | null; raw_conf?: number | null }): string {
  const v = box.raw_conf ?? box.conf;
  return v != null ? `${name} ${Math.round(v * 100)}%` : name;
}

export {
  challengeImageUrl,
  challengePreviewUrl,
  challengeThumbnailUrl,
  preloadChallenge,
  preloadChallengeForward,
  preloadChallengeUrgent,
} from "../store/imageLoader";
