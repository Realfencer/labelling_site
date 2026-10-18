import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { DetectionBox } from "../types";
import type { BoxCompareRole } from "../utils/gtScore";
import { colorForClass } from "../utils/colors";
import {
  abortImageLoad,
  adoptDecodedImage,
  decodeBlobUrl,
  forgetDecodedImage,
  getDecodedImage,
  loadImageImmediate,
  subscribeImage,
  type ImageSource,
} from "../store/imageLoader";
import { getThumbnailBlobUrl, snapshotThumbnailImage, subscribeThumbnail } from "../store/thumbnailLoader";

type ResizeMode = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w" | null;
type Transform = { scale: number; offsetX: number; offsetY: number };
/** A zoom window in image terms (centre in image pixels, zoom relative to "fit"): looks the same on any screen size. */
export type ViewShare = { cx: number; cy: number; z: number };
/** A reason marked on a box (Own Review): a badge in ``color`` with a one-character icon ``text``; ``strong`` = the one in focus. */
export type Callout = { x1: number; y1: number; x2: number; y2: number; text: string; color: string; strong?: boolean; outline?: boolean };

interface Props {
  fullUrl: string;
  /** Mid-tier (~720px) while full source loads. */
  previewUrl?: string;
  thumbUrl?: string;
  width: number;
  height: number;
  boxes: DetectionBox[];
  classNames: string[];
  currentClass: number;
  editable?: boolean;
  selectedIndex?: number | null;
  onBoxesChange?: (boxes: DetectionBox[], selected: number | null, commit?: boolean) => void;
  onSelect?: (index: number | null) => void;
  syncTransform?: Transform | null;
  onTransformChange?: (t: Transform) => void;
  /** Called with the current zoom window whenever it changes (used to show a teammate exactly this spot). */
  onViewChange?: (v: ViewShare) => void;
  /** Smoothly move to this zoom window (a teammate is asking about this spot). */
  focusView?: ViewShare | null;
  /** Reasons to draw around boxes (Own Review). */
  callouts?: Callout[];
  canvasBg?: string;
  sceneId?: string;
  /** Per-box match vs left-pane GT (right panel compare mode). */
  boxCompare?: BoxCompareRole[];
  /** GT boxes missed by miner — ghost overlay on right panel. */
  ghostBoxes?: DetectionBox[];
  /** When false, draw box outlines only (hide class name panels). */
  showLabels?: boolean;
  /** Cross-panel focus: indices into `boxes` that should stay bright while others dim. */
  linkedFocusIdxs?: number[] | null;
  /** Cross-panel focus: indices into `ghostBoxes`. */
  linkedGhostIdxs?: number[] | null;
  /** Report local hover so the other pane can highlight the matched box. */
  onHoverFocus?: (payload: { boxIdx: number | null; ghostIdx: number | null }) => void;
  /** Keyboard inspect: frame this box with a smooth zoom (left pane drives sync). */
  inspectBox?: DetectionBox | null;
  /** Bump to re-run framing when cycling C/V onto another box. */
  inspectSeq?: number;
  /** Right-panel alert solo mode: wrong / extra / missed (null = show all). */
  alertFilter?: "wrong" | "extra" | "missed" | null;
  /** Smart dotted row/column guides that follow the pointer (Own · key R). */
  showGuides?: boolean;
}

const MIN_DRAW = 3;
const HANDLE_HIT = 10;
/** Magnetic reach (screen px) of the 8 resize handles: the nearest one inside it is grabbed, however small the box. */
const HANDLE_MAGNET = 18;
const classCursorCache = new Map<number, string>();

/** Snap pointer to nearby box edges / centers for precise labeling guides. */
function smartGuidePos(
  pt: { x: number; y: number },
  boxes: DetectionBox[],
  snapPx: number,
  draft: DetectionBox | null,
): { x: number; y: number } {
  let gx = pt.x;
  let gy = pt.y;
  let bestDx = snapPx;
  let bestDy = snapPx;
  const consider = (box: DetectionBox) => {
    const { x1, y1, x2, y2 } = boxBounds(box);
    for (const x of [x1, x2, (x1 + x2) / 2]) {
      const d = Math.abs(pt.x - x);
      if (d < bestDx) {
        bestDx = d;
        gx = x;
      }
    }
    for (const y of [y1, y2, (y1 + y2) / 2]) {
      const d = Math.abs(pt.y - y);
      if (d < bestDy) {
        bestDy = d;
        gy = y;
      }
    }
  };
  for (const box of boxes) consider(box);
  if (draft) consider(draft);
  return { x: gx, y: gy };
}

function withAlpha(color: string, a: number): string {
  if (color.startsWith("#")) {
    let hex = color.slice(1);
    if (hex.length === 3) hex = hex.split("").map((c) => c + c).join("");
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  }
  if (color.startsWith("hsl(")) return color.replace("hsl(", "hsla(").replace(")", `, ${a})`);
  return color;
}

function boxBounds(box: DetectionBox): { x1: number; y1: number; x2: number; y2: number } {
  return {
    x1: Math.min(box.x1, box.x2),
    y1: Math.min(box.y1, box.y2),
    x2: Math.max(box.x1, box.x2),
    y2: Math.max(box.y1, box.y2),
  };
}

function hitTestInside(x: number, y: number, boxes: DetectionBox[]): number | null {
  let bestI = -1;
  let bestArea = Infinity;
  for (let i = 0; i < boxes.length; i += 1) {
    const { x1, y1, x2, y2 } = boxBounds(boxes[i]);
    if (x < x1 || x > x2 || y < y1 || y > y2) continue;
    const area = Math.max(1, (x2 - x1) * (y2 - y1));
    if (area < bestArea) {
      bestArea = area;
      bestI = i;
    }
  }
  return bestI >= 0 ? bestI : null;
}

function hitTestBoxOrGhost(
  x: number,
  y: number,
  boxes: DetectionBox[],
  ghosts: DetectionBox[] | undefined,
): { boxIdx: number | null; ghostIdx: number | null } {
  const boxIdx = hitTestInside(x, y, boxes);
  const ghostIdx = ghosts?.length ? hitTestInside(x, y, ghosts) : null;
  if (boxIdx == null) return { boxIdx: null, ghostIdx };
  if (ghostIdx == null) return { boxIdx, ghostIdx: null };
  const b = boxBounds(boxes[boxIdx]);
  const g = boxBounds(ghosts![ghostIdx]);
  const boxArea = Math.max(1, (b.x2 - b.x1) * (b.y2 - b.y1));
  const ghostArea = Math.max(1, (g.x2 - g.x1) * (g.y2 - g.y1));
  // Prefer the tighter hit so stacked EXTRA vs MISSED is unambiguous.
  return boxArea <= ghostArea ? { boxIdx, ghostIdx: null } : { boxIdx: null, ghostIdx };
}

function drawHandle(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  color: string,
  scale: number,
  active = false,
) {
  const r = (active ? 6.6 : 4.4) / scale;
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, r + 2.2 / scale, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0,0,0,0.28)";
  ctx.fill();
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = active ? color : "#fff";
  ctx.fill();
  ctx.lineWidth = 1.6 / scale;
  ctx.strokeStyle = color;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x, y, r * 0.32, 0, Math.PI * 2);
  ctx.fillStyle = active ? "#fff" : color;
  ctx.fill();
  ctx.restore();
}

/** The 8 handles of a box in image coordinates, in drawing order. */
function handlePoints(box: DetectionBox): [Exclude<ResizeMode, null>, number, number][] {
  const [x1, x2] = [box.x1, box.x2].sort((a, b) => a - b);
  const [y1, y2] = [box.y1, box.y2].sort((a, b) => a - b);
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  return [
    ["nw", x1, y1],
    ["n", mx, y1],
    ["ne", x2, y1],
    ["e", x2, my],
    ["se", x2, y2],
    ["s", mx, y2],
    ["sw", x1, y2],
    ["w", x1, my],
  ];
}

function hitTestResize(
  x: number,
  y: number,
  boxes: DetectionBox[],
  scale: number,
  selected: number | null,
): [number, ResizeMode] | null {
  if (selected == null || selected < 0 || selected >= boxes.length) return null;
  const box = boxes[selected];
  // 1) Magnet: the NEAREST of the 8 handles within reach wins. Distances are in screen pixels, so a narrow or
  //    small box still lets you reach every middle handle (before, corners won and middles were switched off).
  let best: Exclude<ResizeMode, null> | null = null;
  let bestD = HANDLE_MAGNET;
  for (const [mode, hx, hy] of handlePoints(box)) {
    const d = Math.hypot(x - hx, y - hy) * scale;
    if (d < bestD) {
      bestD = d;
      best = mode;
    }
  }
  if (best) return [selected, best];
  // 2) Edge grab (anywhere along a side), when the box is big enough on screen for sides to be told apart.
  const tol = Math.max(2, HANDLE_HIT / scale);
  const [x1, x2] = [box.x1, box.x2].sort((a, b) => a - b);
  const [y1, y2] = [box.y1, box.y2].sort((a, b) => a - b);
  if (x < x1 - tol || x > x2 + tol || y < y1 - tol || y > y2 + tol) return null;
  const big = Math.min(x2 - x1, y2 - y1) * scale >= 30;
  if (!big) return null;
  if (Math.abs(y - y1) <= tol) return [selected, "n"];
  if (Math.abs(y - y2) <= tol) return [selected, "s"];
  if (Math.abs(x - x1) <= tol) return [selected, "w"];
  if (Math.abs(x - x2) <= tol) return [selected, "e"];
  return null;
}

function resizedBox(origin: DetectionBox, mode: string, x: number, y: number, imgW: number, imgH: number): DetectionBox {
  let { x1, y1, x2, y2 } = origin;
  if (mode.includes("n")) y1 = y;
  if (mode.includes("s")) y2 = y;
  if (mode.includes("w")) x1 = x;
  if (mode.includes("e")) x2 = x;
  let [left, right] = [x1, x2].sort((a, b) => a - b);
  let [top, bottom] = [y1, y2].sort((a, b) => a - b);
  // Every edge stays on the image, exactly (0 .. W, 0 .. H): a box can never be dragged outside, and it can reach the
  // true right / bottom edge (it used to stop one pixel short).
  left = Math.min(Math.max(left, 0), imgW);
  right = Math.min(Math.max(right, 0), imgW);
  top = Math.min(Math.max(top, 0), imgH);
  bottom = Math.min(Math.max(bottom, 0), imgH);
  if (right - left < MIN_DRAW) {
    if (left + MIN_DRAW <= imgW) right = left + MIN_DRAW;
    else {
      right = imgW;
      left = Math.max(0, imgW - MIN_DRAW);
    }
  }
  if (bottom - top < MIN_DRAW) {
    if (top + MIN_DRAW <= imgH) bottom = top + MIN_DRAW;
    else {
      bottom = imgH;
      top = Math.max(0, imgH - MIN_DRAW);
    }
  }
  return { ...origin, x1: left, y1: top, x2: right, y2: bottom };
}

function cursorForMode(mode: ResizeMode | "draw" | "pan" | "move" | null, classCursor?: string): string {
  if (mode === "n" || mode === "s") return "ns-resize";
  if (mode === "e" || mode === "w") return "ew-resize";
  if (mode === "ne" || mode === "sw") return "nesw-resize";
  if (mode === "nw" || mode === "se") return "nwse-resize";
  if (mode === "pan") return "grab";
  if (mode === "move") return "move";
  if (mode === "draw" && classCursor) return classCursor;
  return classCursor || "crosshair";
}

/** Desktop-style class crosshair: ring + crosshair in the active class color. */
function buildClassCursor(classId: number): string {
  const cached = classCursorCache.get(classId);
  if (cached) return cached;
  const color = colorForClass(classId);
  const size = 22;
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d");
  if (!ctx) return "crosshair";
  const mid = size / 2;
  ctx.clearRect(0, 0, size, size);

  // Thin white halo only for contrast — class color stays dominant.
  ctx.beginPath();
  ctx.arc(mid, mid, 5.5, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(255,255,255,0.55)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Class-colored ring with soft fill
  ctx.beginPath();
  ctx.arc(mid, mid, 5.5, 0, Math.PI * 2);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.fillStyle = `${color}55`;
  ctx.fill();
  ctx.stroke();

  // Crosshair arms — class color
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.75;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(mid, 1);
  ctx.lineTo(mid, 5);
  ctx.moveTo(mid, 17);
  ctx.lineTo(mid, 21);
  ctx.moveTo(1, mid);
  ctx.lineTo(5, mid);
  ctx.moveTo(17, mid);
  ctx.lineTo(21, mid);
  ctx.stroke();

  // Class-colored center dot
  ctx.beginPath();
  ctx.arc(mid, mid, 1.6, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();

  const cursor = `url(${c.toDataURL("image/png")}) ${mid} ${mid}, crosshair`;
  classCursorCache.set(classId, cursor);
  return cursor;
}

/** Boxes are stored in [0,width]×[0,height] — bitmap must fill that rect exactly. */
function imageDestRect(_img: ImageSource, width: number, height: number) {
  return { dx: 0, dy: 0, dw: width, dh: height };
}

function easeOutCubic(t: number) {
  const c = Math.max(0, Math.min(1, t));
  return 1 - (1 - c) ** 3;
}

export default function AnnotationCanvas({
  fullUrl,
  previewUrl,
  thumbUrl,
  width,
  height,
  boxes,
  classNames,
  currentClass,
  editable = false,
  selectedIndex = null,
  onBoxesChange,
  onSelect,
  syncTransform,
  onTransformChange,
  onViewChange,
  focusView,
  callouts,
  canvasBg = "#0b1220",
  sceneId,
  boxCompare,
  ghostBoxes,
  showLabels = true,
  linkedFocusIdxs = null,
  linkedGhostIdxs = null,
  onHoverFocus,
  inspectBox = null,
  inspectSeq = 0,
  alertFilter = null,
  showGuides = false,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<ImageSource | null>(null);
  const fullReadyRef = useRef(false);
  const [hasImage, setHasImage] = useState(false);
  const [reloadGen, setReloadGen] = useState(0);
  const reloadPendingRef = useRef(false);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [cursor, setCursor] = useState("crosshair");
  const classCursorRef = useRef("crosshair");
  const showGuidesRef = useRef(showGuides);
  showGuidesRef.current = showGuides;
  const guidePosRef = useRef<{ x: number; y: number } | null>(null);
  const currentClassRef = useRef(currentClass);
  currentClassRef.current = currentClass;

  useEffect(() => {
    if (!editable) {
      classCursorRef.current = "grab";
      setCursor((c) => (c === "grabbing" || c.includes("resize") ? c : "grab"));
      return;
    }
    const next = buildClassCursor(currentClass);
    classCursorRef.current = next;
    setCursor((c) => {
      if (c === "grab" || c === "grabbing" || c.includes("resize")) return c;
      return next;
    });
  }, [currentClass, editable]);
  const [draftBox, setDraftBox] = useState<DetectionBox | null>(null);
  const dragRef = useRef<{
    kind: string;
    startX: number;
    startY: number;
    origin?: DetectionBox;
    idx?: number;
    mode?: ResizeMode;
    /** Pointer offset from the grabbed edge/corner at pointer-down, so grabbing doesn't snap the edge. */
    grabDx?: number;
    grabDy?: number;
    /** Snapshot at pointer-down so live scoring never double-appends drafts. */
    baseBoxes?: DetectionBox[];
    /** Right-button pan: pointer-down spot, to tell a click (delete a box) from a drag (pan). */
    right?: boolean;
    downX?: number;
    downY?: number;
  } | null>(null);
  const spaceRef = useRef(false);
  const rafRef = useRef(0);
  const hoverIdxRef = useRef<number | null>(null);
  /** Resize handle under the pointer (magnet): drawn bigger so you can see which one a drag will grab. */
  const hoverHandleRef = useRef<ResizeMode>(null);
  const hoverGhostIdxRef = useRef<number | null>(null);
  const lastHoverIdxRef = useRef<number | null>(null);
  const lastHoverGhostIdxRef = useRef<number | null>(null);
  const focusTRef = useRef(0);
  /** Per-box/ghost focus strength 0..1 for smooth hover crossfades (`b:3`, `g:1`). */
  const focusWeightRef = useRef<Map<string, number>>(new Map());
  const fillTRef = useRef(0);
  const prevImgRef = useRef<ImageSource | null>(null);
  const imageFadeTRef = useRef(1);
  const boxesFadeTRef = useRef(1);
  const urlRef = useRef(fullUrl);
  const loadUrlsRef = useRef<string[]>([]);
  const sceneRef = useRef(sceneId);
  const animRafRef = useRef(0);
  const animTsRef = useRef(0);
  const selectedIndexRef = useRef(selectedIndex);
  selectedIndexRef.current = selectedIndex;
  const scaleRef = useRef(1);
  const offsetRef = useRef({ x: 0, y: 0 });
  const targetScaleRef = useRef(1);
  const targetOffsetRef = useRef({ x: 0, y: 0 });
  const frameAnimRef = useRef(0);
  const zoomAnchorRef = useRef({ x: 0, y: 0 });
  const zoomRafRef = useRef(0);
  /** Last pointer position in container CSS px (for Z/X zoom-toward-cursor). */
  const pointerCssRef = useRef<{ x: number; y: number } | null>(null);
  const pointerInsideRef = useRef(false);
  const userZoomedRef = useRef(false);
  const localZoomingRef = useRef(false);
  const lastInspectSeqRef = useRef(0);
  /** True while crossfading strip thumb → full (same scene, quality upgrade only). */
  const qualityUpgradeRef = useRef(false);
  /** Soft pulse phase for EXTRA / WRONG / MISSED overlays (ms-based). */
  const alertPhaseRef = useRef(0);
  const boxCompareRef = useRef(boxCompare);
  boxCompareRef.current = boxCompare;
  const ghostBoxesRef = useRef(ghostBoxes);
  ghostBoxesRef.current = ghostBoxes;
  const linkedFocusIdxsRef = useRef(linkedFocusIdxs);
  linkedFocusIdxsRef.current = linkedFocusIdxs;
  const linkedGhostIdxsRef = useRef(linkedGhostIdxs);
  linkedGhostIdxsRef.current = linkedGhostIdxs;
  const onHoverFocusRef = useRef(onHoverFocus);
  onHoverFocusRef.current = onHoverFocus;
  const showLabelsRef = useRef(showLabels);
  showLabelsRef.current = showLabels;
  const alertFilterRef = useRef(alertFilter);
  alertFilterRef.current = alertFilter;
  /** rAF-coalesce live (non-commit) box publishes so p/t metrics track while resizing. */
  const livePublishRafRef = useRef(0);
  const livePublishBoxesRef = useRef<DetectionBox[] | null>(null);
  const livePublishSelRef = useRef<number | null>(null);
  const onBoxesChangeRef = useRef(onBoxesChange);
  onBoxesChangeRef.current = onBoxesChange;

  const publishLiveBoxes = useCallback((next: DetectionBox[], sel: number | null) => {
    livePublishBoxesRef.current = next;
    livePublishSelRef.current = sel;
    if (livePublishRafRef.current) return;
    livePublishRafRef.current = requestAnimationFrame(() => {
      livePublishRafRef.current = 0;
      const pending = livePublishBoxesRef.current;
      if (!pending) return;
      onBoxesChangeRef.current?.(pending, livePublishSelRef.current, false);
    });
  }, []);

  const reportHoverFocus = useCallback((boxIdx: number | null, ghostIdx: number | null) => {
    onHoverFocusRef.current?.({ boxIdx, ghostIdx });
  }, []);

  const flushLivePublish = useCallback(() => {
    if (livePublishRafRef.current) {
      cancelAnimationFrame(livePublishRafRef.current);
      livePublishRafRef.current = 0;
    }
    livePublishBoxesRef.current = null;
  }, []);

  useEffect(
    () => () => {
      if (livePublishRafRef.current) cancelAnimationFrame(livePublishRafRef.current);
    },
    [],
  );

  const calloutsRef = useRef(callouts);
  calloutsRef.current = callouts;
  const viewCbRef = useRef(onViewChange);
  viewCbRef.current = onViewChange;
  const applyView = useCallback(
    (s: number, ox: number, oy: number, emit = true) => {
      scaleRef.current = s;
      offsetRef.current = { x: ox, y: oy };
      setScale(s);
      setOffset({ x: ox, y: oy });
      if (emit) onTransformChange?.({ scale: s, offsetX: ox, offsetY: oy });
      const cb = viewCbRef.current;
      const container = containerRef.current;
      if (emit && cb && container && width > 0 && height > 0 && container.clientWidth > 0 && container.clientHeight > 0) {
        const cw = container.clientWidth;
        const ch = container.clientHeight;
        const fit = Math.min(cw / width, ch / height);
        cb({ cx: (cw / 2 - ox) / s, cy: (ch / 2 - oy) / s, z: s / fit });
      }
    },
    [onTransformChange, width, height],
  );

  const toImage = useCallback((clientX: number, clientY: number) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const cx = clientX - rect.left;
    const cy = clientY - rect.top;
    const s = scaleRef.current;
    const o = offsetRef.current;
    return { x: (cx - o.x) / s, y: (cy - o.y) / s };
  }, []);

  /** The pointer position as an image point that can never leave the image (drawing from or to the outside snaps to the edge). */
  const toImageClamped = useCallback(
    (clientX: number, clientY: number) => {
      const p = toImage(clientX, clientY);
      return { x: Math.min(Math.max(p.x, 0), width), y: Math.min(Math.max(p.y, 0), height) };
    },
    [toImage, width, height],
  );

  const fitView = useCallback(() => {
    const container = containerRef.current;
    if (!container || !width || !height) return;
    const pad = 10;
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    if (cw <= 0 || ch <= 0) return;
    cancelAnimationFrame(zoomRafRef.current);
    zoomRafRef.current = 0;
    cancelAnimationFrame(frameAnimRef.current);
    frameAnimRef.current = 0;
    userZoomedRef.current = false;
    const s = Math.min((cw - pad * 2) / width, (ch - pad * 2) / height, 1);
    const ox = (cw - width * s) / 2;
    const oy = (ch - height * s) / 2;
    targetScaleRef.current = s;
    targetOffsetRef.current = { x: ox, y: oy };
    applyView(s, ox, oy);
  }, [width, height, applyView]);

  const animateFrameToTarget = useCallback(() => {
    const k = 0.2;
    const s0 = scaleRef.current;
    const o0 = offsetRef.current;
    const s1 = targetScaleRef.current;
    const o1 = targetOffsetRef.current;
    const s = s0 + (s1 - s0) * k;
    const ox = o0.x + (o1.x - o0.x) * k;
    const oy = o0.y + (o1.y - o0.y) * k;
    const done =
      Math.abs(s1 - s) < 0.004 * Math.max(1, s1) &&
      Math.abs(o1.x - ox) < 0.6 &&
      Math.abs(o1.y - oy) < 0.6;
    if (done) {
      applyView(s1, o1.x, o1.y, true);
      frameAnimRef.current = 0;
      localZoomingRef.current = false;
      return;
    }
    applyView(s, ox, oy, true);
    frameAnimRef.current = requestAnimationFrame(animateFrameToTarget);
  }, [applyView]);

  const frameBoxSmooth = useCallback(
    (box: DetectionBox) => {
      const container = containerRef.current;
      if (!container || !width || !height) return;
      const cw = container.clientWidth;
      const ch = container.clientHeight;
      if (cw <= 0 || ch <= 0) return;
      const x1 = Math.min(box.x1, box.x2);
      const x2 = Math.max(box.x1, box.x2);
      const y1 = Math.min(box.y1, box.y2);
      const y2 = Math.max(box.y1, box.y2);
      const bw = Math.max(10, x2 - x1);
      const bh = Math.max(10, y2 - y1);
      const fitScale = Math.min(cw / width, ch / height);
      // Box fills ~48% of the shorter viewport edge — readable without cropping neighbors too hard.
      const desired = Math.min((cw * 0.48) / bw, (ch * 0.48) / bh);
      const s = Math.min(16, Math.max(fitScale * 1.08, Math.min(desired, fitScale * 12)));
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;
      const ox = cw / 2 - cx * s;
      const oy = ch / 2 - cy * s;
      cancelAnimationFrame(zoomRafRef.current);
      zoomRafRef.current = 0;
      userZoomedRef.current = true;
      localZoomingRef.current = true;
      targetScaleRef.current = s;
      targetOffsetRef.current = { x: ox, y: oy };
      if (!frameAnimRef.current) {
        frameAnimRef.current = requestAnimationFrame(animateFrameToTarget);
      }
    },
    [animateFrameToTarget, height, width],
  );

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    const img = imgRef.current;
    const prevImg = prevImgRef.current;
    if (!canvas || !container || (!img && !prevImg)) return;
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    if (cw <= 0 || ch <= 0 || width <= 0 || height <= 0) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const targetW = Math.floor(cw * dpr);
    const targetH = Math.floor(ch * dpr);
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
      canvas.style.width = `${cw}px`;
      canvas.style.height = `${ch}px`;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (canvasBg === "#0b1220" && document.documentElement.classList.contains("dark")) {
      // dark theme: leave the letterbox transparent so the translucent stage (and the site backdrop behind it) shows
      ctx.clearRect(0, 0, cw, ch);
    } else {
      ctx.fillStyle = canvasBg;
      ctx.fillRect(0, 0, cw, ch);
    }
    ctx.save();
    ctx.translate(offset.x, offset.y);
    ctx.scale(scale, scale);

    // Clip everything (image + boxes) to the logical image rectangle so boxes
    // never paint into letterbox / neighboring panes.
    // A margin of 12 screen px: boxes are kept on the image now, so this only lets the edge handles and the full line
    // width show instead of being cut in half exactly at the image border.
    const clipPad = 12 / scale;
    ctx.beginPath();
    ctx.rect(-clipPad, -clipPad, width + clipPad * 2, height + clipPad * 2);
    ctx.clip();

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = fullReadyRef.current ? "high" : "low";

    const fadeT = Math.max(0, Math.min(1, imageFadeTRef.current));
    const boxFade = Math.max(0, Math.min(1, boxesFadeTRef.current));
    const crossing = Boolean(prevImg && img && prevImg !== img && fadeT < 0.995);
    // One shared zoom for image + boxes so labels never drift during upgrade.
    const sceneZoom = crossing ? 1 + 0.014 * (1 - easeOutCubic(fadeT)) : 1;

    ctx.save();
    if (Math.abs(sceneZoom - 1) > 0.0005) {
      ctx.translate(width / 2, height / 2);
      ctx.scale(sceneZoom, sceneZoom);
      ctx.translate(-width / 2, -height / 2);
    }

    const paintPhoto = (src: ImageSource, alpha: number) => {
      if (alpha < 0.01) return;
      const dest = imageDestRect(src, width, height);
      ctx.save();
      ctx.globalAlpha = alpha;
      try {
        ctx.drawImage(src as CanvasImageSource, dest.dx, dest.dy, dest.dw, dest.dh);
      } catch {
        // Closed ImageBitmap — evict and re-fetch (filter remount used to be the only fix).
        forgetDecodedImage(src);
        if (imgRef.current === src) imgRef.current = null;
        if (prevImgRef.current === src) prevImgRef.current = null;
        fullReadyRef.current = false;
        setHasImage(false);
        if (!reloadPendingRef.current) {
          reloadPendingRef.current = true;
          setReloadGen((n) => n + 1);
        }
      }
      ctx.restore();
    };

    if (crossing && prevImg && img) {
      paintPhoto(prevImg, 1 - fadeT);
      paintPhoto(img, fadeT);
    } else if (img) {
      paintPhoto(img, 1);
    } else if (prevImg) {
      paintPhoto(prevImg, 1);
    }

    const showBoxes = Boolean(img || prevImg);
    const hoverT = Math.max(0, Math.min(1, focusTRef.current));
    const fillT = Math.max(0, Math.min(1, fillTRef.current));
    const focusW = focusWeightRef.current;
    const boxFocusW = (i: number) => focusW.get(`b:${i}`) ?? 0;
    const ghostFocusW = (i: number) => focusW.get(`g:${i}`) ?? 0;
    const overlayT = crossing && !qualityUpgradeRef.current ? fadeT : 1;
    let maxFocusW = 0;
    for (const v of focusW.values()) if (v > maxFocusW) maxFocusW = v;
    const hasFocus = hoverT > 0.01 && maxFocusW > 0.01;
    if (hasFocus && overlayT > 0.05) {
      const punch: DetectionBox[] = [];
      for (const [key, w] of focusW) {
        if (w < 0.02) continue;
        if (key.startsWith("b:")) {
          const i = Number(key.slice(2));
          if (i >= 0 && i < boxes.length) punch.push(boxes[i]);
        } else if (key.startsWith("g:")) {
          const ghosts = ghostBoxesRef.current;
          const i = Number(key.slice(2));
          if (ghosts && i >= 0 && i < ghosts.length) punch.push(ghosts[i]);
        }
      }
      if (punch.length) {
        ctx.save();
        ctx.globalAlpha = overlayT;
        // Dim surroundings so the focused box(es) read clearly.
        ctx.fillStyle = `rgba(0, 0, 0, ${0.36 * hoverT * Math.min(1, maxFocusW + 0.15)})`;
        ctx.beginPath();
        ctx.rect(0, 0, width, height);
        for (const box of punch) {
          const hx1 = Math.max(0, Math.min(Math.min(box.x1, box.x2), width));
          const hx2 = Math.max(0, Math.min(Math.max(box.x1, box.x2), width));
          const hy1 = Math.max(0, Math.min(Math.min(box.y1, box.y2), height));
          const hy2 = Math.max(0, Math.min(Math.max(box.y1, box.y2), height));
          ctx.rect(hx1, hy1, Math.max(0, hx2 - hx1), Math.max(0, hy2 - hy1));
        }
        ctx.fill("evenodd");
        ctx.restore();
      }
    }
    const boxA = boxFade * overlayT;
    if (showBoxes && boxA > 0.02) {
      ctx.save();
      const compare = boxCompareRef.current;
      const ghosts = ghostBoxesRef.current;
      const alertCount =
        (ghosts?.length || 0) +
        (compare?.reduce((n, r) => n + (r === "fp" || r === "wrong_class" ? 1 : 0), 0) || 0);
      // Crowded: keep thin outlines, but still flash so alerts stay findable.
      const quiet = alertCount >= 10;
      const veryQuiet = alertCount >= 24;
      const phase = alertPhaseRef.current;
      // Flash only (lighter ↔ darker) — no marching dash / rotate.
      const breath = 0.5 + 0.5 * Math.sin(phase * 0.0055);
      const breathFast = 0.5 + 0.5 * Math.sin(phase * 0.0085);
      // Keep other boxes as weak borders — never erase them.
      const dimStroke = hasFocus ? 0.72 + 0.12 * (1 - hoverT) : 1;

      const clampBox = (box: DetectionBox) => {
        let x1 = Math.min(box.x1, box.x2);
        let x2 = Math.max(box.x1, box.x2);
        let y1 = Math.min(box.y1, box.y2);
        let y2 = Math.max(box.y1, box.y2);
        x1 = Math.max(0, Math.min(x1, width));
        x2 = Math.max(0, Math.min(x2, width));
        y1 = Math.max(0, Math.min(y1, height));
        y2 = Math.max(0, Math.min(y2, height));
        return { x1, y1, x2, y2, bw: x2 - x1, bh: y2 - y1 };
      };

      const drawCornerTicks = (
        x1: number,
        y1: number,
        x2: number,
        y2: number,
        bw: number,
        bh: number,
        color: string,
        pulse = 1,
      ) => {
        const notch = Math.min(bw, bh, ((quiet ? 11 : 15) * (0.85 + 0.25 * pulse)) / scale) * 0.55;
        const t = ((quiet ? 1.8 : 2.4) * (0.9 + 0.25 * pulse)) / scale;
        ctx.fillStyle = color;
        ctx.fillRect(x1, y1, notch, t);
        ctx.fillRect(x1, y1, t, notch);
        ctx.fillRect(x2 - notch, y1, notch, t);
        ctx.fillRect(x2 - t, y1, t, notch);
        ctx.fillRect(x1, y2 - t, notch, t);
        ctx.fillRect(x1, y2 - notch, t, notch);
        ctx.fillRect(x2 - notch, y2 - t, notch, t);
        ctx.fillRect(x2 - t, y2 - notch, t, notch);
      };

      const drawRoleChip = (
        x1: number,
        y1: number,
        _x2: number,
        text: string,
        bg: string,
        border: string,
        pulse = 1,
        faded = false,
      ) => {
        const fontPx = Math.max(quiet ? 9 : 10, (quiet ? 10 : 12) / scale);
        ctx.font = `800 ${fontPx}px Inter, Segoe UI, sans-serif`;
        const padX = 4 / scale;
        const textW = ctx.measureText(text).width;
        const labelH = fontPx + 5 / scale;
        let labelY = y1 - 5 / scale - fontPx;
        if (labelY < 0) labelY = Math.min(y1 + 2 / scale, height - labelH);
        let labelX = x1;
        if (labelX + textW + padX * 2 > width) labelX = Math.max(0, width - textW - padX * 2);
        ctx.shadowColor = border;
        ctx.shadowBlur = faded ? 0 : ((quiet ? 3 : 5) + 4 * pulse) / scale;
        ctx.fillStyle = bg;
        // Keep M/E/W readable on non-focused boxes during hover.
        ctx.globalAlpha = boxA * (faded ? (veryQuiet ? 0.55 : 0.72) : 0.9 + 0.08 * pulse);
        ctx.fillRect(labelX, labelY, textW + padX * 2, labelH);
        ctx.shadowBlur = 0;
        ctx.strokeStyle = border;
        ctx.lineWidth = 1.1 / scale;
        ctx.strokeRect(labelX, labelY, textW + padX * 2, labelH);
        ctx.fillStyle = "#fff";
        ctx.fillText(text, labelX + padX, labelY + fontPx);
        ctx.globalAlpha = boxA;
      };

      // 1) MISSED GT — solid cyan flash (no marching dashes).
      const alertSolo = alertFilterRef.current;
      if (ghosts?.length && (!alertSolo || alertSolo === "missed")) {
        for (let gi = 0; gi < ghosts.length; gi += 1) {
          const box = ghosts[gi];
          const { x1, y1, x2, y2, bw, bh } = clampBox(box);
          if (bw < 1 || bh < 1) continue;
          const fwGi = ghostFocusW(gi);
          const faded = hasFocus && fwGi < 0.45;
          const pulse = faded ? 0.45 : 0.45 + 0.55 * breath * Math.max(fwGi, 0.35);
          const aMul = faded ? dimStroke + (1 - dimStroke) * fwGi : 1;
          ctx.save();
          ctx.globalAlpha = boxA * aMul;
          ctx.fillStyle = `rgba(34, 211, 238, ${quiet ? 0.03 + 0.04 * pulse : 0.05 + 0.06 * pulse})`;
          ctx.fillRect(x1, y1, bw, bh);
          if (!faded) {
            drawCornerTicks(x1, y1, x2, y2, bw, bh, `rgba(6, 182, 212, ${0.75 + 0.25 * pulse})`, pulse);
          }
          ctx.setLineDash([]);
          ctx.strokeStyle = `rgba(103, 232, 249, ${(faded ? 0.7 : 0.75) + 0.2 * pulse})`;
          ctx.lineWidth =
            ((quiet ? 1.5 : 1.9) + (faded ? 0 : 0.65 * pulse) + fwGi * 0.9) / scale;
          ctx.shadowColor = faded ? "transparent" : "rgba(34, 211, 238, 0.55)";
          ctx.shadowBlur = faded ? 0 : ((quiet ? 2.5 : 4) + 6 * pulse) / scale;
          ctx.strokeRect(x1, y1, bw, bh);
          ctx.shadowBlur = 0;
          // Always show M during hover (and normally) — short letter when crowded/dimmed.
          {
            const name = classNames[box.cls] ?? `cls ${box.cls}`;
            const text =
              faded || quiet || !showLabelsRef.current
                ? "M"
                : showLabelsRef.current
                  ? `MISSED · ${name}`
                  : "MISSED";
            drawRoleChip(
              x1,
              y1,
              x2,
              text,
              "rgba(8, 145, 178, 0.94)",
              "rgba(165, 243, 252, 0.95)",
              pulse,
              faded,
            );
          }
          ctx.restore();
        }
      }

      const renderBoxes = (() => {
        if (!draftBox) return boxes;
        const drag = dragRef.current;
        if (drag?.kind === "resize" && drag.idx != null) {
          return boxes.map((b, i) => (i === drag.idx ? draftBox : b));
        }
        return [...boxes, draftBox];
      })();

      const drawOne = (box: DetectionBox, idx: number, preferWrong: boolean) => {
        const drag = dragRef.current;
        const isDraft =
          Boolean(draftBox) &&
          ((drag?.kind === "draw" && idx === boxes.length) ||
            (drag?.kind === "resize" && drag.idx === idx));
        const color = colorForClass(box.cls);
        const selected = idx === selectedIndex || (isDraft && drag?.kind === "draw");
        const fwBi = isDraft ? 1 : boxFocusW(idx);
        const faded = hasFocus && fwBi < 0.45 && !isDraft;
        const role = !isDraft && compare ? compare[idx] : undefined;
        const isFp = role === "fp";
        const isWrongClass = role === "wrong_class";
        const isWrong = isFp || isWrongClass;
        if (preferWrong !== isWrong) return;
        const solo = alertFilterRef.current;
        if (solo === "missed") return;
        if (solo === "wrong" && !isWrongClass) return;
        if (solo === "extra" && !isFp) return;
        const { x1, y1, x2, y2, bw, bh } = clampBox(box);
        if (bw < 1 || bh < 1) return;
        ctx.save();
        ctx.globalAlpha = boxA * (faded ? dimStroke + (1 - dimStroke) * fwBi : 1);
        if (isWrong) {
          const pulse = faded ? 0.4 : 0.4 + 0.6 * (isFp ? breathFast : breath) * Math.max(fwBi, 0.35);
          ctx.fillStyle = isWrongClass
            ? `rgba(251, 146, 60, ${quiet ? 0.04 + 0.04 * pulse : 0.05 + 0.06 * pulse})`
            : `rgba(244, 63, 94, ${quiet ? 0.04 + 0.04 * pulse : 0.05 + 0.06 * pulse})`;
          ctx.fillRect(x1, y1, bw, bh);
          const alertColor = isWrongClass
            ? `rgba(251, 146, 60, ${(faded ? 0.7 : 0.75) + 0.25 * pulse})`
            : `rgba(244, 63, 94, ${(faded ? 0.7 : 0.75) + 0.25 * pulse})`;
          // Solid flash ring — no dash / rotate.
          ctx.setLineDash([]);
          ctx.strokeStyle = alertColor;
          ctx.lineWidth =
            ((quiet ? 1.7 : 2.1) + (faded ? 0 : 0.85 * pulse) + fwBi * 0.95) / scale;
          ctx.shadowColor = faded
            ? "transparent"
            : isWrongClass
              ? "rgba(251, 146, 60, 0.7)"
              : "rgba(244, 63, 94, 0.7)";
          ctx.shadowBlur = faded ? 0 : ((quiet ? 3 : 4.5) + 7 * pulse) / scale;
          ctx.strokeRect(x1, y1, bw, bh);
          ctx.shadowBlur = 0;
          if (!faded) {
            ctx.strokeStyle = isWrongClass ? "rgba(255, 237, 213, 0.7)" : "rgba(255, 228, 230, 0.7)";
            ctx.lineWidth = 1 / scale;
            ctx.strokeRect(x1 + 1.5 / scale, y1 + 1.5 / scale, bw - 3 / scale, bh - 3 / scale);
            drawCornerTicks(x1, y1, x2, y2, bw, bh, alertColor, pulse);
          }
          if (isFp && !faded) {
            const m = Math.min((quiet ? 7 : 9) / scale, bw * 0.2, bh * 0.2);
            ctx.strokeStyle = `rgba(255, 255, 255, ${0.75 + 0.25 * pulse})`;
            ctx.lineWidth = (1.6 + 0.35 * pulse) / scale;
            ctx.beginPath();
            ctx.moveTo(x2 - m - 1 / scale, y1 + 1 / scale);
            ctx.lineTo(x2 - 1 / scale, y1 + m + 1 / scale);
            ctx.moveTo(x2 - 1 / scale, y1 + 1 / scale);
            ctx.lineTo(x2 - m - 1 / scale, y1 + m + 1 / scale);
            ctx.stroke();
          }
          // Always show E / W tags during hover (short when dimmed/crowded).
          {
            const name = classNames[box.cls] ?? `cls ${box.cls}`;
            const short = isWrongClass ? "W" : "E";
            const full = isWrongClass
              ? showLabelsRef.current
                ? `WRONG · ${name}`
                : "WRONG"
              : showLabelsRef.current
                ? (box.raw_conf ?? box.conf) != null
                  ? `EXTRA · ${name} ${(((box.raw_conf ?? box.conf) as number) * 100).toFixed(0)}%`
                  : `EXTRA · ${name}`
                : "EXTRA";
            const text = faded || quiet || !showLabelsRef.current ? short : full;
            drawRoleChip(
              x1,
              y1,
              x2,
              text,
              isWrongClass ? "rgba(234, 88, 12, 0.95)" : "rgba(225, 29, 72, 0.95)",
              "rgba(255,255,255,0.9)",
              pulse,
              faded,
            );
          }
        } else {
          const bodyFill = selected ? 0.18 * fillT : 0;
          if (bodyFill > 0.01 && !faded) {
            ctx.fillStyle = withAlpha(color, bodyFill);
            ctx.fillRect(x1, y1, bw, bh);
          }
          ctx.strokeStyle = faded ? withAlpha(color, 0.62 + 0.1 * (1 - hoverT)) : color;
          const strokeBoost = selected ? 1 : fwBi * hoverT;
          ctx.lineWidth = ((faded ? 1.35 : 1.6) + 1.2 * strokeBoost) / scale;
          ctx.strokeRect(x1, y1, bw, bh);
          if (selected && editable && !faded) {
            const live = dragRef.current;
            const activeMode = live?.kind === "resize" && live.idx === idx ? live.mode : hoverHandleRef.current;
            for (const [mode, hx, hy] of handlePoints(box)) {
              drawHandle(ctx, hx, hy, color, scale, mode === activeMode);
            }
          }
          if (showLabelsRef.current && !faded) {
            const name = classNames[box.cls] ?? `cls ${box.cls}`;
            const shown = box.raw_conf ?? box.conf; // the network's own confidence when the miner reports a gated one
            const label = shown != null ? `${name} ${(shown * 100).toFixed(0)}%` : name;
            const fontPx = Math.max(11, 13 / scale);
            ctx.font = `800 ${fontPx}px Inter, Segoe UI, sans-serif`;
            const padX = 5 / scale;
            const textW = ctx.measureText(label).width;
            const labelH = fontPx + 6 / scale;
            let labelY = y1 - 6 / scale - fontPx;
            if (labelY < 0) labelY = Math.min(y1 + 3 / scale, height - labelH);
            let labelX = x1;
            if (labelX + textW + padX * 2 > width) labelX = Math.max(0, width - textW - padX * 2);
            const labelW = textW + padX * 2;
            ctx.beginPath();
            if (typeof ctx.roundRect === "function") {
              ctx.roundRect(labelX, labelY, labelW, labelH, 3 / scale);
            } else {
              ctx.rect(labelX, labelY, labelW, labelH);
            }
            ctx.fillStyle = withAlpha(color, 0.62);
            ctx.fill();
            ctx.lineWidth = 1 / scale;
            ctx.strokeStyle = withAlpha(color, 0.9);
            ctx.stroke();
            ctx.fillStyle = "#fff";
            ctx.fillText(label, labelX + padX, labelY + fontPx + 1 / scale);
          }
        }
        ctx.restore();
      };

      // 2) EXTRA / WRONG under TPs, then 3) true positives on top.
      renderBoxes.forEach((box, idx) => drawOne(box, idx, true));
      renderBoxes.forEach((box, idx) => drawOne(box, idx, false));
      ctx.restore();
    }

    // Smart dotted row/column guides (Own · R) — class-colored, follow pointer.
    if (showGuidesRef.current && guidePosRef.current) {
      const { x, y } = guidePosRef.current;
      const guideColor = colorForClass(currentClassRef.current);
      ctx.save();
      ctx.setLineDash([5 / scale, 4 / scale]);
      ctx.lineWidth = Math.max(1 / scale, 0.75 / scale);
      // Soft white underlay for contrast on dark/busy images
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.lineWidth = Math.max(2 / scale, 1.25 / scale);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
      ctx.strokeStyle = guideColor;
      ctx.globalAlpha = 0.92;
      ctx.lineWidth = Math.max(1 / scale, 0.75 / scale);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      ctx.fillStyle = guideColor;
      ctx.beginPath();
      ctx.arc(x, y, 2.25 / scale, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Review reasons: one small coloured badge with an icon on the box's lower-right corner (the class label keeps the
    // top-left). The reason in focus gets a bigger badge with a glow. Badges never sit on top of each other.
    const notes = calloutsRef.current;
    if (notes?.length) {
      ctx.save();
      const placed: Array<[number, number, number]> = [];
      for (const n of notes) {
        const r = (n.strong ? 11 : 8) / scale;
        let cx = Math.min(Math.max(n.x1, n.x2), width - r);
        let cy = Math.min(Math.max(n.y1, n.y2), height - r);
        cx = Math.max(r, cx);
        cy = Math.max(r, cy);
        for (let k = 0; k < 6 && placed.some(([px, py, pr]) => Math.hypot(px - cx, py - cy) < pr + r); k += 1) {
          cx = Math.max(r, cx - 2 * r - 2 / scale);
        }
        placed.push([cx, cy, r]);
        ctx.globalAlpha = 1;
        if (n.outline) {
          // A drink nobody labeled has no box of its own: a thin solid outline shows where it is.
          ctx.lineWidth = (n.strong ? 2.5 : 1.5) / scale;
          ctx.strokeStyle = n.color;
          ctx.strokeRect(Math.min(n.x1, n.x2), Math.min(n.y1, n.y2), Math.abs(n.x2 - n.x1), Math.abs(n.y2 - n.y1));
        }
        ctx.shadowColor = n.strong ? n.color : "rgba(0,0,0,0.55)";
        ctx.shadowBlur = (n.strong ? 12 : 4) / scale;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = n.color;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.lineWidth = (n.strong ? 2.5 : 1.5) / scale;
        ctx.strokeStyle = "#ffffff";
        ctx.stroke();
        const fontPx = r * 1.25;
        ctx.font = `900 ${fontPx}px Inter, Segoe UI, sans-serif`;
        ctx.fillStyle = "#0b1220";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(n.text, cx, cy + fontPx * 0.05);
      }
      ctx.restore();
    }

    ctx.restore(); // sceneZoom
    ctx.restore();
  }, [boxes, canvasBg, classNames, currentClass, draftBox, editable, height, offset, scale, selectedIndex, width]);

  const scheduleDraw = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(draw);
  }, [draw]);
  const drawRef = useRef(draw);
  drawRef.current = draw;

  useEffect(() => {
    scheduleDraw();
  }, [showLabels, currentClass, scheduleDraw]);

  useEffect(() => {
    if (!showGuides) {
      guidePosRef.current = null;
      scheduleDraw();
    }
  }, [showGuides, scheduleDraw]);

  const kickAnim = useCallback(() => {
    const step = (now: number) => {
      const last = animTsRef.current || now;
      animTsRef.current = now;
      const dt = Math.min(48, Math.max(0, now - last));
      // Slightly slower hover so focus handoff between boxes reads smoothly.
      const kHover = 1 - Math.exp(-dt / 110);
      const kFocus = 1 - Math.exp(-dt / 95);
      const kScene = 1 - Math.exp(-dt / 85);
      const kUpgrade = 1 - Math.exp(-dt / 145);
      const dragging = dragRef.current;
      const hasLinked =
        (linkedFocusIdxsRef.current?.length ?? 0) > 0 ||
        (linkedGhostIdxsRef.current?.length ?? 0) > 0;
      const targetFocus =
        (hoverIdxRef.current != null || hoverGhostIdxRef.current != null || hasLinked) &&
        dragging?.kind !== "pan" &&
        dragging?.kind !== "resize"
          ? 1
          : 0;
      const targetFill = selectedIndexRef.current != null ? 1 : 0;
      focusTRef.current += (targetFocus - focusTRef.current) * kHover;
      fillTRef.current += (targetFill - fillTRef.current) * kHover;

      // Desired focus keys from linked hover (cross-pane) or local pointer.
      const desired = new Set<string>();
      if (targetFocus > 0.5 || hasLinked || hoverIdxRef.current != null || hoverGhostIdxRef.current != null) {
        const linkedIdxs = linkedFocusIdxsRef.current;
        const linkedGhosts = linkedGhostIdxsRef.current;
        if (linkedIdxs?.length) {
          for (const i of linkedIdxs) desired.add(`b:${i}`);
        } else if (hoverIdxRef.current != null) {
          desired.add(`b:${hoverIdxRef.current}`);
        }
        if (linkedGhosts?.length) {
          for (const i of linkedGhosts) desired.add(`g:${i}`);
        } else if (hoverGhostIdxRef.current != null) {
          desired.add(`g:${hoverGhostIdxRef.current}`);
        }
      }
      const weights = focusWeightRef.current;
      let weightsDone = true;
      const keys = new Set<string>([...weights.keys(), ...desired]);
      for (const key of keys) {
        const target = desired.has(key) ? 1 : 0;
        const cur = weights.get(key) ?? 0;
        const next = cur + (target - cur) * kFocus;
        if (target === 0 && next < 0.004) {
          weights.delete(key);
        } else {
          weights.set(key, next);
          if (Math.abs(next - target) >= 0.004) weightsDone = false;
        }
      }

      boxesFadeTRef.current += (1 - boxesFadeTRef.current) * kScene;
      const hasNew = Boolean(imgRef.current && imgRef.current !== prevImgRef.current);
      const kImage = qualityUpgradeRef.current ? kUpgrade : kScene;
      if (prevImgRef.current && hasNew) {
        imageFadeTRef.current += (1 - imageFadeTRef.current) * kImage;
      } else if (!prevImgRef.current) {
        imageFadeTRef.current = 1;
      }
      const compare = boxCompareRef.current;
      const hasAlert =
        Boolean(ghostBoxesRef.current?.length) ||
        Boolean(compare?.some((r) => r === "fp" || r === "wrong_class"));
      if (hasAlert) alertPhaseRef.current = now;
      const focusDone = Math.abs(focusTRef.current - targetFocus) < 0.004;
      const fillDone = Math.abs(fillTRef.current - targetFill) < 0.004;
      const boxesDone = boxesFadeTRef.current > 0.995;
      const imageDone = imageFadeTRef.current > 0.995 || !prevImgRef.current || !hasNew;
      if (focusDone) focusTRef.current = targetFocus;
      if (fillDone) fillTRef.current = targetFill;
      if (boxesDone) boxesFadeTRef.current = 1;
      if (imageFadeTRef.current > 0.995) {
        imageFadeTRef.current = 1;
        prevImgRef.current = null;
        qualityUpgradeRef.current = false;
      }
      drawRef.current();
      if (!focusDone || !weightsDone || !fillDone || !boxesDone || !imageDone || hasAlert) {
        animRafRef.current = requestAnimationFrame(step);
      } else {
        animRafRef.current = 0;
        animTsRef.current = 0;
      }
    };
    if (!animRafRef.current) {
      animTsRef.current = 0;
      animRafRef.current = requestAnimationFrame(step);
    }
  }, []);

  useEffect(() => {
    scheduleDraw();
    kickAnim();
  }, [alertFilter, scheduleDraw, kickAnim]);

  useLayoutEffect(() => {
    let cancelled = false;
    // Scene identity is block/item — not fullUrl. Collect remaps list index and used to
    // cancel in-flight PNG upgrades whenever the index-bearing URL string changed.
    const sceneKey = sceneId || fullUrl;
    const sceneChanged = urlRef.current !== sceneKey;
    if (sceneChanged && imgRef.current) {
      prevImgRef.current = imgRef.current;
      imageFadeTRef.current = 0;
      boxesFadeTRef.current = 0;
      qualityUpgradeRef.current = false;
    }
    urlRef.current = sceneKey;
    if (sceneChanged) fullReadyRef.current = false;

    const fullHit = getDecodedImage(fullUrl);
    const previewHit = previewUrl ? getDecodedImage(previewUrl) : null;
    const thumbHit = thumbUrl ? getDecodedImage(thumbUrl) : null;
    const thumbSnap = !thumbHit && thumbUrl ? snapshotThumbnailImage(thumbUrl) : null;
    if (thumbSnap && thumbUrl) adoptDecodedImage(thumbUrl, thumbSnap);

    // Only treat full as "ready" when the decoded cache still holds a live bitmap.
    // A closed ImageBitmap left in imgRef paints a black pane until remount/filter switch.
    let previewReady = Boolean(fullHit || previewHit);

    const accept = (src: ImageSource, tier: "thumb" | "preview" | "full") => {
      if (cancelled) return;
      if (tier === "full") {
        if (!fullReadyRef.current && imgRef.current && imgRef.current !== src) {
          prevImgRef.current = imgRef.current;
          imageFadeTRef.current = 0;
          qualityUpgradeRef.current = true;
        }
        fullReadyRef.current = true;
        previewReady = true;
        reloadPendingRef.current = false;
        imgRef.current = src;
        setHasImage(true);
        kickAnim();
        drawRef.current();
        return;
      }
      if (fullReadyRef.current) return;
      if (tier === "preview") {
        if (!previewReady && imgRef.current && imgRef.current !== src) {
          prevImgRef.current = imgRef.current;
          imageFadeTRef.current = 0;
          qualityUpgradeRef.current = true;
        }
        previewReady = true;
        imgRef.current = src;
        setHasImage(true);
        kickAnim();
        drawRef.current();
        return;
      }
      // thumb — only if nothing better is ready
      if (previewReady) return;
      imgRef.current = src;
      setHasImage(true);
      kickAnim();
      drawRef.current();
    };

    const initial = fullHit ?? previewHit ?? thumbHit ?? thumbSnap ?? null;
    if (initial) {
      imgRef.current = initial;
      fullReadyRef.current = !!fullHit;
      previewReady = !!previewHit || !!fullHit;
      setHasImage(true);
      drawRef.current();
      kickAnim();
    } else if (sceneChanged) {
      // Keep previous frame until the new scene's image arrives.
    } else {
      // Same scene but cache miss (bitmaps were closed) — drop stale ref and reload.
      fullReadyRef.current = false;
      previewReady = false;
      imgRef.current = null;
      setHasImage(false);
    }

    if (thumbUrl) {
      const blob = getThumbnailBlobUrl(thumbUrl);
      if (blob && !getDecodedImage(thumbUrl)) {
        void decodeBlobUrl(thumbUrl, blob).then((src) => src && accept(src, "thumb"));
      } else if (!initial) {
        loadImageImmediate(thumbUrl);
      }
    }
    // Preview (~70 KB) first, then the full image (~300 KB): requested together they share the
    // bandwidth and the sharp preview arrives late on a slow link. If the user moves on before
    // the preview lands, the full image is never requested at all.
    const prevLoad = loadUrlsRef.current;
    loadUrlsRef.current = [previewUrl ?? "", fullUrl];
    prevLoad.forEach((u) => {
      if (u && u !== previewUrl && u !== fullUrl) abortImageLoad(u); // skipped image: stop its download
    });
    let fullTimer: number | null = null;
    const needFull = () => {
      if (fullTimer != null) {
        window.clearTimeout(fullTimer);
        fullTimer = null;
      }
      if (!cancelled) loadImageImmediate(fullUrl);
    };
    if (previewUrl && !previewHit && !fullHit) {
      loadImageImmediate(previewUrl);
      fullTimer = window.setTimeout(needFull, 700);
    } else {
      if (previewUrl) loadImageImmediate(previewUrl);
      needFull();
    }

    const unsubs: (() => void)[] = [];
    if (thumbUrl) {
      unsubs.push(subscribeImage(thumbUrl, (src) => accept(src, "thumb")));
      unsubs.push(
        subscribeThumbnail(thumbUrl, (blobUrl) => {
          if (cancelled || fullReadyRef.current || previewReady) return;
          const snap = snapshotThumbnailImage(thumbUrl);
          if (snap) {
            adoptDecodedImage(thumbUrl, snap);
            accept(snap, "thumb");
            return;
          }
          void decodeBlobUrl(thumbUrl, blobUrl).then((src) => src && accept(src, "thumb"));
        }),
      );
    }
    if (previewUrl) {
      unsubs.push(
        subscribeImage(previewUrl, (src) => {
          accept(src, "preview");
          if (fullTimer != null) {
            // preview is on screen: now fetch the sharp one
            window.clearTimeout(fullTimer);
            fullTimer = window.setTimeout(needFull, 0);
          }
        }),
      );
    }
    unsubs.push(subscribeImage(fullUrl, (src) => accept(src, "full")));
    return () => {
      cancelled = true;
      if (fullTimer != null) window.clearTimeout(fullTimer);
      unsubs.forEach((u) => u());
    };
  }, [fullUrl, previewUrl, thumbUrl, sceneId, kickAnim, reloadGen]);

  useEffect(() => {
    if (sceneRef.current === sceneId) return;
    sceneRef.current = sceneId;
    userZoomedRef.current = false;
    boxesFadeTRef.current = 0;
    hoverIdxRef.current = null;
    hoverGhostIdxRef.current = null;
    lastHoverIdxRef.current = null;
    lastHoverGhostIdxRef.current = null;
    focusTRef.current = 0;
    focusWeightRef.current.clear();
    lastInspectSeqRef.current = 0;
    cancelAnimationFrame(frameAnimRef.current);
    frameAnimRef.current = 0;
    reportHoverFocus(null, null);
    kickAnim();
  }, [sceneId, kickAnim, reportHoverFocus]);

  useEffect(() => {
    kickAnim();
  }, [linkedFocusIdxs, linkedGhostIdxs, kickAnim]);

  useEffect(() => {
    if (!inspectBox || !inspectSeq) return;
    if (inspectSeq === lastInspectSeqRef.current) return;
    lastInspectSeqRef.current = inspectSeq;
    frameBoxSmooth(inspectBox);
    kickAnim();
  }, [inspectBox, inspectSeq, frameBoxSmooth, kickAnim]);

  useLayoutEffect(() => {
    // Keep user zoom/pan while labeling — only auto-fit on new scene or before first zoom.
    if (!userZoomedRef.current) fitView();
  }, [width, height, fitView]);

  // A teammate's zoom window: glide there (same spot of the image, whatever our screen size is).
  useEffect(() => {
    const container = containerRef.current;
    if (!focusView || !container || !width || !height) return;
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    if (cw <= 0 || ch <= 0) return;
    const fit = Math.min(cw / width, ch / height);
    const sc = Math.min(16, Math.max(0.05, focusView.z * fit));
    targetScaleRef.current = sc;
    targetOffsetRef.current = { x: cw / 2 - focusView.cx * sc, y: ch / 2 - focusView.cy * sc };
    cancelAnimationFrame(zoomRafRef.current);
    zoomRafRef.current = 0;
    userZoomedRef.current = true;
    localZoomingRef.current = true;
    if (!frameAnimRef.current) frameAnimRef.current = requestAnimationFrame(animateFrameToTarget);
  }, [focusView?.cx, focusView?.cy, focusView?.z, sceneId, width, height, animateFrameToTarget]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!syncTransform) return;
    // Never interrupt an in-progress local wheel zoom (labeling re-renders used to cancel it).
    if (localZoomingRef.current || zoomRafRef.current) return;
    // Remote sync wins over a leftover inspect framing lerp.
    if (frameAnimRef.current) {
      cancelAnimationFrame(frameAnimRef.current);
      frameAnimRef.current = 0;
    }
    const same =
      Math.abs(syncTransform.scale - scaleRef.current) < 1e-4 &&
      Math.abs(syncTransform.offsetX - offsetRef.current.x) < 1e-4 &&
      Math.abs(syncTransform.offsetY - offsetRef.current.y) < 1e-4;
    if (same) return;
    targetScaleRef.current = syncTransform.scale;
    targetOffsetRef.current = { x: syncTransform.offsetX, y: syncTransform.offsetY };
    applyView(syncTransform.scale, syncTransform.offsetX, syncTransform.offsetY, false);
  }, [syncTransform, applyView]);

  useEffect(() => {
    if (hasImage) scheduleDraw();
    kickAnim();
  }, [hasImage, scheduleDraw, kickAnim, boxes, draftBox, offset, scale, selectedIndex, boxCompare, ghostBoxes, callouts]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => {
      // Don't clobber an in-progress user zoom when layout jitters.
      if (!syncTransform && !userZoomedRef.current && !localZoomingRef.current) fitView();
      scheduleDraw();
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [scheduleDraw, fitView, syncTransform, hasImage]);

  const applyZoomFactor = useCallback(
    (factor: number, anchorX: number, anchorY: number, emit = true) => {
      const oldScale = scaleRef.current;
      const next = Math.min(Math.max(oldScale * factor, 0.12), 16);
      const realFactor = next / Math.max(oldScale, 1e-6);
      const o = offsetRef.current;
      const ox = anchorX - (anchorX - o.x) * realFactor;
      const oy = anchorY - (anchorY - o.y) * realFactor;
      applyView(next, ox, oy, emit);
    },
    [applyView],
  );

  const animateZoomStep = useCallback(() => {
    const current = scaleRef.current;
    const target = targetScaleRef.current;
    const diff = target - current;
    const anchor = zoomAnchorRef.current;
    if (Math.abs(diff) < 0.002 * Math.max(1, target)) {
      if (current !== target) {
        applyZoomFactor(target / Math.max(current, 1e-6), anchor.x, anchor.y, true);
      } else {
        // Ensure follower pane gets the final transform.
        applyView(scaleRef.current, offsetRef.current.x, offsetRef.current.y, true);
      }
      zoomRafRef.current = 0;
      localZoomingRef.current = false;
      targetOffsetRef.current = { ...offsetRef.current };
      return;
    }
    // Emit every frame so the other pane tracks live; this pane ignores its own echo.
    const next = current + diff * 0.35;
    applyZoomFactor(next / Math.max(current, 1e-6), anchor.x, anchor.y, true);
    zoomRafRef.current = requestAnimationFrame(animateZoomStep);
  }, [applyView, applyZoomFactor]);

  // Native non-passive wheel on the whole pane so scroll always zooms.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const beginZoomToward = (clientX: number, clientY: number, steps: number) => {
      if (!steps) return;
      if (frameAnimRef.current) {
        cancelAnimationFrame(frameAnimRef.current);
        frameAnimRef.current = 0;
      }
      userZoomedRef.current = true;
      localZoomingRef.current = true;
      const rect = el.getBoundingClientRect();
      zoomAnchorRef.current = { x: clientX - rect.left, y: clientY - rect.top };
      const baseScale = scaleRef.current;
      const nextTarget = Math.min(Math.max(baseScale * Math.pow(1.42, steps), 0.12), 16);
      targetScaleRef.current = nextTarget;
      targetOffsetRef.current = { ...offsetRef.current };
      if (!zoomRafRef.current) {
        zoomRafRef.current = requestAnimationFrame(animateZoomStep);
      }
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      // Prefer vertical delta; fall back to horizontal trackpad gestures.
      const primary = Math.abs(e.deltaY) >= Math.abs(e.deltaX) ? e.deltaY : e.deltaX;
      let steps =
        e.deltaMode === 1 ? -primary : e.deltaMode === 2 ? -primary * 3 : -primary / 100;
      if (steps === 0 && (e.deltaY !== 0 || e.deltaX !== 0)) {
        steps = (e.deltaY || e.deltaX) < 0 ? 0.2 : -0.2;
      }
      if (!steps) return;
      beginZoomToward(e.clientX, e.clientY, steps);
    };
    const onPointerTrack = (e: PointerEvent) => {
      const rect = el.getBoundingClientRect();
      pointerCssRef.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
      pointerInsideRef.current = true;
    };
    const onEnter = (e: PointerEvent) => {
      pointerInsideRef.current = true;
      onPointerTrack(e);
    };
    const onLeave = () => {
      pointerInsideRef.current = false;
    };
    const onKeyZoom = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      if (e.key !== "z" && e.key !== "Z" && e.key !== "x" && e.key !== "X") return;
      if (!pointerInsideRef.current) return;
      e.preventDefault();
      e.stopPropagation();
      const rect = el.getBoundingClientRect();
      const anchor = pointerCssRef.current ?? { x: rect.width / 2, y: rect.height / 2 };
      // One notch per key (~wheel step); hold-repeat zooms smoothly.
      const steps = e.key === "z" || e.key === "Z" ? 0.35 : -0.35;
      beginZoomToward(rect.left + anchor.x, rect.top + anchor.y, steps);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("pointermove", onPointerTrack);
    el.addEventListener("pointerenter", onEnter);
    el.addEventListener("pointerleave", onLeave);
    window.addEventListener("keydown", onKeyZoom);
    return () => {
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("pointermove", onPointerTrack);
      el.removeEventListener("pointerenter", onEnter);
      el.removeEventListener("pointerleave", onLeave);
      window.removeEventListener("keydown", onKeyZoom);
      // Do not cancel an in-flight zoom here — effect deps can churn while labeling.
    };
  }, [animateZoomStep]);

  const onMouseDown = (e: React.MouseEvent) => {
    // Right button: drag pans the image; a click without moving over a box deletes it (decided on release).
    if (e.button === 2) {
      dragRef.current = { kind: "pan", startX: e.clientX, startY: e.clientY, right: true, downX: e.clientX, downY: e.clientY };
      return;
    }
    // Middle mouse (scroll-wheel button) pans — same as Space + left drag.
    if (e.button === 1) {
      e.preventDefault();
      dragRef.current = { kind: "pan", startX: e.clientX, startY: e.clientY };
      setCursor("grabbing");
      return;
    }
    if (e.button !== 0) return;
    const pt = toImage(e.clientX, e.clientY);
    if (spaceRef.current) {
      dragRef.current = { kind: "pan", startX: e.clientX, startY: e.clientY };
      setCursor("grabbing");
      return;
    }
    if (editable) {
      const selected = selectedIndex ?? null;
      const hit = hitTestResize(pt.x, pt.y, boxes, scaleRef.current, selected);
      if (hit) {
        const [idx, mode] = hit;
        dragRef.current = {
          kind: "resize",
          startX: pt.x,
          startY: pt.y,
          origin: { ...boxes[idx] },
          idx,
          mode,
          grabDx: mode?.includes("w") ? boxes[idx].x1 - pt.x : mode?.includes("e") ? boxes[idx].x2 - pt.x : 0,
          grabDy: mode?.includes("n") ? boxes[idx].y1 - pt.y : mode?.includes("s") ? boxes[idx].y2 - pt.y : 0,
          baseBoxes: boxes.map((b) => ({ ...b })),
        };
        hoverIdxRef.current = idx;
        onSelect?.(idx);
        kickAnim();
        return;
      }
      const inside = hitTestInside(pt.x, pt.y, boxes);
      if (inside != null) {
        hoverIdxRef.current = inside;
      } else {
        hoverIdxRef.current = null;
        onSelect?.(null);
      }
      const start = toImageClamped(e.clientX, e.clientY);
      dragRef.current = {
        kind: "draw",
        startX: start.x,
        startY: start.y,
        baseBoxes: boxes.map((b) => ({ ...b })),
      };
      kickAnim();
    } else {
      dragRef.current = { kind: "pan", startX: e.clientX, startY: e.clientY };
    }
  };

  const onMouseMove = (e: React.MouseEvent) => {
    if (showGuidesRef.current) {
      const gpt = toImage(e.clientX, e.clientY);
      const snap = 8 / Math.max(0.001, scaleRef.current);
      guidePosRef.current = smartGuidePos(gpt, boxes, snap, draftBox);
      scheduleDraw();
    }
    const drag = dragRef.current;
    if (!drag) {
      const pt = toImage(e.clientX, e.clientY);
      if (editable && !spaceRef.current) {
        const pt2 = pt;
        const hit = hitTestResize(pt2.x, pt2.y, boxes, scaleRef.current, selectedIndex ?? null);
        const inside = hitTestInside(pt2.x, pt2.y, boxes);
        if (hit) setCursor(cursorForMode(hit[1], classCursorRef.current));
        else setCursor(cursorForMode("draw", classCursorRef.current));
        const nextHandle = hit ? hit[1] : null;
        if (nextHandle !== hoverHandleRef.current) {
          hoverHandleRef.current = nextHandle;
          kickAnim();
        }
        const nextHover = hit ? hit[0] : inside;
        if (nextHover !== hoverIdxRef.current || hoverGhostIdxRef.current != null) {
          hoverIdxRef.current = nextHover;
          hoverGhostIdxRef.current = null;
          reportHoverFocus(nextHover, null);
          kickAnim();
        }
      } else {
        setCursor(spaceRef.current ? "grab" : editable ? classCursorRef.current : "grab");
        if (spaceRef.current) {
          if (hoverIdxRef.current != null || hoverGhostIdxRef.current != null) {
            hoverIdxRef.current = null;
            hoverGhostIdxRef.current = null;
            reportHoverFocus(null, null);
            kickAnim();
          }
        } else {
          const solo = alertFilterRef.current;
          const compare = boxCompareRef.current;
          const ghosts = ghostBoxesRef.current;
          let nextBox: number | null = null;
          let nextGhost: number | null = null;
          if (solo === "missed") {
            nextGhost = ghosts?.length ? hitTestInside(pt.x, pt.y, ghosts) : null;
          } else if (solo === "wrong" || solo === "extra") {
            const want = solo === "wrong" ? "wrong_class" : "fp";
            let bestI = -1;
            let bestArea = Infinity;
            for (let i = 0; i < boxes.length; i += 1) {
              if (compare?.[i] !== want) continue;
              const { x1, y1, x2, y2 } = boxBounds(boxes[i]);
              if (pt.x < x1 || pt.x > x2 || pt.y < y1 || pt.y > y2) continue;
              const area = Math.max(1, (x2 - x1) * (y2 - y1));
              if (area < bestArea) {
                bestArea = area;
                bestI = i;
              }
            }
            nextBox = bestI >= 0 ? bestI : null;
          } else {
            const hit = hitTestBoxOrGhost(pt.x, pt.y, boxes, ghosts);
            nextBox = hit.boxIdx;
            nextGhost = hit.ghostIdx;
          }
          if (nextBox !== hoverIdxRef.current || nextGhost !== hoverGhostIdxRef.current) {
            hoverIdxRef.current = nextBox;
            hoverGhostIdxRef.current = nextGhost;
            reportHoverFocus(nextBox, nextGhost);
            kickAnim();
          }
        }
      }
      return;
    }
    if (drag.kind === "pan") {
      // User pan wins over inspect framing animation.
      if (frameAnimRef.current) {
        cancelAnimationFrame(frameAnimRef.current);
        frameAnimRef.current = 0;
        localZoomingRef.current = false;
      }
      if (drag.right && Math.hypot(e.clientX - (drag.downX ?? e.clientX), e.clientY - (drag.downY ?? e.clientY)) > 4) setCursor("grabbing");
      const dx = e.clientX - drag.startX;
      const dy = e.clientY - drag.startY;
      dragRef.current = { ...drag, startX: e.clientX, startY: e.clientY };
      const o = offsetRef.current;
      const next = { x: o.x + dx, y: o.y + dy };
      targetScaleRef.current = scaleRef.current;
      targetOffsetRef.current = next;
      applyView(scaleRef.current, next.x, next.y);
      return;
    }
    const pt = toImage(e.clientX, e.clientY);
    if (drag.kind === "resize" && drag.origin && drag.idx != null && drag.mode) {
      const updated = resizedBox(drag.origin, drag.mode, pt.x + (drag.grabDx ?? 0), pt.y + (drag.grabDy ?? 0), width, height);
      // Local draft for smooth canvas paint; also stream geometry to App so R1–R3
      // p vs t metrics / EXTRA·WRONG·MISSED update while the boundary moves.
      setDraftBox(updated);
      hoverIdxRef.current = drag.idx;
      kickAnim();
      const base = drag.baseBoxes ?? boxes;
      publishLiveBoxes(
        base.map((b, i) => (i === drag.idx ? updated : b)),
        drag.idx,
      );
    } else if (drag.kind === "draw") {
      const cp = toImageClamped(e.clientX, e.clientY);
      const box: DetectionBox = {
        cls: currentClass,
        x1: Math.min(drag.startX, cp.x),
        y1: Math.min(drag.startY, cp.y),
        x2: Math.max(drag.startX, cp.x),
        y2: Math.max(drag.startY, cp.y),
      };
      setDraftBox(box);
      const w = Math.abs(box.x2 - box.x1);
      const h = Math.abs(box.y2 - box.y1);
      if (w >= MIN_DRAW && h >= MIN_DRAW) {
        const base = drag.baseBoxes ?? boxes;
        publishLiveBoxes([...base, box], base.length);
      }
    }
  };

  const onMouseUp = (e: React.MouseEvent) => {
    const drag = dragRef.current;
    dragRef.current = null;
    setDraftBox(null);
    flushLivePublish();
    if (spaceRef.current) {
      setCursor("grab");
    } else if (editable) {
      setCursor(classCursorRef.current);
    } else {
      setCursor("grab");
    }
    if (drag?.right) {
      if (editable && Math.hypot(e.clientX - (drag.downX ?? e.clientX), e.clientY - (drag.downY ?? e.clientY)) <= 4) {
        const pt = toImage(e.clientX, e.clientY);
        const hits = boxes
          .map((b, i) => ({
            i,
            area: Math.abs(b.x2 - b.x1) * Math.abs(b.y2 - b.y1),
            inside: pt.x >= Math.min(b.x1, b.x2) && pt.x <= Math.max(b.x1, b.x2) && pt.y >= Math.min(b.y1, b.y2) && pt.y <= Math.max(b.y1, b.y2),
          }))
          .filter((h) => h.inside)
          .sort((a, b) => a.area - b.area);
        if (hits.length) {
          onBoxesChange?.(boxes.filter((_, i) => i !== hits[0].i), null, true);
          onSelect?.(null);
        }
      }
      return;
    }
    if (!drag || !editable) return;
    if (drag.kind === "draw") {
      const pt = toImage(e.clientX, e.clientY);
      const cp = toImageClamped(e.clientX, e.clientY);
      const w = Math.abs(cp.x - drag.startX);
      const h = Math.abs(cp.y - drag.startY);
      const base = drag.baseBoxes ?? boxes;
      if (w >= MIN_DRAW && h >= MIN_DRAW) {
        const box: DetectionBox = {
          cls: currentClass,
          x1: Math.min(drag.startX, cp.x),
          y1: Math.min(drag.startY, cp.y),
          x2: Math.max(drag.startX, cp.x),
          y2: Math.max(drag.startY, cp.y),
        };
        const next = [...base, box];
        onBoxesChange?.(next, next.length - 1, true);
        onSelect?.(next.length - 1);
      } else {
        // Undo any live draft that was streamed for scoring mid-gesture.
        if (drag.baseBoxes) onBoxesChange?.(drag.baseBoxes, selectedIndex ?? null, false);
        const hits = base
          .map((b, i) => ({
            i,
            area: Math.abs(b.x2 - b.x1) * Math.abs(b.y2 - b.y1),
            inside:
              pt.x >= Math.min(b.x1, b.x2) &&
              pt.x <= Math.max(b.x1, b.x2) &&
              pt.y >= Math.min(b.y1, b.y2) &&
              pt.y <= Math.max(b.y1, b.y2),
          }))
          .filter((h) => h.inside)
          .sort((a, b) => a.area - b.area);
        onSelect?.(hits.length ? hits[0].i : null);
      }
    } else if (drag.kind === "resize" && drag.origin && drag.idx != null && drag.mode) {
      const pt = toImage(e.clientX, e.clientY);
      const updated = resizedBox(drag.origin, drag.mode, pt.x + (drag.grabDx ?? 0), pt.y + (drag.grabDy ?? 0), width, height);
      const base = drag.baseBoxes ?? boxes;
      const next = base.map((b, i) => (i === drag.idx ? updated : b));
      const moved =
        drag.origin.x1 !== updated.x1 ||
        drag.origin.y1 !== updated.y1 ||
        drag.origin.x2 !== updated.x2 ||
        drag.origin.y2 !== updated.y2;
      if (moved) {
        onBoxesChange?.(next, drag.idx, true);
      } else if (drag.baseBoxes) {
        onBoxesChange?.(drag.baseBoxes, drag.idx, false);
      }
    }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) {
        return;
      }
      if (e.code === "Space") {
        spaceRef.current = e.type === "keydown";
        if (!dragRef.current) {
          setCursor(e.type === "keydown" ? "grab" : editable ? classCursorRef.current : "grab");
        }
      }
      if (e.type === "keydown" && e.key === "f") {
        fitView();
      }
    };
    const onBlur = () => {
      spaceRef.current = false;
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKey);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("blur", onBlur);
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKey);
    };
  }, [classNames.length, editable, fitView]);

  useEffect(() => {
    return () => {
      cancelAnimationFrame(animRafRef.current);
      cancelAnimationFrame(rafRef.current);
      cancelAnimationFrame(frameAnimRef.current);
      cancelAnimationFrame(zoomRafRef.current);
    };
  }, []);

  const thumbBlob = thumbUrl ? getThumbnailBlobUrl(thumbUrl) : null;

  return (
    <div
      ref={containerRef}
      className="relative min-h-[280px] w-full flex-1 overflow-hidden rounded-xl bg-slate-100 dark:bg-ink-950/30"
      style={{ touchAction: "none", overscrollBehavior: "none" }}
    >
      {!hasImage && thumbBlob && (
        <img
          src={thumbBlob}
          alt=""
          draggable={false}
          className="pointer-events-none absolute inset-0 h-full w-full object-contain"
        />
      )}
      {!hasImage && !thumbBlob && (
        <div className="absolute inset-0 bg-slate-200 dark:bg-ink-900" />
      )}
      <canvas
        ref={canvasRef}
        className="block h-full w-full"
        style={{ cursor, touchAction: "none" }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={(e) => {
          hoverIdxRef.current = null;
          hoverGhostIdxRef.current = null;
          reportHoverFocus(null, null);
          if (guidePosRef.current) {
            guidePosRef.current = null;
            scheduleDraw();
          }
          kickAnim();
          onMouseUp(e);
        }}
        onContextMenu={(e) => e.preventDefault()}
        onAuxClick={(e) => {
          // Block browser auto-scroll / new-tab from middle click.
          if (e.button === 1) e.preventDefault();
        }}
      />
    </div>
  );
}
