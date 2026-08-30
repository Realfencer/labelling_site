import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { DetectionBox } from "../types";
import type { BoxCompareRole } from "../utils/gtScore";
import { colorForClass } from "../utils/colors";
import {
  adoptDecodedImage,
  decodeBlobUrl,
  getDecodedImage,
  loadImageImmediate,
  subscribeImage,
  type ImageSource,
} from "../store/imageLoader";
import { getThumbnailBlobUrl, snapshotThumbnailImage, subscribeThumbnail } from "../store/thumbnailLoader";

type ResizeMode = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w" | null;
type Transform = { scale: number; offsetX: number; offsetY: number };

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
  canvasBg?: string;
  sceneId?: string;
  /** Per-box match vs left-pane GT (right panel compare mode). */
  boxCompare?: BoxCompareRole[];
  /** GT boxes missed by miner — ghost overlay on right panel. */
  ghostBoxes?: DetectionBox[];
  /** When false, draw box outlines only (hide class name panels). */
  showLabels?: boolean;
}

const MIN_DRAW = 3;
const HANDLE_HIT = 10;
const classCursorCache = new Map<number, string>();

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

function drawHandle(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  color: string,
  scale: number,
) {
  const r = 4.4 / scale;
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, r + 2.2 / scale, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0,0,0,0.28)";
  ctx.fill();
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();
  ctx.lineWidth = 1.6 / scale;
  ctx.strokeStyle = color;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x, y, r * 0.32, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.restore();
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
  const tol = Math.max(2, HANDLE_HIT / scale);
  const [x1, x2] = [box.x1, box.x2].sort((a, b) => a - b);
  const [y1, y2] = [box.y1, box.y2].sort((a, b) => a - b);
  if (x < x1 - tol || x > x2 + tol || y < y1 - tol || y > y2 + tol) return null;
  const nearL = Math.abs(x - x1) <= tol;
  const nearR = Math.abs(x - x2) <= tol;
  const nearT = Math.abs(y - y1) <= tol;
  const nearB = Math.abs(y - y2) <= tol;
  const minSide = Math.max(1, Math.min(x2 - x1, y2 - y1));
  let mode: ResizeMode = null;
  if (nearT && nearL) mode = "nw";
  else if (nearT && nearR) mode = "ne";
  else if (nearB && nearL) mode = "sw";
  else if (nearB && nearR) mode = "se";
  else if (minSide >= 22 && nearT) mode = "n";
  else if (minSide >= 22 && nearB) mode = "s";
  else if (minSide >= 22 && nearL) mode = "w";
  else if (minSide >= 22 && nearR) mode = "e";
  return mode ? [selected, mode] : null;
}

function resizedBox(origin: DetectionBox, mode: string, x: number, y: number, imgW: number, imgH: number): DetectionBox {
  let { x1, y1, x2, y2 } = origin;
  if (mode.includes("n")) y1 = y;
  if (mode.includes("s")) y2 = y;
  if (mode.includes("w")) x1 = x;
  if (mode.includes("e")) x2 = x;
  let [left, right] = [x1, x2].sort((a, b) => a - b);
  let [top, bottom] = [y1, y2].sort((a, b) => a - b);
  left = Math.min(Math.max(left, 0), imgW - 1);
  right = Math.min(Math.max(right, 0), imgW - 1);
  top = Math.min(Math.max(top, 0), imgH - 1);
  bottom = Math.min(Math.max(bottom, 0), imgH - 1);
  if (right - left < MIN_DRAW) right = left + MIN_DRAW;
  if (bottom - top < MIN_DRAW) bottom = top + MIN_DRAW;
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
  const size = 32;
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d");
  if (!ctx) return "crosshair";
  const mid = size / 2;
  ctx.clearRect(0, 0, size, size);

  // White halo for contrast on dark/light images
  ctx.beginPath();
  ctx.arc(mid, mid, 8, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.lineWidth = 4;
  ctx.stroke();

  // Class-colored ring with soft fill
  ctx.beginPath();
  ctx.arc(mid, mid, 8, 0, Math.PI * 2);
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.fillStyle = `${color}5a`;
  ctx.fill();
  ctx.stroke();

  // Crosshair arms
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(mid, 1);
  ctx.lineTo(mid, 8);
  ctx.moveTo(mid, 24);
  ctx.lineTo(mid, 31);
  ctx.moveTo(1, mid);
  ctx.lineTo(8, mid);
  ctx.moveTo(24, mid);
  ctx.lineTo(31, mid);
  ctx.stroke();

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
  canvasBg = "#0b1220",
  sceneId,
  boxCompare,
  ghostBoxes,
  showLabels = true,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<ImageSource | null>(null);
  const fullReadyRef = useRef(false);
  const [hasImage, setHasImage] = useState(false);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [cursor, setCursor] = useState("crosshair");
  const classCursorRef = useRef("crosshair");

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
    /** Snapshot at pointer-down so live scoring never double-appends drafts. */
    baseBoxes?: DetectionBox[];
  } | null>(null);
  const spaceRef = useRef(false);
  const rafRef = useRef(0);
  const hoverIdxRef = useRef<number | null>(null);
  const lastHoverIdxRef = useRef<number | null>(null);
  const focusTRef = useRef(0);
  const fillTRef = useRef(0);
  const prevImgRef = useRef<ImageSource | null>(null);
  const imageFadeTRef = useRef(1);
  const boxesFadeTRef = useRef(1);
  const urlRef = useRef(fullUrl);
  const sceneRef = useRef(sceneId);
  const animRafRef = useRef(0);
  const animTsRef = useRef(0);
  const selectedIndexRef = useRef(selectedIndex);
  selectedIndexRef.current = selectedIndex;
  const scaleRef = useRef(1);
  const offsetRef = useRef({ x: 0, y: 0 });
  const targetScaleRef = useRef(1);
  const zoomAnchorRef = useRef({ x: 0, y: 0 });
  const zoomRafRef = useRef(0);
  const userZoomedRef = useRef(false);
  const localZoomingRef = useRef(false);
  /** True while crossfading strip thumb → full (same scene, quality upgrade only). */
  const qualityUpgradeRef = useRef(false);
  /** Soft pulse phase for EXTRA / WRONG / MISSED overlays (ms-based). */
  const alertPhaseRef = useRef(0);
  const boxCompareRef = useRef(boxCompare);
  boxCompareRef.current = boxCompare;
  const ghostBoxesRef = useRef(ghostBoxes);
  ghostBoxesRef.current = ghostBoxes;
  const showLabelsRef = useRef(showLabels);
  showLabelsRef.current = showLabels;
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

  const applyView = useCallback(
    (s: number, ox: number, oy: number, emit = true) => {
      scaleRef.current = s;
      offsetRef.current = { x: ox, y: oy };
      setScale(s);
      setOffset({ x: ox, y: oy });
      if (emit) onTransformChange?.({ scale: s, offsetX: ox, offsetY: oy });
    },
    [onTransformChange],
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

  const fitView = useCallback(() => {
    const container = containerRef.current;
    if (!container || !width || !height) return;
    const pad = 10;
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    if (cw <= 0 || ch <= 0) return;
    cancelAnimationFrame(zoomRafRef.current);
    zoomRafRef.current = 0;
    userZoomedRef.current = false;
    const s = Math.min((cw - pad * 2) / width, (ch - pad * 2) / height, 1);
    const ox = (cw - width * s) / 2;
    const oy = (ch - height * s) / 2;
    targetScaleRef.current = s;
    applyView(s, ox, oy);
  }, [width, height, applyView]);

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
    ctx.fillStyle = canvasBg;
    ctx.fillRect(0, 0, cw, ch);
    ctx.save();
    ctx.translate(offset.x, offset.y);
    ctx.scale(scale, scale);

    // Clip everything (image + boxes) to the logical image rectangle so boxes
    // never paint into letterbox / neighboring panes.
    ctx.beginPath();
    ctx.rect(0, 0, width, height);
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
      ctx.drawImage(src as CanvasImageSource, dest.dx, dest.dy, dest.dw, dest.dh);
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
    const hoverIdx = hoverIdxRef.current;
    if (hoverIdx != null) lastHoverIdxRef.current = hoverIdx;
    const paintHoverIdx = hoverIdx ?? lastHoverIdxRef.current;
    const overlayT = crossing && !qualityUpgradeRef.current ? fadeT : 1;
    const hoverBox =
      paintHoverIdx != null && paintHoverIdx >= 0 && paintHoverIdx < boxes.length ? boxes[paintHoverIdx] : null;
    if (hoverT > 0.01 && hoverBox && overlayT > 0.05) {
      const hx1 = Math.max(0, Math.min(Math.min(hoverBox.x1, hoverBox.x2), width));
      const hx2 = Math.max(0, Math.min(Math.max(hoverBox.x1, hoverBox.x2), width));
      const hy1 = Math.max(0, Math.min(Math.min(hoverBox.y1, hoverBox.y2), height));
      const hy2 = Math.max(0, Math.min(Math.max(hoverBox.y1, hoverBox.y2), height));
      ctx.save();
      ctx.globalAlpha = overlayT;
      ctx.fillStyle = `rgba(0, 0, 0, ${0.4 * hoverT})`;
      ctx.beginPath();
      ctx.rect(0, 0, width, height);
      ctx.rect(hx1, hy1, Math.max(0, hx2 - hx1), Math.max(0, hy2 - hy1));
      ctx.fill("evenodd");
      ctx.restore();
    }
    const boxA = boxFade * overlayT;
    if (showBoxes && boxA > 0.02) {
      ctx.save();
      const compare = boxCompareRef.current;
      const ghosts = ghostBoxesRef.current;

      if (ghosts?.length) {
        // Missed GT: cyan wash + marching dashes + corner ticks + MISSED badge.
        const phase = alertPhaseRef.current;
        const pulse = 0.62 + 0.38 * (0.5 + 0.5 * Math.sin(phase * 0.007));
        for (const box of ghosts) {
          let x1 = Math.min(box.x1, box.x2);
          let x2 = Math.max(box.x1, box.x2);
          let y1 = Math.min(box.y1, box.y2);
          let y2 = Math.max(box.y1, box.y2);
          x1 = Math.max(0, Math.min(x1, width));
          x2 = Math.max(0, Math.min(x2, width));
          y1 = Math.max(0, Math.min(y1, height));
          y2 = Math.max(0, Math.min(y2, height));
          if (x2 - x1 < 1 || y2 - y1 < 1) continue;
          const bw = x2 - x1;
          const bh = y2 - y1;
          ctx.save();
          ctx.globalAlpha = boxA;
          ctx.fillStyle = `rgba(34, 211, 238, ${0.14 + 0.12 * pulse})`;
          ctx.fillRect(x1, y1, bw, bh);
          const notch = Math.min(bw, bh, 16 / scale) * 0.55;
          ctx.fillStyle = "rgba(6, 182, 212, 0.98)";
          ctx.fillRect(x1, y1, notch, 2.8 / scale);
          ctx.fillRect(x1, y1, 2.8 / scale, notch);
          ctx.fillRect(x2 - notch, y1, notch, 2.8 / scale);
          ctx.fillRect(x2 - 2.8 / scale, y1, 2.8 / scale, notch);
          ctx.fillRect(x1, y2 - 2.8 / scale, notch, 2.8 / scale);
          ctx.fillRect(x1, y2 - notch, 2.8 / scale, notch);
          ctx.fillRect(x2 - notch, y2 - 2.8 / scale, notch, 2.8 / scale);
          ctx.fillRect(x2 - 2.8 / scale, y2 - notch, 2.8 / scale, notch);
          ctx.setLineDash([7 / scale, 5 / scale]);
          ctx.lineDashOffset = -((phase * 0.045) % 24);
          ctx.strokeStyle = `rgba(103, 232, 249, ${0.75 + 0.25 * pulse})`;
          ctx.lineWidth = (2.4 + 1.1 * pulse) / scale;
          ctx.strokeRect(x1, y1, bw, bh);
          ctx.setLineDash([]);
          ctx.lineDashOffset = 0;
          // Always show a compact role chip so misses stay visible with T (labels off).
          const name = classNames[box.cls] ?? `cls ${box.cls}`;
          const label = showLabelsRef.current ? `MISSED · ${name}` : "MISSED";
          const fontPx = Math.max(11, 13 / scale);
          ctx.font = `800 ${fontPx}px Inter, Segoe UI, sans-serif`;
          const padX = 5 / scale;
          const textW = ctx.measureText(label).width;
          const labelH = fontPx + 6 / scale;
          let labelY = y1 - 6 / scale - fontPx;
          if (labelY < 0) labelY = Math.min(y1 + 3 / scale, height - labelH);
          ctx.shadowColor = "rgba(6, 182, 212, 0.55)";
          ctx.shadowBlur = (6 + 4 * pulse) / scale;
          ctx.fillStyle = "rgba(8, 145, 178, 0.94)";
          ctx.fillRect(x1, labelY, textW + padX * 2, labelH);
          ctx.shadowBlur = 0;
          ctx.strokeStyle = "rgba(165, 243, 252, 0.95)";
          ctx.lineWidth = 1 / scale;
          ctx.strokeRect(x1, labelY, textW + padX * 2, labelH);
          ctx.fillStyle = "#ecfeff";
          ctx.fillText(label, x1 + padX, labelY + fontPx + 1 / scale);
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
      renderBoxes.forEach((box, idx) => {
        const drag = dragRef.current;
        const isDraft =
          Boolean(draftBox) &&
          ((drag?.kind === "draw" && idx === boxes.length) ||
            (drag?.kind === "resize" && drag.idx === idx));
        const color = colorForClass(box.cls);
        const selected = idx === selectedIndex || (isDraft && drag?.kind === "draw");
        const focused = paintHoverIdx != null && idx === paintHoverIdx;
        const role = !isDraft && compare ? compare[idx] : undefined;
        const isFp = role === "fp";
        const isWrongClass = role === "wrong_class";
        const isWrong = isFp || isWrongClass;
        let x1 = Math.min(box.x1, box.x2);
        let x2 = Math.max(box.x1, box.x2);
        let y1 = Math.min(box.y1, box.y2);
        let y2 = Math.max(box.y1, box.y2);
        x1 = Math.max(0, Math.min(x1, width));
        x2 = Math.max(0, Math.min(x2, width));
        y1 = Math.max(0, Math.min(y1, height));
        y2 = Math.max(0, Math.min(y2, height));
        if (x2 - x1 < 1 || y2 - y1 < 1) return;
        const bw = x2 - x1;
        const bh = y2 - y1;
        ctx.save();
        ctx.globalAlpha = boxA;
        if (isWrong) {
          const phase = alertPhaseRef.current;
          const pulse = 0.55 + 0.45 * (0.5 + 0.5 * Math.sin(phase * (isFp ? 0.0085 : 0.0065)));
          // Strong fill so extras / wrong-class pop against correct TPs.
          ctx.fillStyle = isWrongClass
            ? `rgba(251, 146, 60, ${0.22 + 0.14 * pulse})`
            : `rgba(244, 63, 94, ${0.24 + 0.16 * pulse})`;
          ctx.fillRect(x1, y1, bw, bh);
          // Diagonal hatch for false positives — drifts so EXTRA reads as “alive”.
          if (isFp) {
            ctx.save();
            ctx.beginPath();
            ctx.rect(x1, y1, bw, bh);
            ctx.clip();
            ctx.strokeStyle = `rgba(251, 113, 133, ${0.42 + 0.28 * pulse})`;
            ctx.lineWidth = 1.35 / scale;
            const step = 8 / scale;
            const drift = ((phase * 0.04) % step);
            for (let x = x1 - bh - step; x < x2 + bh; x += step) {
              ctx.beginPath();
              ctx.moveTo(x + drift, y1);
              ctx.lineTo(x + drift + bh, y2);
              ctx.stroke();
            }
            ctx.restore();
          } else {
            // Wrong-class: short amber tick hatch (different from EXTRA).
            ctx.save();
            ctx.beginPath();
            ctx.rect(x1, y1, bw, bh);
            ctx.clip();
            ctx.strokeStyle = `rgba(251, 191, 36, ${0.35 + 0.25 * pulse})`;
            ctx.lineWidth = 1.2 / scale;
            const step = 10 / scale;
            for (let y = y1 - 2; y < y2 + 2; y += step) {
              ctx.beginPath();
              ctx.moveTo(x1, y);
              ctx.lineTo(x2, y);
              ctx.stroke();
            }
            ctx.restore();
          }
          const alertColor = isWrongClass
            ? `rgba(251, 146, 60, ${0.78 + 0.22 * pulse})`
            : `rgba(244, 63, 94, ${0.78 + 0.22 * pulse})`;
          if (isWrongClass) {
            ctx.setLineDash([5 / scale, 4 / scale]);
            ctx.lineDashOffset = -((phase * 0.04) % 18);
          }
          ctx.strokeStyle = alertColor;
          ctx.lineWidth = (2.8 + 1.6 * pulse) / scale;
          ctx.shadowColor = isWrongClass ? "rgba(251, 146, 60, 0.55)" : "rgba(244, 63, 94, 0.55)";
          ctx.shadowBlur = (5 + 6 * pulse) / scale;
          ctx.strokeRect(x1, y1, bw, bh);
          ctx.shadowBlur = 0;
          ctx.setLineDash([]);
          ctx.lineDashOffset = 0;
          // Inner ring
          ctx.strokeStyle = isWrongClass ? "rgba(255, 237, 213, 0.9)" : "rgba(255, 228, 230, 0.9)";
          ctx.lineWidth = 1.2 / scale;
          ctx.strokeRect(x1 + 2.5 / scale, y1 + 2.5 / scale, bw - 5 / scale, bh - 5 / scale);
          // Corner X for EXTRA so it is obvious at a glance.
          if (isFp) {
            const m = Math.min(11 / scale, bw * 0.22, bh * 0.22);
            ctx.strokeStyle = `rgba(255, 255, 255, ${0.75 + 0.25 * pulse})`;
            ctx.lineWidth = (2.2 + 0.6 * pulse) / scale;
            ctx.beginPath();
            ctx.moveTo(x2 - m - 2 / scale, y1 + 2 / scale);
            ctx.lineTo(x2 - 2 / scale, y1 + m + 2 / scale);
            ctx.moveTo(x2 - 2 / scale, y1 + 2 / scale);
            ctx.lineTo(x2 - m - 2 / scale, y1 + m + 2 / scale);
            ctx.stroke();
          }
        } else {
          const bodyFill = selected ? 0.18 * fillT : 0;
          if (bodyFill > 0.01) {
            ctx.fillStyle = withAlpha(color, bodyFill);
            ctx.fillRect(x1, y1, bw, bh);
          }
          ctx.strokeStyle = color;
          const strokeBoost = selected ? 1 : focused ? hoverT : 0;
          ctx.lineWidth = (1.6 + 1.1 * strokeBoost) / scale;
          ctx.strokeRect(x1, y1, bw, bh);
        }
        if (selected && editable) {
          for (const [hx, hy] of [
            [x1, y1],
            [(x1 + x2) / 2, y1],
            [x2, y1],
            [x2, (y1 + y2) / 2],
            [x2, y2],
            [(x1 + x2) / 2, y2],
            [x1, y2],
            [x1, (y1 + y2) / 2],
          ]) {
            drawHandle(ctx, hx, hy, color, scale);
          }
        }
        if (showLabelsRef.current || isWrong) {
          const name = classNames[box.cls] ?? `cls ${box.cls}`;
          const roleTag = isWrongClass ? "WRONG · " : isFp ? "EXTRA · " : "";
          const label = isWrong && !showLabelsRef.current
            ? isWrongClass
              ? "WRONG"
              : "EXTRA"
            : box.conf != null
              ? `${roleTag}${name} ${(box.conf * 100).toFixed(0)}%`
              : `${roleTag}${name}`;
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
          if (isWrong) {
            const phase = alertPhaseRef.current;
            const pulse = 0.55 + 0.45 * (0.5 + 0.5 * Math.sin(phase * (isFp ? 0.0085 : 0.0065)));
            ctx.shadowColor = isWrongClass ? "rgba(234, 88, 12, 0.55)" : "rgba(225, 29, 72, 0.55)";
            ctx.shadowBlur = (4 + 5 * pulse) / scale;
          }
          ctx.beginPath();
          if (typeof ctx.roundRect === "function") {
            ctx.roundRect(labelX, labelY, labelW, labelH, 3 / scale);
          } else {
            ctx.rect(labelX, labelY, labelW, labelH);
          }
          ctx.fillStyle = isWrong
            ? isWrongClass
              ? "rgba(234, 88, 12, 0.94)"
              : "rgba(225, 29, 72, 0.94)"
            : withAlpha(color, 0.62);
          ctx.fill();
          ctx.shadowBlur = 0;
          ctx.lineWidth = 1 / scale;
          ctx.strokeStyle = isWrong ? "rgba(255,255,255,0.85)" : withAlpha(color, 0.9);
          ctx.stroke();
          ctx.fillStyle = "#fff";
          ctx.fillText(label, labelX + padX, labelY + fontPx + 1 / scale);
        }
        ctx.restore();
      });
      ctx.restore();
    }
    ctx.restore(); // sceneZoom
    ctx.restore();
  }, [boxes, canvasBg, classNames, draftBox, editable, height, offset, scale, selectedIndex, width]);

  const scheduleDraw = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(draw);
  }, [draw]);
  const drawRef = useRef(draw);
  drawRef.current = draw;

  useEffect(() => {
    scheduleDraw();
  }, [showLabels, scheduleDraw]);

  const kickAnim = useCallback(() => {
    const step = (now: number) => {
      const last = animTsRef.current || now;
      animTsRef.current = now;
      const dt = Math.min(48, Math.max(0, now - last));
      const kHover = 1 - Math.exp(-dt / 80);
      const kScene = 1 - Math.exp(-dt / 85);
      const kUpgrade = 1 - Math.exp(-dt / 145);
      const dragging = dragRef.current;
      const targetFocus =
        hoverIdxRef.current != null && dragging?.kind !== "pan" && dragging?.kind !== "resize" ? 1 : 0;
      const targetFill = selectedIndexRef.current != null ? 1 : 0;
      focusTRef.current += (targetFocus - focusTRef.current) * kHover;
      fillTRef.current += (targetFill - fillTRef.current) * kHover;
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
      if (!focusDone || !fillDone || !boxesDone || !imageDone || hasAlert) {
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

  useLayoutEffect(() => {
    let cancelled = false;
    // Scene identity is block/item — not fullUrl. Collect remaps list index and used to
    // cancel in-flight PNG upgrades whenever the index-bearing URL string changed.
    const sceneKey = sceneId || fullUrl;
    const sceneChanged = urlRef.current !== sceneKey;
    const keepFull = !sceneChanged && fullReadyRef.current && !!imgRef.current;
    if (sceneChanged && imgRef.current) {
      prevImgRef.current = imgRef.current;
      imageFadeTRef.current = 0;
      boxesFadeTRef.current = 0;
      qualityUpgradeRef.current = false;
    }
    urlRef.current = sceneKey;
    if (sceneChanged) fullReadyRef.current = false;
    let previewReady = keepFull;

    const accept = (src: ImageSource, tier: "thumb" | "preview" | "full") => {
      if (cancelled) return;
      if (tier === "full") {
        if (!fullReadyRef.current && imgRef.current && imgRef.current !== src) {
          prevImgRef.current = imgRef.current;
          imageFadeTRef.current = 0;
          qualityUpgradeRef.current = true;
        }
        fullReadyRef.current = true;
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

    const fullHit = getDecodedImage(fullUrl);
    const previewHit = previewUrl ? getDecodedImage(previewUrl) : null;
    const thumbHit = thumbUrl ? getDecodedImage(thumbUrl) : null;
    const thumbSnap = !thumbHit && thumbUrl ? snapshotThumbnailImage(thumbUrl) : null;
    if (thumbSnap && thumbUrl) adoptDecodedImage(thumbUrl, thumbSnap);

    const initial = fullHit ?? previewHit ?? thumbHit ?? thumbSnap ?? null;
    if (initial) {
      imgRef.current = initial;
      fullReadyRef.current = !!fullHit || keepFull;
      previewReady = !!previewHit || !!fullHit || keepFull;
      setHasImage(true);
      drawRef.current();
      kickAnim();
    } else if (keepFull) {
      setHasImage(true);
    } else if (sceneChanged) {
      // Keep previous frame until the new scene's image arrives.
    } else if (!imgRef.current) {
      fullReadyRef.current = false;
      setHasImage(false);
    }

    if (thumbUrl) {
      const blob = getThumbnailBlobUrl(thumbUrl);
      if (blob && !getDecodedImage(thumbUrl)) {
        void decodeBlobUrl(thumbUrl, blob).then((src) => src && accept(src, "thumb"));
      } else if (!initial && !keepFull) {
        loadImageImmediate(thumbUrl);
      }
    }
    if (previewUrl) loadImageImmediate(previewUrl);
    loadImageImmediate(fullUrl);

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
      unsubs.push(subscribeImage(previewUrl, (src) => accept(src, "preview")));
    }
    unsubs.push(subscribeImage(fullUrl, (src) => accept(src, "full")));
    return () => {
      cancelled = true;
      unsubs.forEach((u) => u());
    };
  }, [fullUrl, previewUrl, thumbUrl, sceneId, kickAnim]);

  useEffect(() => {
    if (sceneRef.current === sceneId) return;
    sceneRef.current = sceneId;
    userZoomedRef.current = false;
    boxesFadeTRef.current = 0;
    hoverIdxRef.current = null;
    lastHoverIdxRef.current = null;
    focusTRef.current = 0;
    kickAnim();
  }, [sceneId, kickAnim]);

  useLayoutEffect(() => {
    // Keep user zoom/pan while labeling — only auto-fit on new scene or before first zoom.
    if (!userZoomedRef.current) fitView();
  }, [width, height, fitView]);

  useEffect(() => {
    if (!syncTransform) return;
    // Never interrupt an in-progress local wheel zoom (labeling re-renders used to cancel it).
    if (localZoomingRef.current || zoomRafRef.current) return;
    const same =
      Math.abs(syncTransform.scale - scaleRef.current) < 1e-4 &&
      Math.abs(syncTransform.offsetX - offsetRef.current.x) < 1e-4 &&
      Math.abs(syncTransform.offsetY - offsetRef.current.y) < 1e-4;
    if (same) return;
    targetScaleRef.current = syncTransform.scale;
    applyView(syncTransform.scale, syncTransform.offsetX, syncTransform.offsetY, false);
  }, [syncTransform, applyView]);

  useEffect(() => {
    if (hasImage) scheduleDraw();
    kickAnim();
  }, [hasImage, scheduleDraw, kickAnim, boxes, draftBox, offset, scale, selectedIndex, boxCompare, ghostBoxes]);

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
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      userZoomedRef.current = true;
      localZoomingRef.current = true;
      const rect = el.getBoundingClientRect();
      zoomAnchorRef.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
      // Prefer vertical delta; fall back to horizontal trackpad gestures.
      const primary = Math.abs(e.deltaY) >= Math.abs(e.deltaX) ? e.deltaY : e.deltaX;
      let steps =
        e.deltaMode === 1 ? -primary : e.deltaMode === 2 ? -primary * 3 : -primary / 100;
      if (steps === 0 && (e.deltaY !== 0 || e.deltaX !== 0)) {
        steps = (e.deltaY || e.deltaX) < 0 ? 0.2 : -0.2;
      }
      if (!steps) return;
      const nextTarget = Math.min(Math.max(targetScaleRef.current * Math.pow(1.22, steps), 0.12), 16);
      targetScaleRef.current = nextTarget;
      if (!zoomRafRef.current) {
        zoomRafRef.current = requestAnimationFrame(animateZoomStep);
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      el.removeEventListener("wheel", onWheel);
      // Do not cancel an in-flight zoom here — effect deps can churn while labeling.
    };
  }, [animateZoomStep]);

  const onMouseDown = (e: React.MouseEvent) => {
    if (e.button === 2) {
      if (!editable) return;
      const pt = toImage(e.clientX, e.clientY);
      const hits = boxes
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
      if (hits.length) {
        const next = boxes.filter((_, i) => i !== hits[0].i);
        onBoxesChange?.(next, null, true);
        onSelect?.(null);
      }
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
      dragRef.current = {
        kind: "draw",
        startX: pt.x,
        startY: pt.y,
        baseBoxes: boxes.map((b) => ({ ...b })),
      };
      kickAnim();
    } else {
      dragRef.current = { kind: "pan", startX: e.clientX, startY: e.clientY };
    }
  };

  const onMouseMove = (e: React.MouseEvent) => {
    const drag = dragRef.current;
    if (!drag) {
      const pt = toImage(e.clientX, e.clientY);
      if (editable && !spaceRef.current) {
        const pt2 = pt;
        const hit = hitTestResize(pt2.x, pt2.y, boxes, scaleRef.current, selectedIndex ?? null);
        const inside = hitTestInside(pt2.x, pt2.y, boxes);
        if (hit) setCursor(cursorForMode(hit[1], classCursorRef.current));
        else setCursor(cursorForMode("draw", classCursorRef.current));
        const nextHover = hit ? hit[0] : inside;
        if (nextHover !== hoverIdxRef.current) {
          hoverIdxRef.current = nextHover;
          kickAnim();
        }
      } else {
        setCursor(spaceRef.current ? "grab" : editable ? classCursorRef.current : "grab");
        const nextHover = spaceRef.current ? null : hitTestInside(pt.x, pt.y, boxes);
        if (nextHover !== hoverIdxRef.current) {
          hoverIdxRef.current = nextHover;
          kickAnim();
        }
      }
      return;
    }
    if (drag.kind === "pan") {
      const dx = e.clientX - drag.startX;
      const dy = e.clientY - drag.startY;
      dragRef.current = { ...drag, startX: e.clientX, startY: e.clientY };
      const o = offsetRef.current;
      const next = { x: o.x + dx, y: o.y + dy };
      targetScaleRef.current = scaleRef.current;
      applyView(scaleRef.current, next.x, next.y);
      return;
    }
    const pt = toImage(e.clientX, e.clientY);
    if (drag.kind === "resize" && drag.origin && drag.idx != null && drag.mode) {
      const updated = resizedBox(drag.origin, drag.mode, pt.x, pt.y, width, height);
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
      const box: DetectionBox = {
        cls: currentClass,
        x1: Math.min(drag.startX, pt.x),
        y1: Math.min(drag.startY, pt.y),
        x2: Math.max(drag.startX, pt.x),
        y2: Math.max(drag.startY, pt.y),
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
    if (!drag || !editable) return;
    if (drag.kind === "draw") {
      const pt = toImage(e.clientX, e.clientY);
      const w = Math.abs(pt.x - drag.startX);
      const h = Math.abs(pt.y - drag.startY);
      const base = drag.baseBoxes ?? boxes;
      if (w >= MIN_DRAW && h >= MIN_DRAW) {
        const box: DetectionBox = {
          cls: currentClass,
          x1: Math.min(drag.startX, pt.x),
          y1: Math.min(drag.startY, pt.y),
          x2: Math.max(drag.startX, pt.x),
          y2: Math.max(drag.startY, pt.y),
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
      const updated = resizedBox(drag.origin, drag.mode, pt.x, pt.y, width, height);
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
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKey);
    };
  }, [classNames.length, editable, fitView]);

  useEffect(() => {
    return () => {
      cancelAnimationFrame(animRafRef.current);
      cancelAnimationFrame(rafRef.current);
    };
  }, []);

  const thumbBlob = thumbUrl ? getThumbnailBlobUrl(thumbUrl) : null;

  return (
    <div
      ref={containerRef}
      className="relative min-h-[280px] w-full flex-1 overflow-hidden rounded-xl bg-slate-100 dark:bg-ink-950"
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
          kickAnim();
          onMouseUp(e);
        }}
        onContextMenu={(e) => e.preventDefault()}
      />
    </div>
  );
}
