import type { DetectionBox, TopVsLeftScore } from "../types";

type ScoredBox = { cls: number; x1: number; y1: number; x2: number; y2: number; conf: number };

// TurboVision smart IoU (Aug 2026): required match IoU scales with GT box size.
const ADAPTIVE_IOU_ANCHORS: Array<[number, number]> = [
  [0.0005, 0.3],
  [0.01, 0.5],
  [0.05, 0.5],
  [0.25, 0.7],
];
const DEFAULT_SCORE_IMAGE_WIDTH = 960;
const DEFAULT_SCORE_IMAGE_HEIGHT = 540;

function area(b: ScoredBox): number {
  return Math.abs(b.x2 - b.x1) * Math.abs(b.y2 - b.y1);
}

function boxIou(a: ScoredBox, b: ScoredBox): number {
  const [ax1, ax2] = [a.x1, a.x2].sort((x, y) => x - y);
  const [ay1, ay2] = [a.y1, a.y2].sort((x, y) => x - y);
  const [bx1, bx2] = [b.x1, b.x2].sort((x, y) => x - y);
  const [by1, by2] = [b.y1, b.y2].sort((x, y) => x - y);
  const interW = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
  const interH = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
  const inter = interW * interH;
  if (inter <= 0) return 0;
  const union = area(a) + area(b) - inter;
  return union > 0 ? inter / union : 0;
}

function toScoredBox(b: DetectionBox): ScoredBox {
  return {
    cls: b.cls,
    x1: b.x1,
    y1: b.y1,
    x2: b.x2,
    y2: b.y2,
    conf: b.conf ?? 0,
  };
}

function gtAreaRatio(gt: ScoredBox, imageWidth: number, imageHeight: number): number {
  const imageArea = Math.max(1, imageWidth * imageHeight);
  const gtArea = Math.max(0, Math.abs(gt.x2 - gt.x1) * Math.abs(gt.y2 - gt.y1));
  return gtArea / imageArea;
}

function adaptiveIouThresholdForGt(gt: ScoredBox, imageWidth: number, imageHeight: number): number {
  let ratio = gtAreaRatio(gt, imageWidth, imageHeight);
  const [rMin, _tMin] = ADAPTIVE_IOU_ANCHORS[0];
  const [_rMax, tMax] = ADAPTIVE_IOU_ANCHORS[ADAPTIVE_IOU_ANCHORS.length - 1];
  ratio = Math.max(rMin, Math.min(_rMax, ratio));
  for (let i = 0; i < ADAPTIVE_IOU_ANCHORS.length - 1; i += 1) {
    const [r0, t0] = ADAPTIVE_IOU_ANCHORS[i];
    const [r1, t1] = ADAPTIVE_IOU_ANCHORS[i + 1];
    if (ratio >= r0 && ratio <= r1) {
      if (t0 === t1) return t0;
      const frac = (Math.log10(ratio) - Math.log10(r0)) / (Math.log10(r1) - Math.log10(r0));
      return t0 + (t1 - t0) * frac;
    }
  }
  return tMax;
}

function resolveScoreImageSize(
  predBoxes: ScoredBox[],
  gtBoxes: ScoredBox[],
  imageWidth?: number,
  imageHeight?: number,
): { width: number; height: number } {
  if (imageWidth && imageHeight && imageWidth > 1 && imageHeight > 1) {
    return { width: imageWidth, height: imageHeight };
  }
  let maxX = 0;
  let maxY = 0;
  for (const b of [...predBoxes, ...gtBoxes]) {
    maxX = Math.max(maxX, b.x1, b.x2);
    maxY = Math.max(maxY, b.y1, b.y2);
  }
  if (maxX >= 256 && maxY >= 256) {
    return { width: Math.round(maxX), height: Math.round(maxY) };
  }
  return { width: DEFAULT_SCORE_IMAGE_WIDTH, height: DEFAULT_SCORE_IMAGE_HEIGHT };
}

function requiredIouForGtMatch(
  gt: ScoredBox,
  iouThreshold: number,
  useAdaptiveIou: boolean,
  imageWidth: number,
  imageHeight: number,
): number {
  if (useAdaptiveIou) return adaptiveIouThresholdForGt(gt, imageWidth, imageHeight);
  return iouThreshold;
}

function greedyMatch(
  predBoxes: ScoredBox[],
  gtBoxes: ScoredBox[],
  iouThreshold = 0.5,
  useAdaptiveIou = true,
  imageWidth?: number,
  imageHeight?: number,
): { tp: number; fp: number; fn: number } {
  const { width, height } = resolveScoreImageSize(predBoxes, gtBoxes, imageWidth, imageHeight);
  const byClsPred = new Map<number, ScoredBox[]>();
  const byClsGt = new Map<number, ScoredBox[]>();
  for (const b of predBoxes) {
    const list = byClsPred.get(b.cls) || [];
    list.push(b);
    byClsPred.set(b.cls, list);
  }
  for (const b of gtBoxes) {
    const list = byClsGt.get(b.cls) || [];
    list.push(b);
    byClsGt.set(b.cls, list);
  }
  const classIds = new Set([...byClsPred.keys(), ...byClsGt.keys()]);
  let tp = 0;
  let fp = 0;
  let fn = 0;
  for (const classId of classIds) {
    const preds = (byClsPred.get(classId) || []).sort((a, b) => b.conf - a.conf);
    const gts = [...(byClsGt.get(classId) || [])];
    const gtUsed = gts.map(() => false);
    for (const pred of preds) {
      let bestIou = 0;
      let bestGt = -1;
      gts.forEach((gt, gi) => {
        if (gtUsed[gi]) return;
        const iou = boxIou(pred, gt);
        if (iou > bestIou) {
          bestIou = iou;
          bestGt = gi;
        }
      });
      const required =
        bestGt >= 0
          ? requiredIouForGtMatch(gts[bestGt], iouThreshold, useAdaptiveIou, width, height)
          : iouThreshold;
      if (bestGt >= 0 && bestIou >= required) {
        gtUsed[bestGt] = true;
        tp += 1;
      } else {
        fp += 1;
      }
    }
    fn += gtUsed.filter((u) => !u).length;
  }
  return { tp, fp, fn };
}

function perClassTpFlags(
  predBoxes: ScoredBox[],
  gtBoxes: ScoredBox[],
  iouThreshold = 0.5,
  useAdaptiveIou = true,
  imageWidth?: number,
  imageHeight?: number,
): Map<number, { flags: boolean[]; nGt: number }> {
  const { width, height } = resolveScoreImageSize(predBoxes, gtBoxes, imageWidth, imageHeight);
  const byClsPred = new Map<number, ScoredBox[]>();
  const byClsGt = new Map<number, ScoredBox[]>();
  for (const b of predBoxes) {
    const list = byClsPred.get(b.cls) || [];
    list.push(b);
    byClsPred.set(b.cls, list);
  }
  for (const b of gtBoxes) {
    const list = byClsGt.get(b.cls) || [];
    list.push(b);
    byClsGt.set(b.cls, list);
  }
  const result = new Map<number, { flags: boolean[]; nGt: number }>();
  const classIds = new Set([...byClsPred.keys(), ...byClsGt.keys()]);
  for (const classId of classIds) {
    const preds = (byClsPred.get(classId) || []).sort((a, b) => b.conf - a.conf);
    const gts = [...(byClsGt.get(classId) || [])];
    const gtUsed = gts.map(() => false);
    const flags: boolean[] = [];
    for (const pred of preds) {
      let bestIou = 0;
      let bestGt = -1;
      gts.forEach((gt, gi) => {
        if (gtUsed[gi]) return;
        const iou = boxIou(pred, gt);
        if (iou > bestIou) {
          bestIou = iou;
          bestGt = gi;
        }
      });
      const required =
        bestGt >= 0
          ? requiredIouForGtMatch(gts[bestGt], iouThreshold, useAdaptiveIou, width, height)
          : iouThreshold;
      if (bestGt >= 0 && bestIou >= required) {
        gtUsed[bestGt] = true;
        flags.push(true);
      } else {
        flags.push(false);
      }
    }
    result.set(classId, { flags, nGt: gts.length });
  }
  return result;
}

function averagePrecision(flags: boolean[], nGt: number): number {
  // Desktop VOC-style area-under-PR (`_average_precision_from_matches`).
  if (nGt <= 0) return Number.NaN;
  if (!flags.length) return 0;
  let tpCum = 0;
  let fpCum = 0;
  const recalls: number[] = [];
  const precisions: number[] = [];
  for (const isTp of flags) {
    if (isTp) tpCum += 1;
    else fpCum += 1;
    recalls.push(tpCum / nGt);
    precisions.push(tpCum / Math.max(tpCum + fpCum, 1e-9));
  }
  const mrec = [0, ...recalls, 1];
  const mpre = [1, ...precisions, 0];
  for (let i = mpre.length - 1; i > 0; i -= 1) {
    mpre[i - 1] = Math.max(mpre[i - 1], mpre[i]);
  }
  let ap = 0;
  for (let i = 0; i < mrec.length - 1; i += 1) {
    if (mrec[i + 1] !== mrec[i]) {
      ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1];
    }
  }
  return ap;
}

function parseScore(value: unknown): number | null {
  if (value == null) return null;
  const n = Number(value);
  return Number.isNaN(n) ? null : n;
}

export type BoxCompareRole = "tp" | "fp" | "wrong_class";

/** Classify each prediction vs left-pane GT (same greedy/adaptive IoU as scoring). */
export function comparePredToGt(
  predBoxes: DetectionBox[],
  gtBoxes: DetectionBox[],
  imageWidth?: number,
  imageHeight?: number,
): { roles: BoxCompareRole[]; missedGt: DetectionBox[] } {
  const roles: BoxCompareRole[] = predBoxes.map(() => "fp");
  if (!gtBoxes.length) return { roles: predBoxes.map(() => "tp"), missedGt: [] };
  if (!predBoxes.length) return { roles, missedGt: [...gtBoxes] };

  const preds = predBoxes.map(toScoredBox);
  const gts = gtBoxes.map(toScoredBox);
  const { width, height } = resolveScoreImageSize(preds, gts, imageWidth, imageHeight);
  const gtUsed = gts.map(() => false);

  const byClsPredIdx = new Map<number, number[]>();
  predBoxes.forEach((b, i) => {
    const list = byClsPredIdx.get(b.cls) || [];
    list.push(i);
    byClsPredIdx.set(b.cls, list);
  });

  for (const [, indices] of byClsPredIdx) {
    const sorted = [...indices].sort((a, b) => (predBoxes[b].conf ?? 0) - (predBoxes[a].conf ?? 0));
    for (const pi of sorted) {
      const pred = preds[pi];
      const gtIndices = gts.map((_, i) => i).filter((i) => gts[i].cls === pred.cls);
      let bestIou = 0;
      let bestGi = -1;
      for (const gi of gtIndices) {
        if (gtUsed[gi]) continue;
        const iou = boxIou(pred, gts[gi]);
        if (iou > bestIou) {
          bestIou = iou;
          bestGi = gi;
        }
      }
      if (bestGi >= 0) {
        const required = requiredIouForGtMatch(gts[bestGi], 0.5, true, width, height);
        if (bestIou >= required) {
          gtUsed[bestGi] = true;
          roles[pi] = "tp";
          continue;
        }
      }
      let wrongIou = 0;
      for (let gi = 0; gi < gts.length; gi += 1) {
        if (gts[gi].cls === pred.cls) continue;
        const iou = boxIou(pred, gts[gi]);
        if (iou > wrongIou) wrongIou = iou;
      }
      roles[pi] = wrongIou >= 0.3 ? "wrong_class" : "fp";
    }
  }

  const missedGt = gtBoxes.filter((_, i) => !gtUsed[i]);
  return { roles, missedGt };
}

export function scoreTopAgainstLeftGt(
  predBoxes: DetectionBox[],
  gtBoxes: DetectionBox[],
  record: Record<string, unknown>,
  imageWidth?: number,
  imageHeight?: number,
): TopVsLeftScore | null {
  if (!predBoxes.length) return null;
  if (!gtBoxes.length) return null;

  const preds = predBoxes.map(toScoredBox);
  const gts = gtBoxes.map(toScoredBox);
  const { tp, fp, fn } = greedyMatch(preds, gts, 0.5, true, imageWidth, imageHeight);

  const perClass = perClassTpFlags(preds, gts, 0.5, true, imageWidth, imageHeight);
  const aps: number[] = [];
  perClass.forEach(({ flags, nGt }) => {
    if (nGt <= 0) return;
    const ap = averagePrecision(flags, nGt);
    if (!Number.isNaN(ap)) aps.push(ap);
  });

  const nGt = gtBoxes.length;
  const nPred = predBoxes.length;
  const map50 = aps.length ? aps.reduce((a, b) => a + b, 0) / aps.length : nGt > 0 ? 0 : 1;
  const ffpi = fp;
  const falsePositive = Math.max(0, 1 - ffpi / 10);
  const composite = 0.6 * map50 + 0.4 * falsePositive;

  const platformScore = parseScore(record.total_weighted);
  const platformMap50 = parseScore(record.map50);
  const platformFp = parseScore(record.false_positive);

  return {
    composite,
    map50,
    false_positive: falsePositive,
    ffpi,
    tp,
    fp,
    fn,
    n_gt: nGt,
    n_pred: nPred,
    platform_score: platformScore,
    platform_map50: platformMap50,
    platform_fp: platformFp,
    delta_vs_platform: platformScore != null ? composite - platformScore : null,
  };
}
