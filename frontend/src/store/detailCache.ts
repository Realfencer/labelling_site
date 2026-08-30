import type { ChallengeDetail } from "../types";

/** Keyed by challenge.item_key so collect re-sorts cannot serve the wrong left boxes. */
const cache = new Map<string, ChallengeDetail>();

function cacheKey(detailOrKey: ChallengeDetail | string, index?: number): string {
  if (typeof detailOrKey === "string") return detailOrKey;
  const itemKey = detailOrKey.challenge?.item_key;
  if (itemKey) return itemKey;
  return `idx:${detailOrKey.index ?? index ?? -1}`;
}

export function getCachedDetail(index: number, itemKey?: string | null): ChallengeDetail | undefined {
  if (!itemKey) return undefined;
  const hit = cache.get(itemKey);
  if (hit) return { ...hit, index };
  // Exact item_key only — never fuzzy-match Own stems (v01/v00 collisions).
  return undefined;
}

export function getCachedDetailByItemKey(itemKey: string | null | undefined): ChallengeDetail | undefined {
  if (!itemKey) return undefined;
  return cache.get(itemKey);
}

export function setCachedDetail(index: number, detail: ChallengeDetail) {
  const key = cacheKey(detail, index);
  cache.set(key, { ...detail, index });
  if (cache.size > 64) {
    const oldest = cache.keys().next().value;
    if (oldest != null) cache.delete(oldest);
  }
}

export function clearDetailCache() {
  cache.clear();
}

export function updateCachedDetailBoxes(
  itemKey: string,
  boxes: import("../types").DetectionBox[],
  reviewed: boolean,
) {
  const hit = cache.get(itemKey);
  if (hit) {
    const updated: ChallengeDetail = {
      ...hit,
      boxes,
      challenge: {
        ...hit.challenge,
        reviewed,
        cached_left_boxes: boxes.map((b) => ({
          cls: b.cls,
          x1: b.x1,
          y1: b.y1,
          x2: b.x2,
          y2: b.y2,
          conf: b.conf ?? null,
        })),
      },
    };
    cache.set(itemKey, updated);
  }
}

export function preloadDetails(
  indices: number[],
  loader: (index: number) => Promise<ChallengeDetail>,
  itemKeyForIndex?: (index: number) => string | undefined,
  onCached?: () => void,
) {
  for (const index of indices) {
    if (index < 0) continue;
    const itemKey = itemKeyForIndex?.(index);
    if (itemKey && cache.has(itemKey)) continue;
    if (!itemKey) continue;
    loader(index)
      .then((detail) => {
        setCachedDetail(index, detail);
        onCached?.();
      })
      .catch(() => undefined);
  }
}
