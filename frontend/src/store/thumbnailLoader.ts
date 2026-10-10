/** Fast filmstrip thumbnails — batch fetch + blob cache. */

import { adoptDecodedImage, challengeThumbnailUrl, decodeBlobUrl } from "./imageLoader";
import { tabSessionHeaders } from "../sessionId";

const blobCache = new Map<string, string>();
const listeners = new Map<string, Set<(url: string) => void>>();
const batchInflight = new Map<string, Promise<void>>();
const queue: { url: string; priority: number }[] = [];
const inflight = new Map<string, Promise<string>>();
let active = 0;
const MAX_CONCURRENT = 8;
const MAX_CACHE = 400;
const BATCH_SIZE = 24;

function trimCache() {
  while (blobCache.size > MAX_CACHE) {
    const oldest = blobCache.keys().next().value;
    if (oldest == null) break;
    const blobUrl = blobCache.get(oldest);
    if (blobUrl) URL.revokeObjectURL(blobUrl);
    blobCache.delete(oldest);
  }
}

function notify(url: string, blobUrl: string) {
  listeners.get(url)?.forEach((fn) => fn(blobUrl));
}

function storeBlob(url: string, bytes: Uint8Array) {
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  const blob = new Blob([copy], { type: "image/jpeg" });
  const blobUrl = URL.createObjectURL(blob);
  blobCache.set(url, blobUrl);
  trimCache();
  notify(url, blobUrl);
  return blobUrl;
}

export function getThumbnailBlobUrl(url: string): string | null {
  return blobCache.get(url) ?? null;
}

/** Instant canvas source from the filmstrip blob — no extra network. */
export function snapshotThumbnailImage(url: string): HTMLImageElement | null {
  const blobUrl = blobCache.get(url);
  if (!blobUrl) return null;
  const img = new Image();
  img.decoding = "sync";
  img.src = blobUrl;
  if (img.complete && img.naturalWidth > 0 && img.naturalHeight > 0) return img;
  return null;
}

export function primeDecodedThumbnail(url: string) {
  if (!url) return;
  const snap = snapshotThumbnailImage(url);
  if (snap) {
    adoptDecodedImage(url, snap);
    return;
  }
  const blobUrl = blobCache.get(url);
  if (blobUrl) void decodeBlobUrl(url, blobUrl);
}

export function subscribeThumbnail(url: string, fn: (blobUrl: string) => void): () => void {
  const hit = blobCache.get(url);
  if (hit) {
    fn(hit);
    return () => undefined;
  }
  let set = listeners.get(url);
  if (!set) {
    set = new Set();
    listeners.set(url, set);
  }
  set.add(fn);
  return () => {
    set!.delete(fn);
    if (set!.size === 0) listeners.delete(url);
  };
}

function decodeBase64(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) out[i] = bin.charCodeAt(i);
  return out;
}

export async function batchLoadStripThumbnails(
  indices: number[],
  mode: string,
  blocks: ReadonlyMap<number, number>,
  cacheToken?: string,
  meta?: ReadonlyMap<number, { item_key?: string; media_rev?: number }>,
): Promise<void> {
  const pending = indices.filter((index) => {
    const block = blocks.get(index);
    if (block == null) return false;
    const m = meta?.get(index);
    const url = challengeThumbnailUrl(index, mode, block, true, cacheToken, m?.item_key, m?.media_rev);
    return url && !blobCache.has(url);
  });
  if (!pending.length) return;

  const pendingBlocks = pending
    .map((index) => blocks.get(index))
    .filter((block): block is number => block != null);

  await batchLoadStripByBlocks(pendingBlocks, mode, blocks, cacheToken, meta, pending);
}

export async function batchLoadStripByBlocks(
  blocksWanted: number[],
  mode: string,
  indexByBlock: ReadonlyMap<number, number>,
  cacheToken?: string,
  meta?: ReadonlyMap<number, { item_key?: string; media_rev?: number }>,
  indicesHint?: number[],
): Promise<void> {
  const blockToIndex = new Map<number, number>();
  if (indicesHint?.length) {
    indicesHint.forEach((index) => {
      const block = indexByBlock.get(index);
      if (block != null) blockToIndex.set(block, index);
    });
  } else {
    indexByBlock.forEach((block, index) => {
      if (!blockToIndex.has(block)) blockToIndex.set(block, index);
    });
  }

  const pendingBlocks = blocksWanted.filter((block) => {
    const index = blockToIndex.get(block);
    if (index == null) return false;
    const m = meta?.get(index);
    const url = challengeThumbnailUrl(index, mode, block, true, cacheToken, m?.item_key, m?.media_rev);
    return url && !blobCache.has(url);
  });
  if (!pendingBlocks.length) return;

  const chunks: number[][] = [];
  for (let start = 0; start < pendingBlocks.length; start += BATCH_SIZE) {
    chunks.push(pendingBlocks.slice(start, start + BATCH_SIZE));
  }

  await Promise.all(
    chunks.map(async (chunk) => {
      const key = `blocks:${cacheToken || ""}:${chunk.join(",")}`;
      let job = batchInflight.get(key);
      if (!job) {
        job = (async () => {
          const res = await fetch("/api/media/thumbnails/batch", {
            method: "POST",
            credentials: "include",
            headers: tabSessionHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({ blocks: chunk, mode, strip: true }),
          });
          if (!res.ok) return;
          const data = (await res.json()) as { thumbnails?: Record<string, string> };
          const thumbs = data.thumbnails || {};
          for (const [blockStr, b64] of Object.entries(thumbs)) {
            const block = Number(blockStr);
            const index = blockToIndex.get(block);
            if (index == null || !b64) continue;
            const m = meta?.get(index);
            const url = challengeThumbnailUrl(index, mode, block, true, cacheToken, m?.item_key, m?.media_rev);
            if (!url || blobCache.has(url)) continue;
            storeBlob(url, decodeBase64(b64));
          }
        })().finally(() => batchInflight.delete(key));
        batchInflight.set(key, job);
      }
      await job;
    }),
  );
}

async function fetchThumbnail(url: string): Promise<string> {
  const hit = blobCache.get(url);
  if (hit) return hit;

  let pending = inflight.get(url);
  if (pending) return pending;

  pending = (async () => {
    let res = await fetch(url, {
      cache: url.includes("strip=1") || /[?&]v=/.test(url) ? "force-cache" : "no-cache",
      credentials: "include",
      priority: "high",
    } as RequestInit);
    if (!res.ok) {
      // Bypass the browser cache once: a stored error answer must not hide the real thumbnail.
      const retry = await fetch(url, { cache: "reload", credentials: "include", priority: "high" } as RequestInit);
      if (!retry.ok) throw new Error(`Thumbnail fetch failed: ${retry.status}`);
      res = retry;
    }
    const buf = new Uint8Array(await res.arrayBuffer());
    return storeBlob(url, buf);
  })();

  inflight.set(url, pending);
  try {
    return await pending;
  } finally {
    inflight.delete(url);
  }
}

function pumpQueue() {
  while (active < MAX_CONCURRENT && queue.length > 0) {
    queue.sort((a, b) => b.priority - a.priority);
    const job = queue.shift();
    if (!job || blobCache.has(job.url) || inflight.has(job.url)) continue;
    active += 1;
    fetchThumbnail(job.url)
      .catch(() => undefined)
      .finally(() => {
        active -= 1;
        pumpQueue();
      });
  }
}

export function preloadThumbnail(url: string, priority = 10) {
  if (!url || blobCache.has(url) || inflight.has(url)) return;
  const existing = queue.find((j) => j.url === url);
  if (existing) {
    existing.priority = Math.max(existing.priority, priority);
    return;
  }
  queue.push({ url, priority });
  pumpQueue();
}

export function preloadThumbnailBurst(urls: string[], basePriority = 80) {
  urls.forEach((url, i) => {
    if (url) preloadThumbnail(url, basePriority - Math.min(i, 40));
  });
}

export function preloadThumbnailWindow(urls: string[], centerPriority = 180) {
  const mid = Math.floor(urls.length / 2);
  urls.forEach((url, i) => {
    if (!url) return;
    preloadThumbnail(url, centerPriority - Math.abs(i - mid) * 4);
  });
}

let batchScrollTimer: number | null = null;

export function scheduleBatchStripLoad(
  indices: number[],
  mode: string,
  blocks: ReadonlyMap<number, number>,
  cacheToken?: string,
  meta?: ReadonlyMap<number, { item_key?: string; media_rev?: number }>,
) {
  if (batchScrollTimer != null) window.clearTimeout(batchScrollTimer);
  batchScrollTimer = window.setTimeout(() => {
    batchScrollTimer = null;
    void batchLoadStripThumbnails(indices, mode, blocks, cacheToken, meta);
  }, 4);
}

export function batchLoadAllStrip(
  indices: number[],
  mode: string,
  blocks: ReadonlyMap<number, number>,
  cacheToken?: string,
  meta?: ReadonlyMap<number, { item_key?: string; media_rev?: number }>,
) {
  void batchLoadStripThumbnails(indices, mode, blocks, cacheToken, meta);
}

export function clearThumbnailBlobCache() {
  for (const blobUrl of blobCache.values()) {
    URL.revokeObjectURL(blobUrl);
  }
  blobCache.clear();
  // Do NOT clear listeners — FilmstripThumb mounts/subscribes before parent
  // review_mode effects run; wiping listeners leaves thumbs dark forever while
  // batch/media fetches still succeed in the network tab.
  batchInflight.clear();
  inflight.clear();
  queue.length = 0;
}
