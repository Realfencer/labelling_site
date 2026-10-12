/** Progressive decoded-image cache: thumb → preview → full source. */

import { withTabSessionParam } from "../sessionId";

export type ImageSource = HTMLImageElement | ImageBitmap;

const decoded = new Map<string, ImageSource>();
const inflight = new Map<string, Promise<ImageSource>>();
const queue: { url: string; priority: number }[] = [];
let active = 0;
let immediateActive = 0;
const MAX_CONCURRENT = 6;
const MAX_CACHE = 160;

const listeners = new Map<string, Set<(src: ImageSource) => void>>();

let lastStaleSignal = 0;
export function signalStaleIndex(status: number) {
  const now = Date.now();
  if ((status === 409 || status === 404) && now - lastStaleSignal > 2500) {
    lastStaleSignal = now;
    window.dispatchEvent(new CustomEvent("labeler-stale-index", { detail: { status } }));
  }
}

function trimCache() {
  while (decoded.size > MAX_CACHE) {
    const oldest = decoded.keys().next().value;
    if (oldest == null) break;
    const entry = decoded.get(oldest);
    if (entry instanceof ImageBitmap) {
      try {
        entry.close();
      } catch {
        /* already closed */
      }
    }
    decoded.delete(oldest);
  }
}

function notify(url: string, src: ImageSource) {
  listeners.get(url)?.forEach((fn) => fn(src));
}

export function getDecodedImage(url: string): ImageSource | null {
  const hit = decoded.get(url);
  if (!hit) return null;
  decoded.delete(url);
  decoded.set(url, hit);
  return hit;
}

/** Drop a decoded entry (e.g. closed ImageBitmap) so the next load re-fetches. */
export function forgetDecodedImage(src: ImageSource | null | undefined) {
  if (!src) return;
  for (const [url, entry] of decoded) {
    if (entry === src) {
      decoded.delete(url);
      inflight.delete(url);
      if (entry instanceof ImageBitmap) {
        try {
          entry.close();
        } catch {
          /* already closed */
        }
      }
      return;
    }
  }
}

/** Put an already-decoded source into the cache (filmstrip blob snapshot). */
export function adoptDecodedImage(url: string, src: ImageSource) {
  if (!url || !src || decoded.has(url)) return;
  decoded.set(url, src);
  trimCache();
  notify(url, src);
}

export function subscribeImage(url: string, fn: (src: ImageSource) => void): () => void {
  if (!url) return () => undefined;
  const existing = decoded.get(url);
  if (existing) {
    fn(existing);
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

async function blobToImage(blob: Blob): Promise<ImageSource> {
  if (typeof createImageBitmap === "function") {
    return createImageBitmap(blob);
  }
  const img = new Image();
  img.decoding = "async";
  const objectUrl = URL.createObjectURL(blob);
  try {
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error("Image decode failed"));
      img.src = objectUrl;
    });
    try {
      await img.decode();
    } catch {
      /* decode optional */
    }
    return img;
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

function isAbort(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

const aborters = new Map<string, AbortController>();

/** Stop downloading an image nobody is waiting for any more (the user moved on): frees the
 *  connection and the bandwidth for the image on screen. No-op once it is decoded. */
export function abortImageLoad(url: string) {
  if (!url || decoded.has(url)) return;
  const i = queue.findIndex((j) => j.url === url);
  if (i >= 0) queue.splice(i, 1);
  aborters.get(url)?.abort();
}

export async function decodeUrl(url: string, fetchPriority: "high" | "low" | "auto" = "auto"): Promise<ImageSource> {
  const hit = decoded.get(url);
  if (hit) return hit;

  let pending = inflight.get(url);
  if (pending) return pending;

  const ac = new AbortController();
  aborters.set(url, ac);
  pending = (async () => {
    // Strip thumbs + versioned media URLs are content-addressed via `v=` / immutable
    // Cache-Control. Prefer HTTP cache (Ultralytics-style instant revisit). Keep
    // no-cache only for unversioned / Own paths that can replace pixels in place.
    const immutable =
      url.includes("strip=1") ||
      /[?&]v=/.test(url) ||
      url.includes("/api/media/");
    const init = {
      cache: immutable ? "force-cache" : "no-cache",
      credentials: "include",
      priority: fetchPriority,
      signal: ac.signal,
    } as RequestInit;
    try {
      let res = await fetch(url, init);
      if (!res.ok) {
        // Ask the server once more, bypassing the browser cache: with "force-cache" a stored error answer
        // (e.g. a 404 from just after a server restart) would otherwise be served again and again.
        const retry = await fetch(url, { ...init, cache: "reload" } as RequestInit);
        if (!retry.ok) {
          if (retry.status === 404 || retry.status === 409) signalStaleIndex(retry.status);
          throw new Error(`Image fetch failed: ${retry.status}`);
        }
        res = retry;
      }
      const blob = await res.blob();
      const src = await blobToImage(blob);
      decoded.set(url, src);
      trimCache();
      notify(url, src);
      return src;
    } catch (err) {
      if (isAbort(err)) throw err;
      throw err;
    }
  })();

  inflight.set(url, pending);
  try {
    return await pending;
  } finally {
    inflight.delete(url);
    aborters.delete(url);
  }
}

export async function decodeBlobUrl(cacheKey: string, blobUrl: string): Promise<ImageSource | null> {
  const hit = decoded.get(cacheKey);
  if (hit) return hit;
  try {
    const res = await fetch(blobUrl);
    if (!res.ok) return null;
    const src = await blobToImage(await res.blob());
    decoded.set(cacheKey, src);
    trimCache();
    notify(cacheKey, src);
    return src;
  } catch {
    return null;
  }
}

function pumpQueue() {
  const cap = Math.max(1, MAX_CONCURRENT - immediateActive);
  while (active < cap && queue.length > 0) {
    queue.sort((a, b) => b.priority - a.priority);
    const job = queue.shift();
    if (!job || decoded.has(job.url) || inflight.has(job.url)) continue;
    active += 1;
    decodeUrl(job.url, job.priority >= 50 ? "auto" : "low")
      .catch(() => undefined)
      .finally(() => {
        active -= 1;
        pumpQueue();
      });
  }
}

/** Visible-pane loads skip the background queue so they never wait behind neighbors. */
export function loadImageImmediate(url: string) {
  if (!url || decoded.has(url) || inflight.has(url)) return;
  const idx = queue.findIndex((j) => j.url === url);
  if (idx >= 0) queue.splice(idx, 1);
  immediateActive += 1;
  decodeUrl(url, "high")
    .catch(() => undefined)
    .finally(() => {
      immediateActive = Math.max(0, immediateActive - 1);
      pumpQueue();
    });
}

/** Drop queued full-image work for other blocks when the pane changes. */
export function preferImageUrls(urls: string[]) {
  const keep = new Set(urls.filter(Boolean));
  for (let i = queue.length - 1; i >= 0; i -= 1) {
    const url = queue[i].url;
    if (url.includes("/image?") && !keep.has(url)) queue.splice(i, 1);
  }
}

export function preloadImage(url: string, priority = 10) {
  if (!url || decoded.has(url) || inflight.has(url)) return;
  if (priority >= 90) {
    loadImageImmediate(url);
    return;
  }
  const existing = queue.find((j) => j.url === url);
  if (existing) {
    existing.priority = Math.max(existing.priority, priority);
    return;
  }
  queue.push({ url, priority });
  pumpQueue();
}

export function preloadImageUrgent(url: string) {
  loadImageImmediate(url);
}

export function clearDecodedImages() {
  for (const entry of decoded.values()) {
    if (typeof ImageBitmap !== "undefined" && entry instanceof ImageBitmap) {
      try {
        entry.close();
      } catch {
        /* already closed */
      }
    }
  }
  decoded.clear();
  inflight.clear();
  queue.length = 0;
  immediateActive = 0;
  active = 0;
}

function mediaCacheV(
  block: number,
  cacheToken?: string,
  itemKey?: string,
  mediaRev?: number | string,
): string {
  const key = itemKey || "";
  const rev = mediaRev != null && mediaRev !== "" ? String(mediaRev) : "0";
  return encodeURIComponent(
    cacheToken ? `${cacheToken}:${block}:${key}:${rev}` : `block:${block}:${key}:${rev}`,
  );
}

/** Block-stable media URLs — filmstrip re-index during collect must not change the path.
 *  itemKey + mediaRev bust Own synthetic block 1..N collisions and file replaces.
 */
export function challengeImageUrl(
  _index: number,
  mode: string,
  block?: number,
  cacheToken?: string,
  itemKey?: string,
  mediaRev?: number | string,
) {
  if (block == null) return "";
  const v = mediaCacheV(block, cacheToken, itemKey, mediaRev);
  const stem = itemKey ? `&stem=${encodeURIComponent(itemKey)}` : "";
  return withTabSessionParam(`/api/media/image?mode=${mode}&block=${block}${stem}&v=${v}`);
}

export function challengePreviewUrl(
  _index: number,
  mode: string,
  block?: number,
  cacheToken?: string,
  itemKey?: string,
  mediaRev?: number | string,
) {
  if (block == null) return "";
  const v = mediaCacheV(block, cacheToken, itemKey, mediaRev);
  const stem = itemKey ? `&stem=${encodeURIComponent(itemKey)}` : "";
  return withTabSessionParam(`/api/media/preview?mode=${mode}&block=${block}${stem}&v=${v}`);
}

export function challengeThumbnailUrl(
  _index: number,
  mode: string,
  block?: number,
  strip = false,
  cacheToken?: string,
  itemKey?: string,
  mediaRev?: number | string,
) {
  if (block == null) return "";
  const v = mediaCacheV(block, cacheToken, itemKey, mediaRev);
  const stem = itemKey ? `&stem=${encodeURIComponent(itemKey)}` : "";
  const base = `/api/media/thumbnail?mode=${mode}&block=${block}${stem}&v=${v}`;
  return withTabSessionParam(strip ? `${base}&strip=1` : base);
}

export function preloadChallenge(
  index: number,
  mode: string,
  block?: number,
  priority = 30,
  cacheToken?: string,
  itemKey?: string,
  mediaRev?: number | string,
) {
  preloadImage(challengeThumbnailUrl(index, mode, block, true, cacheToken, itemKey, mediaRev), priority + 12);
  // Full image only at lower priority — thumb paints first.
  preloadImage(challengeImageUrl(index, mode, block, cacheToken, itemKey, mediaRev), Math.max(1, priority - 8));
}

export function preloadChallengeUrgent(
  index: number,
  mode: string,
  block?: number,
  cacheToken?: string,
  itemKey?: string,
  mediaRev?: number | string,
  opts?: { full?: boolean },
) {
  const thumb = challengeThumbnailUrl(index, mode, block, true, cacheToken, itemKey, mediaRev);
  preferImageUrls([thumb]);
  loadImageImmediate(thumb);
  // Default: thumb only (hover). Selected open passes { full: true }.
  if (opts?.full) {
    const full = challengeImageUrl(index, mode, block, cacheToken, itemKey, mediaRev);
    preferImageUrls([full, thumb]);
    loadImageImmediate(full);
  }
}

/** Preload the next few images while the user labels the current one. */
export function preloadChallengeForward(
  fromIndex: number,
  mode: string,
  challenges: ReadonlyArray<{ loaded?: boolean; block?: number; item_key?: string; media_rev?: number }>,
  count = 4,
  cacheToken?: string,
  nextIndices?: number[],
) {
  const targets = nextIndices?.length
    ? nextIndices
    : Array.from({ length: count }, (_, k) => fromIndex + k + 1);
  targets.forEach((i, d) => {
    if (i < 0 || i >= challenges.length) return;
    const ch = challenges[i];
    if (!ch?.loaded || ch.block == null) return;
    const pri = 52 - (d + 1) * 8;
    preloadImage(
      challengeThumbnailUrl(i, mode, ch.block, true, cacheToken, ch.item_key, ch.media_rev),
      pri + 10,
    );
    // The small sharp preview (~70 KB) for EVERY neighbour: moving to any of them (forward, back, two ahead) paints sharp
    // at once from the decoded cache instead of showing the blurry thumbnail until a network round trip finishes.
    preloadImage(challengePreviewUrl(i, mode, ch.block, cacheToken, ch.item_key, ch.media_rev), pri + 6);
    // Full image for the three most likely next stops (+1, -1, +2).
    if (d <= 2) {
      preloadImage(challengeImageUrl(i, mode, ch.block, cacheToken, ch.item_key, ch.media_rev), Math.max(1, pri - 2));
    }
  });
}
