import { api, HttpError } from "../api";
import { tabSessionHeaders } from "../sessionId";
import type { DetectionBox } from "../types";
import { useLabeler } from "./useLabeler";
import { useTeam } from "./useTeam";

/**
 * Durable save queue. A save that fails (network drop, 5xx, timeout) used to be swallowed and the
 * boxes were lost once you navigated away. Failed saves now stay here — latest boxes per image —
 * and are retried with backoff until the server accepts them, or refuses for a real reason:
 * 423 (a teammate owns the image / you are watching) or 409 (a teammate changed it meanwhile).
 */

type Pending = {
  itemKey: string;
  index: number;
  boxes: DetectionBox[];
  reviewed: boolean;
  attempt: number;
  timer: number | null;
};

const pending = new Map<string, Pending>();
let conflictHandler: ((itemKey: string, message: string) => void) | null = null;

export function setSaveConflictHandler(fn: ((itemKey: string, message: string) => void) | null): void {
  conflictHandler = fn;
}

export function hasPendingSaves(): boolean {
  return pending.size > 0;
}

function currentIndexFor(p: Pending): number {
  const app = useLabeler.getState().appState;
  const i = app?.challenges.findIndex((c) => c.item_key === p.itemKey) ?? -1;
  return i >= 0 ? i : p.index;
}

function delayFor(attempt: number): number {
  return Math.min(10000, 800 * 2 ** Math.min(attempt, 4));
}

async function attempt(p: Pending): Promise<void> {
  p.timer = null;
  const team = useTeam.getState();
  try {
    const res = await api.saveBoxes(currentIndexFor(p), p.boxes, p.reviewed, p.itemKey, team.revs[p.itemKey]);
    // A newer edit may have replaced this entry while the request was in flight.
    if (pending.get(p.itemKey) === p) pending.delete(p.itemKey);
    useTeam.getState().noteRev(p.itemKey, res.rev);
    useTeam.getState().noteLabel(p.itemKey, res.labeled_by, res.rev);
    if (pending.size === 0) useTeam.getState().setSave("saved");
  } catch (err) {
    if (pending.get(p.itemKey) !== p) return; // superseded
    const status = err instanceof HttpError ? err.status : 0;
    const message = err instanceof Error ? err.message : String(err);
    if (status === 423) {
      pending.delete(p.itemKey);
      useTeam.getState().setSave("locked", message);
      return;
    }
    if (status === 409) {
      pending.delete(p.itemKey);
      useTeam.getState().setSave("conflict", message);
      conflictHandler?.(p.itemKey, message);
      return;
    }
    if (status >= 400 && status < 500 && status !== 408 && status !== 429) {
      pending.delete(p.itemKey);
      useTeam.getState().setSave("error", message);
      return;
    }
    p.attempt += 1;
    useTeam.getState().setSave("retrying", message);
    p.timer = window.setTimeout(() => void attempt(p), delayFor(p.attempt));
  }
}

/** Queue (or replace) the retry for an image whose save just failed. */
export function retrySave(itemKey: string, index: number, boxes: DetectionBox[], reviewed: boolean): void {
  if (!itemKey) return;
  const prev = pending.get(itemKey);
  if (prev?.timer != null) window.clearTimeout(prev.timer);
  const p: Pending = { itemKey, index, boxes, reviewed, attempt: prev ? prev.attempt : 0, timer: null };
  pending.set(itemKey, p);
  useTeam.getState().setSave("retrying", "Not saved yet — retrying");
  p.timer = window.setTimeout(() => void attempt(p), delayFor(p.attempt));
}

/** The user edited this image again: a queued older version must not overwrite it later. */
export function dropPendingSave(itemKey: string | null | undefined): void {
  if (!itemKey) return;
  const p = pending.get(itemKey);
  if (p?.timer != null) window.clearTimeout(p.timer);
  pending.delete(itemKey);
}

/** Tab hiding/closing: push everything unsaved with keepalive so the browser finishes it. */
export function flushOnUnload(extra?: { itemKey: string; index: number; boxes: DetectionBox[]; reviewed: boolean }): void {
  const jobs: Array<{ itemKey: string; index: number; boxes: DetectionBox[]; reviewed: boolean }> = [];
  for (const p of pending.values()) jobs.push({ itemKey: p.itemKey, index: currentIndexFor(p), boxes: p.boxes, reviewed: p.reviewed });
  if (extra && !jobs.some((j) => j.itemKey === extra.itemKey)) jobs.push(extra);
  const revs = useTeam.getState().revs;
  for (const j of jobs) {
    try {
      void fetch(`/api/challenges/${j.index}/boxes`, {
        method: "PUT",
        keepalive: true,
        credentials: "include",
        headers: tabSessionHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({
          boxes: j.boxes,
          reviewed: j.reviewed,
          item_key: j.itemKey,
          ...(revs[j.itemKey] != null ? { base_rev: revs[j.itemKey] } : {}),
        }),
      });
    } catch {
      /* best effort */
    }
  }
}
