import { create } from "zustand";
import type { ReviewItemStatus, ReviewStatus, ReviewSummary } from "../api";

/** Live label-audit findings for the Own "Review" list (polled by useReviewSync while it is open). */
interface ReviewStore {
  version: number;
  items: Record<string, ReviewItemStatus | null>;
  summary: ReviewSummary | null;
  /** Signature of the server's list (it grows when an edit makes an old image risky again). */
  keysSig: string | null;
  /** "Looks right" was just pressed on this image: it moves on at ``until`` unless undone. */
  lookRight: { key: string; until: number } | null;
  apply: (r: ReviewStatus) => void;
  setLookRight: (v: { key: string; until: number } | null) => void;
  patchItem: (key: string, st: ReviewItemStatus | null, version?: number) => void;
  reset: () => void;
}

export const useReview = create<ReviewStore>((set) => ({
  version: -1,
  items: {},
  summary: null,
  keysSig: null,
  lookRight: null,
  apply: (r) =>
    set((s) => ({
      version: r.version,
      summary: r.summary,
      items: r.items ? r.items : s.items,
      keysSig: r.keys_sig ?? s.keysSig,
    })),
  setLookRight: (lookRight) => set({ lookRight }),
  patchItem: (key, st, version) =>
    set((s) => ({ items: { ...s.items, [key]: st }, version: version ?? s.version })),
  reset: () => set({ version: -1, items: {}, summary: null, keysSig: null, lookRight: null }),
}));

/** Still to look at: listed, not fixed, not marked "looks right". Unknown (not loaded yet) counts as open. */
export function reviewIsOpen(st: ReviewItemStatus | null | undefined): boolean {
  return st === undefined ? true : Boolean(st?.open);
}
