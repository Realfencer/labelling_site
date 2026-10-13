import { create } from "zustand";

export type TeamUser = {
  name: string;
  color: string;
  online: boolean;
  item_key: string | null;
  label: string;
  workspace: string;
  idle: boolean;
  watching: string | null;
  watchers: string[];
  labeled: number;
  today: number;
  per_hour: number;
};

export type LabelInfo = { u: string | null; t: number; rev: number; e?: string | null };

export type TeamSyncResponse = {
  ok: boolean;
  reason?: string;
  server_time?: number;
  me?: {
    user: string;
    color: string;
    item_key: string | null;
    read_only: boolean;
    reason: "" | "claimed" | "watching";
    owner: string | null;
    watching: string | null;
    watchers: string[];
    rev: number;
  };
  users?: TeamUser[];
  claims?: Record<string, string>;
  labels?: {
    full: boolean;
    seq: number;
    changes: Array<{ k: string; u: string | null; t: number; a: string; rev: number; e: string }>;
    map?: Record<string, LabelInfo>;
  };
  watch_error?: string;
  feed?: {
    user: string;
    item_key: string | null;
    label: string;
    draft: { seq: number; boxes: Array<Record<string, number | null>> } | null;
  };
};

export type SaveStatus = "idle" | "saving" | "saved" | "retrying" | "error" | "locked" | "conflict";

interface TeamStore {
  me: { user: string; color: string } | null;
  users: TeamUser[];
  colors: Record<string, string>;
  claims: Record<string, string>;
  labels: Record<string, LabelInfo>;
  seq: number | null;
  readOnly: boolean;
  reason: "" | "claimed" | "watching";
  owner: string | null;
  watching: string | null;
  watchers: string[];
  feed: TeamSyncResponse["feed"] | null;
  notice: string;
  /** Latest revision of each image we know of; sent as base_rev so stale saves are refused. */
  revs: Record<string, number>;
  /** One-shot instruction for the next heartbeat. */
  watchIntent: "start" | "stop" | null;
  /** Who the next "start" should watch (a name), or null for "whoever labeled most recently". */
  watchTarget: string | null;
  connected: boolean;
  save: { status: SaveStatus; message: string; at: number | null };
  setMe: (me: { user: string; color: string } | null) => void;
  applySync: (r: TeamSyncResponse) => void;
  noteRev: (itemKey: string, rev: number | undefined) => void;
  /** Our own save came back with the item's labeler: show the tag now instead of waiting for the next poll. */
  noteLabel: (itemKey: string, user: string | null | undefined, rev: number | undefined) => void;
  setWatchIntent: (i: "start" | "stop" | null, target?: string | null) => void;
  setSave: (status: SaveStatus, message?: string) => void;
  setNotice: (m: string) => void;
  setConnected: (v: boolean) => void;
  reset: () => void;
}

function sameJson(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  try {
    return JSON.stringify(a) === JSON.stringify(b);
  } catch {
    return false;
  }
}

export const useTeam = create<TeamStore>((set, get) => ({
  me: null,
  users: [],
  colors: {},
  claims: {},
  labels: {},
  seq: null,
  readOnly: false,
  reason: "",
  owner: null,
  watching: null,
  watchers: [],
  feed: null,
  notice: "",
  revs: {},
  watchIntent: null,
  watchTarget: null,
  connected: true,
  save: { status: "idle", message: "", at: null },

  setMe: (me) => set({ me }),

  applySync: (r) => {
    if (!r.ok || !r.me) return;
    const cur = get();
    const patch: Partial<TeamStore> = { connected: true };
    if (!sameJson(cur.users, r.users)) {
      patch.users = r.users ?? [];
      const colors: Record<string, string> = {};
      for (const u of r.users ?? []) colors[u.name] = u.color;
      if (!sameJson(cur.colors, colors)) patch.colors = colors;
    }
    if (!sameJson(cur.claims, r.claims)) patch.claims = r.claims ?? {};
    const ro = r.me.read_only;
    if (cur.readOnly !== ro) patch.readOnly = ro;
    if (cur.reason !== r.me.reason) patch.reason = r.me.reason;
    if (cur.owner !== r.me.owner) patch.owner = r.me.owner;
    if (cur.watching !== r.me.watching) patch.watching = r.me.watching;
    if (!sameJson(cur.watchers, r.me.watchers)) patch.watchers = r.me.watchers;
    if (!sameJson(cur.feed, r.feed ?? null)) patch.feed = r.feed ?? null;
    if (!cur.me || cur.me.user !== r.me.user || cur.me.color !== r.me.color) {
      patch.me = { user: r.me.user, color: r.me.color };
    }
    if (r.watch_error) patch.notice = r.watch_error;
    // Label log: full map first time (or after a gap), deltas afterwards.
    const lb = r.labels;
    if (lb) {
      if (lb.full && lb.map) {
        patch.labels = lb.map;
        const revs = { ...cur.revs };
        for (const [k, v] of Object.entries(lb.map)) revs[k] = Math.max(revs[k] ?? 0, v.rev);
        patch.revs = revs;
      } else if (lb.changes.length) {
        const labels = { ...cur.labels };
        const revs = { ...cur.revs };
        for (const c of lb.changes) {
          labels[c.k] = { u: c.u, t: c.t, rev: c.rev, e: c.e };
          revs[c.k] = Math.max(revs[c.k] ?? 0, c.rev);
        }
        patch.labels = labels;
        patch.revs = revs;
      }
      patch.seq = lb.seq;
    }
    set(patch);
  },

  noteRev: (itemKey, rev) => {
    if (!itemKey || rev == null) return;
    const revs = get().revs;
    if ((revs[itemKey] ?? 0) >= rev) return;
    set({ revs: { ...revs, [itemKey]: rev } });
  },

  noteLabel: (itemKey, user, rev) => {
    if (!itemKey || user === undefined) return;
    const labels = get().labels;
    const cur = labels[itemKey];
    if (user === null) {
      if (!cur) return;
      const next = { ...labels };
      delete next[itemKey];
      set({ labels: next });
      return;
    }
    if (cur && cur.u === user && (cur.rev ?? 0) >= (rev ?? 0)) return;
    set({ labels: { ...labels, [itemKey]: { u: user, t: Date.now() / 1000, rev: Math.max(rev ?? 0, cur?.rev ?? 0), e: user } } });
  },

  setWatchIntent: (watchIntent, target = null) => set({ watchIntent, watchTarget: watchIntent === "start" ? target : null }),
  setSave: (status, message = "") =>
    set({ save: { status, message, at: status === "saved" ? Date.now() : get().save.at } }),
  setNotice: (notice) => set({ notice }),
  setConnected: (connected) => {
    if (get().connected !== connected) set({ connected });
  },
  reset: () =>
    set({
      users: [],
      claims: {},
      labels: {},
      seq: null,
      readOnly: false,
      reason: "",
      owner: null,
      watching: null,
      watchers: [],
      feed: null,
      notice: "",
      revs: {},
      watchIntent: null,
      watchTarget: null,
      save: { status: "idle", message: "", at: null },
    }),
}));

/** Display helper: "3 min ago". */
export function timeAgo(tsSeconds: number, nowMs = Date.now()): string {
  const s = Math.max(0, Math.round(nowMs / 1000 - tsSeconds));
  if (s < 10) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ${m % 60}m ago`;
  return `${Math.floor(h / 24)}d ago`;
}
