import { useEffect } from "react";
import { create } from "zustand";

/** hero: always-dark showcase screens (full quality) · vivid: same scene at a 30 fps cap for content-heavy pages · calm: working screens (labeler): low fps, freezes while you work · soft: faint. All but hero only in dark theme. */
export type BackdropMode = "hero" | "vivid" | "calm" | "soft";

type State = {
  requests: Map<symbol, BackdropMode>;
  add: (id: symbol, mode: BackdropMode) => void;
  remove: (id: symbol) => void;
};

/** Which ambient background the visible page wants. Several requests can overlap (route change): hero beats soft. */
export const useBackdrop = create<State>((set) => ({
  requests: new Map(),
  add: (id, mode) =>
    set((s) => {
      const next = new Map(s.requests);
      next.set(id, mode);
      return { requests: next };
    }),
  remove: (id) =>
    set((s) => {
      if (!s.requests.has(id)) return s;
      const next = new Map(s.requests);
      next.delete(id);
      return { requests: next };
    }),
}));

export function effectiveMode(requests: Map<symbol, BackdropMode>): BackdropMode | null {
  let mode: BackdropMode | null = null;
  for (const m of requests.values()) {
    if (m === "hero") return "hero";
    if (m === "vivid") mode = "vivid";
    else if (m === "calm" && mode !== "vivid") mode = "calm";
    else if (mode === null) mode = "soft";
  }
  return mode;
}

/** Ask the site-wide backdrop for a mode while this component is mounted. Pages that do not ask get none. */
export function useSiteBackdrop(mode: BackdropMode) {
  useEffect(() => {
    const id = Symbol("backdrop");
    useBackdrop.getState().add(id, mode);
    // Hero screens are always dark: give the page itself a dark base too, so they can never show white
    // (first paint, canvas not drawn yet, or a browser without canvas).
    const root = document.documentElement;
    if (mode === "hero") root.classList.add("bd-hero");
    return () => {
      useBackdrop.getState().remove(id);
      if (mode === "hero") root.classList.remove("bd-hero");
    };
  }, [mode]);
}
