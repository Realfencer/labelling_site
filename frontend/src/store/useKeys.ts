import { create } from "zustand";
import { api, type HfModelRow, type KeysView } from "../api";

interface KeysStore {
  view: KeysView | null;
  models: HfModelRow[];
  open: boolean;
  setOpen: (v: boolean) => void;
  setView: (v: KeysView) => void;
  refresh: () => Promise<void>;
  refreshModels: () => Promise<void>;
}

/** Shared (server-side) API keys: only masked status lives here, never a secret. */
export const useKeys = create<KeysStore>((set, get) => ({
  view: null,
  models: [],
  open: false,
  setOpen: (open) => set({ open }),
  setView: (view) => set({ view }),
  refresh: async () => {
    try {
      const view = await api.keysGet();
      set({ view });
      if (view.hf.some((a) => a.n_models > 0) || get().models.length) await get().refreshModels();
    } catch {
      /* not signed in yet / transient */
    }
  },
  refreshModels: async () => {
    try {
      set({ models: (await api.keysModels()).models });
    } catch {
      /* ignore */
    }
  },
}));
