import { create } from "zustand";
import type { AppState, ChallengeDetail, DetectionBox } from "../types";

export type Theme = "light" | "dark";

const THEME_KEY = "sn44-labeler-theme";

function readTheme(): Theme {
  if (typeof localStorage === "undefined") return "dark";
  return localStorage.getItem(THEME_KEY) === "light" ? "light" : "dark";
}

interface LabelerStore {
  theme: Theme;
  toggleTheme: () => void;
  appState: AppState | null;
  setAppState: (s: AppState | null) => void;
  detail: ChallengeDetail | null;
  setDetail: (d: ChallengeDetail | null) => void;
  boxes: DetectionBox[];
  setBoxes: (boxes: DetectionBox[]) => void;
  refBoxes: DetectionBox[];
  setRefBoxes: (boxes: DetectionBox[]) => void;
  selected: number | null;
  setSelected: (index: number | null) => void;
  currentClass: number;
  setCurrentClass: (cls: number) => void;
  error: string;
  setError: (msg: string) => void;
  busy: boolean;
  setBusy: (busy: boolean) => void;
}

export const useLabeler = create<LabelerStore>((set) => ({
  theme: readTheme(),
  toggleTheme: () =>
    set((s) => {
      const theme: Theme = s.theme === "dark" ? "light" : "dark";
      localStorage.setItem(THEME_KEY, theme);
      const root = document.documentElement;
      root.classList.add("theme-fade"); // colours cross-fade for a moment instead of flashing
      root.classList.toggle("dark", theme === "dark");
      window.setTimeout(() => root.classList.remove("theme-fade"), 350);
      return { theme };
    }),
  appState: null,
  setAppState: (appState) => set({ appState }),
  detail: null,
  setDetail: (detail) => set({ detail }),
  boxes: [],
  setBoxes: (boxes) => set({ boxes }),
  refBoxes: [],
  setRefBoxes: (refBoxes) => set({ refBoxes }),
  selected: null,
  setSelected: (selected) => set({ selected }),
  currentClass: 0,
  setCurrentClass: (currentClass) => set({ currentClass }),
  error: "",
  setError: (error) => set({ error }),
  busy: false,
  setBusy: (busy) => set({ busy }),
}));

if (typeof document !== "undefined") {
  document.documentElement.classList.toggle("dark", readTheme() === "dark");
}
