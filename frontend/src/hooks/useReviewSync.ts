import { useEffect } from "react";
import { api } from "../api";
import { useReview } from "../store/useReview";
import { useTeam } from "../store/useTeam";

const POLL_MS = 3000;
const HIDDEN_MS = 15000;

let kick: (() => void) | null = null;
/** Ask for fresh findings now (after a save or a "looks right"). */
export function syncReviewNow(): void {
  kick?.();
}

/** While the Own Review list is open: keep its findings live. Every save (ours or a teammate's) re-scores the image on
 *  the server; this picks that up within a few seconds, and right away after our own save. */
export function useReviewSync(active: boolean): void {
  useEffect(() => {
    if (!active) {
      useReview.getState().reset();
      return;
    }
    let stopped = false;
    let timer: number | null = null;
    let inflight = false;
    let again = false;

    const schedule = () => {
      if (stopped) return;
      if (timer != null) window.clearTimeout(timer);
      timer = window.setTimeout(() => void tick(), document.hidden ? HIDDEN_MS : POLL_MS);
    };
    const tick = async () => {
      if (stopped) return;
      if (inflight) {
        again = true;
        return;
      }
      inflight = true;
      try {
        const r = await api.ownReviewStatus(useReview.getState().version);
        if (!stopped) useReview.getState().apply(r);
      } catch {
        /* next beat */
      } finally {
        inflight = false;
        if (again) {
          again = false;
          void tick();
        } else {
          schedule();
        }
      }
    };
    kick = () => {
      if (timer != null) window.clearTimeout(timer);
      void tick();
    };
    // Our own save was stored: the server has already re-scored that image.
    const unsub = useTeam.subscribe((s, prev) => {
      if (s.save.at !== prev.save.at && s.save.status === "saved") kick?.();
    });
    void tick();
    return () => {
      stopped = true;
      kick = null;
      unsub();
      if (timer != null) window.clearTimeout(timer);
    };
  }, [active]);
}
