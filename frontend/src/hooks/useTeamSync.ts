import { useEffect } from "react";
import { api } from "../api";
import { getTabSessionId } from "../sessionId";
import { useLabeler } from "../store/useLabeler";
import { useTeam } from "../store/useTeam";
import { workspaceFromState } from "../api";

const BASE_MS = 1500;
const WATCH_MS = 600;
const HIDDEN_MS = 5000;

/** Heartbeat + team state (presence, claims, label log deltas, live feed). */
export function useTeamSync(enabled: boolean, topicSlug: string): void {
  useEffect(() => {
    if (!enabled) return;
    useTeam.getState().reset();

    let stopped = false;
    let timer: number | null = null;
    let inflight = false;
    let again = false;
    let lastInput = Date.now();
    let failures = 0;

    const markInput = () => {
      lastInput = Date.now();
    };
    const inputEvents: Array<keyof WindowEventMap> = ["pointerdown", "keydown", "wheel", "pointermove"];
    for (const ev of inputEvents) window.addEventListener(ev, markInput, { passive: true });

    const schedule = () => {
      if (stopped) return;
      if (timer != null) window.clearTimeout(timer);
      const t = useTeam.getState();
      let delay = document.hidden ? HIDDEN_MS : t.watching ? WATCH_MS : BASE_MS;
      if (failures > 0) delay = Math.min(10000, delay * (1 + failures));
      timer = window.setTimeout(() => void tick(), delay);
    };

    const tick = async () => {
      if (stopped) return;
      if (inflight) {
        again = true;
        return;
      }
      inflight = true;
      const team = useTeam.getState();
      const intent = team.watchIntent;
      const live = useLabeler.getState();
      const challenge = live.detail?.challenge;
      try {
        const resp = await api.teamSync({
          item_key: challenge?.item_key ?? null,
          label: challenge?.display_name ?? "",
          workspace: live.appState ? workspaceFromState(live.appState) : "",
          idle_s: Math.round((Date.now() - lastInput) / 1000),
          ...(intent === "start" ? { watch: team.watchTarget ?? "auto" } : {}),
          ...(intent === "stop" ? { stop_watch: true } : {}),
          // Keep an active watch alive on every beat (server drops it if the target left).
          ...(!intent && team.watching ? { watch: team.watching } : {}),
          since: team.seq,
        });
        if (stopped) return;
        failures = 0;
        useTeam.getState().applySync(resp);
        if (intent && useTeam.getState().watchIntent === intent) useTeam.getState().setWatchIntent(null);
      } catch {
        failures = Math.min(failures + 1, 5);
        // One slow/failed beat (busy server, brief network blip) is normal — only flag a real outage.
        if (failures >= 3) useTeam.getState().setConnected(false);
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

    const syncNow = () => {
      if (timer != null) window.clearTimeout(timer);
      void tick();
    };

    // Immediate beat when the open image changes (this is what claims the image).
    let lastKey = useLabeler.getState().detail?.challenge?.item_key ?? null;
    const unsubDetail = useLabeler.subscribe((s) => {
      const key = s.detail?.challenge?.item_key ?? null;
      if (key !== lastKey) {
        lastKey = key;
        syncNow();
      }
    });
    // …and when watching starts / switches person / stops (G, a teammate chip, Esc).
    const unsubIntent = useTeam.subscribe((s, prev) => {
      if (s.watchIntent && (s.watchIntent !== prev.watchIntent || s.watchTarget !== prev.watchTarget)) syncNow();
    });
    const onVisible = () => {
      if (!document.hidden) syncNow();
    };
    document.addEventListener("visibilitychange", onVisible);

    // Tab closing: free the image right away instead of waiting for the heartbeat to lapse.
    const onPageHide = () => {
      try {
        navigator.sendBeacon(`/api/team/release?sid=${encodeURIComponent(getTabSessionId())}`, new Blob([], { type: "text/plain" }));
      } catch {
        /* best effort */
      }
    };
    window.addEventListener("pagehide", onPageHide);

    void tick();
    return () => {
      stopped = true;
      if (timer != null) window.clearTimeout(timer);
      unsubDetail();
      unsubIntent();
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("pagehide", onPageHide);
      for (const ev of inputEvents) window.removeEventListener(ev, markInput);
    };
  }, [enabled, topicSlug]);
}
