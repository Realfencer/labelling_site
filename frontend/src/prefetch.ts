import { api, type TopicInfo } from "./api";

type TopicsResponse = { topics: TopicInfo[] };

let topicsPromise: Promise<TopicsResponse> | null = null;
let topicsStartedAt = 0;

/** Start GET /topics right away, in parallel with the auth check, instead of after it (one round trip less). */
export function startTopicsPrefetch() {
  if (topicsPromise) return;
  topicsStartedAt = Date.now();
  topicsPromise = api.listTopics();
  topicsPromise.catch(() => undefined); // handled here; the consumer decides what a failure means
}

/** The prefetched topic list while it is fresh; null → ask normally. */
export function takePrefetchedTopics(): Promise<TopicsResponse> | null {
  if (!topicsPromise || Date.now() - topicsStartedAt > 15000) return null;
  return topicsPromise;
}

let labelerWarm = false;

/** Fetch the labeler bundle while the browser is idle, so opening a workspace is instant. */
export function prefetchLabeler() {
  if (labelerWarm) return;
  labelerWarm = true;
  const run = () => void import("./App");
  const w = window as Window & { requestIdleCallback?: (cb: () => void, opts?: { timeout: number }) => number };
  if (typeof w.requestIdleCallback === "function") w.requestIdleCallback(run, { timeout: 2500 });
  else window.setTimeout(run, 900);
}
