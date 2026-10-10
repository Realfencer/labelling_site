import type { TopicInfo } from "./api";

/** Score Studio–style element naming: dedupe "Detect", split camelCase, title-case. */
export function formatTopicTitle(raw: string): string {
  const text = (raw ?? "").trim();
  if (!text) return "Topic";
  const tail = text.includes("/") ? text.slice(text.lastIndexOf("/") + 1) : text;
  let spaced = tail
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const words = spaced.split(" ").filter(Boolean);
  const deduped: string[] = [];
  for (const word of words) {
    if (deduped.length && deduped[deduped.length - 1].toLowerCase() === word.toLowerCase()) continue;
    deduped.push(word);
  }
  while (deduped.length > 1 && /^\d+$/.test(deduped[deduped.length - 1])) deduped.pop();
  return deduped
    .map((w) => {
      const lower = w.toLowerCase();
      if (lower === "detect") return "Detect";
      return lower.charAt(0).toUpperCase() + lower.slice(1);
    })
    .join(" ")
    .trim();
}

export function topicStatus(topic: TopicInfo): "active" | "completed" | "upcoming" {
  const s = (topic.status || "active").toLowerCase();
  if (s === "completed") return "completed";
  if (s === "upcoming") return "upcoming";
  return "active";
}

export function topicPlatformImages(topic: TopicInfo): number {
  return topic.platform_images ?? topic.local_images ?? 0;
}

export function topicCanEnter(topic: TopicInfo): boolean {
  return Boolean(topic.can_label ?? topic.label_ready);
}

export type TopicMetric = { label: string; value: string };

export function topicMetrics(topic: TopicInfo): TopicMetric[] {
  const status = topicStatus(topic);
  const platform = topicPlatformImages(topic);
  if (status === "upcoming") {
    return [
      { label: "Starter", value: String(topic.evaluation_count || "—") },
      { label: "Block", value: topic.latest_block ? String(topic.latest_block) : "TBD" },
      { label: "Classes", value: String(topic.objects?.length || 0) },
    ];
  }
  if (status === "completed") {
    return [
      { label: "Images", value: platform > 0 ? platform.toLocaleString() : "—" },
      { label: "Evals", value: topic.evaluation_count.toLocaleString() },
      { label: "Latest", value: topic.latest_block ? String(topic.latest_block) : "—" },
    ];
  }
  return [
    { label: "Evals", value: topic.evaluation_count.toLocaleString() },
    { label: "Cached", value: platform > 0 ? platform.toLocaleString() : "—" },
    { label: "Latest", value: topic.latest_block ? String(topic.latest_block) : "—" },
  ];
}

export function topicActionLabel(topic: TopicInfo): string {
  const status = topicStatus(topic);
  const platform = topicPlatformImages(topic);
  if (status === "upcoming") return "Not started yet";
  if (!topicCanEnter(topic)) return "Dataset pending";
  if (status === "completed" && platform > 0) return "Label now";
  if (status === "active" && platform > 0) return "Continue";
  if (status === "active") return "Open";
  if (status === "completed") return "Import & label";
  return "Open";
}

export function matchesTopicSearch(topic: TopicInfo, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  const hay = [
    topic.title,
    topic.subtitle,
    topic.element_id,
    topic.slug,
    ...(topic.objects || []),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return hay.includes(q);
}

export function groupTopics(topics: TopicInfo[]) {
  const active: TopicInfo[] = [];
  const completed: TopicInfo[] = [];
  const upcoming: TopicInfo[] = [];
  for (const topic of topics) {
    const status = topicStatus(topic);
    if (status === "completed") completed.push(topic);
    else if (status === "upcoming") upcoming.push(topic);
    else active.push(topic);
  }
  return { active, completed, upcoming };
}

export function topicHighlights(topics: TopicInfo[]) {
  const ready = topics.filter((t) => topicCanEnter(t) && topicPlatformImages(t) > 0);
  const active = topics.filter((t) => topicStatus(t) === "active");
  const completed = topics.filter((t) => topicStatus(t) === "completed");
  const labelableCompleted = completed.filter((t) => topicCanEnter(t));
  return {
    readyCount: ready.length,
    activeCount: active.length,
    completedCount: completed.length,
    labelableCompletedCount: labelableCompleted.length,
    topReady: ready.sort((a, b) => topicPlatformImages(b) - topicPlatformImages(a))[0] ?? null,
  };
}
