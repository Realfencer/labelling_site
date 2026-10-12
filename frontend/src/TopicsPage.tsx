import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import BrandLoader from "./components/BrandLoader";
import { useSiteBackdrop } from "./store/useBackdrop";
import BrandMark from "./components/BrandMark";
import WorkspaceChooser from "./components/WorkspaceChooser";
import { Search } from "lucide-react";
import { api, type TopicInfo } from "./api";
import { prefetchLabeler, takePrefetchedTopics } from "./prefetch";
import {
  formatTopicTitle,
  groupTopics,
  matchesTopicSearch,
  topicActionLabel,
  topicCanEnter,
  topicHighlights,
  topicMetrics,
  topicPlatformImages,
  topicStatus,
} from "./topicUtils";

function statusMeta(topic: TopicInfo) {
  const status = topicStatus(topic);
  if (status === "completed") {
    return { label: "Completed", tagClass: "v1-tag--success", banner: "public · completed" };
  }
  if (status === "upcoming") {
    return { label: "Upcoming", tagClass: "v1-tag--warning", banner: "public · upcoming" };
  }
  return { label: "Active", tagClass: "v1-tag--live", banner: "public · live" };
}

function HighlightsRow({ topics }: { topics: TopicInfo[] }) {
  const h = topicHighlights(topics);
  if (!topics.length) return null;

  return (
    <section className="grid grid-cols-1 gap-3 md:grid-cols-3">
      <div className="rounded-2xl border border-manako-border bg-manako-card p-4">
        <div className="text-[10px] font-bold uppercase tracking-wider text-manako-muted">Ready to label</div>
        <div className="mt-1 text-2xl font-semibold text-white">{h.readyCount}</div>
        <div className="mt-1 font-mono text-[11px] text-manako-muted">
          {h.topReady ? formatTopicTitle(h.topReady.title) : "Import a topic to begin"}
        </div>
      </div>
      <div className="rounded-2xl border border-manako-border bg-manako-card p-4">
        <div className="text-[10px] font-bold uppercase tracking-wider text-manako-muted">Active public</div>
        <div className="mt-1 text-2xl font-semibold text-white">{h.activeCount}</div>
        <div className="mt-1 font-mono text-[11px] text-manako-muted">Live subnet eval streams</div>
      </div>
      <div className="rounded-2xl border border-manako-border bg-manako-card p-4">
        <div className="text-[10px] font-bold uppercase tracking-wider text-manako-muted">Completed archive</div>
        <div className="mt-1 text-2xl font-semibold text-white">{h.completedCount}</div>
        <div className="mt-1 font-mono text-[11px] text-manako-muted">
          {h.labelableCompletedCount} wired for labelling
        </div>
      </div>
    </section>
  );
}

function TopicCard({ topic, compact = false }: { topic: TopicInfo; compact?: boolean }) {
  const [previewOk, setPreviewOk] = useState(Boolean(topic.preview_ready));
  const [imgSrc, setImgSrc] = useState(`${topic.preview_url}?v=1`);
  const meta = statusMeta(topic);
  const status = topicStatus(topic);
  const canEnter = topicCanEnter(topic);
  const platform = topicPlatformImages(topic);
  const metrics = topicMetrics(topic);
  const actionLabel = topicActionLabel(topic);
  const title = formatTopicTitle(topic.title);
  const objects = (topic.objects || []).slice(0, 4);

  useEffect(() => {
    setPreviewOk(Boolean(topic.preview_ready));
    // Stable URL (no Date.now()) so the browser/HTTP cache can serve repeat visits.
    setImgSrc(`${topic.preview_url}?v=${topic.preview_ready ? 1 : 0}`);
  }, [topic.preview_url, topic.preview_ready, topic.slug]);

  const preview = (
    <div className={`topic-card-preview ${compact ? "topic-card-preview--compact" : ""}`}>
      {previewOk ? (
        <img
          src={imgSrc}
          alt=""
          loading="lazy"
          onError={() => setPreviewOk(false)}
          onLoad={() => setPreviewOk(true)}
        />
      ) : (
        <div className="flex h-full w-full items-center justify-center bg-manako-panel">
          <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-manako-muted">No preview</span>
        </div>
      )}
      {!previewOk && (
        <img
          src={`${topic.preview_url}?retry=1`}
          alt=""
          className="pointer-events-none absolute inset-0 h-full w-full object-cover opacity-0"
          loading="lazy"
          onLoad={(e) => {
            setImgSrc((e.target as HTMLImageElement).src);
            setPreviewOk(true);
          }}
        />
      )}
      {canEnter && platform > 0 && (
        <span className="absolute right-2 top-2 v1-tag v1-tag--success">Ready</span>
      )}
    </div>
  );

  return (
    <article className={`topic-card ${compact ? "topic-card--compact" : ""} ${status === "active" ? "topic-card--active" : ""}`}>
      {canEnter && status !== "upcoming" ? (
        <Link to={`/t/${topic.slug}?workspace=platform`} className="relative block">
          {preview}
        </Link>
      ) : (
        <div className="relative block cursor-not-allowed opacity-90">{preview}</div>
      )}

      <div className="topic-card-banner">
        <span>{meta.banner}</span>
        {status === "active" && (
          <span className="inline-flex items-center gap-1.5 text-manako-success">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-manako-success" />
            Live
          </span>
        )}
      </div>

      <div className={`flex flex-1 flex-col ${compact ? "gap-2 p-3" : "gap-3 p-4"}`}>
        {!compact && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="v1-tag v1-tag--outline">General</span>
            <span className="v1-tag v1-tag--outline">Public</span>
            <span className={`v1-tag ${meta.tagClass}`}>{meta.label}</span>
          </div>
        )}

        <div>
          <h2 className={`font-medium leading-tight text-white ${compact ? "text-xs" : "text-sm"}`}>{title}</h2>
          {objects.length > 0 ? (
            <p className={`mt-1 truncate font-mono text-manako-muted ${compact ? "text-[10px]" : "text-[11px]"}`}>
              {objects.join(" · ")}
            </p>
          ) : (
            <p className={`mt-1 truncate font-mono text-manako-muted ${compact ? "text-[10px]" : "text-[11px]"}`}>
              {topic.subtitle || topic.element_id}
            </p>
          )}
        </div>

        <div className={compact ? "topic-metric-grid topic-metric-grid--compact" : "topic-metric-grid"}>
          {metrics.map((m) => (
            <div key={m.label} className="topic-metric-cell">
              <div className="topic-metric-label">{m.label}</div>
              <div className="topic-metric-value truncate">{m.value}</div>
            </div>
          ))}
        </div>

        <div className="mt-auto flex gap-2">
          {canEnter && status !== "upcoming" ? (
            <>
              <Link
                to={`/t/${topic.slug}?workspace=platform`}
                className="topic-btn topic-btn--platform flex-[1.4]"
              >
                {actionLabel}
              </Link>
              <Link to={`/t/${topic.slug}/own/days`} className="topic-btn topic-btn--own">
                Own
              </Link>
            </>
          ) : (
            <div className="topic-btn topic-btn--disabled">{actionLabel}</div>
          )}
        </div>
      </div>
    </article>
  );
}

function UpcomingPane({ topics }: { topics: TopicInfo[] }) {
  if (!topics.length) return null;

  return (
    <aside className="topics-upcoming-pane">
      <div className="topics-upcoming-pane-inner">
        <div className="topics-section-title topics-upcoming-pane-title">
          <span>Upcoming</span>
          <span className="font-mono text-xs text-manako-muted">{topics.length}</span>
        </div>
        <div className="topics-upcoming-stack">
          {topics.map((topic) => (
            <TopicCard key={topic.slug} topic={topic} compact />
          ))}
        </div>
      </div>
    </aside>
  );
}

function TopicSection({
  title,
  topics,
  layout,
}: {
  title: string;
  topics: TopicInfo[];
  layout: "grid" | "archive" | "list";
}) {
  if (!topics.length) return null;

  const gridClass =
    layout === "archive"
      ? "topics-archive-grid"
      : layout === "list"
        ? "topics-card-grid topics-card-grid--list"
        : "topics-card-grid";

  return (
    <section>
      <div className="topics-section-title">
        <span>{title}</span>
        <span className="font-mono text-xs text-manako-muted">{topics.length}</span>
      </div>
      <div className={gridClass}>
        {topics.map((topic) => (
          <TopicCard key={topic.slug} topic={topic} />
        ))}
      </div>
    </section>
  );
}

type FilterTab = "all" | "active" | "completed" | "upcoming";

export default function TopicsPage() {
  useSiteBackdrop("hero"); // always-dark screen on the shared 3D backdrop
  const [topics, setTopics] = useState<TopicInfo[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [tab, setTab] = useState<FilterTab>("all");

  useEffect(() => {
    let alive = true;
    setLoading(true);
    prefetchLabeler();
    (takePrefetchedTopics() ?? api.listTopics())
      .catch(() => api.listTopics()) // prefetch failed (e.g. not signed in yet): one normal attempt
      .then((res) => {
        if (!alive) return;
        const list = res.topics || [];
        setTopics(list);
        setError("");
      })
      .catch((e) => {
        if (!alive) return;
        setError(String(e));
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, []);

  const filtered = useMemo(
    () => topics.filter((t) => matchesTopicSearch(t, search)),
    [topics, search],
  );

  const grouped = useMemo(() => groupTopics(filtered), [filtered]);

  const visible = useMemo(() => {
    if (tab === "all") return grouped;
    return {
      active: tab === "active" ? grouped.active : [],
      completed: tab === "completed" ? grouped.completed : [],
      upcoming: tab === "upcoming" ? grouped.upcoming : [],
    };
  }, [grouped, tab]);

  const tabs: { id: FilterTab; label: string; count: number }[] = [
    { id: "all", label: "All", count: filtered.length },
    { id: "active", label: "Active", count: grouped.active.length },
    { id: "completed", label: "Completed", count: grouped.completed.length },
    { id: "upcoming", label: "Upcoming", count: grouped.upcoming.length },
  ];

  const [meName, setMeName] = useState<string | null>(null);
  useEffect(() => {
    api
      .authStatus()
      .then((a) => setMeName(a.username ?? null))
      .catch(() => undefined);
  }, []);
  if (loading) return <BrandLoader label="Loading topics" hint="Fetching your workspaces" accent="violet" />;
  if (!error && topics.length === 1) {
    const only = topics[0];
    return (
      <WorkspaceChooser
        slug={only.slug}
        title={formatTopicTitle(only.title)}
        platformImages={topicPlatformImages(only)}
        user={meName}
      />
    );
  }

  return (
    <div className="topics-shell rt-in">
      <header className="topics-header">
        <div className="flex w-full items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <BrandMark size={36} />
            <div>
              <div className="text-sm font-semibold text-white">Score Labeler</div>
              <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-manako-muted">
                Bittensor SN44 · Public Elements
              </div>
            </div>
          </div>
          <div className="relative hidden max-w-xs flex-1 sm:block">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-manako-muted" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search topics, classes…"
              className="h-9 w-full rounded-lg border border-manako-border bg-manako-card pl-9 pr-3 text-sm text-white placeholder:text-manako-muted focus:border-manako-muted focus:outline-none"
            />
          </div>
        </div>
      </header>

      <main className="topics-main mx-auto w-full max-w-[90rem] space-y-8 px-4 py-8 sm:px-6 md:py-10">
        <div className="relative sm:hidden">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-manako-muted" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search topics, classes…"
            className="h-10 w-full rounded-lg border border-manako-border bg-manako-card pl-9 pr-3 text-sm text-white placeholder:text-manako-muted focus:border-manako-muted focus:outline-none"
          />
        </div>

        <div className="topics-tab-row flex gap-2 overflow-x-auto pb-1">
          {tabs.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              className={`v1-tag ${tab === t.id ? "border-white text-white" : "v1-tag--outline"}`}
            >
              {t.label} · {t.count}
            </button>
          ))}
        </div>

        {error && (
          <div className="rounded-xl border border-manako-error/40 bg-manako-error/10 px-4 py-3 text-sm text-rose-200">
            {error}
          </div>
        )}

        {filtered.length === 0 ? (
          <div className="rounded-2xl border border-manako-border bg-manako-card px-6 py-12 text-center">
            <p className="text-sm text-manako-muted">No topics match your search.</p>
          </div>
        ) : (
          <div className={tab === "all" ? "topics-layout" : "space-y-10"}>
            <div className="topics-main-column space-y-10">
              {tab === "all" && !search && <HighlightsRow topics={topics} />}
              {(tab === "all" || tab === "active") && (
                <TopicSection title="Active" topics={visible.active} layout="grid" />
              )}
              {(tab === "all" || tab === "completed") && (
                <TopicSection title="Completed" topics={visible.completed} layout="archive" />
              )}
              {tab === "upcoming" && (
                <TopicSection title="Upcoming" topics={visible.upcoming} layout="list" />
              )}
              {tab === "all" && visible.upcoming.length > 0 && (
                <div className="topics-upcoming-mobile">
                  <TopicSection title="Upcoming" topics={visible.upcoming} layout="list" />
                </div>
              )}
            </div>
            {tab === "all" && visible.upcoming.length > 0 && (
              <UpcomingPane topics={visible.upcoming} />
            )}
          </div>
        )}
      </main>
    </div>
  );
}
