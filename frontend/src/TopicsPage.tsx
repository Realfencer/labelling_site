import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type TopicInfo } from "./api";
import { cn } from "./lib/utils";

function TopicCard({ topic }: { topic: TopicInfo }) {
  const [previewOk, setPreviewOk] = useState(Boolean(topic.preview_ready));
  const [imgSrc, setImgSrc] = useState(`${topic.preview_url}?v=1`);

  useEffect(() => {
    setPreviewOk(Boolean(topic.preview_ready));
    setImgSrc(`${topic.preview_url}?v=${Date.now()}`);
  }, [topic.preview_url, topic.preview_ready, topic.slug]);

  return (
    <div className="group flex flex-col overflow-hidden rounded-2xl border border-slate-700/80 bg-[#12161e] shadow-[0_12px_40px_rgba(0,0,0,0.35)] transition hover:-translate-y-0.5 hover:border-sky-500/60 hover:shadow-[0_16px_48px_rgba(14,165,233,0.12)]">
      <Link to={`/t/${topic.slug}?workspace=platform`} className="block">
        <div className="relative aspect-[16/10] w-full overflow-hidden bg-black">
          {previewOk ? (
            <img
              src={imgSrc}
              alt=""
              className="h-full w-full object-cover opacity-95 transition duration-500 group-hover:scale-[1.03]"
              onError={() => setPreviewOk(false)}
              onLoad={() => setPreviewOk(true)}
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center bg-gradient-to-b from-slate-900 to-black">
              <span className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-600">
                loading preview…
              </span>
            </div>
          )}
          {!previewOk && (
            <img
              src={`${topic.preview_url}?retry=1`}
              alt=""
              className="pointer-events-none absolute inset-0 h-full w-full object-cover opacity-0"
              onLoad={(e) => {
                setImgSrc((e.target as HTMLImageElement).src);
                setPreviewOk(true);
              }}
            />
          )}
        </div>
      </Link>
      <div className="flex flex-1 flex-col gap-3 p-4">
        <div className="flex items-center gap-2">
          <span className="rounded border border-slate-600/80 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-slate-400">
            General
          </span>
          <span className="rounded border border-slate-600/80 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-slate-400">
            Public
          </span>
        </div>
        <div>
          <h2 className="text-xl font-extrabold tracking-tight text-white">{topic.title}</h2>
          <p className="mt-1 truncate text-xs text-slate-400">{topic.subtitle || topic.element_id}</p>
        </div>
        <div className="grid grid-cols-3 gap-2 border-t border-slate-800 pt-3">
          <div>
            <div className="text-[9px] font-bold uppercase tracking-wider text-slate-500">Evals</div>
            <div className="text-sm font-extrabold text-slate-100">{topic.evaluation_count}</div>
          </div>
          <div>
            <div className="text-[9px] font-bold uppercase tracking-wider text-slate-500">Own</div>
            <div className="text-sm font-extrabold text-slate-100">{topic.local_images}</div>
          </div>
          <div>
            <div className="text-[9px] font-bold uppercase tracking-wider text-slate-500">Latest</div>
            <div className="truncate text-sm font-extrabold text-slate-100">{topic.latest_block || "—"}</div>
          </div>
        </div>
        <div className="mt-auto flex gap-1.5">
          <Link
            to={`/t/${topic.slug}?workspace=platform`}
            className={cn(
              "flex h-8 flex-1 items-center justify-center rounded-lg border border-sky-500/50 bg-sky-500/10 text-xs font-extrabold text-sky-300 transition",
              "hover:border-sky-400 hover:bg-sky-500/20",
            )}
          >
            Platform
          </Link>
          <Link
            to={`/t/${topic.slug}?workspace=own`}
            className={cn(
              "flex h-8 flex-1 items-center justify-center rounded-lg border border-violet-500/50 bg-violet-500/10 text-xs font-extrabold text-violet-300 transition",
              "hover:border-violet-400 hover:bg-violet-500/20",
            )}
          >
            Own
          </Link>
        </div>
      </div>
    </div>
  );
}

export default function TopicsPage() {
  const [topics, setTopics] = useState<TopicInfo[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    api
      .listTopics()
      .then((res) => {
        if (!alive) return;
        setTopics(res.topics || []);
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

  return (
    <div className="min-h-screen bg-[#080b12] text-slate-100">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-10 sm:px-6">
        <header className="space-y-2">
          <p className="text-[11px] font-bold uppercase tracking-[0.22em] text-sky-400/90">
            Bittensor Subnet 44 · Public Elements
          </p>
          <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">Choose a topic</h1>
          <p className="max-w-2xl text-sm text-slate-400">
            Pick <span className="font-semibold text-sky-300">Platform</span> for eval images from the score subnet, or{" "}
            <span className="font-semibold text-violet-300">Own</span> to upload and label your own images per topic.
          </p>
        </header>

        {error && (
          <div className="rounded-xl border border-rose-800/60 bg-rose-950/40 px-4 py-3 text-sm text-rose-200">{error}</div>
        )}

        {loading ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-2">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="h-80 animate-pulse rounded-2xl bg-slate-900/80" />
            ))}
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-2">
            {topics.map((topic) => (
              <TopicCard key={topic.slug} topic={topic} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
