import { useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Images, Upload } from "lucide-react";
import KeysButton from "./KeysButton";

type Props = {
  slug: string;
  title: string;
  platformImages: number;
  user?: string | null;
};

/** Home screen for a single-topic deployment: pick Platform or Own, then go straight in. */
export default function WorkspaceChooser({ slug, title, platformImages, user }: Props) {
  const navigate = useNavigate();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "p" || e.key === "P") navigate(`/t/${slug}?workspace=platform`);
      if (e.key === "o" || e.key === "O") navigate(`/t/${slug}?workspace=own`);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navigate, slug]);

  const card =
    "group flex flex-1 flex-col gap-3 rounded-2xl border border-slate-700/70 bg-[#12161e] p-6 text-left shadow-2xl transition hover:-translate-y-0.5 hover:border-sky-500/70 focus:outline-none focus-visible:border-sky-400";

  return (
    <div className="flex min-h-screen items-center justify-center bg-[#0b1220] px-4 py-10">
      <div className="w-full max-w-3xl">
        <div className="mb-6 text-center">
          <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-slate-500">
            {user ? `Signed in as ${user}` : "Score Labeler"}
          </div>
          <h1 className="mt-1 text-2xl font-extrabold text-white">{title}</h1>
          <p className="mt-1 text-[13px] text-slate-400">Choose where you want to label.</p>
        </div>

        <div className="flex flex-col gap-4 sm:flex-row">
          <Link to={`/t/${slug}?workspace=platform`} className={card} autoFocus>
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-sky-500 text-white">
              <Images className="h-5 w-5" />
            </div>
            <div>
              <div className="text-lg font-extrabold text-white">Platform</div>
              <div className="text-[12px] text-slate-400">
                Score challenge images with miner references and scoring.
              </div>
            </div>
            <div className="mt-auto flex items-center justify-between">
              <span className="text-[12px] font-semibold text-sky-300">
                {platformImages > 0 ? `${platformImages.toLocaleString()} images` : "Open"}
              </span>
              <kbd className="rounded border border-slate-600 px-1.5 py-0.5 text-[10px] text-slate-400">P</kbd>
            </div>
          </Link>

          <Link to={`/t/${slug}?workspace=own`} className={card}>
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-violet-500 text-white">
              <Upload className="h-5 w-5" />
            </div>
            <div>
              <div className="text-lg font-extrabold text-white">Own</div>
              <div className="text-[12px] text-slate-400">Your uploaded images, grouped by day.</div>
            </div>
            <div className="mt-auto flex items-center justify-between">
              <span className="text-[12px] font-semibold text-violet-300">Pick a day</span>
              <kbd className="rounded border border-slate-600 px-1.5 py-0.5 text-[10px] text-slate-400">O</kbd>
            </div>
          </Link>
        </div>

        <div className="mt-4 flex items-center justify-center gap-4">
          <KeysButton />
          <Link to="/collected" className="text-[12px] font-semibold text-slate-400 hover:text-sky-300">
            Collected miner solutions →
          </Link>
        </div>
      </div>
    </div>
  );
}
