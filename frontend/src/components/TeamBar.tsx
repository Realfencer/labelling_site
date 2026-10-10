import { memo, useEffect, useMemo, useRef, useState } from "react";
import { Eye, Library, LogOut, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";
import { useTeam, type TeamUser } from "../store/useTeam";
import { cn } from "../lib/utils";
import KeysButton from "./KeysButton";

function fmtEta(hours: number): string {
  if (!isFinite(hours) || hours <= 0) return "";
  const totalMin = Math.round(hours * 60);
  if (totalMin < 60) return `~${totalMin}m`;
  const h = Math.floor(totalMin / 60);
  const m = totalMin % 60;
  return `~${h}h${m ? ` ${m}m` : ""}`;
}

const HOT_MS = 12000;

/** What this person is doing right now, as one small coloured pill. */
function StatusPill({ u }: { u: TeamUser }) {
  if (u.watching) {
    return (
      <span className="inline-flex shrink-0 items-center gap-0.5 rounded bg-violet-500/15 px-1 py-px text-[9px] font-bold uppercase leading-none text-violet-500 dark:text-violet-300">
        <Eye className="h-2.5 w-2.5" /> {u.watching}
      </span>
    );
  }
  if (!u.online) {
    return <span className="shrink-0 text-[9px] font-semibold uppercase leading-none text-slate-400">offline</span>;
  }
  if (u.idle) {
    return <span className="shrink-0 text-[9px] font-semibold uppercase leading-none text-slate-400">idle</span>;
  }
  // Actively labeling: say where (Platform = sky, Own = violet — same colours as the workspace switch).
  const own = u.workspace === "own";
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-0.5 rounded px-1 py-px text-[9px] font-extrabold uppercase leading-none",
        own
          ? "bg-violet-500/20 text-violet-600 dark:text-violet-300"
          : "bg-sky-500/20 text-sky-600 dark:text-sky-300",
      )}
      title={own ? "Labeling in the Own workspace" : "Labeling in the Platform workspace"}
    >
      ✎ {own ? "Own" : "Platform"}
    </span>
  );
}

function UserCard({ u, isMe, labeled, scopeLabel }: { u: TeamUser; isMe: boolean; labeled: number; scopeLabel?: string }) {
  // "hot" for a few seconds after this person saves an image (their "today" count goes up).
  const [hotUntil, setHotUntil] = useState(0);
  const prevToday = useRef(u.today);
  useEffect(() => {
    if (u.today > prevToday.current) setHotUntil(Date.now() + HOT_MS);
    prevToday.current = u.today;
  }, [u.today]);
  useEffect(() => {
    if (hotUntil <= Date.now()) return;
    const t = window.setTimeout(() => setHotUntil(0), hotUntil - Date.now());
    return () => window.clearTimeout(t);
  }, [hotUntil]);

  const working = u.online && !u.idle && !u.watching;
  const hot = working && hotUntil > Date.now();
  return (
    <div
      className={cn(
        "team-card flex h-7 min-w-0 shrink-0 items-center gap-1.5 rounded-md border px-2 transition-colors",
        isMe ? "border-slate-300 bg-slate-50 dark:border-slate-600 dark:bg-slate-800/60" : "border-slate-200 dark:border-slate-700",
        !u.online && "opacity-55",
        hot ? "team-card--hot" : working ? "team-card--working" : "",
      )}
      style={{ ["--tc" as string]: u.color }}
      title={[
        `${u.name}${isMe ? " (you)" : ""}`,
        u.label ? `on ${u.label}` : "",
        `${labeled} labeled${scopeLabel ? ` in ${scopeLabel.toLowerCase()}` : " total"}`,
        `${u.today} today · ${u.per_hour}/h`,
      ]
        .filter(Boolean)
        .join(" · ")}
    >
      <span className="relative flex h-2 w-2 shrink-0">
        {working && (
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-70" style={{ backgroundColor: u.color }} />
        )}
        <span className="relative h-2 w-2 rounded-full" style={{ backgroundColor: u.color }} />
      </span>
      <span className="max-w-[90px] truncate text-[11px] font-bold leading-none text-slate-800 dark:text-slate-100">{u.name}</span>
      {isMe && <span className="text-[9px] font-semibold leading-none text-slate-400">you</span>}
      <StatusPill u={u} />
      <span className="ml-0.5 text-[13px] font-extrabold leading-none tabular-nums text-slate-900 dark:text-white">{labeled}</span>
      <span className="text-[9px] leading-none tabular-nums text-slate-500 dark:text-slate-400">
        {u.today}
        <span className="opacity-70"> today</span> · {u.per_hour}/h
      </span>
    </div>
  );
}

/** Top bar (one slim row): who is online/doing what, and overall progress in real time. */
function TeamBar({
  total,
  reviewed,
  scopeLabel,
  perUser,
  onLogout,
}: {
  total: number;
  reviewed: number;
  /** e.g. "Day 2026-09-08": progress is for this cluster only, not the whole pool. */
  scopeLabel?: string;
  /** Done images per person inside the scope (when scoped). */
  perUser?: Record<string, number>;
  onLogout: () => void;
}) {
  const me = useTeam((s) => s.me);
  const users = useTeam((s) => s.users);
  const connected = useTeam((s) => s.connected);

  const stats = useMemo(() => {
    const countOf = (u: TeamUser) => (perUser ? perUser[u.name] ?? 0 : u.labeled);
    const attributed = users.reduce((n, u) => n + countOf(u), 0);
    const other = Math.max(0, reviewed - attributed);
    const rate = users.reduce((n, u) => n + (u.online ? u.per_hour : 0), 0);
    const remaining = Math.max(0, total - reviewed);
    return { other, rate, remaining, pct: total > 0 ? (reviewed / total) * 100 : 0 };
  }, [users, reviewed, total, perUser]);

  const ordered = useMemo(() => {
    const mine = users.filter((u) => u.name === me?.user);
    // Offline people with no work today are just noise: hide them.
    const rest = users.filter((u) => u.name !== me?.user && (u.online || u.today > 0));
    if (me && mine.length === 0) {
      mine.push({
        name: me.user,
        color: me.color,
        online: true,
        item_key: null,
        label: "",
        workspace: "",
        idle: false,
        watching: null,
        watchers: [],
        labeled: 0,
        today: 0,
        per_hour: 0,
      });
    }
    return [...mine, ...rest];
  }, [users, me]);

  return (
    <div className="flex h-[36px] w-full shrink-0 items-center gap-2.5 overflow-x-auto rounded-lg border border-slate-200 bg-white/95 px-2.5 shadow-soft dark:border-slate-700/60 dark:bg-ink-900/90">
      <div className="flex shrink-0 items-center gap-1.5">
        {ordered.map((u) => (
          <UserCard
            key={u.name}
            u={u}
            isMe={u.name === me?.user}
            labeled={perUser ? perUser[u.name] ?? 0 : u.labeled}
            scopeLabel={scopeLabel}
          />
        ))}
      </div>

      <div className="flex min-w-[260px] flex-1 items-center gap-2">
        <span className="shrink-0 text-[10px] font-semibold tabular-nums text-slate-600 dark:text-slate-300">
          {scopeLabel ? <span className="mr-1 text-slate-400">{scopeLabel} ·</span> : null}
          Done {reviewed}/{total} · {stats.pct.toFixed(1)}%
        </span>
        <div className="flex h-2 min-w-[60px] flex-1 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700" title="Progress by person">
          {total > 0 &&
            users
              .filter((u) => (perUser ? perUser[u.name] ?? 0 : u.labeled) > 0)
              .map((u) => {
                const n = perUser ? perUser[u.name] ?? 0 : u.labeled;
                return <div key={u.name} style={{ width: `${(Math.min(n, total) / total) * 100}%`, backgroundColor: u.color }} />;
              })}
          {total > 0 && stats.other > 0 && (
            <div className="bg-slate-400/70" style={{ width: `${(stats.other / total) * 100}%` }} title="Labeled before teams were tracked" />
          )}
        </div>
        <span className="shrink-0 text-[10px] tabular-nums text-slate-500 dark:text-slate-400">
          {stats.remaining} left{stats.rate > 0 ? ` · ${stats.rate.toFixed(0)}/h ${fmtEta(stats.remaining / stats.rate)}` : ""}
        </span>
      </div>

      {!connected && (
        <span
          className="shrink-0 rounded bg-amber-500/90 px-1.5 py-0.5 text-[10px] font-bold text-amber-950"
          title="Can't reach the server right now (it may be busy). Your labels are kept and are saved automatically as soon as it answers. Team progress is paused until then."
        >
          reconnecting…
        </span>
      )}
      <KeysButton />
      <Link
        to="/collected"
        title="Collected miner solutions (beverage)"
        className="flex shrink-0 items-center gap-1 rounded-md border border-slate-300 px-1.5 py-1 text-[10px] font-bold text-slate-600 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-800"
      >
        <Library className="h-3 w-3" /> Collected
      </Link>
      <Link
        to="/studio"
        title="Studio: generate challenge-look training images (OpenAI / Gemini)"
        className="flex shrink-0 items-center gap-1 rounded-md border border-violet-300 px-1.5 py-1 text-[10px] font-bold text-violet-700 hover:bg-violet-50 dark:border-violet-700 dark:text-violet-300 dark:hover:bg-violet-950/30"
      >
        <Sparkles className="h-3 w-3" /> Studio
      </Link>
      <button
        type="button"
        onClick={onLogout}
        title={`Sign out${me ? ` (${me.user})` : ""}`}
        className="flex shrink-0 items-center rounded-md border border-slate-300 p-1 text-slate-600 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-800"
      >
        <LogOut className="h-3 w-3" />
      </button>
    </div>
  );
}

export default memo(TeamBar);
