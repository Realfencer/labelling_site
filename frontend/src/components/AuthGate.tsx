import { useState, type FormEvent } from "react";
import { Loader2, Lock, ShieldCheck } from "lucide-react";
import BrandMark from "./BrandMark";
import { useSiteBackdrop } from "../store/useBackdrop";
import { api, type AuthStatus } from "../api";
import { cn } from "../lib/utils";

export default function AuthGate({
  onAuthenticated,
}: {
  status?: AuthStatus;
  onAuthenticated: (next: AuthStatus) => void;
}) {
  useSiteBackdrop("hero");
  const [username, setUsername] = useState(() => {
    try {
      return localStorage.getItem("labeler_username") || "";
    } catch {
      return "";
    }
  });
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const login = async (e?: FormEvent) => {
    e?.preventDefault();
    setBusy(true);
    setError("");
    try {
      await api.login(password, username.trim());
      try {
        localStorage.setItem("labeler_username", username.trim());
      } catch {
        /* private mode */
      }
      const next = await api.authStatus();
      onAuthenticated(next);
    } catch (err) {
      setError(String(err).replace(/^Error:\s*/i, ""));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4 py-10">
      <div className="mb-6 flex flex-col items-center text-center">
        <div className="relative">
          <div className="absolute inset-2 rounded-[28px] bg-sky-500/30 blur-2xl" aria-hidden />
          <BrandMark size={88} className="relative drop-shadow-2xl" />
        </div>
        <div className="mt-4 text-xl font-extrabold text-white">Score Labeler</div>
        <div className="mt-0.5 text-[11px] font-bold uppercase tracking-[0.2em] text-slate-500">Beverage labeling · SN44</div>
      </div>
      <div className="w-full max-w-md rounded-2xl border border-slate-700/80 bg-[#12161e] p-6 shadow-2xl">
        <div className="mb-5 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-sky-500 text-white">
            <ShieldCheck className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-lg font-extrabold text-white">Sign in</h1>
            <p className="text-[12px] text-slate-400">Pick your name (your partner sees it on your work) and enter the shared password.</p>
          </div>
        </div>

        <form className="space-y-3" onSubmit={(e) => void login(e)}>
          <label className="block">
            <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">Your name</span>
            <input
              type="text"
              required
              minLength={2}
              maxLength={24}
              autoComplete="username"
              autoFocus={!username}
              value={username}
              onChange={(ev) => setUsername(ev.target.value)}
              disabled={busy}
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-sky-500"
              placeholder="e.g. Alex"
            />
          </label>

          <label className="block">
            <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">Password</span>
            <input
              type="password"
              required
              autoComplete="current-password"
              autoFocus={Boolean(username)}
              value={password}
              onChange={(ev) => setPassword(ev.target.value)}
              disabled={busy}
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-sky-500"
              placeholder="••••••••"
            />
          </label>

          {error && <p className="text-[12px] leading-relaxed text-rose-300">{error}</p>}

          <button
            type="submit"
            disabled={busy || !password || username.trim().length < 2}
            className={cn(
              "inline-flex w-full items-center justify-center gap-2 rounded-lg bg-sky-600 px-3 py-2.5 text-sm font-bold text-white hover:bg-sky-500 disabled:opacity-40",
            )}
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Lock className="h-4 w-4" />}
            {busy ? "Please wait…" : "Unlock"}
          </button>
        </form>
      </div>
    </div>
  );
}
