import { useState, type FormEvent } from "react";
import { Loader2, Mail, ShieldCheck } from "lucide-react";
import { api, type AuthStatus } from "../api";
import { cn } from "../lib/utils";

export default function AuthGate({
  status,
  onAuthenticated,
}: {
  status: AuthStatus;
  onAuthenticated: (next: AuthStatus) => void;
}) {
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [requested, setRequested] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");

  const requestCode = async (e?: FormEvent) => {
    e?.preventDefault();
    setBusy(true);
    setError("");
    setInfo("");
    try {
      const res = await api.requestAccess(email.trim());
      setRequested(true);
      setInfo(res.message);
    } catch (err) {
      setError(String(err).replace(/^Error:\s*/i, ""));
    } finally {
      setBusy(false);
    }
  };

  const verify = async (e?: FormEvent) => {
    e?.preventDefault();
    setBusy(true);
    setError("");
    try {
      await api.verifyAccess(email.trim(), code.trim());
      const next = await api.authStatus();
      onAuthenticated(next);
    } catch (err) {
      setError(String(err).replace(/^Error:\s*/i, ""));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-[#0b1220] px-4 py-10">
      <div className="w-full max-w-md rounded-2xl border border-slate-700/80 bg-[#12161e] p-6 shadow-2xl">
        <div className="mb-5 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-sky-500 text-white">
            <ShieldCheck className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-lg font-extrabold text-white">Request access</h1>
            <p className="text-[12px] text-slate-400">
              Codes go to the site owner ({status.admin_email || "admin"}). After approval, enter your code here.
            </p>
          </div>
        </div>

        {!status.smtp_configured && (
          <div className="mb-4 rounded-lg border border-amber-700/50 bg-amber-950/40 px-3 py-2 text-[12px] text-amber-100">
            Email sending is not configured yet on the server (SMTP). Access requests cannot be delivered until the
            owner sets AUTH_SMTP_USER / AUTH_SMTP_PASSWORD.
          </div>
        )}

        <form className="space-y-3" onSubmit={(e) => void (requested ? verify(e) : requestCode(e))}>
          <label className="block">
            <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">Your email</span>
            <input
              type="email"
              required
              autoComplete="email"
              value={email}
              onChange={(ev) => setEmail(ev.target.value)}
              disabled={busy}
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-sky-500"
              placeholder="you@example.com"
            />
          </label>

          {requested && (
            <label className="block">
              <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-slate-500">
                Verification code
              </span>
              <input
                inputMode="numeric"
                pattern="[0-9]*"
                maxLength={6}
                required
                value={code}
                onChange={(ev) => setCode(ev.target.value.replace(/\D/g, "").slice(0, 6))}
                disabled={busy}
                className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 font-mono text-lg tracking-[0.35em] text-white outline-none focus:border-sky-500"
                placeholder="••••••"
              />
            </label>
          )}

          {info && <p className="text-[12px] leading-relaxed text-sky-300">{info}</p>}
          {error && <p className="text-[12px] leading-relaxed text-rose-300">{error}</p>}

          <button
            type="submit"
            disabled={busy || !email.trim() || (requested && code.length !== 6)}
            className={cn(
              "inline-flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2.5 text-sm font-bold text-white disabled:opacity-40",
              requested ? "bg-emerald-600 hover:bg-emerald-500" : "bg-sky-600 hover:bg-sky-500",
            )}
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : requested ? <ShieldCheck className="h-4 w-4" /> : <Mail className="h-4 w-4" />}
            {busy ? "Please wait…" : requested ? "Verify & enter" : "Request access"}
          </button>
        </form>

        {requested && (
          <button
            type="button"
            disabled={busy}
            onClick={() => void requestCode()}
            className="mt-3 w-full text-center text-[11px] font-semibold text-slate-400 hover:text-sky-300"
          >
            Resend request to owner
          </button>
        )}
      </div>
    </div>
  );
}
