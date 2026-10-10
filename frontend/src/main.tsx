import { StrictMode, Suspense, lazy, useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes, useParams, useSearchParams } from "react-router-dom";
import App from "./App";
import AuthGate from "./components/AuthGate";
import KeysModal from "./components/KeysModal";
import { useKeys } from "./store/useKeys";
import TopicsPage from "./TopicsPage";
const CollectedPage = lazy(() => import("./pages/CollectedPage"));
import { api, type AuthStatus } from "./api";
import "./index.css";

/**
 * /t/<topic>                      Platform
 * /t/<topic>/own                  Own workspace (as you left it)
 * /t/<topic>/own/days             Own: the day-selection page
 * /t/<topic>/own/days/2026-10-07  Own: that day (every day has its own link)
 * (?workspace=own|platform on /t/<topic> keeps working.) One route with a splat, so moving between these
 * does not remount the labeler.
 */
function LabelerRoute() {
  const { slug, "*": rest } = useParams();
  const [searchParams] = useSearchParams();
  if (!slug) return <Navigate to="/" replace />;
  const parts = (rest || "").split("/").filter(Boolean);
  const pathOwn = parts[0] === "own";
  const q = searchParams.get("workspace");
  const workspace = pathOwn ? "own" : q === "own" || q === "platform" ? q : undefined;
  const dayPart = pathOwn && parts[1] === "days" ? parts[2] : undefined;
  const day = dayPart && /^\d{4}-\d{2}-\d{2}$/.test(dayPart) ? dayPart : undefined;
  const daysPage = pathOwn && parts[1] === "days" && !day;
  return <App topicSlug={slug} initialWorkspace={workspace} initialOwnDay={day} ownDaysPage={daysPage} />;
}

function Root() {
  const [auth, setAuth] = useState<AuthStatus | null>(null);
  const [authError, setAuthError] = useState("");

  const refreshAuth = useCallback(async () => {
    try {
      const status = await api.authStatus();
      setAuth(status);
      setAuthError("");
    } catch (e) {
      setAuthError(String(e).replace(/^Error:\s*/i, ""));
      setAuth({
        enabled: true,
        authenticated: false,
      });
    }
  }, []);

  useEffect(() => {
    void refreshAuth();
  }, [refreshAuth]);

  // Shared API keys (masked status only) — loaded once we are signed in.
  const signedIn = Boolean(auth && (!auth.enabled || auth.authenticated));
  useEffect(() => {
    if (signedIn) void useKeys.getState().refresh();
  }, [signedIn]);

  useEffect(() => {
    const onNeedAuth = () => {
      setAuth((prev) =>
        prev
          ? { ...prev, authenticated: false, email: null }
          : { enabled: true, authenticated: false },
      );
    };
    window.addEventListener("labeler-auth-required", onNeedAuth);
    return () => window.removeEventListener("labeler-auth-required", onNeedAuth);
  }, []);

  if (!auth) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#0b1220] text-sm text-slate-400">
        Checking access…
      </div>
    );
  }

  if (auth.enabled && !auth.authenticated) {
    return (
      <>
        {authError && (
          <div className="fixed left-0 right-0 top-0 z-50 bg-rose-950/90 px-3 py-2 text-center text-[12px] text-rose-100">
            {authError}
          </div>
        )}
        <AuthGate status={auth} onAuthenticated={setAuth} />
      </>
    );
  }

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<TopicsPage />} />
        <Route
          path="/collected"
          element={
            <Suspense fallback={<div className="flex min-h-screen items-center justify-center bg-[#0b1220] text-sm text-slate-400">Loading…</div>}>
              <CollectedPage />
            </Suspense>
          }
        />
        <Route path="/:slug" element={<LabelerRoute />} />
        <Route path="/t/:slug/*" element={<LabelerRoute />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <KeysModal />
    </BrowserRouter>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);
