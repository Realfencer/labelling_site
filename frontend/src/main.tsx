import { StrictMode, useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes, useParams, useSearchParams } from "react-router-dom";
import App from "./App";
import AuthGate from "./components/AuthGate";
import ChutesBalanceToast from "./components/ChutesBalanceToast";
import TopicsPage from "./TopicsPage";
import { api, type AuthStatus } from "./api";
import "./index.css";

function LabelerRoute() {
  const { slug } = useParams();
  const [searchParams] = useSearchParams();
  const workspace = searchParams.get("workspace");
  if (!slug) return <Navigate to="/" replace />;
  return <App topicSlug={slug} initialWorkspace={workspace === "own" || workspace === "platform" ? workspace : undefined} />;
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
        admin_email: "matthewsliger0302@gmail.com",
        smtp_configured: false,
      });
    }
  }, []);

  useEffect(() => {
    void refreshAuth();
  }, [refreshAuth]);

  useEffect(() => {
    const onNeedAuth = () => {
      setAuth((prev) =>
        prev
          ? { ...prev, authenticated: false, email: null }
          : { enabled: true, authenticated: false, admin_email: "matthewsliger0302@gmail.com" },
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
        <Route path="/:slug" element={<LabelerRoute />} />
        <Route path="/t/:slug" element={<LabelerRoute />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <ChutesBalanceToast />
    </BrowserRouter>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);
