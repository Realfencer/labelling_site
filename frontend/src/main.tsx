import { StrictMode, Suspense, lazy, useCallback, useEffect, useReducer, useRef, useState, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes, useLocation, useParams, useSearchParams } from "react-router-dom";
import AuthGate from "./components/AuthGate";
import BrandLoader from "./components/BrandLoader";
import KeysModal from "./components/KeysModal";
import SiteBackdrop from "./components/SiteBackdrop";
import { startTopicsPrefetch } from "./prefetch";
import { useKeys } from "./store/useKeys";
import TopicsPage from "./TopicsPage";
// The labeler is ~40% of the JavaScript: load it only when a workspace is opened (prefetched while idle).
const App = lazy(() => import("./App"));
const CollectedPage = lazy(() => import("./pages/CollectedPage"));
const StudioPage = lazy(() => import("./pages/StudioPage"));
import { api, type AuthStatus } from "./api";
import "./index.css";

// Images fade in when their pixels arrive (first load of each <img> only: later src swaps stay instant). `load` does not
// bubble, so listen in the capture phase.
document.addEventListener(
  "load",
  (e) => {
    const t = e.target;
    if (t instanceof HTMLImageElement && !t.dataset.fx && !t.hasAttribute("data-nofade")) {
      t.dataset.fx = "1";
      t.classList.add("img-fade");
      t.addEventListener("animationend", () => t.classList.remove("img-fade"), { once: true });
    }
  },
  true,
);

// Ask for the topic list now, together with the auth check (they used to run one after the other).
startTopicsPrefetch();

/**
 * /t/<topic>                      Platform
 * /t/<topic>/own                  Own workspace (as you left it)
 * /t/<topic>/own/days             Own: the day-selection page
 * /t/<topic>/own/days/2026-10-07  Own: that day (every day has its own link)
 * /t/<topic>/own/days/review      Own: the images the label audit flagged, riskiest first
 * /t/<topic>/own/openai/days[/…]  Own: only the images OpenAI made (…/own/gemini/… for Gemini)
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
  // /own/<openai|gemini>/days[/<date>] = one generator's images; /own/days[/<date>] = all
  const field = pathOwn && (parts[1] === "openai" || parts[1] === "gemini") ? parts[1] : undefined;
  const after = pathOwn ? parts.slice(field ? 2 : 1) : [];
  const dayPart = after[0] === "days" ? after[1] : undefined;
  // a calendar day, or "review" (the images the label audit flagged)
  const day = dayPart && (/^\d{4}-\d{2}-\d{2}$/.test(dayPart) || dayPart === "review") ? dayPart : undefined;
  const daysPage = pathOwn && after[0] === "days" && !day;
  return (
    <Suspense fallback={<BrandLoader label="Opening workspace" hint="Loading the labeling tools" accent="emerald" />}>
      <App topicSlug={slug} initialWorkspace={workspace} initialOwnDay={day} initialOwnField={field} ownDaysPage={daysPage} />
    </Suspense>
  );
}

/** A fade for a lazy page that has just arrived (the loader fades out with the Suspense fallback, the page fades in). */
function PageFade({ children }: { children: ReactNode }) {
  return (
    <div className="rt-in">{children}</div>
  );
}

/** Which screen family a path belongs to: moving WITHIN a family (e.g. labeler -> Own days -> a day) never animates. */
function routeGroup(pathname: string): string {
  const seg = pathname.split("/")[1] || "";
  if (seg === "") return "home";
  if (seg === "collected" || seg === "studio") return seg;
  return "labeler";
}

const EXIT_MS = 120;

/**
 * Routes with a smooth hand-over: when the screen family changes, the current page fades out (120 ms) and the next one
 * fades in over the persistent 3D backdrop. Inside one family the location updates synchronously, so the labeler
 * keeps its state and never remounts or flickers.
 */
function AppRoutes() {
  const location = useLocation();
  const shown = useRef(location);
  const [, bump] = useReducer((x: number) => x + 1, 0);
  const sameGroup = routeGroup(shown.current.pathname) === routeGroup(location.pathname);
  if (sameGroup) shown.current = location;
  const leaving = !sameGroup;
  const group = routeGroup(shown.current.pathname);
  useEffect(() => {
    if (!leaving) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const t = window.setTimeout(() => {
      shown.current = location;
      bump();
    }, reduce ? 0 : EXIT_MS);
    return () => window.clearTimeout(t);
  }, [leaving, location]);
  return (
    <div
      key={group}
      // the labeler is a full-height app (children use h-full); the other screens grow with their content, so their
      // sticky headers keep working past the first screen
      className={`${leaving ? "rt-out" : "rt-in"}${group === "labeler" ? " rt-fill" : ""}`}
    >
    <Routes location={shown.current}>
      <Route path="/" element={<TopicsPage />} />
      <Route
        path="/collected"
        element={
          <Suspense fallback={<BrandLoader label="Collected miners" hint="Scores and latency checks" accent="amber" />}>
            <PageFade><CollectedPage /></PageFade>
          </Suspense>
        }
      />
      <Route
        path="/studio"
        element={
          <Suspense fallback={<BrandLoader label="Studio" hint="Image generation workspace" accent="fuchsia" />}>
            <PageFade><StudioPage /></PageFade>
          </Suspense>
        }
      />
      <Route path="/:slug" element={<LabelerRoute />} />
      <Route path="/t/:slug/*" element={<LabelerRoute />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
    </div>
  );
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

  // One persistent backdrop for every branch below (loader, sign-in, app): it only draws when a page asks for it.
  if (!auth) {
    return (
      <>
        <SiteBackdrop />
        <BrandLoader label="Checking access" hint="Securing your session" accent="sky" />
      </>
    );
  }

  if (auth.enabled && !auth.authenticated) {
    return (
      <>
        <SiteBackdrop />
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
      <SiteBackdrop />
      <AppRoutes />
      <KeysModal />
    </BrowserRouter>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);
