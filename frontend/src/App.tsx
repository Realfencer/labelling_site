import BrandLoader from "./components/BrandLoader";
import { useSiteBackdrop } from "./store/useBackdrop";
import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState, type ComponentProps, type ReactNode } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import {
  CloudDownload,
  CloudUpload,
  Crosshair,
  Download,
  FolderUp,
  Upload,
  Moon,
  Rocket,
  Sparkles,
  GitCompareArrows,
  Square,
  SquareDashedMousePointer,
  Sun,
  Trash2,
  Zap,
  Loader2,
  Send,
} from "lucide-react";
import { api, workspaceFromState, HttpError, type OwnField, type PeekInfo, type ReviewIssue, type WorkspaceMode } from "./api";
import TeamBar from "./components/TeamBar";
import ImageStatusBar from "./components/ImageStatusBar";
import StripStatusBar, { STRIP_CODE, type StripCodes } from "./components/StripStatusBar";
import { useTeam } from "./store/useTeam";
import { useKeys } from "./store/useKeys";
import { useTeamSync } from "./hooks/useTeamSync";
import { dropPendingSave, flushOnUnload, hasPendingSaves, retrySave, setSaveConflictHandler } from "./store/saveRetry";
import AnnotationCanvas, { type Callout, type ViewShare } from "./components/AnnotationCanvas";
import ReviewPanel, { REVIEW_COLOR, REVIEW_ICON } from "./components/ReviewPanel";
import { useReviewSync, syncReviewNow } from "./hooks/useReviewSync";
import { useReview, reviewIsOpen } from "./store/useReview";
const FindBestModal = lazy(() => import("./components/FindBestModal"));
const ManualMergeModal = lazy(() => import("./components/ManualMergeModal"));
import SubmitModeModal from "./components/SubmitModeModal";
const UploadLabelsModal = lazy(() => import("./components/UploadLabelsModal"));
const DownloadGtModal = lazy(() => import("./components/DownloadGtModal"));
const OwnSam3Modal = lazy(() => import("./components/OwnSam3Modal"));
const SendGtModal = lazy(() => import("./components/SendGtModal"));
const UploadUltralyticsModal = lazy(() => import("./components/UploadUltralyticsModal"));
const DeepPostProcModal = lazy(() => import("./components/DeepPostProcModal"));
const CompareHfModelsModal = lazy(() => import("./components/CompareHfModelsModal"));
const SelectBaseModelModal = lazy(() => import("./components/SelectBaseModelModal"));
const OwnDeleteRangeModal = lazy(() => import("./components/OwnDeleteRangeModal"));
const OwnUploadModal = lazy(() => import("./components/OwnUploadModal"));
const OwnHfImportModal = lazy(() => import("./components/OwnHfImportModal"));
import OwnDayList from "./components/OwnDayList";
import { defaultWorkspaceMetrics, type OwnFilter } from "./components/WorkspaceBar";
import WorkspaceSwitch from "./components/WorkspaceSwitch";
import FilmstripThumb from "./components/FilmstripThumb";
import { clearDetailCache, dropCachedDetail, getCachedDetail, getCachedDetailByItemKey, preloadDetails, setCachedDetail } from "./store/detailCache";
import {
  challengePreviewUrl,
  challengeThumbnailUrl,
  clearDecodedImages,
  preloadImage,
} from "./store/imageLoader";
import {
  clearThumbnailBlobCache,
  primeDecodedThumbnail,
  scheduleBatchStripLoad,
} from "./store/thumbnailLoader";
import { useLabeler } from "./store/useLabeler";
import type { AppState, ChallengeDetail, ChallengeItem, DetectionBox, TopVsLeftScore } from "./types";
import { comparePredToGt, scoreTopAgainstLeftGt } from "./utils/gtScore";
import { itemKeysMatch, stateMatchesTopic, topicSlugFromSkill } from "./utils/skillSlug";
import { getTabSessionId } from "./sessionId";
import { boxFromDict, challengeImageUrl, colorForClass, preloadChallengeForward, preloadChallengeUrgent } from "./utils/colors";
import { cn } from "./lib/utils";
import DeleteImageConfirm from "./components/DeleteImageConfirm";

/** item_keys marked Done in the UI before saveBoxes / SSE has caught up. */
const pendingDoneKeys = new Set<string>();
/** item_keys cleared by Again (A) so sticky Done cannot revive them. */
const pendingNewKeys = new Set<string>();

function notePendingDone(itemKey: string | undefined | null) {
  if (!itemKey) return;
  pendingDoneKeys.add(itemKey);
  pendingNewKeys.delete(itemKey);
}

function notePendingNew(itemKey: string | undefined | null) {
  if (!itemKey) return;
  pendingNewKeys.add(itemKey);
  pendingDoneKeys.delete(itemKey);
}

function challengeItemKey(
  ch: { item_key?: string | null } | null | undefined,
  fallback?: string | null,
): string | null {
  return ch?.item_key || fallback || null;
}

/** URL topic this browser is labeling — never adopt another user's skill over SSE/API. */
let pinnedTopicSlug: string | null = null;

function skillPathFromItemKey(itemKey: string | null | undefined): string | null {
  if (!itemKey) return null;
  // Platform keys are "skill:block". Own keys are bare stems (e.g. local_8769600_v00)
  // — never treat those as a skill path or applyDetail / applyState will drop them.
  const cut = itemKey.lastIndexOf(":");
  if (cut <= 0) return null;
  return itemKey.slice(0, cut);
}

function payloadMatchesPinnedTopic(state: { active_skill?: { element_path?: string } | null; challenge?: ChallengeItem | null }): boolean {
  if (!pinnedTopicSlug) return true;
  // Own pool items are topic-scoped by the active project, not by item_key skill prefix.
  if (state.challenge?.source_type === "local") return true;
  if (state.active_skill?.element_path) {
    return stateMatchesTopic(state, pinnedTopicSlug);
  }
  const skillPath = skillPathFromItemKey(state.challenge?.item_key);
  if (!skillPath) return true;
  return stateMatchesTopic({ active_skill: { element_path: skillPath } }, pinnedTopicSlug);
}

function applyState(s: AppState) {
  if (!payloadMatchesPinnedTopic(s)) return;
  const prev = useLabeler.getState().appState;
  const stickyChecks = readStripChecksStore(pinnedTopicSlug);
  // Never merge Platform ↔ Own challenge lists — they are private pools.
  const modeChanged = Boolean(prev && prev.review_mode !== s.review_mode);
  if (
    prev &&
    !modeChanged &&
    prev.session_id === s.session_id &&
    Array.isArray(prev.challenges) &&
    Array.isArray(s.challenges)
  ) {
    const prevByKey = new Map(prev.challenges.map((c) => [c.item_key, c]));
    const findOld = (itemKey: string) => {
      const direct = prevByKey.get(itemKey);
      if (direct) return direct;
      return prev.challenges.find((c) => itemKeysMatch(c.item_key, itemKey));
    };
    const challenges = s.challenges.map((c) => {
      const old = findOld(c.item_key);
      const sticky = stickyChecks.get(c.item_key);
      const oldLeft = normalizeBoxes(old?.cached_left_boxes);
      const newLeft = normalizeBoxes(c.cached_left_boxes);
      const serverBoxesChanged = Boolean(old) && !boxesEqual(oldLeft, newLeft);
      const serverReviewChanged = Boolean(old) && Boolean(old!.reviewed) !== Boolean(c.reviewed);
      const trustServer = serverBoxesChanged || serverReviewChanged;
      const top = c.top_boxes?.length ? c.top_boxes : old?.top_boxes;
      const leftRaw = trustServer
        ? (c.cached_left_boxes ?? [])
        : c.cached_left_boxes?.length
          ? c.cached_left_boxes
          : (old?.cached_left_boxes ?? []);
      const left = leftRaw;
      const hasPendingDone = pendingDoneKeys.has(c.item_key);
      const hasPendingNew = pendingNewKeys.has(c.item_key);

      // Strip NEW/DONE/SAM3 must track the user's labeling, not flicker from SSE / Own pool
      // reloads. Never regress Done→New unless Again (pendingNew) was pressed.
      // pendingDone (M) always wins — even if the server still briefly reports a SAM3 draft
      // before the .done marker is visible. Do not clear pendingDone on has_label drafts.
      let reviewed = Boolean(c.reviewed);
      let has_label = Boolean(c.has_label);
      if (hasPendingNew) {
        // Again (A): Done/SAM3 → New (winner). Keep pending until server drops both flags.
        reviewed = false;
        has_label = false;
        if (!c.reviewed && !c.has_label) pendingNewKeys.delete(c.item_key);
      } else if (hasPendingDone) {
        // M / explicit Done — keep Done until server confirms reviewed=true.
        reviewed = true;
        has_label = true;
        if (c.reviewed) pendingDoneKeys.delete(c.item_key);
      } else if (old?.reviewed && !c.reviewed && !c.has_label) {
        // Sticky Done only when server dropped reviewed without a SAM3 draft label.
        reviewed = true;
        has_label = true;
      } else {
        reviewed = Boolean(c.reviewed);
        // SAM3 draft = has_label && !reviewed; Done implies label.
        has_label = Boolean(c.has_label) || reviewed;
        // Trust server SAM3 drafts over sticky Done (no pending M).
        if (Boolean(c.has_label) && !c.reviewed) {
          reviewed = false;
          has_label = true;
        }
      }

      let checks_passed = c.checks_passed ?? null;
      let checks_total = c.checks_total ?? null;
      let checks = c.checks ?? null;
      const checksStale = Boolean(c.checks_kind && c.checks_kind !== CURRENT_CHECKS_KIND);
      if (checksStale) {
        checks_passed =
          old?.checks_kind === CURRENT_CHECKS_KIND
            ? (old?.checks_passed ?? null)
            : sticky?.checks_kind === CURRENT_CHECKS_KIND
              ? sticky.checks_passed
              : null;
        checks_total =
          old?.checks_kind === CURRENT_CHECKS_KIND
            ? (old?.checks_total ?? null)
            : sticky?.checks_kind === CURRENT_CHECKS_KIND
              ? sticky.checks_total
              : null;
        checks =
          old?.checks_kind === CURRENT_CHECKS_KIND
            ? (old?.checks ?? null)
            : sticky?.checks_kind === CURRENT_CHECKS_KIND
              ? (sticky.checks ?? null)
              : null;
      }
      if (!reviewed) {
        checks_passed = null;
        checks_total = null;
        checks = null;
      } else if (c.checks_kind === CURRENT_CHECKS_KIND && c.checks_passed != null) {
        checks_passed = c.checks_passed;
        checks_total = c.checks_total ?? null;
        checks = c.checks ?? null;
      } else if (old?.checks_passed != null && (c.checks_passed == null || hasPendingDone)) {
        // Keep prior n/N while server recompute is in flight — stops strip tags bouncing.
        checks_passed = old.checks_passed;
        checks_total = old.checks_total ?? checks_total;
        checks = old.checks ?? checks;
      } else if (sticky?.checks_passed != null && c.checks_passed == null) {
        // Survives full page refresh / server restart until disk hydrate catches up.
        checks_passed = sticky.checks_passed;
        checks_total = sticky.checks_total ?? checks_total;
        checks = sticky.checks ?? checks;
      } else if (c.checks_passed != null) {
        checks_passed = c.checks_passed;
        checks_total = c.checks_total ?? null;
        checks = c.checks ?? null;
      }
      if (
        !old &&
        top === c.top_boxes &&
        left === c.cached_left_boxes &&
        reviewed === c.reviewed &&
        has_label === Boolean(c.has_label) &&
        checks_passed === c.checks_passed
      ) {
        return c;
      }
      return {
        ...c,
        reviewed,
        has_label,
        top_boxes: top || c.top_boxes || [],
        cached_left_boxes: left,
        checks_passed,
        checks_total,
        checks,
        checks_kind: c.checks_kind ?? old?.checks_kind,
      };
    });
    const notReviewed = challenges.filter((c) => c.loaded && !c.reviewed).length;
    const ready = challenges.filter((c) => c.loaded && c.reviewed && !c.tuned).length;
    s = {
      ...s,
      challenges,
      metrics: {
        ...s.metrics,
        not_reviewed: notReviewed,
        ready_to_tune: ready,
      },
    };
  } else if (Array.isArray(s.challenges) && stickyChecks.size) {
    // Fresh session / page reload: restore strip n/N from sessionStorage until server hydrates.
    s = {
      ...s,
      challenges: s.challenges.map((c) => {
        if (!c.reviewed || c.checks_passed != null) return c;
        const sticky = stickyChecks.get(c.item_key);
        if (!sticky || sticky.checks_passed == null) return c;
        return {
          ...c,
          checks_passed: sticky.checks_passed,
          checks_total: sticky.checks_total ?? c.checks_total,
          checks: sticky.checks ?? c.checks,
          checks_kind: sticky.checks_kind || c.checks_kind,
        };
      }),
    };
  }
  useLabeler.getState().setAppState(s);
  if (Array.isArray(s.challenges)) writeStripChecksStore(pinnedTopicSlug, s.challenges);
}

function patchChallengeChecks(index: number, patch: Partial<ChallengeItem>) {
  const app = useLabeler.getState().appState;
  if (!app || index < 0 || index >= app.challenges.length) return;
  const ch = app.challenges[index];
  patchChallengeChecksByKey(ch.item_key, patch);
}

function patchChallengeChecksByKey(itemKey: string | null | undefined, patch: Partial<ChallengeItem>) {
  if (!itemKey) return;
  const app = useLabeler.getState().appState;
  if (!app) return;
  const idx = app.challenges.findIndex((c) => itemKeysMatch(c.item_key, itemKey));
  if (idx < 0) return;
  const ch = app.challenges[idx];
  if (
    ch.checks_passed === patch.checks_passed &&
    ch.checks_total === patch.checks_total &&
    ch.checks?.r1 === patch.checks?.r1 &&
    ch.checks?.r2 === patch.checks?.r2 &&
    ch.checks?.r3 === patch.checks?.r3 &&
    Boolean(ch.reviewed) === Boolean(patch.reviewed ?? ch.reviewed)
  ) {
    return;
  }
  const challenges = app.challenges.slice();
  challenges[idx] = { ...ch, ...patch };
  useLabeler.getState().setAppState({ ...app, challenges });
  writeStripChecksStore(pinnedTopicSlug, challenges);
}

function syncStripChecks(
  _index: number,
  ch: ChallengeItem,
  gt: DetectionBox[],
  variantsRaw?: unknown,
) {
  if (!ch.reviewed) return;
  const patch = checksPatchForItem(ch, gt, variantsRaw);
  patchChallengeChecksByKey(ch.item_key, { ...patch, reviewed: true });
}

/** Filmstrip slot order: filtered rows plus the open image when the filter would hide it. */
function filmstripIndices(
  app: AppState,
  stripFilter: StripFilter,
  pinIndex?: number,
  ownFilter: OwnFilter = "all",
): number[] {
  const indices = filteredStripIndices(app, stripFilter, ownFilter);
  if (pinIndex == null || pinIndex < 0 || !app.challenges[pinIndex]?.loaded) return indices;
  const c =
    app.review_mode === "local"
      ? app.challenges[pinIndex]
      : challengeForStrip(app.challenges[pinIndex], pinIndex);
  if (indices.includes(pinIndex)) return indices;
  if (!matchesFilmstripFilter(app, c, stripFilter, ownFilter)) return indices;
  return [...indices, pinIndex].sort((a, b) => a - b);
}

const LABEL_CHECK_DELTA = 0.06;

/** Holding G this long turns "watch" into "ask the team for help on this spot". */
const G_HOLD_MS = 300;
const CURRENT_CHECKS_KIND = "r123-v9";
const STRIP_CHECKS_STORE = "labeler_strip_checks_v1";

type StoredStripChecks = {
  checks_passed: number | null;
  checks_total: number | null;
  checks: ChallengeItem["checks"];
  checks_kind: string;
};

function stripChecksStoreKey(topic: string | null | undefined): string {
  return `${STRIP_CHECKS_STORE}:${topic || "default"}`;
}

const OWN_CURSOR_STORE = "labeler_own_cursor_v1";

function ownCursorStoreKey(topic: string | null | undefined): string {
  return `${OWN_CURSOR_STORE}:${topic || "default"}`;
}

function readOwnCursor(topic: string | null | undefined): string | null {
  if (!topic) return null;
  try {
    const raw = localStorage.getItem(ownCursorStoreKey(topic));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { item_key?: string };
    return parsed?.item_key ? String(parsed.item_key) : null;
  } catch {
    return null;
  }
}

function writeOwnCursor(topic: string | null | undefined, itemKey: string | null | undefined): void {
  if (!topic || !itemKey) return;
  try {
    localStorage.setItem(ownCursorStoreKey(topic), JSON.stringify({ item_key: itemKey }));
  } catch {
    /* ignore quota */
  }
}

function readStripChecksStore(topic: string | null | undefined): Map<string, StoredStripChecks> {
  const out = new Map<string, StoredStripChecks>();
  try {
    const raw = sessionStorage.getItem(stripChecksStoreKey(topic));
    if (!raw) return out;
    const parsed = JSON.parse(raw) as Record<string, StoredStripChecks>;
    for (const [key, value] of Object.entries(parsed || {})) {
      if (!value || value.checks_kind !== CURRENT_CHECKS_KIND) continue;
      if (value.checks_passed == null) continue;
      out.set(key, value);
    }
  } catch {
    /* ignore */
  }
  return out;
}

function writeStripChecksStore(topic: string | null | undefined, rows: ChallengeItem[]): void {
  try {
    const prev = readStripChecksStore(topic);
    for (const c of rows) {
      if (!c?.item_key || !c.reviewed) continue;
      if (c.checks_kind !== CURRENT_CHECKS_KIND || c.checks_passed == null) continue;
      prev.set(c.item_key, {
        checks_passed: c.checks_passed,
        checks_total: c.checks_total ?? null,
        checks: c.checks ?? null,
        checks_kind: CURRENT_CHECKS_KIND,
      });
    }
    const payload: Record<string, StoredStripChecks> = {};
    for (const [key, value] of prev) payload[key] = value;
    sessionStorage.setItem(stripChecksStoreKey(topic), JSON.stringify(payload));
  } catch {
    /* ignore */
  }
}

function checksTotal(c: ChallengeItem): number {
  if (c.checks_total != null && c.checks_total > 0) return c.checks_total;
  const slots = [c.checks?.r1, c.checks?.r2, c.checks?.r3];
  const active = slots.filter((v) => v !== null && v !== undefined);
  if (active.length > 0) return active.length;
  return 3;
}

function checksPassComplete(c: ChallengeItem): boolean {
  const total = checksTotal(c);
  if (total <= 0) return false;
  return (c.checks_passed ?? 0) >= total;
}

type StripFilter = "all" | "new" | "fail" | "mid" | "pass";
/** OpenAI / Gemini images (beverage) — module-level so every strip helper honours it without extra parameters. */
type GenFilter = "all" | "openai" | "gemini";
let genFilterNow: GenFilter = "all";

function matchesOwnFilter(c: ChallengeItem | undefined, filter: OwnFilter): boolean {
  if (!c?.loaded) return false;
  if (filter === "new") return !c.reviewed && !c.has_label;
  if (filter === "sam3") return !c.reviewed && Boolean(c.has_label);
  if (filter === "done") return Boolean(c.reviewed);
  return true;
}

function matchesFilmstripFilter(
  app: AppState,
  c: ChallengeItem | undefined,
  stripFilter: StripFilter,
  ownFilter: OwnFilter,
): boolean {
  if (genFilterNow !== "all" && c?.generator !== genFilterNow) return false;
  if (app.review_mode === "local") return matchesOwnFilter(c, ownFilter);
  return matchesStripFilter(c, stripFilter);
}

function stripBucket(c: ChallengeItem): "new" | "fail" | "mid" | "pass" {
  if (!c.reviewed) return "new";
  if (checksPassComplete(c)) return "pass";
  if (c.checks_passed === 0) return "fail";
  return "mid";
}

function minerPasses(
  pS: number | null | undefined,
  pM: number | null | undefined,
  pF: number | null | undefined,
  tS: number | null | undefined,
  tM: number | null | undefined,
  tF: number | null | undefined,
): boolean {
  if (pS == null || pM == null || pF == null || tS == null || tM == null || tF == null) return false;
  return (Math.abs(tS - pS) + Math.abs(tM - pM) + Math.abs(tF - pF)) / 3 <= LABEL_CHECK_DELTA;
}

function checksPatchForItem(ch: ChallengeItem, gt: DetectionBox[], variantsRaw?: unknown): Partial<ChallengeItem> {
  if (!gt.length) return { checks_passed: null, checks: null };
  let variants = normalizeReferenceVariants(variantsRaw);
  if (!variants.length) {
    const winner = (ch.top_boxes || []).map(boxFromDict);
    if (!winner.length) return { checks_passed: null, checks: null };
    const rec = ch.record || {};
    const n = (v: unknown) => {
      if (v == null || v === "") return null;
      const x = Number(v);
      return Number.isFinite(x) ? x : null;
    };
    variants = [
      {
        source: "top-winner",
        boxes: winner,
        total_weighted: n(rec.total_weighted),
        map50: n(rec.map50),
        false_positive: n(rec.false_positive),
        is_winner: true,
      },
    ];
  }
  const available = variants.filter((v) => v.boxes.length > 0).slice(0, 3);
  if (!available.length) return { checks_passed: null, checks: null, checks_total: null };
  const flags = available.map((v) => {
    const scored = scoreTopAgainstLeftGt(
      v.boxes,
      gt,
      {
        total_weighted: v.total_weighted,
        map50: v.map50,
        false_positive: v.false_positive,
      },
      ch.image_width || undefined,
      ch.image_height || undefined,
    );
    return minerPasses(
      v.total_weighted,
      v.map50,
      v.false_positive,
      scored?.composite,
      scored?.map50,
      scored?.false_positive,
    );
  });
  const slotFlags: Array<boolean | null> = [null, null, null];
  flags.forEach((flag, i) => {
    slotFlags[i] = flag;
  });
  return {
    checks_passed: flags.filter(Boolean).length,
    checks_total: available.length,
    checks: { r1: slotFlags[0], r2: slotFlags[1], r3: slotFlags[2] },
    checks_kind: CURRENT_CHECKS_KIND,
  };
}

/** Platform: NEW or reviewed but not 3/3. Own: NEW only (no miner n/3). */
function needsWork(c: ChallengeItem | undefined, app?: AppState | null): boolean {
  if (!c?.loaded) return false;
  if (!c.reviewed) return true;
  if (app?.review_mode === "local") return false;
  return c.checks_passed !== checksTotal(c);
}

function needsWorkStripIndices(app: AppState): number[] {
  const out: number[] = [];
  for (let i = 0; i < app.challenges.length; i += 1) {
    if (needsWork(app.challenges[i], app)) out.push(i);
  }
  return out;
}

/** N/B targets: always NEW / needs-work. Own Done filter must not empty the N/B list. */
function needsWorkNavIndices(app: AppState, _stripFilter: StripFilter, _ownFilter: OwnFilter): number[] {
  // Own Review: N/B (and the step after M) go to the next image that is still to check.
  const all =
    app.review_mode === "local" && app.own_day === "review"
      ? app.challenges.flatMap((c, i) => (c.loaded && reviewIsOpen(useReview.getState().items[c.item_key]) ? [i] : []))
      : needsWorkStripIndices(app);
  // Avoid duplicate work: N/B and auto-advance skip images a teammate has open right now.
  const claims = useTeam.getState().claims;
  if (!Object.keys(claims).length) return all;
  return all.filter((i) => !claims[app.challenges[i]?.item_key]);
}

function isAbortish(err: unknown): boolean {
  const msg = String(err).toLowerCase();
  return msg.includes("abort") || msg.includes("timed out") || msg.includes("timeout");
}

function isStaleChallengeError(err: unknown): boolean {
  const msg = String(err).toLowerCase();
  return (
    msg.includes("404") ||
    msg.includes("409") ||
    msg.includes("invalid challenge index") ||
    msg.includes("unknown item_key") ||
    msg.includes("not found") ||
    msg.includes("block mismatch")
  );
}

function neighborNeedsWorkIndex(
  from: number,
  dir: 1 | -1,
  stripFilter: StripFilter = "all",
  ownFilter: OwnFilter = "all",
): number {
  const app = useLabeler.getState().appState;
  if (!app) return -1;
  const indices = needsWorkNavIndices(app, stripFilter, ownFilter);
  let pos = indices.indexOf(from);
  if (pos < 0) {
    const key = app.challenges[from]?.item_key;
    if (key) pos = indices.findIndex((i) => itemKeysMatch(app.challenges[i]?.item_key, key));
  }
  let res = -1;
  if (pos < 0) {
    if (dir > 0) res = indices.find((i) => i > from) ?? -1;
    else {
      for (let k = indices.length - 1; k >= 0; k -= 1) {
        if (indices[k] < from) {
          res = indices[k];
          break;
        }
      }
    }
  } else {
    res = indices[pos + dir] ?? -1;
  }
  // Review: the list is worked until empty, so going past the end wraps to what is still left.
  if (res < 0 && app.review_mode === "local" && app.own_day === "review") {
    const rest = indices.filter((i) => i !== from);
    if (rest.length) res = dir > 0 ? rest[0] : rest[rest.length - 1];
  }
  return res;
}

function matchesStripFilter(c: ChallengeItem | undefined, filter: StripFilter): boolean {
  if (!c?.loaded) return false;
  if (filter === "all") return true;
  return stripBucket(c) === filter;
}

/** Strip tags/filters use server n/3 (whole-dataset pass runs in background on topic enter). */
function challengeForStrip(c: ChallengeItem, _i: number): ChallengeItem {
  if (!c?.loaded) return c;
  if (!c.reviewed) return { ...c, checks_passed: null, checks: null };
  return c;
}

function filteredStripIndices(app: AppState, stripFilter: StripFilter, ownFilter: OwnFilter = "all"): number[] {
  const out: number[] = [];
  for (let i = 0; i < app.challenges.length; i += 1) {
    const c = app.review_mode === "local" ? app.challenges[i] : challengeForStrip(app.challenges[i], i);
    if (matchesFilmstripFilter(app, c, stripFilter, ownFilter)) out.push(i);
  }
  return out;
}

function filteredNeighbors(from: number, stripFilter: StripFilter, count: number, ownFilter: OwnFilter = "all"): number[] {
  const app = useLabeler.getState().appState;
  if (!app) return [];
  const indices = filteredStripIndices(app, stripFilter, ownFilter);
  let pos = indices.indexOf(from);
  if (pos < 0) {
    const key = app.challenges[from]?.item_key;
    if (key) pos = indices.findIndex((i) => itemKeysMatch(app.challenges[i]?.item_key, key));
  }
  if (pos < 0) return [];
  const out: number[] = [];
  for (let k = 1; k <= count; k += 1) {
    if (indices[pos + k] != null) out.push(indices[pos + k]);
    if (indices[pos - k] != null) out.push(indices[pos - k]);
  }
  return out;
}

function stripBadge(c: ChallengeItem): { text: string; className: string; title: string } {
  if (!c.reviewed) {
    return { text: "new", className: "bg-amber-400 text-amber-950", title: `${c.display_name} · unlabeled` };
  }
  const r1 = c.checks?.r1;
  const r2 = c.checks?.r2;
  const r3 = c.checks?.r3;
  const total = checksTotal(c);
  const bits =
    r1 == null && r2 == null && r3 == null
      ? "checks pending"
      : [
          r1 == null ? null : `R1${r1 ? "✓" : "✗"}`,
          r2 == null ? null : `R2${r2 ? "✓" : "✗"}`,
          r3 == null ? null : `R3${r3 ? "✓" : "✗"}`,
        ]
          .filter(Boolean)
          .join(" ");
  const n = c.checks_passed;
  if (n == null) {
    return { text: `?/${total}`, className: "bg-slate-500 text-white", title: `${c.display_name} · Done · ${bits}` };
  }
  if (checksPassComplete(c)) {
    return { text: `${total}/${total}`, className: "bg-emerald-500 text-white", title: `${c.display_name} · Pass · ${bits}` };
  }
  if (n === 0) {
    return { text: `0/${total}`, className: "bg-rose-500 text-white", title: `${c.display_name} · Fail · ${bits}` };
  }
  return { text: `${n}/${total}`, className: "bg-amber-500 text-white", title: `${c.display_name} · Partial · ${bits}` };
}

function cloneBoxes(boxes: DetectionBox[]): DetectionBox[] {
  return boxes.map((b) => ({ ...b }));
}

function boxesEqual(a: DetectionBox[], b: DetectionBox[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    const x = a[i];
    const y = b[i];
    if (x.cls !== y.cls || x.x1 !== y.x1 || x.y1 !== y.y1 || x.x2 !== y.x2 || x.y2 !== y.y2) return false;
  }
  return true;
}

function normalizeBoxes(raw: unknown): DetectionBox[] {
  if (!Array.isArray(raw)) return [];
  return raw.map((b) => boxFromDict((b || {}) as Record<string, unknown>));
}

type RefVariant = {
  source: string;
  boxes: DetectionBox[];
  hotkey?: string;
  total_weighted?: number | null;
  map50?: number | null;
  false_positive?: number | null;
  composite_score?: number | null;
  is_winner?: boolean;
};

function normalizeReferenceVariants(raw: unknown): RefVariant[] {
  if (!Array.isArray(raw)) return [];
  const out: RefVariant[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const source = String((item as { source?: unknown }).source || "reference");
    const boxes = normalizeBoxes((item as { boxes?: unknown }).boxes);
    const rec = item as Record<string, unknown>;
    const hasScores =
      typeof rec.total_weighted === "number" ||
      typeof rec.map50 === "number" ||
      typeof rec.false_positive === "number";
    if (!boxes.length && !hasScores) continue;
    out.push({
      source,
      boxes,
      hotkey: typeof rec.hotkey === "string" ? rec.hotkey : undefined,
      total_weighted: typeof rec.total_weighted === "number" ? rec.total_weighted : null,
      map50: typeof rec.map50 === "number" ? rec.map50 : null,
      false_positive: typeof rec.false_positive === "number" ? rec.false_positive : null,
      composite_score: typeof rec.composite_score === "number" ? rec.composite_score : null,
      is_winner: Boolean(rec.is_winner),
    });
  }
  return out;
}

function referenceVariantsFromChallenge(ch: ChallengeItem | null | undefined): RefVariant[] {
  if (!ch) return [];
  const boxes = (ch.top_boxes || []).map(boxFromDict);
  if (!boxes.length) return [];
  const rec = (ch.record || {}) as Record<string, unknown>;
  return referenceVariantsFromBoxes(boxes, rec);
}

function referenceVariantsFromBoxes(boxes: DetectionBox[], rec: Record<string, unknown>): RefVariant[] {
  if (!boxes.length) return [];
  const n = (v: unknown) => {
    if (v == null || v === "") return null;
    const x = Number(v);
    return Number.isFinite(x) ? x : null;
  };
  return [
    {
      source: "top-winner",
      boxes,
      hotkey: typeof rec.hotkey === "string" ? rec.hotkey : undefined,
      total_weighted: n(rec.total_weighted),
      map50: n(rec.map50),
      false_positive: n(rec.false_positive),
      is_winner: true,
    },
  ];
}

function effectiveReferenceVariants(
  raw: unknown,
  challenge?: ChallengeItem | null,
  fallbackChallenge?: ChallengeItem | null,
  referenceBoxes?: DetectionBox[],
): RefVariant[] {
  const fromDetail = normalizeReferenceVariants(raw || []);
  if (fromDetail.length) return fromDetail;
  const fromCh = referenceVariantsFromChallenge(challenge ?? fallbackChallenge);
  if (fromCh.length) return fromCh;
  if (referenceBoxes?.length) {
    const rec = ((challenge ?? fallbackChallenge)?.record || {}) as Record<string, unknown>;
    return referenceVariantsFromBoxes(referenceBoxes, rec);
  }
  return [];
}

/** Per-image winner-miner defaults for instant Again (A). Keyed by item_key. */
const winnerPredCache = new Map<string, DetectionBox[]>();

/** Only cache preds that are actually winner-model output, not reviewed GT mis-tagged as preds. */
/** Winner-model snapshot for dual base-pane. Key A always re-infers fresh. */
function rememberWinnerPreds(
  itemKey: string | null | undefined,
  preds: DetectionBox[],
  left: DetectionBox[],
  reviewed: boolean,
) {
  if (!itemKey || !preds.length) return;
  // Always keep a winner-model snapshot for dual base-pane + Again (A).
  // Previously we skipped when Done GT == preds (accept-without-edit), which
  // left Own dual-view base detection and key A empty on new/unedited images.
  void left;
  void reviewed;
  winnerPredCache.set(itemKey, preds);
}

/** Winner-model defaults kept for dual-pane paint — key A always re-infers. */

function variantSetScore(variants: { boxes?: unknown[] }[] | null | undefined): number {
  if (!variants?.length) return 0;
  let boxed = 0;
  for (const v of variants) {
    if (Array.isArray(v.boxes) && v.boxes.length) boxed += 1;
  }
  return variants.length * 10 + boxed;
}

/** Prefer the richer R1/R2/R3 payload so a late R1-only response cannot wipe alts. */
function preferReferenceVariants<T extends { boxes?: unknown[] }>(
  incoming: T[] | null | undefined,
  previous: T[] | null | undefined,
  fallback: T[] = [],
): T[] {
  const a = incoming?.length ? incoming : [];
  const b = previous?.length ? previous : [];
  const c = fallback;
  const best = [a, b, c].sort((x, y) => variantSetScore(y) - variantSetScore(x))[0];
  return best.length ? best : [];
}

/**
 * Live R1/R2/R3 selection. Late disk/SSE payloads always ship winner (R1) as
 * `reference_boxes` / `top_boxes` — writers must re-apply this index or the right
 * pane snaps back to R1 while the middle rail still highlights R2/R3.
 */
const activeRefVariantIdxGate = { current: 0 };

/** Resolve boxes for the currently selected miner variant (falls back to `fallback`). */
function refBoxesForActiveVariant(
  variants: { boxes?: unknown[] }[] | null | undefined,
  fallback: DetectionBox[],
  idx = activeRefVariantIdxGate.current,
): DetectionBox[] {
  if (!variants?.length) return fallback;
  const clamped = Math.max(0, Math.min(idx, variants.length - 1));
  const picked = normalizeBoxes(variants[clamped]?.boxes);
  return picked.length ? picked : fallback;
}

/** Keep store.refBoxes + detail.reference_boxes on the selected R# after a late merge. */
function reapplyActiveRefVariant(detail?: ChallengeDetail | null) {
  const state = useLabeler.getState();
  const current = detail ?? state.detail;
  if (!current) return;
  const navI = current.index ?? state.appState?.current_index ?? -1;
  const ch = navI >= 0 ? state.appState?.challenges[navI] : current.challenge;
  const variants = effectiveReferenceVariants(
    current.reference_variants,
    ch,
    current.challenge,
    state.refBoxes,
  );
  if (!variants.length) return;
  const next = refBoxesForActiveVariant(variants, state.refBoxes);
  if (!next.length) return;
  if (!boxesEqual(state.refBoxes, next)) state.setRefBoxes(next);
  if (!boxesEqual(normalizeBoxes(current.reference_boxes), next)) {
    state.setDetail({
      ...current,
      reference_boxes: next as unknown as Record<string, unknown>[],
    });
  }
}

function applyDetail(d: ChallengeDetail) {
  if (!payloadMatchesPinnedTopic({ challenge: d.challenge })) return;
  const { setDetail, setBoxes, setRefBoxes, setSelected } = useLabeler.getState();
  const prev = useLabeler.getState().detail;
  const left = normalizeBoxes(d.boxes);
  const ref = normalizeBoxes(d.reference_boxes);
  const preds = normalizeBoxes(d.prediction_boxes);
  const reviewed = Boolean(d.challenge?.reviewed);
  // Avoid width/height jitter from late server responses resetting canvas zoom.
  const sameItem =
    prev?.challenge?.item_key &&
    d.challenge?.item_key &&
    itemKeysMatch(prev.challenge.item_key, d.challenge.item_key);
  const width = sameItem && prev.width > 0 ? prev.width : d.width;
  const height = sameItem && prev.height > 0 ? prev.height : d.height;
  const fallbackVariants = referenceVariantsFromChallenge(d.challenge).map((v) => ({
    source: v.source,
    boxes: v.boxes as unknown as Record<string, unknown>[],
    hotkey: v.hotkey,
    total_weighted: v.total_weighted,
    map50: v.map50,
    false_positive: v.false_positive,
    composite_score: v.composite_score,
    is_winner: v.is_winner,
  }));
  const keepVariants = preferReferenceVariants(
    d.reference_variants || [],
    sameItem ? prev?.reference_variants : undefined,
    fallbackVariants,
  );
  const keepRef =
    ref.length > 0
      ? ref
      : sameItem && (prev?.reference_boxes?.length ?? 0) > 0
        ? normalizeBoxes(prev!.reference_boxes)
        : ref;
  // Same image refresh must not yank the right pane back to R1/winner.
  const activeRef = sameItem
    ? refBoxesForActiveVariant(keepVariants, keepRef.length ? keepRef : ref)
    : keepRef.length
      ? keepRef
      : ref;
  setDetail({
    ...d,
    width,
    height,
    boxes: left,
    reference_boxes: (activeRef.length ? activeRef : keepRef.length ? keepRef : ref) as unknown as Record<
      string,
      unknown
    >[],
    reference_variants: keepVariants,
    prediction_boxes: preds,
  });
  setBoxes(left);
  setRefBoxes(activeRef.length ? activeRef : keepRef.length ? keepRef : ref);
  setSelected(null);
  rememberWinnerPreds(challengeItemKey(d.challenge), preds, left, reviewed);
}

function boxesFromChallenge(ch: AppState["challenges"][number]) {
  const left = (ch.cached_left_boxes || []).map(boxFromDict);
  const ref = (ch.top_boxes || []).map(boxFromDict);
  return { left, ref };
}

function applyChallengeInstant(
  index: number,
  ch: AppState["challenges"][number],
  app: AppState,
  imageUrl: string,
  w: number,
  h: number,
) {
  const { left, ref } = boxesFromChallenge(ch);
  const winner = (ch.item_key && winnerPredCache.get(ch.item_key)) || [];
  let seededLeft: DetectionBox[];
  if (ch.reviewed) {
    // DONE → saved GT labels
    seededLeft = left;
  } else if (Boolean(ch.has_label) && !ch.reviewed) {
    // Own SAM3 draft (label on disk, not Done) → SAM3 labels, never winner.
    seededLeft = left;
  } else {
    // NEW → winner preds; if cache still cold, show top detections so left is never blank.
    seededLeft = winner.length ? winner : left.length ? left : ref;
  }
  useLabeler.getState().setDetail({
    index,
    challenge: {
      ...ch,
      source_type: app.review_mode === "local" ? "local" : ch.source_type || "challenge",
    },
    image_url: imageUrl,
    width: w,
    height: h,
    boxes: seededLeft,
    reference_boxes: ch.top_boxes || [],
    // Seed winner as R1 immediately so middle/right panes never flash empty.
    reference_variants: referenceVariantsFromChallenge(ch).map((v) => ({
      source: v.source,
      boxes: v.boxes as unknown as Record<string, unknown>[],
      hotkey: v.hotkey,
      total_weighted: v.total_weighted,
      map50: v.map50,
      false_positive: v.false_positive,
      composite_score: v.composite_score,
      is_winner: v.is_winner,
    })),
    prediction_boxes: winner.length ? winner : seededLeft,
    class_names: app.model_class_names,
    // Right pane = platform miner boxes → taxonomy / Score Studio order, not local ONNX order.
    reference_class_names: app.model_class_names.length ? app.model_class_names : app.miner_class_names,
    score: null,
    missing_classes: [],
    record: (ch.record as Record<string, unknown>) || {},
  });
  useLabeler.getState().setBoxes(seededLeft);
  useLabeler.getState().setRefBoxes(ref);
  useLabeler.getState().setSelected(null);
}

export default function App({
  topicSlug,
  initialWorkspace,
  initialOwnDay,
  initialOwnField,
  ownDaysPage = false,
}: {
  topicSlug: string;
  initialWorkspace?: WorkspaceMode;
  /** /t/<topic>/own/days/<date>: open this Own day right away. */
  initialOwnDay?: string;
  /** /t/<topic>/own/<openai|gemini>/…: only that generator's Own images. */
  initialOwnField?: OwnField;
  /** /t/<topic>/own/days: show the day-selection page (not a day). */
  ownDaysPage?: boolean;
}) {
  pinnedTopicSlug = topicSlug || null;
  const theme = useLabeler((s) => s.theme);
  const toggleTheme = useLabeler((s) => s.toggleTheme);
  const appState = useLabeler((s) => s.appState);
  // Ambient 3D scene (dark theme only). The Own day list is a light static page → lively; the labeling stage → calm:
  // low fps, and frozen while you are working (pointer / keys), so it costs nothing while you draw boxes.
  const onDayList = ownDaysPage || (appState?.review_mode === "local" && !appState?.own_day);
  useSiteBackdrop(onDayList ? "vivid" : "calm");
  const detail = useLabeler((s) => s.detail);
  const boxes = useLabeler((s) => s.boxes);
  const refBoxes = useLabeler((s) => s.refBoxes);
  const selected = useLabeler((s) => s.selected);
  const currentClass = useLabeler((s) => s.currentClass);
  const setCurrentClass = useLabeler((s) => s.setCurrentClass);
  const setSelected = useLabeler((s) => s.setSelected);
  const setBoxes = useLabeler((s) => s.setBoxes);
  const setRefBoxes = useLabeler((s) => s.setRefBoxes);
  const error = useLabeler((s) => s.error);
  const setError = useLabeler((s) => s.setError);
  const [syncTransform, setSyncTransform] = useState<{ scale: number; offsetX: number; offsetY: number } | null>(null);
  /** Cross-pane hover: highlight matched GT / miner / missed boxes together. */
  const [linkedBoxFocus, setLinkedBoxFocus] = useState<{
    gtIdxs: number[];
    predIdxs: number[];
    ghostIdxs: number[];
  } | null>(null);
  /** Keyboard C/V box inspect — zoom + cross-pane highlight. */
  const [boxInspect, setBoxInspect] = useState<{ idx: number; seq: number; box?: DetectionBox } | null>(null);
  const boxInspectIdxRef = useRef<number | null>(null);
  /** Right-pane solo: wrong (W) / extra (E) / missed (R). null = all. */
  const [minerAlertFilter, setMinerAlertFilter] = useState<"wrong" | "extra" | "missed" | null>(null);
  /** Own: dual labeled + base-detection panes (default on; toggle with E). */
  const [ownDualView, setOwnDualView] = useState(false);
  /** Own: smart dotted row/column guides while labeling (toggle with R). */
  const [ownGuides, setOwnGuides] = useState(() => {
    try {
      return localStorage.getItem("sn44-guides") !== "0"; // dotted guide lines: on unless switched off
    } catch {
      return true;
    }
  });
  const [downloadBusy, setDownloadBusy] = useState(false);
  const [downloadOpen, setDownloadOpen] = useState(false);
  const [sendGtOpen, setSendGtOpen] = useState(false);
  const [ultralyticsOpen, setUltralyticsOpen] = useState(false);
  const [dppOpen, setDppOpen] = useState(false);
  const [hfCompareOpen, setHfCompareOpen] = useState(false);
  const [baseModelOpen, setBaseModelOpen] = useState(false);
  const [stopCollectBusy, setStopCollectBusy] = useState(false);
  const [stripFilter, setStripFilter] = useState<StripFilter>("all");
  const [genFilter, setGenFilterState] = useState<GenFilter>(() => {
    try {
      return (localStorage.getItem("labeler_gen_filter") as GenFilter) || "all";
    } catch {
      return "all";
    }
  });
  genFilterNow = genFilter;
  const setGenFilter = (g: GenFilter) => {
    genFilterNow = g;
    setGenFilterState(g);
    try {
      localStorage.setItem("labeler_gen_filter", g);
    } catch {
      /* private mode */
    }
  };
  const stripFilterRef = useRef<StripFilter>("all");
  stripFilterRef.current = stripFilter;
  const [ownFilter, setOwnFilter] = useState<OwnFilter>("all");
  const ownFilterRef = useRef<OwnFilter>("all");
  ownFilterRef.current = ownFilter;
  const [modeBusy, setModeBusy] = useState(false);
  const [ownUploadOpen, setOwnUploadOpen] = useState(false);
  const [ownHfImportOpen, setOwnHfImportOpen] = useState(false);
  const [sam3Busy, setSam3Busy] = useState(false);
  const [sam3Open, setSam3Open] = useState(false);
  const [ownDeleteOpen, setOwnDeleteOpen] = useState(false);
  const [deleteImageAsk, setDeleteImageAsk] = useState<{ key: string; index: number; name: string; hasLabel: boolean } | null>(null);
  const [daySel, setDaySel] = useState<string[]>([]); // days ticked on the Own day list
  const pendingWorkspaceRef = useRef<WorkspaceMode | null>(initialWorkspace ?? null);
  const navigate = useNavigate();
  const location = useLocation();
  const initialOwnDayRef = useRef(initialOwnDay);
  initialOwnDayRef.current = initialOwnDay;
  // Own "field" (OpenAI / Gemini images, null = all): chosen on the day page, kept while you open and leave days
  const [ownField, setOwnFieldState] = useState<OwnField | null>(() => {
    if (initialOwnField) return initialOwnField;
    if (initialWorkspace === "own") return null; // a link without a field means "all"
    try {
      const v = localStorage.getItem("labeler_own_field");
      return v === "openai" || v === "gemini" ? v : null;
    } catch {
      return null;
    }
  });
  const ownFieldRef = useRef(ownField);
  ownFieldRef.current = ownField;
  const setOwnField = (f: OwnField | null) => {
    ownFieldRef.current = f;
    setOwnFieldState(f);
    try {
      localStorage.setItem("labeler_own_field", f || "");
    } catch {
      /* private mode */
    }
  };
  const ownDaysPageRef = useRef(ownDaysPage);
  ownDaysPageRef.current = ownDaysPage;
  const enterOwnDayRef = useRef<((date: string) => Promise<void>) | null>(null);
  const leaveOwnDayRef = useRef<(() => Promise<void>) | null>(null);
  const [urlSync, setUrlSync] = useState(false); // false until the link's own day/page has been opened
  const [stripTick, setStripTick] = useState(0);
  const stripTickRafRef = useRef(0);
  const bumpStripTick = useCallback(() => {
    if (stripTickRafRef.current) return;
    stripTickRafRef.current = window.requestAnimationFrame(() => {
      stripTickRafRef.current = 0;
      setStripTick((n) => n + 1);
    });
  }, []);
  const [findBestOpen, setFindBestOpen] = useState(false);
  const [submitModeOpen, setSubmitModeOpen] = useState(false);
  const [manualMergeOpen, setManualMergeOpen] = useState(false);
  const [uploadLabelsOpen, setUploadLabelsOpen] = useState(false);
  const [showBoxLabels, setShowBoxLabels] = useState(() => {
    try {
      return localStorage.getItem("sn44-box-labels") === "1"; // class labels: hidden unless switched on
    } catch {
      return false;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem("sn44-guides", ownGuides ? "1" : "0");
      localStorage.setItem("sn44-box-labels", showBoxLabels ? "1" : "0");
    } catch {
      /* private mode */
    }
  }, [ownGuides, showBoxLabels]);
  const modalOpenRef = useRef(false);
  modalOpenRef.current =
    downloadOpen || sendGtOpen || findBestOpen || uploadLabelsOpen || ownUploadOpen || ownHfImportOpen || ownDeleteOpen || deleteImageAsk != null || ultralyticsOpen || dppOpen || hfCompareOpen ||
    submitModeOpen || manualMergeOpen || baseModelOpen || sam3Open || useKeys.getState().open;

  const [topicReady, setTopicReady] = useState(false);
  // Team: heartbeat (presence/claims), label log, live feed. Read-only when a teammate owns
  // the open image or when we are watching (G).
  useTeamSync(topicReady, topicSlug);
  const teamReadOnly = useTeam((s) => s.readOnly);
  const teamClaims = useTeam((s) => s.claims);
  const teamLabels = useTeam((s) => s.labels);
  const teamColors = useTeam((s) => s.colors);
  const [refVariantIdx, setRefVariantIdx] = useState(0);
  const refVariantIdxRef = useRef(0);
  refVariantIdxRef.current = refVariantIdx;
  activeRefVariantIdxGate.current = refVariantIdx;

  const saveTimer = useRef<number | null>(null);
  const detailIndexRef = useRef<number | null>(null);
  /** Stable identity of the open image — survives collect prepends that shift indices. */
  const openItemKeyRef = useRef<string | null>(null);
  /** React state mirror of detailIndexRef so filmstrip highlight stays in sync mid-load. */
  const [viewIndex, setViewIndex] = useState<number | null>(null);
  const reviewActionBusyRef = useRef(false);
  const loadSeqRef = useRef(0);
  const loadAbortRef = useRef<AbortController | null>(null);
  const loadRetryTimerRef = useRef<number | null>(null);
  const navStepRef = useRef(0);
  const navRafRef = useRef(0);
  /** Separate coalescing so Arrow ←/→ never steal N/B (and vice versa). */
  const needsNavStepRef = useRef(0);
  const needsNavRafRef = useRef(0);
  const filmstripPickAtRef = useRef(0);
  const filmstripRef = useRef<HTMLDivElement>(null);
  const prevLoadedCountRef = useRef(0);
  const prevBoxesSeqRef = useRef(0);
  const followCollectRef = useRef(true); // true = near left end (latest); collect must not steal selection
  const topicReadyRef = useRef(false);
  const THUMB_SLOT = 118;
  const [stripRange, setStripRange] = useState({ first: 0, last: 12 });
  const blocksMapRef = useRef<Map<number, number>>(new Map());

  const buildBlocksMap = useCallback((app: AppState) => {
    const map = new Map<number, number>();
    app.challenges.forEach((c, i) => {
      if (c.loaded && c.block != null) map.set(i, c.block);
    });
    blocksMapRef.current = map;
    return map;
  }, []);

  // Same block ids exist across skills (car-wash / crime / fire). Include project+skill
  // in image URLs so browser/memory caches never reuse another model's frames.
  const imageCacheToken = useMemo(() => {
    if (!appState) return "";
    const skill = appState.active_skill?.element_path || appState.hf_model_id || "skill";
    // Session + project + skill — block ids collide across topics and users.
    return `${appState.session_id || "sid"}:${appState.project_id}:${skill}`;
  }, [appState?.session_id, appState?.project_id, appState?.active_skill?.element_path, appState?.hf_model_id]);

  const imageCacheTokenRef = useRef(imageCacheToken);
  useEffect(() => {
    // URLs already include session+project+skill — keep decoded thumbs across topic
    // switches so revisiting a topic stays instant (LRU trims memory).
    imageCacheTokenRef.current = imageCacheToken;
  }, [imageCacheToken]);

  // Dropping a folder onto Chrome otherwise navigates to file:// and blanks the app.
  useEffect(() => {
    const blockNav = (e: DragEvent) => {
      e.preventDefault();
    };
    window.addEventListener("dragover", blockNav);
    window.addEventListener("drop", blockNav);
    return () => {
      window.removeEventListener("dragover", blockNav);
      window.removeEventListener("drop", blockNav);
    };
  }, []);

  const preloadFilmstripWindow = useCallback(() => {
    const el = filmstripRef.current;
    const app = useLabeler.getState().appState;
    if (!el || !app) return;
    const mode = app.review_mode;
    const blocks = blocksMapRef.current.size ? blocksMapRef.current : buildBlocksMap(app);
    const filter = stripFilterRef.current;
    const own = ownFilterRef.current;
    const cur = detailIndexRef.current ?? app.current_index;
    const indicesAll = filmstripIndices(app, filter, cur, own);
    const visibleCount = indicesAll.length;
    // Ultralytics-style: only the visible slots + a tiny scroll buffer.
    const pad = 3;
    const firstSlot = Math.max(0, Math.floor(el.scrollLeft / THUMB_SLOT) - pad);
    const lastSlot = Math.min(
      Math.max(0, visibleCount - 1),
      Math.ceil((el.scrollLeft + Math.max(el.clientWidth, 1)) / THUMB_SLOT) + pad,
    );
    setStripRange((prev) =>
      prev.first === firstSlot && prev.last === lastSlot ? prev : { first: firstSlot, last: lastSlot },
    );
    const indices = indicesAll.slice(firstSlot, lastSlot + 1);
    if (!indices.length) return;
    const mediaMeta = new Map(
      app.challenges.map((c, i) => [i, { item_key: c.item_key, media_rev: c.media_rev }]),
    );
    scheduleBatchStripLoad(indices, mode, blocks, imageCacheToken, mediaMeta);
  }, [THUMB_SLOT, buildBlocksMap, imageCacheToken]);

  const scrollThumbIntoView = useCallback((index: number, behavior: ScrollBehavior = "auto") => {
    const el = filmstripRef.current;
    const app = useLabeler.getState().appState;
    if (!el || !app) return;
    const filter = stripFilterRef.current;
    const own = ownFilterRef.current;
    const cur = detailIndexRef.current ?? app.current_index;
    const indices = filmstripIndices(app, filter, cur, own);
    const slot = indices.indexOf(index);
    if (slot < 0) return;
    const left = slot * THUMB_SLOT - el.clientWidth / 2 + THUMB_SLOT / 2;
    el.scrollTo({ left: Math.max(0, left), behavior });
  }, [THUMB_SLOT]);

  const neighborFilteredIndex = useCallback((from: number, dir: 1 | -1) => {
    const app = useLabeler.getState().appState;
    const filter = stripFilterRef.current;
    const own = ownFilterRef.current;
    if (!app) return -1;
    const indices = filteredStripIndices(app, filter, own);
    let pos = indices.indexOf(from);
    if (pos < 0) {
      const key = app.challenges[from]?.item_key;
      if (key) pos = indices.findIndex((i) => itemKeysMatch(app.challenges[i]?.item_key, key));
    }
    if (pos < 0) {
      if (dir > 0) return indices.find((i) => i > from) ?? -1;
      for (let k = indices.length - 1; k >= 0; k -= 1) {
        if (indices[k] < from) return indices[k];
      }
      return -1;
    }
    return indices[pos + dir] ?? -1;
  }, []);

  const onFilmstripScroll = useCallback(() => {
    const el = filmstripRef.current;
    if (!el) return;
    // Latest images live on the left — "follow collect" only when user is already there.
    followCollectRef.current = el.scrollLeft < 140;
    preloadFilmstripWindow();
  }, [preloadFilmstripWindow]);

  const onFilmstripWheel = useCallback(
    (e: React.WheelEvent<HTMLDivElement>) => {
      const el = filmstripRef.current;
      if (!el || Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
      e.preventDefault();
      el.scrollLeft += e.deltaY;
      onFilmstripScroll();
    },
    [onFilmstripScroll],
  );

  const saveSnapshotRef = useRef<{
    index: number;
    itemKey: string;
    boxes: DetectionBox[];
    /** Save as Done (false = Own SAM3 draft stays a draft). */
    done?: boolean;
    gen: number;
  } | null>(null);
  const historyRef = useRef<Map<string, { undo: DetectionBox[][]; redo: DetectionBox[][] }>>(new Map());
  const editSessionRef = useRef(false);
  const dirtyIndexRef = useRef<number | null>(null);
  const dirtyItemKeyRef = useRef<string | null>(null);
  const saveGenRef = useRef(0);
  const againGenRef = useRef(0);

  const resolveSaveTarget = useCallback((index: number | null, itemKey?: string | null) => {
    const app = useLabeler.getState().appState;
    const key = itemKey || (index != null ? app?.challenges[index]?.item_key : null) || null;
    if (key && app) {
      const ni = app.challenges.findIndex((c) => itemKeysMatch(c.item_key, key));
      if (ni >= 0) return { index: ni, itemKey: app.challenges[ni].item_key };
    }
    if (index == null) return null;
    return { index, itemKey: key || "" };
  }, []);

  const flushSave = useCallback(async (index: number | null, itemKey?: string | null) => {
    if (index == null && !itemKey) return;
    if (saveTimer.current) {
      window.clearTimeout(saveTimer.current);
      saveTimer.current = null;
    }
    const snap = saveSnapshotRef.current;
    const matchesSnap =
      snap &&
      ((itemKey && snap.itemKey === itemKey) ||
        (index != null && (snap.index === index || snap.itemKey === itemKey)));
    const pending = matchesSnap ? snap!.boxes : null;
    const gen = matchesSnap ? snap!.gen : 0;
    const snapKey = matchesSnap ? snap!.itemKey : itemKey || "";
    if (matchesSnap) {
      saveSnapshotRef.current = null;
    }
    if (!pending) return;
    const target = resolveSaveTarget(index, snapKey);
    if (!target) return;
    const app = useLabeler.getState().appState;
    const row =
      (snapKey && app?.challenges.find((c) => itemKeysMatch(c.item_key, snapKey))) ||
      (target.index != null ? app?.challenges[target.index] : null);
    // Own SAM3 draft only: label on disk, not yet reviewed. NEW with pending boxes → Done.
    const sam3Draft =
      app?.review_mode === "local" &&
      Boolean(row && !row.reviewed && Boolean(row.has_label)) &&
      !pendingDoneKeys.has(snapKey);
    const markDone = !sam3Draft;
    if (markDone && snapKey) notePendingDone(snapKey);
    const flushKey = target.itemKey || snapKey;
    const teamNow = useTeam.getState();
    teamNow.setSave("saving");
    try {
      const res = await api.saveBoxes(target.index, pending, markDone, target.itemKey || undefined, teamNow.revs[flushKey]);
      useTeam.getState().noteRev(flushKey, res.rev);
      useTeam.getState().noteLabel(flushKey, res.labeled_by, res.rev);
      if (!hasPendingSaves()) useTeam.getState().setSave("saved");
      if (saveGenRef.current !== gen) return;
      if (dirtyItemKeyRef.current && itemKeysMatch(dirtyItemKeyRef.current, snapKey)) {
        dirtyItemKeyRef.current = null;
        dirtyIndexRef.current = null;
      } else if (dirtyIndexRef.current === target.index) {
        dirtyIndexRef.current = null;
      }
    } catch (err) {
      // Never block navigation, but never lose the labels either: keep them queued.
      const status = err instanceof HttpError ? err.status : 0;
      if (status === 423 || status === 409) {
        useTeam.getState().setSave(status === 423 ? "locked" : "conflict", err instanceof Error ? err.message : String(err));
      } else {
        retrySave(flushKey, target.index, pending, markDone);
      }
    }
  }, [resolveSaveTarget]);

  const clearLoadRetry = useCallback(() => {
    if (loadRetryTimerRef.current != null) {
      window.clearTimeout(loadRetryTimerRef.current);
      loadRetryTimerRef.current = null;
    }
  }, []);

  const loadStillCurrent = useCallback((seq: number, index: number, itemKey: string) => {
    if (seq !== loadSeqRef.current) return false;
    if (detailIndexRef.current !== index) return false;
    const app = useLabeler.getState().appState;
    const ch = app?.challenges[index];
    return Boolean(ch && itemKeysMatch(ch.item_key, itemKey));
  }, []);

  const selectChallenge = useCallback(
    async (index: number, reason = "select") => {
      if (switchingWorkspaceRef.current) return;
      const app = useLabeler.getState().appState;
      if (!app || !app.challenges[index]?.loaded) return;

      const ch = app.challenges[index];
      const itemKey = ch.item_key;
      // Already on OR loading this image — never abort our own in-flight getChallenge.
      if (detailIndexRef.current === index && openItemKeyRef.current === itemKey) {
        return;
      }

      const prev = detailIndexRef.current;
      const prevCh = prev != null && prev !== index ? app.challenges[prev] : null;
      const prevKey =
        prevCh &&
        (saveSnapshotRef.current?.itemKey ||
          dirtyItemKeyRef.current ||
          prevCh.item_key ||
          null);

      // Pin desire BEFORE aborting so filmstrip / keys agree immediately.
      detailIndexRef.current = index;
      openItemKeyRef.current = itemKey;
      setViewIndex(index);
      reviewActionBusyRef.current = false;
      if (app.review_mode === "local" && itemKey) {
        writeOwnCursor(topicSlug, itemKey);
      }

      const seq = ++loadSeqRef.current;
      clearLoadRetry();
      loadAbortRef.current?.abort();
      const controller = new AbortController();
      loadAbortRef.current = controller;

      const mode = app.review_mode;
      const imageUrl = challengeImageUrl(index, mode, ch.block, imageCacheToken, ch.item_key, ch.media_rev);
      const w = ch.image_width && ch.image_width > 0 ? ch.image_width : 1024;
      const h = ch.image_height && ch.image_height > 0 ? ch.image_height : 1024;

      setRefVariantIdx(0);
      activeRefVariantIdxGate.current = 0;
      setMinerAlertFilter(null);
      setSyncTransform(null);
      setLinkedBoxFocus(null);
      boxInspectIdxRef.current = null;
      setBoxInspect(null);
      setError("");

      // 1) Paint next image + boxes immediately (before any network / save work).
      const cached = getCachedDetail(index, itemKey) || getCachedDetailByItemKey(itemKey);
      const stateLeft = ch.cached_left_boxes?.length ? normalizeBoxes(ch.cached_left_boxes) : [];
      const cacheHasBoxes =
        Boolean(cached) &&
        (ch.reviewed
          ? normalizeBoxes(cached!.boxes).length > 0
          : normalizeBoxes(cached!.prediction_boxes?.length ? cached!.prediction_boxes : cached!.boxes).length > 0);
      const cacheOk =
        Boolean(cached) &&
        Boolean(itemKey) &&
        itemKeysMatch(cached!.challenge?.item_key, itemKey) &&
        (ch.reviewed || cacheHasBoxes || stateLeft.length > 0);
      if (cacheOk && cached) {
        const instantRefs = cached.reference_variants?.length
          ? cached.reference_variants
          : referenceVariantsFromChallenge(ch).map((v) => ({
              source: v.source,
              boxes: v.boxes as unknown as Record<string, unknown>[],
              hotkey: v.hotkey,
              total_weighted: v.total_weighted,
              map50: v.map50,
              false_positive: v.false_positive,
              composite_score: v.composite_score,
              is_winner: v.is_winner,
            }));
        if (ch.reviewed) {
          const gt = stateLeft.length ? stateLeft : normalizeBoxes(cached.boxes);
          applyDetail({
            ...cached,
            index,
            boxes: gt,
            reference_variants: instantRefs,
            challenge: { ...cached.challenge, ...ch, item_key: itemKey, reviewed: true },
          });
          if (instantRefs.length >= 3) syncStripChecks(index, ch, gt, instantRefs);
        } else {
          const preds = normalizeBoxes(
            cached.prediction_boxes?.length ? cached.prediction_boxes : cached.boxes,
          );
          const leftBoxes = stateLeft.length ? stateLeft : preds.length ? preds : normalizeBoxes(cached.boxes);
          applyDetail({
            ...cached,
            index,
            boxes: leftBoxes,
            prediction_boxes: preds.length ? preds : leftBoxes,
            reference_variants: instantRefs,
            challenge: { ...cached.challenge, ...ch, item_key: itemKey, reviewed: false },
          });
        }
        // Guard: if applyDetail was dropped (topic mismatch), still paint from the row.
        const liveKey = useLabeler.getState().detail?.challenge?.item_key;
        if (liveKey !== itemKey) {
          applyChallengeInstant(index, ch, app, imageUrl, w, h);
        }
      } else {
        applyChallengeInstant(index, ch, app, imageUrl, w, h);
      }
      useLabeler.getState().setSelected(null);

      primeDecodedThumbnail(challengeThumbnailUrl(index, mode, ch.block, true, imageCacheToken, ch.item_key, ch.media_rev));
      preloadChallengeUrgent(index, mode, ch.block, imageCacheToken, ch.item_key, ch.media_rev, { full: true });
      preloadChallengeForward(
        index,
        mode,
        app.challenges,
        3,
        imageCacheToken,
        filteredNeighbors(index, stripFilterRef.current, 3, ownFilterRef.current),
      );

      // Save previous labels in the background — never block left-pane paint.
      if (prev != null && prev !== index && prevCh) {
        const prevIsSam3 =
          app.review_mode === "local" && Boolean(prevCh.has_label) && !prevCh.reviewed;
        if (
          prevKey &&
          !prevIsSam3 &&
          (prevCh.reviewed ||
            itemKeysMatch(dirtyItemKeyRef.current, prevKey) ||
            itemKeysMatch(saveSnapshotRef.current?.itemKey, prevKey) ||
            dirtyIndexRef.current === prev)
        ) {
          notePendingDone(prevKey);
        }
        void flushSave(prev, prevKey).then(() => {
          if (dirtyItemKeyRef.current === prevKey) {
            dirtyItemKeyRef.current = null;
            dirtyIndexRef.current = null;
          } else if (dirtyIndexRef.current === prev) {
            dirtyIndexRef.current = null;
          }
        });
        editSessionRef.current = false;
      }

      // Load R1/R2/R3 after first paint — always refresh per item_key (never reuse prior block's count).
      window.setTimeout(() => {
        if (app.review_mode === "local") return;
        if (!loadStillCurrent(seq, index, itemKey)) return;
        void api
          .getChallengeVariants(index, controller.signal, itemKey)
          .then((payload) => {
            if (!loadStillCurrent(seq, index, itemKey)) return;
            const state = useLabeler.getState();
            const prev = state.detail;
            if (!prev || prev.index !== index || prev.challenge?.item_key !== itemKey) return;
            const variants = (payload.reference_variants || []) as NonNullable<
              ChallengeDetail["reference_variants"]
            >;
            if (!variants.length) return;
            const merged = preferReferenceVariants(
              variants,
              prev.reference_variants as typeof variants | undefined,
              [],
            );
            const refBoxes = normalizeBoxes(payload.reference_boxes);
            state.setDetail({
              ...prev,
              reference_variants: merged,
              reference_boxes: (payload.reference_boxes ||
                prev.reference_boxes) as ChallengeDetail["reference_boxes"],
            } as ChallengeDetail);
            if (refBoxes.length && !state.refBoxes.length) state.setRefBoxes(refBoxes);
            reapplyActiveRefVariant(useLabeler.getState().detail);
            setCachedDetail(index, {
              ...prev,
              reference_variants: merged,
              reference_boxes: (payload.reference_boxes || prev.reference_boxes) as ChallengeDetail["reference_boxes"],
            });
            const gt = normalizeBoxes(state.boxes);
            const reviewed = Boolean(
              prev.challenge?.reviewed || useLabeler.getState().appState?.challenges[index]?.reviewed,
            );
            if (reviewed && gt.length && merged.length >= 1) {
              syncStripChecks(index, { ...ch, reviewed: true }, gt, merged);
              bumpStripTick();
            }
          })
          .catch(() => undefined);
      }, 0);

      try {
        // 2) Fast disk read. DONE keeps painted GT; NEW always takes winner-model boxes.
        const d = await api.getChallenge(index, true, controller.signal, true, reason, itemKey);
        if (!loadStillCurrent(seq, index, itemKey) || controller.signal.aborted) return;
        if (
          dirtyIndexRef.current === index ||
          itemKeysMatch(dirtyItemKeyRef.current, itemKey) ||
          editSessionRef.current
        ) {
          const preds = normalizeBoxes(d.prediction_boxes);
          const state = useLabeler.getState();
          const prev = state.detail;
          if (preds.length) {
            rememberWinnerPreds(
              itemKey,
              preds,
              normalizeBoxes(state.boxes),
              Boolean(prev?.challenge?.reviewed || ch.reviewed),
            );
          }
          // Keep left GT stable, but always merge R1/R2/R3 into the middle rail.
          if (prev && (d.reference_variants?.length ?? 0) > 0) {
            const merged = preferReferenceVariants(
              d.reference_variants || [],
              prev.reference_variants,
              [],
            );
            state.setDetail({
              ...prev,
              reference_variants: merged,
              // Do not force winner reference_boxes — reapplyActiveRefVariant picks R#.
              reference_boxes: prev.reference_boxes,
              prediction_boxes: preds.length ? preds : prev.prediction_boxes,
              record: d.record ?? prev.record,
            });
            reapplyActiveRefVariant(useLabeler.getState().detail);
            const gt = normalizeBoxes(state.boxes);
            if ((prev.challenge?.reviewed || ch.reviewed) && gt.length && merged.length >= 1) {
              syncStripChecks(index, { ...ch, reviewed: true }, gt, merged);
              bumpStripTick();
            }
          } else {
            reapplyActiveRefVariant(prev);
          }
          return;
        }
        const paintedReviewed = Boolean(ch.reviewed || (d.challenge as { reviewed?: boolean })?.reviewed);
        const diskLeft = normalizeBoxes(d.boxes);
        // Prefer item_key over filmstrip index — Own day list can shift under us.
        const liveApp = useLabeler.getState().appState;
        const liveIdx =
          liveApp?.challenges.findIndex((c) => itemKeysMatch(c.item_key, itemKey)) ?? -1;
        const paintIdx = liveIdx >= 0 ? liveIdx : index;
        if (paintIdx !== detailIndexRef.current) {
          detailIndexRef.current = paintIdx;
          openItemKeyRef.current = itemKey;
          setViewIndex(paintIdx);
        }
        const serverSam3 =
          app.review_mode === "local" &&
          Boolean(ch.has_label || (d.challenge as { has_label?: boolean })?.has_label) &&
          !Boolean((d.challenge as { reviewed?: boolean })?.reviewed) &&
          !ch.reviewed;
        if (paintedReviewed && !serverSam3) {
          if (!loadStillCurrent(seq, paintIdx, itemKey) && !loadStillCurrent(seq, index, itemKey)) return;
          applyDetail({ ...d, index: paintIdx, boxes: diskLeft });
          setCachedDetail(paintIdx, {
            ...d,
            index: paintIdx,
            boxes: diskLeft,
            challenge: { ...d.challenge, reviewed: true },
          });
          // Own dual-view base pane: prediction_cache may still be warming async —
          // poll without blocking media (sync ONNX used to freeze the full image).
          if (app.review_mode === "local" && !normalizeBoxes(d.prediction_boxes).length) {
            const tryBase = (delayMs: number) => {
              window.setTimeout(() => {
                if (!loadStillCurrent(seq, paintIdx, itemKey) || controller.signal.aborted) return;
                const prev = useLabeler.getState().detail;
                if (normalizeBoxes(prev?.prediction_boxes).length) return;
                void api
                  .getChallenge(paintIdx, true, controller.signal, false, undefined, itemKey)
                  .then((warmDetail) => {
                    if (!loadStillCurrent(seq, paintIdx, itemKey)) return;
                    const warm = normalizeBoxes(warmDetail.prediction_boxes || []);
                    if (!warm.length) return;
                    rememberWinnerPreds(itemKey, warm, normalizeBoxes(useLabeler.getState().boxes), true);
                    const live = useLabeler.getState().detail;
                    if (!live || !itemKeysMatch(live.challenge?.item_key, itemKey)) return;
                    useLabeler.getState().setDetail({
                      ...live,
                      prediction_boxes: warm,
                    });
                    setCachedDetail(paintIdx, { ...live, prediction_boxes: warm });
                    bumpStripTick();
                  })
                  .catch(() => undefined);
              }, delayMs);
            };
            tryBase(400);
            tryBase(1200);
            tryBase(3000);
            tryBase(7000);
          }
        } else {
          // SAM3 draft only when the server/strip says has_label && !reviewed.
          // Do NOT treat winner preds on NEW (diskLeft.length > 0) as SAM3 — that
          // falsely set has_label and made manual labeling save as SAM3 instead of Done.
          if (serverSam3) {
            // Own SAM3: left = draft labels from disk; keep winner preds for dual base pane.
            if (!loadStillCurrent(seq, paintIdx, itemKey) && !loadStillCurrent(seq, index, itemKey)) return;
            const sam3Preds = normalizeBoxes(d.prediction_boxes || []);
            if (sam3Preds.length) rememberWinnerPreds(itemKey, sam3Preds, diskLeft, false);
            applyDetail({
              ...d,
              index: paintIdx,
              boxes: diskLeft,
              prediction_boxes: sam3Preds,
              challenge: { ...d.challenge, reviewed: false, has_label: true },
            });
            setCachedDetail(paintIdx, {
              ...d,
              index: paintIdx,
              boxes: diskLeft,
              prediction_boxes: sam3Preds,
              challenge: { ...d.challenge, reviewed: false, has_label: true },
            });
            // Warm winner prediction_cache for dual base pane (same as Done).
            if (!sam3Preds.length) {
              const tryBase = (delayMs: number) => {
                window.setTimeout(() => {
                  if (!loadStillCurrent(seq, paintIdx, itemKey) || controller.signal.aborted) return;
                  const prev = useLabeler.getState().detail;
                  if (normalizeBoxes(prev?.prediction_boxes).length) return;
                  void api
                    .getChallenge(paintIdx, true, controller.signal, false, undefined, itemKey)
                    .then((warmDetail) => {
                      if (!loadStillCurrent(seq, paintIdx, itemKey)) return;
                      const warm = normalizeBoxes(warmDetail.prediction_boxes || []);
                      if (!warm.length) return;
                      rememberWinnerPreds(itemKey, warm, normalizeBoxes(useLabeler.getState().boxes), false);
                      const live = useLabeler.getState().detail;
                      if (!live || !itemKeysMatch(live.challenge?.item_key, itemKey)) return;
                      useLabeler.getState().setDetail({
                        ...live,
                        prediction_boxes: warm,
                      });
                      setCachedDetail(paintIdx, { ...live, prediction_boxes: warm });
                      bumpStripTick();
                    })
                    .catch(() => undefined);
                }, delayMs);
              };
              tryBase(400);
              tryBase(1200);
              tryBase(3000);
              tryBase(7000);
            }
          } else {
          // NEW: always paint winner-model boxes on the left (never keep empty / top-only).
          const preds = normalizeBoxes(
            d.prediction_boxes?.length ? d.prediction_boxes : d.boxes,
          );
          let leftBoxes = preds.length ? preds : normalizeBoxes(d.boxes);
          if (!loadStillCurrent(seq, index, itemKey)) return;
          applyDetail({
            ...d,
            index,
            boxes: leftBoxes,
            prediction_boxes: preds.length ? preds : leftBoxes,
            challenge: { ...d.challenge, reviewed: false, has_label: false },
          });
          setCachedDetail(index, {
            ...d,
            index,
            boxes: leftBoxes,
            prediction_boxes: preds.length ? preds : leftBoxes,
            challenge: { ...d.challenge, reviewed: false, has_label: false },
          });
          // Platform cold-cache fallback only — Own uses activate=True ONNX on the server.
          if (!leftBoxes.length && app.review_mode !== "local") {
            void api
              .getChallenge(index, false, controller.signal, false, undefined, itemKey)
              .then((full) => {
                if (!loadStillCurrent(seq, index, itemKey)) return;
                if (useLabeler.getState().appState?.challenges[index]?.reviewed) return;
                if (dirtyIndexRef.current === index || editSessionRef.current) return;
                const warm = normalizeBoxes(
                  full.prediction_boxes?.length ? full.prediction_boxes : full.boxes,
                );
                const variants = full.reference_variants || [];
                if (warm.length) {
                  applyDetail({
                    ...full,
                    boxes: warm,
                    prediction_boxes: warm,
                    challenge: { ...full.challenge, reviewed: false },
                  });
                  reapplyActiveRefVariant(useLabeler.getState().detail);
                  setCachedDetail(index, { ...full, boxes: warm, prediction_boxes: warm });
                  bumpStripTick();
                } else if (variants.length) {
                  const state = useLabeler.getState();
                  const prev = state.detail;
                  if (prev && prev.index === index) {
                    state.setDetail({ ...prev, reference_variants: variants });
                    reapplyActiveRefVariant(useLabeler.getState().detail);
                  }
                }
              })
              .catch(() => undefined);
          } else if (app.review_mode === "local") {
            // Own NEW: always poll for real winner prediction_cache (top_boxes /
            // empty left must not freeze base detection or key-A origin).
            const needWinnerPreds = () => {
              const live = useLabeler.getState().detail;
              if (!live || !itemKeysMatch(live.challenge?.item_key, itemKey)) return true;
              return !normalizeBoxes(live.prediction_boxes).length;
            };
            if (!leftBoxes.length || needWinnerPreds()) {
              const tryWarm = (delayMs: number) => {
                window.setTimeout(() => {
                  if (!loadStillCurrent(seq, index, itemKey) || controller.signal.aborted) return;
                  if (useLabeler.getState().appState?.challenges[index]?.reviewed) return;
                  if (!needWinnerPreds() && useLabeler.getState().boxes.length) return;
                  void api
                    .getChallenge(index, true, controller.signal, false, undefined, itemKey)
                    .then((warmDetail) => {
                      if (!loadStillCurrent(seq, index, itemKey)) return;
                      if (useLabeler.getState().appState?.challenges[index]?.reviewed) return;
                      if (dirtyIndexRef.current === index || editSessionRef.current) return;
                      const warm = normalizeBoxes(
                        warmDetail.prediction_boxes?.length
                          ? warmDetail.prediction_boxes
                          : warmDetail.boxes,
                      );
                      if (!warm.length) return;
                      rememberWinnerPreds(
                        itemKey,
                        warm,
                        normalizeBoxes(useLabeler.getState().boxes),
                        false,
                      );
                      applyDetail({
                        ...warmDetail,
                        boxes: warm,
                        prediction_boxes: warm,
                        challenge: {
                          ...warmDetail.challenge,
                          reviewed: false,
                          source_type: "local",
                          cached_left_boxes: warm.map((b) => ({
                            cls_id: b.cls,
                            x1: b.x1,
                            y1: b.y1,
                            x2: b.x2,
                            y2: b.y2,
                            conf: b.conf ?? null,
                          })),
                        },
                      });
                      setCachedDetail(index, {
                        ...warmDetail,
                        boxes: warm,
                        prediction_boxes: warm,
                      });
                      bumpStripTick();
                    })
                    .catch(() => undefined);
                }, delayMs);
              };
              tryWarm(300);
              tryWarm(900);
              tryWarm(2000);
              tryWarm(4500);
              tryWarm(9000);
            }
          }
          } // end !isSam3Draft (true New / winner path)
        }
        if (!loadStillCurrent(seq, index, itemKey)) return;
        const liveGt = normalizeBoxes(useLabeler.getState().boxes);
        const reviewed = Boolean(ch.reviewed || d.challenge?.reviewed);
        if (reviewed && liveGt.length) {
          const patch = checksPatchForItem(
            { ...ch, ...(d.challenge as ChallengeItem), reviewed: true },
            liveGt,
            d.reference_variants,
          );
          if (patch.checks_passed != null) patchChallengeChecks(index, patch);
        }
        detailIndexRef.current = index;
        openItemKeyRef.current = itemKey;
        preloadChallengeForward(
          index,
          mode,
          app.challenges,
          3,
          imageCacheToken,
          filteredNeighbors(index, stripFilterRef.current, 3, ownFilterRef.current),
        );
        preloadDetails(
          filteredNeighbors(index, stripFilterRef.current, 2, ownFilterRef.current),
          (i) => api.getChallenge(i, true, undefined, false, undefined, app.challenges[i]?.item_key),
          (i) => app.challenges[i]?.item_key,
          bumpStripTick,
        );
      } catch (e) {
        if (controller.signal.aborted || !loadStillCurrent(seq, index, itemKey)) return;
        const msg = String(e);
        if (msg.toLowerCase().includes("abort")) return;
        if (isStaleChallengeError(e)) {
          setError("Image index changed — pick another block or refresh the topic.");
          return;
        }
        if (!msg.toLowerCase().includes("failed to fetch")) {
          setError(msg);
          return;
        }
        setError("Server connection lost — retrying…");
        loadRetryTimerRef.current = window.setTimeout(() => {
          loadRetryTimerRef.current = null;
          if (loadStillCurrent(seq, index, itemKey)) {
            selectChallenge(index, "retry").catch(() => undefined);
          }
        }, 1200);
      }
    },
    [bumpStripTick, clearLoadRetry, flushSave, imageCacheToken, loadStillCurrent, setError, topicSlug],
  );

  const loadChallenge = selectChallenge;

  const requestRelativeNav = useCallback((dir: 1 | -1) => {
    navStepRef.current = Math.max(-24, Math.min(24, navStepRef.current + dir));
    if (navRafRef.current) return;
    navRafRef.current = window.requestAnimationFrame(() => {
      navRafRef.current = 0;
      const steps = navStepRef.current;
      navStepRef.current = 0;
      if (!steps) return;
      const app = useLabeler.getState().appState;
      let cur = detailIndexRef.current ?? app?.current_index ?? -1;
      if (!app || cur < 0) return;
      const stepDir: 1 | -1 = steps > 0 ? 1 : -1;
      let target = cur;
      for (let i = 0; i < Math.abs(steps); i += 1) {
        const next = neighborFilteredIndex(target, stepDir);
        if (next < 0) break;
        target = next;
      }
      if (target !== cur) {
        void loadChallenge(target, "arrow").catch((err) => {
          if (!isAbortish(err) && !isStaleChallengeError(err)) setError(String(err));
        });
      }
    });
  }, [loadChallenge, neighborFilteredIndex, setError]);

  const requestNeedsWorkNav = useCallback((dir: 1 | -1) => {
    needsNavStepRef.current = Math.max(-24, Math.min(24, needsNavStepRef.current + dir));
    if (needsNavRafRef.current) return;
    needsNavRafRef.current = window.requestAnimationFrame(() => {
      needsNavRafRef.current = 0;
      const steps = needsNavStepRef.current;
      needsNavStepRef.current = 0;
      if (!steps) return;
      const app = useLabeler.getState().appState;
      let cur = detailIndexRef.current ?? app?.current_index ?? -1;
      if (!app || cur < 0) return;
      const stepDir: 1 | -1 = steps > 0 ? 1 : -1;
      let target = cur;
      for (let i = 0; i < Math.abs(steps); i += 1) {
        const next = neighborNeedsWorkIndex(target, stepDir, stripFilterRef.current, ownFilterRef.current);
        if (next < 0) break;
        target = next;
      }
      if (target !== cur) {
        void loadChallenge(target, "needs").catch((err) => {
          if (!isAbortish(err) && !isStaleChallengeError(err)) setError(String(err));
        });
      }
    });
  }, [loadChallenge, setError]);

  useEffect(
    () => () => {
      if (navRafRef.current) cancelAnimationFrame(navRafRef.current);
      if (needsNavRafRef.current) cancelAnimationFrame(needsNavRafRef.current);
    },
    [],
  );

  const workspacePoolsRef = useRef<{ platform: AppState | null; own: AppState | null }>({
    platform: null,
    own: null,
  });
  const switchingWorkspaceRef = useRef(false);

  const clearLabelerPanes = useCallback(() => {
    loadSeqRef.current += 1;
    loadAbortRef.current?.abort();
    detailIndexRef.current = null;
    openItemKeyRef.current = null;
    setViewIndex(null);
    useLabeler.getState().setDetail(null);
    useLabeler.getState().setBoxes([]);
    useLabeler.getState().setRefBoxes([]);
    useLabeler.getState().setSelected(null);
    setRefVariantIdx(0);
    activeRefVariantIdxGate.current = 0;
    setSyncTransform(null);
    setLinkedBoxFocus(null);
    clearDetailCache();
    clearDecodedImages();
    clearThumbnailBlobCache();
  }, []);

  const switchWorkspace = useCallback(
    (next: WorkspaceMode) => {
      const app = useLabeler.getState().appState;
      if (!app || workspaceFromState(app) === next || switchingWorkspaceRef.current) return;
      switchingWorkspaceRef.current = true;
      setModeBusy(true);
      loadSeqRef.current += 1;
      loadAbortRef.current?.abort();
      setError("");

      const prevMode = workspaceFromState(app);
      // Capture save targets BEFORE clearing panes.
      const prevIdx = detailIndexRef.current ?? app.current_index;
      const prevKey =
        dirtyItemKeyRef.current ||
        saveSnapshotRef.current?.itemKey ||
        (prevIdx >= 0 ? app.challenges[prevIdx]?.item_key : null) ||
        null;

      // Snapshot current pool so return trips feel instant.
      workspacePoolsRef.current[prevMode] = structuredClone
        ? structuredClone({ ...app, challenges: app.challenges })
        : { ...app, challenges: app.challenges.map((c) => ({ ...c })) };

      // Instant UI swap — Platform images must disappear immediately.
      const cached = workspacePoolsRef.current[next];
      const optimistic: AppState = cached
        ? {
            ...cached,
            review_mode: next === "own" ? "local" : "challenge",
            workspace: next,
            own_day: next === "own" ? null : cached.own_day ?? null,
            challenges: next === "own" ? [] : cached.challenges,
            current_index: next === "own" ? -1 : cached.current_index,
            platform_metrics: app.platform_metrics ?? cached.platform_metrics,
            own_metrics: app.own_metrics ?? cached.own_metrics,
            collect: next === "own" ? { running: false, message: "" } : cached.collect,
          }
        : {
            ...app,
            review_mode: next === "own" ? "local" : "challenge",
            workspace: next,
            own_day: next === "own" ? null : app.own_day ?? null,
            challenges: [],
            current_index: -1,
            metrics:
              next === "own"
                ? (app.own_metrics ?? defaultWorkspaceMetrics())
                : (app.platform_metrics ?? app.metrics),
            collect: next === "own" ? { running: false, message: "" } : app.collect,
          };
      applyState(optimistic);
      clearLabelerPanes();
      setOwnFilter("all");
      setStripFilter("all");

      // Background confirm — UI already swapped.
      void (async () => {
        try {
          if (prevMode === "platform" && prevIdx >= 0) {
            await flushSave(prevIdx, prevKey);
          }
          const preferKey =
            next === "own" && topicSlug ? readOwnCursor(topicSlug) : null;
          const s = await api.setWorkspace(next, preferKey);
          if (workspaceFromState(useLabeler.getState().appState || s) !== next) {
            // User flipped again while we were in flight.
            return;
          }
          workspacePoolsRef.current[next] = s;
          applyState(s);
          // Own day-list mode: no filmstrip until a day is entered.
          if (next === "own" && !s.own_day) {
            clearLabelerPanes();
            return;
          }
          const preferIdx =
            preferKey != null
              ? s.challenges.findIndex((c) => c.loaded && itemKeysMatch(c.item_key, preferKey))
              : -1;
          const idx =
            preferIdx >= 0
              ? preferIdx
              : s.current_index >= 0
                ? s.current_index
                : s.challenges.findIndex((c) => c.loaded);
          if (idx >= 0) {
            // loadChallenge() does nothing while a workspace switch is in progress, so end the switch first —
            // otherwise the image was never requested and "Loading image…" stayed until you clicked another one.
            switchingWorkspaceRef.current = false;
            await loadChallenge(idx, "workspace");
            scrollThumbIntoView(idx);
          } else {
            clearLabelerPanes();
          }
        } catch (e) {
          if (!isStaleChallengeError(e)) setError(String(e));
        } finally {
          switchingWorkspaceRef.current = false;
          setModeBusy(false);
        }
      })();
    },
    [clearLabelerPanes, flushSave, loadChallenge, scrollThumbIntoView, setError, topicSlug],
  );

  const onUploadOwnDone = useCallback(
    async (s: AppState & { upload?: { added: number; skipped: number; total: number } }) => {
      setError("");
      clearDetailCache();
      clearDecodedImages();
      clearThumbnailBlobCache();
      workspacePoolsRef.current.own = s;
      applyState(s);
      // Upload / HF import auto-enters today's day cluster.
      const idx = s.current_index >= 0 ? s.current_index : s.challenges.findIndex((c) => c.loaded);
      if (idx >= 0) await loadChallenge(idx, "upload");
      else clearLabelerPanes();
    },
    [clearLabelerPanes, loadChallenge],
  );

  const enterOwnDay = useCallback(
    async (date: string) => {
      setError("");
      setModeBusy(true);
      try {
        const preferKey = topicSlug ? readOwnCursor(topicSlug) : null;
        const s = await api.setOwnDay(date, preferKey, ownFieldRef.current);
        workspacePoolsRef.current.own = s;
        applyState(s);
        const preferIdx =
          preferKey != null
            ? s.challenges.findIndex((c) => c.loaded && itemKeysMatch(c.item_key, preferKey))
            : -1;
        const idx =
          preferIdx >= 0
            ? preferIdx
            : s.current_index >= 0
              ? s.current_index
              : s.challenges.findIndex((c) => c.loaded);
        if (idx >= 0) {
          await loadChallenge(idx, "workspace");
          scrollThumbIntoView(idx);
        } else {
          clearLabelerPanes();
        }
      } catch (e) {
        setError(String(e));
      } finally {
        setModeBusy(false);
      }
    },
    [clearLabelerPanes, loadChallenge, scrollThumbIntoView, topicSlug],
  );

  const leaveOwnDay = useCallback(async () => {
    setError("");
    setModeBusy(true);
    try {
      const idx = detailIndexRef.current;
      const key = useLabeler.getState().detail?.challenge?.item_key || null;
      if (idx != null && idx >= 0) await flushSave(idx, key);
      const s = await api.setOwnDay(null);
      workspacePoolsRef.current.own = s;
      applyState(s);
      clearLabelerPanes();
    } catch (e) {
      setError(String(e));
    } finally {
      setModeBusy(false);
    }
  }, [clearLabelerPanes, flushSave]);

  enterOwnDayRef.current = enterOwnDay;
  leaveOwnDayRef.current = leaveOwnDay;

  // The address bar always shows where you are, so a link can be sent to someone:
  //   /t/<topic>  ·  /t/<topic>/own[/<openai|gemini>]/days  ·  …/days/<date>
  useEffect(() => {
    if (!urlSync || !appState) return;
    const ws = workspaceFromState(appState);
    const wantPath =
      ws === "own"
        ? `/t/${topicSlug}/own${ownField ? `/${ownField}` : ""}/days${appState.own_day ? `/${appState.own_day}` : ""}`
        : `/t/${topicSlug}`;
    const wantSearch = ws === "own" ? "" : "?workspace=platform";
    if (location.pathname !== wantPath || location.search !== wantSearch) {
      navigate({ pathname: wantPath, search: wantSearch }, { replace: true });
    }
  }, [urlSync, appState?.own_day, appState?.workspace, appState?.review_mode, ownField, topicSlug, location.pathname, location.search, navigate]);

  /** Own day list: delete every image of these days (the list page shows its own confirmation). */
  const deleteOwnDaysFromList = useCallback(
    async (days: string[]) => {
      setError("");
      const next = await api.deleteOwnDays(days);
      workspacePoolsRef.current.own = next;
      applyState(next);
      clearDetailCache();
      clearDecodedImages();
      clearThumbnailBlobCache();
      clearLabelerPanes();
    },
    [clearLabelerPanes],
  );

  const clearOwnDayNew = useCallback(async () => {
    const app = useLabeler.getState().appState;
    if (!app?.own_day) {
      setError("Enter an Own day cluster first");
      return;
    }
    const newCount = app.challenges.filter((c) => c.loaded && !c.reviewed).length;
    if (newCount <= 0) {
      setError("No New images in this day to clear");
      return;
    }
    const ok = window.confirm(
      `Clear ${newCount} New image(s) from ${app.own_day}?\n\nDone labels are kept. This cannot be undone.`,
    );
    if (!ok) return;
    setError("");
    setModeBusy(true);
    try {
      const idx = detailIndexRef.current;
      const key = useLabeler.getState().detail?.challenge?.item_key || null;
      if (idx != null && idx >= 0) await flushSave(idx, key);
      const s = await api.clearOwnDayNew();
      workspacePoolsRef.current.own = s;
      applyState(s);
      clearDetailCache();
      clearDecodedImages();
      clearThumbnailBlobCache();
      if (!s.own_day) {
        clearLabelerPanes();
        return;
      }
      const nextIdx = s.current_index >= 0 ? s.current_index : s.challenges.findIndex((c) => c.loaded);
      if (nextIdx >= 0) {
        await loadChallenge(nextIdx, "workspace");
        scrollThumbIntoView(nextIdx);
      } else {
        clearLabelerPanes();
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setModeBusy(false);
    }
  }, [clearLabelerPanes, flushSave, loadChallenge, scrollThumbIntoView]);

  const openSam3Modal = useCallback(() => {
    const app = useLabeler.getState().appState;
    if (!app || app.review_mode !== "local") {
      setError("Open an Own day cluster first");
      return;
    }
    if (!app.own_day) {
      setError("Enter an Own day cluster before running SAM3");
      return;
    }
    setError("");
    setSam3Open(true);
  }, [setError]);

  const runSam3LabelOne = useCallback(
    async (opts: { index: number; itemKey: string }) => {
      // The Chutes key comes from the shared key vault on the server.
      const res = await api.ownSam3Label({
        index: opts.index,
        item_key: opts.itemKey,
      });
      // Do NOT clearDecodedImages here — closing ImageBitmaps while the canvas still
      // holds them leaves a black pane, and loadChallenge no-ops if already on this item.
      clearDetailCache();
      // Prefer server row for this item_key — list index can drift after applyState.
      const appNow = useLabeler.getState().appState;
      const rowFromRes =
        res.challenges.find((c) => itemKeysMatch(c.item_key, opts.itemKey)) ||
        (typeof res.sam3?.index === "number" ? res.challenges[res.sam3.index] : undefined);
      if (appNow && rowFromRes) {
        applyState({
          ...res,
          challenges: appNow.challenges.map((c) =>
            itemKeysMatch(c.item_key, opts.itemKey)
              ? {
                  ...c,
                  ...rowFromRes,
                  reviewed: false,
                  has_label: true,
                  cached_left_boxes: rowFromRes.cached_left_boxes ?? c.cached_left_boxes,
                  image_width: rowFromRes.image_width || c.image_width,
                  image_height: rowFromRes.image_height || c.image_height,
                }
              : c,
          ),
        });
      } else {
        applyState(res);
      }
      const liveIdx = useLabeler.getState().appState?.challenges.findIndex((c) =>
        itemKeysMatch(c.item_key, opts.itemKey),
      );
      const detail = res.detail;
      if (detail && liveIdx != null && liveIdx >= 0) {
        const boxes = normalizeBoxes(
          detail.boxes?.length ? detail.boxes : rowFromRes?.cached_left_boxes || [],
        );
        const w =
          detail.width ||
          rowFromRes?.image_width ||
          useLabeler.getState().detail?.width ||
          0;
        const h =
          detail.height ||
          rowFromRes?.image_height ||
          useLabeler.getState().detail?.height ||
          0;
        // Keep winner preds for dual base pane (NEW left / prediction_cache).
        const prev = useLabeler.getState().detail;
        const prevPreds =
          prev && itemKeysMatch(prev.challenge?.item_key, opts.itemKey)
            ? normalizeBoxes(prev.prediction_boxes || [])
            : [];
        const cachedPreds = winnerPredCache.get(opts.itemKey) || [];
        const serverPreds = normalizeBoxes(detail.prediction_boxes || []);
        const prediction_boxes = serverPreds.length
          ? serverPreds
          : prevPreds.length
            ? prevPreds
            : cachedPreds;
        if (prediction_boxes.length) {
          rememberWinnerPreds(opts.itemKey, prediction_boxes, boxes, false);
        }
        const painted = {
          ...detail,
          index: liveIdx,
          boxes,
          width: w,
          height: h,
          prediction_boxes,
          challenge: {
            ...detail.challenge,
            reviewed: false,
            has_label: true,
            item_key: opts.itemKey,
            image_width: w || detail.challenge?.image_width,
            image_height: h || detail.challenge?.image_height,
          },
        };
        // Paint SAM3 boxes when this image is (or becomes) the open one.
        const curKey = useLabeler.getState().detail?.challenge?.item_key;
        if (!curKey || itemKeysMatch(curKey, opts.itemKey)) {
          applyDetail(painted);
          setCachedDetail(liveIdx, painted);
        } else {
          setCachedDetail(liveIdx, painted);
        }
      }
      bumpStripTick();
    },
    [bumpStripTick],
  );

  const onSam3RunComplete = useCallback(
    async (opts: { labeled: number; failed: number; stopped: boolean; total: number }) => {
      setSam3Busy(false);
      if (opts.labeled <= 0 && opts.failed <= 0) return;
      if (opts.failed > 0) {
        setError(
          `SAM3 labeled ${opts.labeled}/${opts.total}` +
            (opts.stopped ? " (stopped)" : "") +
            `; ${opts.failed} failed`,
        );
      } else if (opts.stopped) {
        setError(`SAM3 stopped after ${opts.labeled}/${opts.total}`);
      }
      const app = useLabeler.getState().appState;
      if (!app) return;
      const firstSam3 = app.challenges.findIndex((c) => c.loaded && !c.reviewed && Boolean(c.has_label));
      if (firstSam3 >= 0 && opts.labeled > 0) {
        setOwnFilter("sam3");
        // Force a real reload even if this image is already selected — otherwise
        // selectChallenge early-returns and the pane can stay blank after SAM3.
        detailIndexRef.current = null;
        openItemKeyRef.current = null;
        await loadChallenge(firstSam3, "sam3");
        scrollThumbIntoView(firstSam3);
      }
      bumpStripTick();
    },
    [bumpStripTick, loadChallenge, scrollThumbIntoView, setError],
  );

  useEffect(() => {
    if (!topicReady || !appState) return;
    const want = pendingWorkspaceRef.current;
    if (!want || workspaceFromState(appState) === want) {
      pendingWorkspaceRef.current = null;
      return;
    }
    pendingWorkspaceRef.current = null;
    void switchWorkspace(want);
  }, [topicReady, appState, switchWorkspace]);

  // While labeling the current block, keep the next few full images warming in the background.
  useEffect(() => {
    const app = useLabeler.getState().appState;
    const d = useLabeler.getState().detail;
    if (!app || !d) return;
    const index = d.index ?? app.current_index;
    if (index < 0) return;
    preloadChallengeForward(
      index,
      app.review_mode,
      app.challenges,
      3,
      imageCacheToken,
      filteredNeighbors(index, stripFilterRef.current, 3, ownFilterRef.current),
    );
  }, [detail?.index, detail?.challenge?.item_key, appState?.challenges.length, appState?.review_mode, imageCacheToken, stripFilter, genFilter]);

  // The URL's ?workspace= is rewritten by switchWorkspace(); re-entering the topic on that
  // change cleared all panes and raced the in-flight setWorkspace request. Only the topic
  // itself should (re)trigger the enter effect.
  const initialWorkspaceRef = useRef(initialWorkspace);
  initialWorkspaceRef.current = initialWorkspace;

  useEffect(() => {
    if (!topicSlug) return;
    const initialWorkspace = initialWorkspaceRef.current;
    const prevApp = useLabeler.getState().appState;
    const keepStrip = Boolean(prevApp && stateMatchesTopic(prevApp, topicSlug));
    pinnedTopicSlug = topicSlug;
    setTopicReady(false);
    topicReadyRef.current = false;
    detailIndexRef.current = null;
    openItemKeyRef.current = null;
    setViewIndex(null);
    winnerPredCache.clear();
    pendingDoneKeys.clear();
    pendingNewKeys.clear();
    dirtyItemKeyRef.current = null;
    dirtyIndexRef.current = null;
    saveSnapshotRef.current = null;
    historyRef.current.clear();
    useLabeler.getState().setDetail(null);
    useLabeler.getState().setBoxes([]);
    useLabeler.getState().setRefBoxes([]);
    // Keep strip for same topic while re-entering after refresh — avoids "No images loaded" flash.
    if (!keepStrip) {
      useLabeler.getState().setAppState(null);
    }
    useLabeler.getState().setCurrentClass(0);
    setRefVariantIdx(0);
    activeRefVariantIdxGate.current = 0;
    clearDetailCache();
    setStripRange({ first: 0, last: 12 });
    setStripFilter("all");
    setOwnFilter("all");
    pendingWorkspaceRef.current = initialWorkspace ?? null;
    let cancelled = false;
    const preferOwnKey =
      initialWorkspace === "own" ? readOwnCursor(topicSlug) : null;
    api
      .enterTopic(topicSlug, {
        workspace: initialWorkspace ?? undefined,
        preferItemKey: preferOwnKey,
      })
      .then((state) => {
        if (cancelled) return;
        applyState(state);
        // Own-first enter already returned the right workspace — skip second switch.
        if (initialWorkspace && workspaceFromState(state) === initialWorkspace) {
          pendingWorkspaceRef.current = null;
        }
        topicReadyRef.current = true;
        setTopicReady(true);
        setError("");
        // A link to an Own day / the day page: open exactly that (each day has its own URL).
        if (initialWorkspace === "own") {
          const wantDay = initialOwnDayRef.current;
          const run = async () => {
            if (wantDay) await enterOwnDayRef.current?.(wantDay);
            else if (ownDaysPageRef.current && state.own_day) await leaveOwnDayRef.current?.();
          };
          run()
            .catch(() => undefined)
            .finally(() => !cancelled && setUrlSync(true));
        } else {
          setUrlSync(true);
        }
      })
      .catch((e) => {
        if (cancelled) return;
        topicReadyRef.current = false;
        setTopicReady(false);
        setError(String(e));
      });
    return () => {
      cancelled = true;
      setUrlSync(false);
    };
  }, [topicSlug, setError]);

  useEffect(() => {
    pendingWorkspaceRef.current = initialWorkspace ?? null;
  }, [initialWorkspace]);

  useEffect(() => {
    let source: EventSource | null = null;
    let fallback: number | null = null;
    const boot = async () => {
      if (topicSlug) return;
      for (let attempt = 0; attempt < 5; attempt += 1) {
        try {
          applyState(await api.getState());
          setError("");
          return;
        } catch (e) {
          if (attempt === 4) {
            const msg = String(e);
            setError(
              msg.includes("Failed to fetch")
                ? "Cannot reach the labeler server — run: cd web && ./run.sh"
                : msg,
            );
          } else {
            await new Promise((r) => window.setTimeout(r, 400 * (attempt + 1)));
          }
        }
      }
    };
    boot();
    const applySseState = (raw: AppState) => {
      if (topicSlug && !topicReadyRef.current) return;
      if (topicSlug && !stateMatchesTopic(raw, topicSlug)) return;
      // Never let SSE yank us between Platform/Own mid-switch, or bleed pools together.
      if (switchingWorkspaceRef.current) return;
      const live = useLabeler.getState().appState;
      if (live && raw.review_mode && live.review_mode && raw.review_mode !== live.review_mode) {
        return;
      }
      // Pin current_index to the image THIS tab has open. Server index follows every
      // activate=true getChallenge (and other tabs) and must not drive our selection.
      let incoming = raw;
      const openKey = openItemKeyRef.current;
      if (openKey && Array.isArray(raw.challenges) && raw.challenges.length) {
        const pinned = raw.challenges.findIndex(
          (c) => c.item_key === openKey || itemKeysMatch(c.item_key, openKey),
        );
        if (pinned >= 0 && pinned !== raw.current_index) {
          incoming = { ...raw, current_index: pinned };
        }
      } else if (
        detailIndexRef.current != null &&
        detailIndexRef.current >= 0 &&
        live &&
        live.review_mode === raw.review_mode
      ) {
        incoming = { ...raw, current_index: detailIndexRef.current };
      }
      // While labeling, keep local boxes stable but still sync collect / strip tags.
      if (dirtyItemKeyRef.current || editSessionRef.current || saveSnapshotRef.current) {
        const prev = useLabeler.getState().appState;
        if (
          prev &&
          prev.session_id === incoming.session_id &&
          prev.review_mode === incoming.review_mode &&
          Array.isArray(prev.challenges) &&
          Array.isArray(incoming.challenges)
        ) {
          const dirtyKey = dirtyItemKeyRef.current || saveSnapshotRef.current?.itemKey || null;
          const byKey = new Map(incoming.challenges.map((c) => [c.item_key, c]));
          const mergedChallenges = prev.challenges.map((pc) => {
            const rc =
              byKey.get(pc.item_key) ||
              incoming.challenges.find((c) => itemKeysMatch(c.item_key, pc.item_key));
            if (!rc) return pc;
            const pendingDone = pendingDoneKeys.has(pc.item_key);
            const pendingNew = pendingNewKeys.has(pc.item_key);
            let reviewed = Boolean(rc.reviewed);
            if (pendingDone) reviewed = true;
            else if (pendingNew) reviewed = false;
            else if (pc.reviewed && !rc.reviewed) reviewed = true; // sticky Done
            const editingThis =
              dirtyKey != null &&
              (pc.item_key === dirtyKey || itemKeysMatch(pc.item_key, dirtyKey));
            return {
              ...pc,
              reviewed,
              tuned: pendingDone ? false : rc.tuned,
              checks_passed:
                pc.reviewed || pendingDone
                  ? (pc.checks_passed ?? rc.checks_passed)
                  : (rc.checks_passed ?? pc.checks_passed),
              checks:
                pc.reviewed || pendingDone
                  ? (pc.checks ?? rc.checks)
                  : (rc.checks ?? pc.checks),
              checks_kind:
                pc.reviewed || pendingDone
                  ? (pc.checks_kind ?? rc.checks_kind)
                  : (rc.checks_kind ?? pc.checks_kind),
              top_boxes: rc.top_boxes?.length ? rc.top_boxes : pc.top_boxes,
              cached_left_boxes: editingThis
                ? pc.cached_left_boxes?.length
                  ? pc.cached_left_boxes
                  : rc.cached_left_boxes
                : rc.cached_left_boxes?.length
                  ? rc.cached_left_boxes
                  : pc.cached_left_boxes,
            };
          });
          applyState({
            ...prev,
            challenges: mergedChallenges,
            collect: incoming.collect,
            collect_seq: incoming.collect_seq,
            boxes_seq: incoming.boxes_seq,
            label_seq: incoming.label_seq,
            status: incoming.status,
            metrics: incoming.metrics,
            current_index: incoming.current_index,
          });
          return;
        }
      }
      applyState(incoming);
    };
    try {
      source = new EventSource(`/api/events?sid=${encodeURIComponent(getTabSessionId())}`);
      source.onmessage = (ev) => {
        try {
          applySseState(JSON.parse(ev.data) as AppState);
        } catch {
          /* ignore malformed frames */
        }
      };
      source.onerror = () => {
        // EventSource auto-reconnects; only fall back to polling if the socket stays dead.
        // Closing here permanently forced a 2s /api/state storm and burned server CPU.
        if (source?.readyState === EventSource.CLOSED) {
          source = null;
          if (fallback == null) {
            fallback = window.setInterval(() => {
              api.getState().then(applySseState).catch(() => undefined);
            }, 5000);
          }
        }
      };
      source.onopen = () => {
        if (fallback != null) {
          window.clearInterval(fallback);
          fallback = null;
        }
      };
    } catch {
      fallback = window.setInterval(() => {
        api.getState().then(applySseState).catch(() => undefined);
      }, 5000);
    }
    return () => {
      source?.close();
      if (fallback != null) window.clearInterval(fallback);
    };
  }, [setError, topicSlug]);

  useEffect(() => {
    if (!topicSlug || !topicReady) return;
    let debounce: number | null = null;
    const onStale = () => {
      if (debounce != null) window.clearTimeout(debounce);
      debounce = window.setTimeout(() => {
        debounce = null;
        api
          .getState()
          .then((state) => {
            if (stateMatchesTopic(state, topicSlug)) {
              applyState(state);
            }
          })
          .catch(() => undefined);
      }, 800);
    };
    window.addEventListener("labeler-stale-index", onStale);
    return () => {
      window.removeEventListener("labeler-stale-index", onStale);
      if (debounce != null) window.clearTimeout(debounce);
    };
  }, [topicSlug, topicReady]);

  useEffect(() => {
    clearDetailCache();
    clearDecodedImages();
    clearThumbnailBlobCache();
    // Rebuild strip thumbs after the blob cache wipe (Own↔Platform switch).
    window.requestAnimationFrame(() => preloadFilmstripWindow());
  }, [appState?.review_mode, preloadFilmstripWindow]);

  useEffect(() => {
    if (!topicReady || !appState) return;
    buildBlocksMap(appState);
    // Nearby JPEG pyramid only (backend caps Own to selected ±2).
    const prefer =
      appState.current_index >= 0
        ? appState.current_index
        : appState.challenges.findIndex((c) => c.loaded);
    if (prefer >= 0) {
      void api.warmMediaCache(prefer).catch(() => undefined);
      // Scroll strip near resume target; first-open will refine.
      window.requestAnimationFrame(() => {
        scrollThumbIntoView(prefer);
        preloadFilmstripWindow();
      });
    } else {
      window.requestAnimationFrame(() => preloadFilmstripWindow());
    }
  }, [
    topicReady,
    appState?.active_skill?.element_path,
    appState?.review_mode,
    buildBlocksMap,
    preloadFilmstripWindow,
    scrollThumbIntoView,
    imageCacheToken,
  ]);

  // Open last-edited (Own) or current_index once when the pool appears — not on every SSE.
  useEffect(() => {
    if (!topicReady || !appState) return;
    if (switchingWorkspaceRef.current) return;
    if (detailIndexRef.current != null) return;
    const preferKey =
      appState.review_mode === "local" ? readOwnCursor(topicSlug) : null;
    let idx =
      preferKey != null
        ? appState.challenges.findIndex(
            (c) => c.loaded && itemKeysMatch(c.item_key, preferKey),
          )
        : -1;
    if (idx < 0 && appState.current_index >= 0 && appState.challenges[appState.current_index]?.loaded) {
      idx = appState.current_index;
    }
    if (idx < 0) idx = appState.challenges.findIndex((c) => c.loaded);
    if (idx >= 0) {
      followCollectRef.current = false;
      scrollThumbIntoView(idx);
      loadChallenge(idx, "first-open").catch((e) => {
        if (!isStaleChallengeError(e)) setError(String(e));
      });
    }
  }, [
    topicReady,
    topicSlug,
    appState?.challenges.length,
    appState?.review_mode,
    appState?.current_index,
    loadChallenge,
    scrollThumbIntoView,
    setError,
  ]);

  // Collect prepends newer blocks at index 0 — keep the open image by item_key, not index.
  // If the open image is not in the active pool (Platform↔Own switch), drop it immediately.
  //
  // CRITICAL: do NOT depend on `appState.challenges` identity. Label saves / applyState create
  // a new array every edit; that used to re-run this effect with a STALE detail.item_key and
  // yank detailIndexRef back to the previous image mid-navigation (N/B/←/→ / filmstrip broken).
  useEffect(() => {
    if (!appState || !topicReady) return;

    // Prefer the item_key of the image the user opened (not challenges[index],
    // which is a different image after collect prepends).
    const detail = useLabeler.getState().detail;
    const key = openItemKeyRef.current || detail?.challenge?.item_key || null;

    const snap = saveSnapshotRef.current;
    if (snap?.itemKey) {
      const ni = appState.challenges.findIndex((c) => c.item_key === snap.itemKey);
      if (ni >= 0) saveSnapshotRef.current = { ...snap, index: ni };
    }
    if (dirtyItemKeyRef.current) {
      const ni = appState.challenges.findIndex((c) => c.item_key === dirtyItemKeyRef.current);
      dirtyIndexRef.current = ni >= 0 ? ni : null;
    }
    if (!key) {
      if (detailIndexRef.current != null && appState.challenges.length === 0) {
        clearLabelerPanes();
      }
      return;
    }
    // Exact stem match only — never fuzzy Own v01 collisions.
    const nextIdx = appState.challenges.findIndex((c) => c.item_key === key);
    if (nextIdx < 0) {
      // Pool empty (workspace swap) → clear. Otherwise keep the open image; a brief
      // SSE mismatch must not wipe the canvas and re-trigger auto-open thrash.
      if (appState.challenges.length === 0) {
        clearLabelerPanes();
        setError("");
      }
      return;
    }
    if (nextIdx === detailIndexRef.current) {
      // Same slot — keep in-flight getChallenge; only refresh challenge row metadata.
      if (detail && detail.index === nextIdx) {
        const row = appState.challenges[nextIdx];
        if (row && detail.challenge?.item_key === row.item_key) {
          if (detail.challenge.reviewed !== row.reviewed) {
            useLabeler.getState().setDetail({
              ...detail,
              challenge: { ...detail.challenge, ...row },
            });
          }
        }
      }
      return;
    }
    // Key moved to a new list index (collect re-sort) — remap without aborting unless needed.
    detailIndexRef.current = nextIdx;
    openItemKeyRef.current = key;
    setViewIndex(nextIdx);
    if (detail) {
      useLabeler.getState().setDetail({
        ...detail,
        index: nextIdx,
        challenge: { ...detail.challenge, ...appState.challenges[nextIdx] },
      });
    }
  }, [
    appState?.collect_seq,
    appState?.review_mode,
    appState?.challenges.length,
    topicReady,
    clearLabelerPanes,
    setError,
  ]);

  useEffect(() => {
    if (!appState) return;
    buildBlocksMap(appState);
    window.requestAnimationFrame(() => preloadFilmstripWindow());
  }, [appState?.collect_seq, appState?.challenges.length, appState?.review_mode, buildBlocksMap, imageCacheToken, preloadFilmstripWindow]);

  useEffect(() => {
    preloadFilmstripWindow();
  }, [detail?.index, appState?.current_index, preloadFilmstripWindow]);

  useEffect(() => {
    const currentIdx = detail?.index ?? appState?.current_index ?? -1;
    if (currentIdx < 0) return;
    scrollThumbIntoView(currentIdx);
  }, [detail?.index, appState?.current_index, scrollThumbIntoView]);

  useEffect(() => {
    if (!appState) return;
    if (switchingWorkspaceRef.current) return;
    setStripRange({ first: 0, last: 12 });
    if (filmstripRef.current) filmstripRef.current.scrollLeft = 0;
    window.requestAnimationFrame(() => preloadFilmstripWindow());
    const cur = detailIndexRef.current ?? appState.current_index;
    const ch = cur >= 0 ? appState.challenges[cur] : null;
    const indices = filteredStripIndices(appState, stripFilter, ownFilter);
    const curView = ch ? (appState.review_mode === "local" ? ch : challengeForStrip(ch, cur)) : null;
    const curMatches =
      cur >= 0 &&
      ch?.loaded &&
      curView &&
      matchesFilmstripFilter(appState, curView, stripFilter, ownFilter);
    if (cur >= 0 && ch?.loaded) {
      void flushSave(cur, ch.item_key).finally(() => {
        if (curMatches) {
          scrollThumbIntoView(cur, "auto");
          bumpStripTick();
        }
      });
      if (curMatches) return;
    }
    // Only jump when the strip filter hides the open image — never on workspace switch
    // (switchWorkspace already opens the right frame).
    if (indices.length) void selectChallenge(indices[0], "filter");
  }, [stripFilter, ownFilter, genFilter]);

  useEffect(() => {
    if (!appState) return;
    const boxesSeq = appState.boxes_seq ?? 0;
    if (boxesSeq === prevBoxesSeqRef.current) return;
    prevBoxesSeqRef.current = boxesSeq;
    const idx = detailIndexRef.current ?? appState.current_index;
    if (idx == null || idx < 0 || !appState.challenges[idx]?.loaded) return;
    // Never clobber in-progress local edits or a pending save with a late SSE refresh.
    if (
      (dirtyItemKeyRef.current && itemKeysMatch(dirtyItemKeyRef.current, appState.challenges[idx]?.item_key)) ||
      dirtyIndexRef.current === idx ||
      saveSnapshotRef.current?.index === idx ||
      (saveSnapshotRef.current?.itemKey &&
        itemKeysMatch(saveSnapshotRef.current.itemKey, appState.challenges[idx]?.item_key))
    ) {
      return;
    }
    if (editSessionRef.current && detailIndexRef.current === idx) return;
    const ch = appState.challenges[idx];
    const { left, ref } = boxesFromChallenge(ch);
    if (detailIndexRef.current === idx) {
      const painted = useLabeler.getState().boxes;
      const reviewed = Boolean(ch.reviewed || useLabeler.getState().detail?.challenge?.reviewed);
      // Enrichment may fill top_boxes; do not rewrite left GT / already-painted seeds.
      if (!reviewed && !painted.length && left.length) {
        setBoxes(left);
        if (ch.item_key) rememberWinnerPreds(ch.item_key, left, painted, false);
      }
      const current = useLabeler.getState().detail;
      if (current && (current.index === idx || itemKeysMatch(current.challenge?.item_key, ch?.item_key)) && (left.length || ref.length)) {
        useLabeler.getState().setDetail({
          ...current,
          boxes: reviewed || painted.length ? painted : left.length ? left : current.boxes,
          // Never force top_boxes (R1) onto the right pane — selected R2/R3 must stick.
          reference_boxes: current.reference_boxes,
        });
        reapplyActiveRefVariant(useLabeler.getState().detail);
      } else if (ref.length && activeRefVariantIdxGate.current === 0) {
        useLabeler.getState().setRefBoxes(ref);
      }
      // Keep strip n/3 aligned when server recompute lands — do not rescore here
      // (scoring all R1–R3 on every boxes_seq made labeling feel sticky).
      if (reviewed && ch.checks_passed != null) {
        bumpStripTick();
      }
    }
  }, [appState?.boxes_seq, appState?.current_index, bumpStripTick, setBoxes]);

  useEffect(() => {
    if (!appState) return;
    const loadedCount = appState.challenges.filter((c) => c.loaded).length;
    onFilmstripScroll();
    // Collect adds newest at left (index 0). Only nudge scroll if user is already on the left —
    // never change the selected labeling image.
    if (appState.collect.running && loadedCount > prevLoadedCountRef.current && followCollectRef.current) {
      if (filmstripRef.current) filmstripRef.current.scrollLeft = 0;
      scrollThumbIntoView(0, "auto");
    }
    prevLoadedCountRef.current = loadedCount;
  }, [appState?.collect_seq, appState?.challenges.length, appState?.collect.running, onFilmstripScroll, scrollThumbIntoView]);

  // Live drag boxes -> partner (only while someone else is looking at this image).
  const draftSentAtRef = useRef(0);
  const streamDraft = useCallback((boxesNow: DetectionBox[]) => {
    const team = useTeam.getState();
    const key = useLabeler.getState().detail?.challenge?.item_key;
    if (!key) return;
    const someoneThere =
      team.watchers.length > 0 || team.users.some((u) => u.name !== team.me?.user && u.online && u.item_key === key);
    if (!someoneThere) return;
    const now = performance.now();
    if (now - draftSentAtRef.current < 150) return;
    draftSentAtRef.current = now;
    void api.teamDraft(key, boxesNow).catch(() => undefined);
  }, []);

  // 409: a teammate changed this image after we opened it -> show their version.
  const reloadOpenImageRef = useRef<((itemKey: string) => void) | null>(null);
  reloadOpenImageRef.current = (itemKey: string) => {
    dropCachedDetail(itemKey);
    useTeam.getState().setNotice("Your partner changed this image — reloaded their version");
    const app = useLabeler.getState().appState;
    const idx = app?.challenges.findIndex((c) => c.item_key === itemKey) ?? -1;
    if (idx < 0 || openItemKeyRef.current !== itemKey) return;
    openItemKeyRef.current = null;
    detailIndexRef.current = null;
    void selectChallenge(idx, "conflict-reload").catch(() => undefined);
  };
  useEffect(() => {
    setSaveConflictHandler((key) => reloadOpenImageRef.current?.(key));
    return () => setSaveConflictHandler(null);
  }, []);

  const persistBoxes = useCallback(
    (next: DetectionBox[], sel: number | null, commit = true, recordHistory = true) => {
      if (useTeam.getState().readOnly) return; // teammate owns this image / we are watching
      const index = detailIndexRef.current;
      if (!commit) streamDraft(next);
      const appEarly = useLabeler.getState().appState;
      const histKey =
        (index != null ? appEarly?.challenges[index]?.item_key : null) ||
        useLabeler.getState().detail?.challenge?.item_key ||
        null;
      if (recordHistory && histKey) {
        const prev = useLabeler.getState().boxes;
        if (!boxesEqual(prev, next)) {
          if (!commit && !editSessionRef.current) {
            const hist = historyRef.current.get(histKey) ?? { undo: [], redo: [] };
            hist.undo.push(cloneBoxes(prev));
            if (hist.undo.length > 80) hist.undo.shift();
            hist.redo = [];
            historyRef.current.set(histKey, hist);
            editSessionRef.current = true;
          } else if (commit && !editSessionRef.current) {
            const hist = historyRef.current.get(histKey) ?? { undo: [], redo: [] };
            hist.undo.push(cloneBoxes(prev));
            if (hist.undo.length > 80) hist.undo.shift();
            hist.redo = [];
            historyRef.current.set(histKey, hist);
          }
        }
      }
      if (commit) editSessionRef.current = false;
      setBoxes(next);
      setSelected(sel);
      const current = useLabeler.getState().detail;
      const app = useLabeler.getState().appState;
      const asDicts = next.map((b) => ({
        cls: b.cls,
        x1: b.x1,
        y1: b.y1,
        x2: b.x2,
        y2: b.y2,
        conf: b.conf ?? null,
      }));
      const itemKey =
        histKey ||
        (index != null ? app?.challenges[index]?.item_key : null) ||
        current?.challenge?.item_key ||
        null;

      if (current) {
        const score = scoreTopAgainstLeftGt(useLabeler.getState().refBoxes, next, current.record || {});
        // Own SAM3 draft (strip has_label, not reviewed): keep unreviewed until M.
        // NEW (no has_label): commit → Done. Prefer strip flags — detail.has_label can be stale.
        const stripRow =
          (index != null ? app?.challenges[index] : null) ||
          (itemKey ? app?.challenges.find((c) => itemKeysMatch(c.item_key, itemKey)) : null);
        const wasSam3 =
          app?.review_mode === "local" &&
          Boolean(stripRow?.has_label) &&
          !Boolean(stripRow?.reviewed) &&
          !pendingDoneKeys.has(itemKey || "");
        const markDone = Boolean(commit && !wasSam3);
        const updated = {
          ...current,
          boxes: next,
          score,
          ...(commit
            ? {
                challenge: {
                  ...current.challenge,
                  reviewed: markDone ? true : false,
                  has_label: true,
                  tuned: false,
                  cached_left_boxes: asDicts,
                },
              }
            : {}),
        };
        useLabeler.getState().setDetail(updated);
        setCachedDetail(current.index, updated);
      }

      if (commit && index != null && app) {
        const stripRow = app.challenges[index];
        const wasSam3 =
          app.review_mode === "local" &&
          Boolean(stripRow?.has_label) &&
          !Boolean(stripRow?.reviewed) &&
          !pendingDoneKeys.has(itemKey || "");
        const markDone = !wasSam3;
        if (markDone) notePendingDone(itemKey);
        const wasReviewed = Boolean(stripRow?.reviewed);
        const asDictsLocal = asDicts;
        const nextLocal = next;
        const idxLocal = index;
        const variants = useLabeler.getState().detail?.reference_variants;
        const row = app.challenges[idxLocal];
        const checksPatch = markDone
          ? checksPatchForItem(
              { ...row, reviewed: true, cached_left_boxes: asDictsLocal },
              nextLocal,
              variants,
            )
          : null;
        const rowPatch = {
          reviewed: markDone,
          has_label: true,
          tuned: false,
          cached_left_boxes: asDictsLocal,
          ...(checksPatch || { checks_passed: null, checks: null }),
        };
        applyState({
          ...app,
          challenges: app.challenges.map((c, i) => (i === idxLocal ? { ...c, ...rowPatch } : c)),
          metrics:
            wasReviewed || !markDone
              ? app.metrics
              : {
                  ...app.metrics,
                  ready_to_tune: (app.metrics.ready_to_tune || 0) + 1,
                  not_reviewed: Math.max(0, (app.metrics.not_reviewed || 0) - 1),
                },
        });
        bumpStripTick();
        if (markDone && checksPatch && checksPatch.checks_passed == null && itemKey) {
          window.setTimeout(() => {
            const live = useLabeler.getState();
            const v = live.detail?.reference_variants;
            if (!v?.length) return;
            if (!itemKeysMatch(live.detail?.challenge?.item_key, itemKey)) return;
            const appLive = live.appState;
            if (!appLive) return;
            const rowLive = appLive.challenges.find((c) => itemKeysMatch(c.item_key, itemKey));
            if (!rowLive) return;
            const retryPatch = checksPatchForItem({ ...rowLive, reviewed: true }, nextLocal, v);
            if (retryPatch.checks_passed == null) return;
            patchChallengeChecksByKey(itemKey, retryPatch);
            bumpStripTick();
          }, 0);
        }
      }

      if (!commit) return;
      if (index == null || !itemKey) return;
      dirtyIndexRef.current = index;
      dirtyItemKeyRef.current = itemKey;
      const snapshot = next.map((b) => ({ ...b }));
      const gen = ++saveGenRef.current;
      const appNow = useLabeler.getState().appState;
      const rowNow = appNow?.challenges[index];
      // NEW → Done; SAM3 draft (has_label, !reviewed) stays draft until M.
      const saveAsDone =
        pendingDoneKeys.has(itemKey) ||
        !(
          appNow?.review_mode === "local" &&
          rowNow &&
          !rowNow.reviewed &&
          Boolean(rowNow.has_label)
        );
      saveSnapshotRef.current = { index, itemKey, boxes: snapshot, gen, done: saveAsDone };
      if (saveTimer.current) window.clearTimeout(saveTimer.current);
      const runSave = () => {
        const snap = saveSnapshotRef.current;
        if (!snap || snap.gen !== gen || !itemKeysMatch(snap.itemKey, itemKey)) return;
        const target = resolveSaveTarget(snap.index, snap.itemKey);
        if (!target) return;
        const teamKey = target.itemKey || itemKey;
        // A fresh edit supersedes any older failed save still waiting to retry.
        dropPendingSave(teamKey);
        useTeam.getState().setSave("saving");
        api
          .saveBoxes(target.index, snap.boxes, saveAsDone, target.itemKey || undefined, useTeam.getState().revs[teamKey])
          .then((res) => {
            useTeam.getState().noteRev(teamKey, res.rev);
            useTeam.getState().noteLabel(teamKey, res.labeled_by, res.rev);
            if (saveGenRef.current !== gen) return;
            if (!hasPendingSaves()) useTeam.getState().setSave("saved");
            if (saveSnapshotRef.current?.gen && saveSnapshotRef.current.gen > gen) return;
            if (itemKeysMatch(saveSnapshotRef.current?.itemKey, itemKey)) {
              saveSnapshotRef.current = null;
            }
            if (itemKeysMatch(dirtyItemKeyRef.current, itemKey)) {
              dirtyItemKeyRef.current = null;
              dirtyIndexRef.current = null;
            }
          })
          .catch((err) => {
            if (saveGenRef.current !== gen) return; // a newer edit owns the status now
            const status = err instanceof HttpError ? err.status : 0;
            const msg = err instanceof Error ? err.message : String(err);
            if (status === 423) {
              useTeam.getState().setSave("locked", msg);
              useTeam.getState().setNotice(msg);
              return;
            }
            if (status === 409) {
              useTeam.getState().setSave("conflict", msg);
              reloadOpenImageRef.current?.(teamKey);
              return;
            }
            retrySave(teamKey, target.index, snap.boxes, saveAsDone);
          });
      };
      // Coalesce rapid commits (draw/resize/delete/hotkeys) into one network write.
      saveTimer.current = window.setTimeout(runSave, 120);
    },
    [bumpStripTick, resolveSaveTarget, setBoxes, setSelected],
  );

  const undoBoxes = useCallback(() => {
    const index = detailIndexRef.current;
    if (index == null) return;
    const itemKey =
      useLabeler.getState().appState?.challenges[index]?.item_key ||
      useLabeler.getState().detail?.challenge?.item_key;
    if (!itemKey) return;
    const hist = historyRef.current.get(itemKey);
    if (!hist?.undo.length) return;
    const current = useLabeler.getState().boxes;
    hist.redo.push(cloneBoxes(current));
    const prev = hist.undo.pop()!;
    persistBoxes(prev, null, true, false);
  }, [persistBoxes]);

  const redoBoxes = useCallback(() => {
    const index = detailIndexRef.current;
    if (index == null) return;
    const itemKey =
      useLabeler.getState().appState?.challenges[index]?.item_key ||
      useLabeler.getState().detail?.challenge?.item_key;
    if (!itemKey) return;
    const hist = historyRef.current.get(itemKey);
    if (!hist?.redo.length) return;
    const current = useLabeler.getState().boxes;
    hist.undo.push(cloneBoxes(current));
    const next = hist.redo.pop()!;
    persistBoxes(next, null, true, false);
  }, [persistBoxes]);

  const clearPendingSaveFor = useCallback((index: number, itemKey?: string | null) => {
    const key =
      itemKey ||
      useLabeler.getState().appState?.challenges[index]?.item_key ||
      useLabeler.getState().detail?.challenge?.item_key ||
      null;
    if (saveTimer.current) {
      window.clearTimeout(saveTimer.current);
      saveTimer.current = null;
    }
    if (
      saveSnapshotRef.current &&
      (saveSnapshotRef.current.index === index || (key && itemKeysMatch(saveSnapshotRef.current.itemKey, key)))
    ) {
      saveSnapshotRef.current = null;
    }
    if (dirtyIndexRef.current === index || (key && itemKeysMatch(dirtyItemKeyRef.current, key))) {
      dirtyIndexRef.current = null;
      dirtyItemKeyRef.current = null;
    }
    editSessionRef.current = false;
    // Invalidate in-flight saveBoxes responses so they cannot re-mark reviewed after A/M/H.
    saveGenRef.current += 1;
  }, []);

  const resetToWinnerOrigin = useCallback(() => {
    if (useTeam.getState().readOnly) {
      useTeam.getState().setNotice("Read-only — you can't change this image right now");
      return;
    }
    const app = useLabeler.getState().appState;
    const current = useLabeler.getState().detail;
    const index = detailIndexRef.current;
    if (!app || !current || index == null) return;
    // Same sync gate as M/H: only act when the loaded detail matches the selected row.
    if (current.index !== index) return;
    const ch = app.challenges[index];
    if (!ch?.loaded) return;

    const gen = ++againGenRef.current;
    setError("");

    // Desktop Again (A): always re-run current base miner.py — do not paint stale
    // winnerPredCache / prediction_cache from a previous model.
    const itemKey = ch.item_key;
    const prev = useLabeler.getState().boxes;
    if (itemKey) winnerPredCache.delete(itemKey);

    clearPendingSaveFor(index, itemKey);

    // Instant filmstrip: Done/SAM3 → New (do not wait for the API).
    notePendingNew(itemKey);
    const challenges = app.challenges.map((c, i) =>
      i === index
        ? {
            ...c,
            reviewed: false,
            has_label: false,
            tuned: false,
            checks_passed: null,
            checks: null,
            cached_left_boxes: [],
          }
        : c,
    );
    applyState({
      ...app,
      challenges,
      metrics: {
        ...app.metrics,
        ready_to_tune: Math.max(0, (app.metrics.ready_to_tune || 0) - (ch.reviewed ? 1 : 0)),
        not_reviewed: (app.metrics.not_reviewed || 0) + (ch.reviewed ? 1 : 0),
      },
    });

    bumpStripTick();
    // Clear left pane until fresh miner.py result arrives (usually <100ms).
    if (prev.length) {
      const hist = historyRef.current.get(itemKey) ?? { undo: [], redo: [] };
      hist.undo.push(cloneBoxes(prev));
      if (hist.undo.length > 80) hist.undo.shift();
      hist.redo = [];
      if (itemKey) historyRef.current.set(itemKey, hist);
    }
    setBoxes([]);
    setSelected(null);
    const pending = {
      ...current,
      boxes: [],
      prediction_boxes: [],
      score: null,
      challenge: { ...current.challenge, reviewed: false, has_label: false, tuned: false },
    };
    useLabeler.getState().setDetail(pending);
    setCachedDetail(index, pending);

    const mergeAgainState = (resultState: AppState) => {
      const appNow = useLabeler.getState().appState;
      if (!appNow) {
        applyState(resultState);
        return;
      }
      const row = resultState.challenges[index];
      if (!row) {
        applyState({ ...appNow, metrics: resultState.metrics });
        return;
      }
      // Merge only this row + metrics. Never adopt a stolen current_index from a late response.
      applyState({
        ...appNow,
        metrics: resultState.metrics,
        challenges: appNow.challenges.map((c, i) =>
          i === index
            ? {
                ...c,
                reviewed: false,
                has_label: false,
                tuned: row.tuned,
                cached_left_boxes: row.cached_left_boxes,
              }
            : c,
        ),
      });
    };

    void api
      .reviewAgain(index)
      .then((result) => {
        const stale = againGenRef.current !== gen;
        const navigatedAway = detailIndexRef.current !== index;

        mergeAgainState(result.state);

        if (stale || navigatedAway) return;

        const boxes = normalizeBoxes(result.detail.prediction_boxes || result.detail.boxes);
        rememberWinnerPreds(itemKey, boxes, boxes, false);
        applyDetail({
          ...result.detail,
          boxes,
          challenge: { ...result.detail.challenge, reviewed: false, has_label: false },
        });
        setCachedDetail(index, {
          ...result.detail,
          boxes,
          challenge: { ...result.detail.challenge, reviewed: false, has_label: false },
        });
        bumpStripTick();
      })
      .catch((e) => {
        if (againGenRef.current === gen) setError(String(e));
      });
  }, [bumpStripTick, clearPendingSaveFor, setBoxes, setError, setSelected]);

  const advanceAfterReview = useCallback(
    (currentRow: number) => {
      const next = neighborNeedsWorkIndex(currentRow, 1, stripFilterRef.current, ownFilterRef.current);
      reviewActionBusyRef.current = false;
      if (next >= 0) {
        void loadChallenge(next, "mark-next").catch((err) => {
          if (!isAbortish(err) && !isStaleChallengeError(err)) setError(String(err));
        });
      }
    },
    [loadChallenge, setError],
  );

  /** Desktop mark_current_as_reviewed (M): save left boxes, mark reviewed, advance. */
  const markReviewedAndNext = useCallback(() => {
    if (useTeam.getState().readOnly) {
      useTeam.getState().setNotice("Read-only — you can't change this image right now");
      return;
    }
    if (reviewActionBusyRef.current) return;
    const app = useLabeler.getState().appState;
    const detail = useLabeler.getState().detail;
    const cur = detailIndexRef.current ?? app?.current_index ?? -1;
    if (!app || cur < 0) return;
    const ch = app.challenges[cur];
    if (!ch?.loaded) return;
    // Match by item_key — index can lag after Own pool reload / filter changes.
    const openKey = openItemKeyRef.current || detail?.challenge?.item_key;
    if (!detail || !openKey || !itemKeysMatch(openKey, ch.item_key)) return;

    reviewActionBusyRef.current = true;
    setError("");

    if (ch.reviewed) {
      const pendingSave = itemKeysMatch(saveSnapshotRef.current?.itemKey, ch.item_key);
      const dirty = itemKeysMatch(dirtyItemKeyRef.current, ch.item_key);
      if (pendingSave || dirty) {
        void flushSave(cur, ch.item_key).finally(() => {
          reviewActionBusyRef.current = false;
          advanceAfterReview(cur);
        });
        return;
      }
      advanceAfterReview(cur);
      return;
    }

    const boxes = cloneBoxes(useLabeler.getState().boxes);
    const asDicts = boxes.map((b) => ({
      cls: b.cls,
      x1: b.x1,
      y1: b.y1,
      x2: b.x2,
      y2: b.y2,
      conf: b.conf ?? null,
    }));
    clearPendingSaveFor(cur, ch.item_key);

    const variants = useLabeler.getState().detail?.reference_variants;
    const checksPatch = checksPatchForItem({ ...ch, reviewed: true }, boxes, variants);
    notePendingDone(ch.item_key);
    const challenges = app.challenges.map((c, i) =>
      i === cur
        ? {
            ...c,
            reviewed: true,
            tuned: false,
            cached_left_boxes: asDicts,
            ...checksPatch,
          }
        : c,
    );
    applyState({
      ...app,
      challenges,
      metrics: {
        ...app.metrics,
        ready_to_tune: (app.metrics.ready_to_tune || 0) + 1,
        not_reviewed: Math.max(0, (app.metrics.not_reviewed || 0) - 1),
      },
    });
    bumpStripTick();
    const score = scoreTopAgainstLeftGt(useLabeler.getState().refBoxes, boxes, detail.record || {});
    const updated = {
      ...detail,
      boxes,
      score,
      challenge: { ...detail.challenge, reviewed: true, cached_left_boxes: asDicts },
    };
    useLabeler.getState().setDetail(updated);
    setCachedDetail(cur, updated);

    const itemKey = ch.item_key;
    void api
      .saveBoxes(cur, boxes, true, itemKey)
      .then(() => {
        // Keep pendingDone until applyState sees server reviewed=true — clearing here
        // let SSE flip prior strip tags back to NEW before the label hit disk.
        bumpStripTick();
      })
      .catch((e) => setError(String(e)));
    advanceAfterReview(cur);
  }, [advanceAfterReview, bumpStripTick, clearPendingSaveFor, flushSave, setError]);

  /** Desktop accept_reference_labels_and_next (H): copy selected right reference → left, save, advance. */
  const acceptRefAndNext = useCallback(() => {
    if (useTeam.getState().readOnly) {
      useTeam.getState().setNotice("Read-only — you can't change this image right now");
      return;
    }
    if (reviewActionBusyRef.current) return;
    const app = useLabeler.getState().appState;
    const detail = useLabeler.getState().detail;
    const cur = detailIndexRef.current ?? app?.current_index ?? -1;
    if (!app || cur < 0) return;
    // Own has no miner reference rail — H is Platform-only.
    if (app.review_mode === "local") return;
    const ch = app.challenges[cur];
    if (!ch?.loaded) return;
    if (!detail || !itemKeysMatch(detail.challenge?.item_key, ch.item_key)) return;

    const variants = normalizeReferenceVariants(detail.reference_variants || []);
    const pickedIdx = variants.length ? Math.max(0, Math.min(refVariantIdx, variants.length - 1)) : 0;
    const selectedRef = variants[pickedIdx]?.boxes?.length
      ? variants[pickedIdx].boxes
      : cloneBoxes(useLabeler.getState().refBoxes);
    if (!selectedRef.length) {
      setError("No reference boxes on the right pane to accept.");
      return;
    }

    reviewActionBusyRef.current = true;
    setError("");
    clearPendingSaveFor(cur, ch.item_key);

    const cloned = cloneBoxes(selectedRef);
    const asDicts = cloned.map((b) => ({
      cls: b.cls,
      x1: b.x1,
      y1: b.y1,
      x2: b.x2,
      y2: b.y2,
      conf: b.conf ?? null,
    }));
    const checksPatch = checksPatchForItem({ ...ch, reviewed: true }, cloned, variants);
    notePendingDone(ch.item_key);
    const challenges = app.challenges.map((c, i) =>
      i === cur
        ? {
            ...c,
            reviewed: true,
            tuned: false,
            cached_left_boxes: asDicts,
            ...checksPatch,
          }
        : c,
    );
    applyState({
      ...app,
      challenges,
      metrics: {
        ...app.metrics,
        ready_to_tune: (app.metrics.ready_to_tune || 0) + (ch.reviewed ? 0 : 1),
        not_reviewed: ch.reviewed
          ? app.metrics.not_reviewed
          : Math.max(0, (app.metrics.not_reviewed || 0) - 1),
      },
    });
    bumpStripTick();
    const score = scoreTopAgainstLeftGt(cloned, cloned, detail.record || {});
    const updated = {
      ...detail,
      boxes: cloned,
      score,
      challenge: { ...detail.challenge, reviewed: true, cached_left_boxes: asDicts },
    };
    useLabeler.getState().setDetail(updated);
    setCachedDetail(cur, updated);
    setBoxes(cloned);
    setSelected(null);

    const itemKey = ch.item_key;
    void api
      .acceptReference(cur, pickedIdx, itemKey)
      .then((s) => {
        // Keep pendingDone until applyState sees server reviewed=true (same as M).
        // Merge only this row's server checks — do not clobber the image we navigated to.
        const live = useLabeler.getState().appState;
        if (!live) {
          applyState(s);
          return;
        }
        const serverRow = s.challenges.find((c) => itemKeysMatch(c.item_key, itemKey));
        if (!serverRow) {
          bumpStripTick();
          return;
        }
        applyState({
          ...live,
          challenges: live.challenges.map((c) =>
            itemKeysMatch(c.item_key, itemKey)
              ? {
                  ...c,
                  reviewed: true,
                  cached_left_boxes: serverRow.cached_left_boxes ?? c.cached_left_boxes,
                  checks_passed: serverRow.checks_passed ?? c.checks_passed,
                  checks: serverRow.checks ?? c.checks,
                  checks_kind: serverRow.checks_kind ?? c.checks_kind,
                }
              : c,
          ),
          metrics: s.metrics,
        });
        bumpStripTick();
      })
      .catch((e) => setError(String(e)));
    advanceAfterReview(cur);
  }, [advanceAfterReview, bumpStripTick, clearPendingSaveFor, refVariantIdx, setBoxes, setError, setSelected]);

  const handleClassHotkey = useCallback(
    (classId: number) => {
      if (classId < 0 || classId >= (appState?.model_class_names.length ?? 0)) return;
      const sel = useLabeler.getState().selected;
      const current = useLabeler.getState().boxes;
      if (sel != null && sel >= 0 && sel < current.length) {
        const old = current[sel];
        if (old.cls !== classId) {
          const next = current.map((b, i) => (i === sel ? { ...b, cls: classId } : b));
          persistBoxes(next, sel, true);
        }
      }
      setCurrentClass(classId);
    },
    [appState?.model_class_names.length, persistBoxes, setCurrentClass],
  );

  // Never lose labels on tab close / hide: warn if something is unsaved, flush with keepalive.
  useEffect(() => {
    const unsaved = () =>
      hasPendingSaves() || saveSnapshotRef.current != null || dirtyItemKeyRef.current != null;
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      if (!unsaved()) return;
      e.preventDefault();
      e.returnValue = "";
    };
    const flushNow = () => {
      const snap = saveSnapshotRef.current;
      if (saveTimer.current) {
        window.clearTimeout(saveTimer.current);
        saveTimer.current = null;
      }
      flushOnUnload(
        snap && !useTeam.getState().readOnly
          ? { itemKey: snap.itemKey, index: snap.index, boxes: snap.boxes, reviewed: snap.done !== false }
          : undefined,
      );
    };
    const onVisibility = () => {
      if (document.visibilityState === "hidden" && unsaved()) flushNow();
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    window.addEventListener("pagehide", flushNow);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("beforeunload", onBeforeUnload);
      window.removeEventListener("pagehide", flushNow);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  // ---- Team: watch a partner (G / click their chip): their image + live boxes in the main frame -------------
  /** Stop watching: the main frame shows our own image again (we never left it). */
  const stopWatch = useCallback(() => {
    const team = useTeam.getState();
    if (!team.watching && team.watchIntent !== "start") return;
    team.setWatchIntent("stop");
  }, []);

  /** Peek at a live labeler's current image (read-only). `name` null = whoever labeled most recently. */
  const startWatch = useCallback((name: string | null) => {
    const team = useTeam.getState();
    team.setNotice("");
    team.setWatchIntent("start", name);
  }, []);

  /** G: peek at the first live labeler; G again moves to the next one (when only one is live it ends the peek). */
  const toggleWatch = useCallback(() => {
    const team = useTeam.getState();
    const live = team.users
      .filter((u) => u.online && !u.idle && !u.watching && u.item_key && u.name !== team.me?.user)
      .map((u) => u.name)
      .sort();
    if (!team.watching && team.watchIntent !== "start") {
      // Somebody holding G (asking for help) comes first.
      const askers = team.users.filter((u) => u.help && u.online && u.item_key && u.name !== team.me?.user);
      startWatch((askers[0] ?? team.users.find((u) => u.online && !u.idle && u.item_key && u.name !== team.me?.user))?.name ?? null);
    } else if (live.length >= 2) {
      const at = team.watching ? live.indexOf(team.watching) : -1;
      startWatch(live[(at + 1) % live.length]);
    } else {
      stopWatch();
    }
  }, [startWatch, stopWatch]);

  // ---- "I don't understand this one": hold G on a zoomed spot ----------------------------------
  const [helpOn, setHelpOn] = useState(false);
  const helpOnRef = useRef(false);
  const myViewRef = useRef<ViewShare | null>(null);
  const helpPostTimerRef = useRef<number | null>(null);
  const helpKeepRef = useRef<number | null>(null);
  const gHoldTimerRef = useRef<number | null>(null);
  const gDownAtRef = useRef(0);
  const postHelp = useCallback((on: boolean) => {
    const key = useLabeler.getState().detail?.challenge?.item_key;
    if (!key) return;
    void api.teamHelp({ item_key: key, on, view: on ? myViewRef.current : null }).catch(() => {});
  }, []);
  const onMainView = useCallback(
    (v: ViewShare) => {
      myViewRef.current = v;
      if (!helpOnRef.current || helpPostTimerRef.current != null) return;
      // The zoom/pan the asker is doing is sent a few times a second so the others follow it live.
      helpPostTimerRef.current = window.setTimeout(() => {
        helpPostTimerRef.current = null;
        if (helpOnRef.current) postHelp(true);
      }, 120);
    },
    [postHelp],
  );
  const startHelp = useCallback(() => {
    const team = useTeam.getState();
    if (helpOnRef.current || team.watching || team.watchIntent === "start") return; // can't ask while looking at someone else's image
    if (!useLabeler.getState().detail?.challenge?.item_key) return;
    helpOnRef.current = true;
    setHelpOn(true);
    postHelp(true);
    helpKeepRef.current = window.setInterval(() => postHelp(true), 2000);
  }, [postHelp]);
  const stopHelp = useCallback(() => {
    if (!helpOnRef.current) return;
    helpOnRef.current = false;
    setHelpOn(false);
    if (helpKeepRef.current != null) window.clearInterval(helpKeepRef.current);
    if (helpPostTimerRef.current != null) window.clearTimeout(helpPostTimerRef.current);
    helpKeepRef.current = null;
    helpPostTimerRef.current = null;
    postHelp(false);
  }, [postHelp]);
  useEffect(() => stopHelp, [stopHelp]);
  // Tell me when somebody asks for help.
  const askersKey = useTeam((s) => s.users.filter((u) => u.help && u.name !== s.me?.user).map((u) => u.name).join(","));
  const prevAskersRef = useRef("");
  useEffect(() => {
    const prev = new Set(prevAskersRef.current.split(",").filter(Boolean));
    const fresh = askersKey.split(",").filter((n) => n && !prev.has(n));
    prevAskersRef.current = askersKey;
    if (fresh.length) useTeam.getState().setNotice(`${fresh.join(", ")} needs help — press G to look`);
  }, [askersKey]);

  // Peek: the main frame (only) shows the image the watched labeler is on, wherever it lives (another day, the other
  // workspace). Our own image, filmstrip, day and workspace stay exactly as they are; Esc / G just change the frame back.
  // Optimistic: the chip / G names the person, so the frame starts moving before the server has confirmed the watch.
  const pendingTarget = useTeam((s) => (s.watchIntent === "start" ? s.watchTarget : null));
  const watchedName = useTeam((s) => (s.watchIntent === "start" && s.watchTarget ? s.watchTarget : s.watching));
  const peekUser = useTeam((s) => (watchedName ? s.users.find((u) => u.name === watchedName) : undefined));
  const feedItemKey = useTeam((s) => (s.watching ? s.feed?.item_key ?? null : null));
  // While a switch to someone else is on its way, the old feed (the previous person) must not win.
  const peekKey = pendingTarget ? peekUser?.item_key ?? null : feedItemKey ?? peekUser?.item_key ?? null;
  const peekWorkspace = peekUser?.workspace ?? "";
  const peekRev = useTeam((s) => (peekKey ? s.labels[peekKey]?.rev ?? 0 : 0));
  const feedLive = useTeam((s) => s.feed);
  const [peek, setPeek] = useState<PeekInfo | null>(null);
  // Recent peeks, newest last: pressing G on someone whose image we already know is instant.
  const peekCacheRef = useRef(new Map<string, PeekInfo>());
  const fetchPeek = useCallback(async (key: string, workspace: string, rev: number): Promise<PeekInfo> => {
    const ck = `${key}:${rev}`;
    const cache = peekCacheRef.current;
    const hit = cache.get(ck);
    if (hit) return hit;
    const info = await api.teamPeek(key, workspace);
    cache.set(ck, info);
    while (cache.size > 24) cache.delete(cache.keys().next().value as string);
    return info;
  }, []);
  useEffect(() => {
    if (!watchedName || !peekKey) {
      setPeek(null);
      return;
    }
    const hit = peekCacheRef.current.get(`${peekKey}:${peekRev}`);
    if (hit) {
      setPeek(hit);
      return;
    }
    let dead = false;
    // The previous peek stays on screen until the new one is ready: the frame changes in one smooth step.
    fetchPeek(peekKey, peekWorkspace, peekRev)
      .then((r) => {
        if (!dead) setPeek(r);
      })
      .catch(() => {
        if (!dead) {
          setPeek(null);
          useTeam.getState().setNotice("Can't show that image right now.");
        }
      });
    return () => {
      dead = true;
    };
  }, [watchedName, peekKey, peekWorkspace, peekRev, fetchPeek]);
  // Get ready before G is pressed: what each online teammate is on (their labels + the image itself) is fetched quietly.
  const labelersKey = useTeam((s) =>
    s.users
      .filter((u) => u.online && !u.idle && u.item_key && u.name !== s.me?.user)
      .slice(0, 4)
      .map((u) => `${u.name}|${u.item_key}|${u.workspace}`)
      .join(";"),
  );
  useEffect(() => {
    if (!labelersKey) return;
    for (const entry of labelersKey.split(";")) {
      const [, key, workspace] = entry.split("|");
      if (!key) continue;
      const rev = useTeam.getState().labels[key]?.rev ?? 0;
      void fetchPeek(key, workspace, rev)
        .then((r) => {
          const ch = r.challenge;
          const token = imageCacheToken;
          preloadImage(challengePreviewUrl(0, r.mode, ch.block, token, ch.item_key, ch.media_rev), 6);
          preloadImage(challengeThumbnailUrl(0, r.mode, ch.block, true, token, ch.item_key, ch.media_rev), 6);
        })
        .catch(() => {});
    }
  }, [labelersKey, fetchPeek, imageCacheToken]);
  const peekView = useMemo(() => {
    if (!watchedName || !peekKey || !peek) return null;
    const ch = peek.challenge;
    const live = feedLive?.draft && feedLive.item_key === ch.item_key ? feedLive.draft.boxes : peek.boxes;
    // They are holding G on a zoomed spot: show exactly that spot (the people list carries it, so it is there at once).
    const focus = feedLive?.item_key === ch.item_key && feedLive?.help?.view ? feedLive.help.view : peekUser?.help_view ?? null;
    const props: Partial<ComponentProps<typeof AnnotationCanvas>> = {
      sceneId: `peek:${ch.item_key}`,
      fullUrl: challengeImageUrl(0, peek.mode, ch.block, imageCacheToken, ch.item_key, ch.media_rev),
      previewUrl: challengePreviewUrl(0, peek.mode, ch.block, imageCacheToken, ch.item_key, ch.media_rev),
      thumbUrl: challengeThumbnailUrl(0, peek.mode, ch.block, true, imageCacheToken, ch.item_key, ch.media_rev),
      width: peek.width,
      height: peek.height,
      boxes: normalizeBoxes(live as unknown as Array<Record<string, unknown>>),
      editable: false,
      selectedIndex: null,
      inspectBox: null,
      inspectSeq: 0,
      // Their image is not ours: its zoom/pan must not drag our reference pane along.
      syncTransform: null,
      onTransformChange: undefined,
      onViewChange: undefined,
      focusView: focus,
      linkedFocusIdxs: null,
    };
    return { title: `${watchedName} is labeling · ${peek.mode === "local" ? "Own" : "Platform"} · ${ch.display_name || ch.item_key}`, props };
  }, [watchedName, peekKey, peek, peekUser, feedLive, imageCacheToken]);
  const peekProps = peekView?.props ?? {};

  // Navigating to another image of our own (filmstrip, N/B, arrows) ends the peek: we are back to work.
  const ownOpenKey = detail?.challenge?.item_key ?? null;
  const prevOwnKeyRef = useRef<string | null>(ownOpenKey);
  useEffect(() => {
    const prev = prevOwnKeyRef.current;
    prevOwnKeyRef.current = ownOpenKey;
    if (prev && ownOpenKey && prev !== ownOpenKey) stopWatch();
  }, [ownOpenKey, stopWatch]);

  // Read-only live view: paint the owner's current boxes (drag-level while they edit).
  const feedSeq = useTeam((s) => s.feed?.draft?.seq ?? 0);
  const feedFor = useTeam((s) => s.feed?.item_key ?? null);
  useEffect(() => {
    const team = useTeam.getState();
    if (!team.readOnly || !team.feed?.draft) return;
    const openKey = useLabeler.getState().detail?.challenge?.item_key;
    if (!openKey || team.feed.item_key !== openKey) return;
    useLabeler.getState().setBoxes(normalizeBoxes(team.feed.draft.boxes as unknown as Array<Record<string, unknown>>));
    useLabeler.getState().setSelected(null);
  }, [feedSeq, feedFor, detail?.challenge?.item_key]);

  // A teammate saved an image: drop our cached copy so opening it shows their labels.
  const seenRevsRef = useRef<Record<string, number>>({});
  useEffect(() => {
    const me = useTeam.getState().me?.user;
    for (const [key, info] of Object.entries(teamLabels)) {
      if (seenRevsRef.current[key] === info.rev) continue;
      seenRevsRef.current[key] = info.rev;
      if (info.e && info.e !== me) dropCachedDetail(key);
    }
  }, [teamLabels]);

  /** Own Review: ` / Shift+` = next / previous reason on this image (set once the panel is on screen). */
  const reviewCycleRef = useRef<((dir: 1 | -1) => void) | null>(null);
  /** Own Review: the list has nothing left to check — leave it (set below). */
  const reviewFinishRef = useRef<(() => void) | null>(null);
  const lookRightTimerRef = useRef<number | null>(null);
  /** "Looks right" calls still on their way, per image: the list drop waits for them so the server agrees. */
  const lookRightInflightRef = useRef(new Map<string, Promise<unknown>>());
  /** Own Review: Enter = these labels look right, Enter again (or Undo) = no. Unless undone within 5 s, the next image to
   *  check opens and this one leaves the list. */
  const setReviewLooksRight = useCallback(
    (on: boolean) => {
      const key = useLabeler.getState().detail?.challenge?.item_key;
      if (!key) return;
      const st = useReview.getState().items[key];
      if (!st || (!st.open && !st.dismissed)) return; // nothing to do / fixed already
      if (Boolean(st.dismissed) === on) return;
      if (lookRightTimerRef.current != null) window.clearTimeout(lookRightTimerRef.current);
      lookRightTimerRef.current = null;
      useReview.getState().setLookRight(on ? { key, until: Date.now() + 5000 } : null);
      if (on) {
        lookRightTimerRef.current = window.setTimeout(() => {
          lookRightTimerRef.current = null;
          useReview.getState().setLookRight(null);
          if (useLabeler.getState().detail?.challenge?.item_key !== key) return;
          const app = useLabeler.getState().appState;
          const cur = detailIndexRef.current ?? app?.current_index ?? -1;
          if (neighborNeedsWorkIndex(cur, 1) >= 0) requestNeedsWorkNav(1);
          else reviewFinishRef.current?.();
        }, 5000);
      }
      const call = api
        .ownReviewDismiss(key, on)
        .then((r) => {
          useReview.getState().patchItem(key, r.status, r.version);
          syncReviewNow();
        })
        .catch((err) => setError(String(err)))
        .finally(() => {
          if (lookRightInflightRef.current.get(key) === call) lookRightInflightRef.current.delete(key);
        });
      lookRightInflightRef.current.set(key, call);
    },
    [requestNeedsWorkNav, setError],
  );
  const toggleReviewLooksRight = useCallback(() => {
    const key = useLabeler.getState().detail?.challenge?.item_key;
    const st = key ? useReview.getState().items[key] : undefined;
    if (st) setReviewLooksRight(!st.dismissed);
  }, [setReviewLooksRight]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const app = useLabeler.getState().appState;
      if (!app || e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) {
        return;
      }
      if (modalOpenRef.current) return;
      if (e.key === "Enter" && !(e.ctrlKey || e.metaKey || e.altKey) && app.review_mode === "local" && app.own_day === "review") {
        e.preventDefault(); // Enter = looks right (Enter again within 5 s = undo)
        if (!e.repeat) toggleReviewLooksRight();
        return;
      }
      if (e.code === "Backquote" && !(e.ctrlKey || e.metaKey || e.altKey) && app.review_mode === "local" && app.own_day === "review") {
        e.preventDefault(); // ` = next reason on this image, Shift+` = previous (zooms to it)
        reviewCycleRef.current?.(e.shiftKey ? -1 : 1);
        return;
      }
      const cur = detailIndexRef.current ?? app.current_index;
      const mod = e.ctrlKey || e.metaKey;
      if (mod && (e.key === "z" || e.key === "Z")) {
        e.preventDefault();
        if (e.shiftKey) redoBoxes();
        else undoBoxes();
        return;
      }
      if (mod && (e.key === "y" || e.key === "Y")) {
        e.preventDefault();
        redoBoxes();
        return;
      }
      if (mod) return;
      const peeking = Boolean(useTeam.getState().watching || useTeam.getState().watchIntent === "start");
      if (e.key === "Escape" && peeking) {
        e.preventDefault();
        stopWatch();
        return;
      }
      // Our own image is not on screen while peeking: don't label / reset / delete it blind.
      if (peeking && /^h$/i.test(e.key)) {
        e.preventDefault();
        if (!e.repeat) stopWatch();
        return;
      }
      if (peeking && (/^[ma1-9]$/i.test(e.key) || e.key === "Delete" || e.key === "Backspace")) return;
      if (e.key === "g" || e.key === "G") {
        e.preventDefault();
        if (e.repeat) return;
        // Tap (on release) = watch a labeler / next one. Hold = "I don't understand this": ask the team to look at the spot I'm zoomed on.
        // Already peeking: G acts at once (next labeler / back) — nothing to hold for.
        const tm = useTeam.getState();
        if (tm.watching || tm.watchIntent === "start") {
          gDownAtRef.current = 0;
          toggleWatch();
          return;
        }
        gDownAtRef.current = Date.now();
        if (gHoldTimerRef.current != null) window.clearTimeout(gHoldTimerRef.current);
        gHoldTimerRef.current = window.setTimeout(() => {
          gHoldTimerRef.current = null;
          startHelp();
        }, G_HOLD_MS);
        return;
      }
      if (e.key === "n" || e.key === "N") {
        e.preventDefault(); // next image that still needs work
        requestNeedsWorkNav(1);
        return;
      }
      if (e.key === "b" || e.key === "B") {
        e.preventDefault(); // previous image that still needs work
        requestNeedsWorkNav(-1);
        return;
      }
      if (e.key === "ArrowRight" || e.key === "Tab") {
        e.preventDefault();
        requestRelativeNav(1);
        return;
      }
      if (e.key === "ArrowLeft" || e.key === "q" || e.key === "Q") {
        e.preventDefault(); // Tab = same as →, Q = same as ←
        requestRelativeNav(-1);
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setReferenceVariant(refVariantIdxRef.current - 1);
        return;
      }
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setReferenceVariant(refVariantIdxRef.current + 1);
        return;
      }
      if ((e.key === "m" || e.key === "M") && cur >= 0) {
        e.preventDefault();
        if (e.repeat) return;
        markReviewedAndNext();
        return;
      }
      if ((e.key === "h" || e.key === "H") && cur >= 0) {
        e.preventDefault();
        if (e.repeat) return;
        acceptRefAndNext();
        return;
      }
      if ((e.key === "a" || e.key === "A") && cur >= 0) {
        e.preventDefault();
        if (e.repeat) return;
        resetToWinnerOrigin();
        return;
      }
      if (e.key === "t" || e.key === "T") {
        e.preventDefault();
        if (e.repeat) return;
        setShowBoxLabels((v) => !v);
        return;
      }
      if (e.key >= "1" && e.key <= "9") {
        handleClassHotkey(parseInt(e.key, 10) - 1);
        return;
      }
      if (e.key === "Delete" && e.shiftKey) {
        // Shift+Delete: remove the whole image from the dataset (asks first; Enter confirms).
        e.preventDefault();
        if (e.repeat) return;
        const lab = useLabeler.getState();
        const st = lab.appState;
        const idx = detailIndexRef.current ?? st?.current_index ?? -1;
        const ch = st && idx >= 0 ? st.challenges[idx] : null;
        if (!st || !ch) return;
        if (workspaceFromState(st) !== "own") {
          lab.setError("Platform images come from the validator and can't be deleted. Shift+Delete works for Own images.");
          return;
        }
        if (useTeam.getState().readOnly) {
          lab.setError("A teammate has this image open — you can't delete it now.");
          return;
        }
        setDeleteImageAsk({ key: ch.item_key, index: idx, name: ch.display_name || ch.item_key, hasLabel: Boolean(ch.has_label || ch.reviewed) });
        return;
      }
      if (e.key === "Delete" || e.key === "Backspace") {
        const sel = useLabeler.getState().selected;
        if (sel != null) {
          const next = useLabeler.getState().boxes.filter((_, i) => i !== sel);
          persistBoxes(next, null, true);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    acceptRefAndNext,
    handleClassHotkey,
    markReviewedAndNext,
    persistBoxes,
    redoBoxes,
    requestNeedsWorkNav,
    requestRelativeNav,
    resetToWinnerOrigin,
    startHelp,
    stopWatch,
    toggleReviewLooksRight,
    toggleWatch,
    undoBoxes,
  ]);

  // G released: a short press is the watch toggle; after a hold it just ends the help request.
  useEffect(() => {
    const onUp = (e: KeyboardEvent) => {
      if (e.key !== "g" && e.key !== "G") return;
      if (gHoldTimerRef.current != null) {
        window.clearTimeout(gHoldTimerRef.current);
        gHoldTimerRef.current = null;
        if (gDownAtRef.current) toggleWatch();
      }
      gDownAtRef.current = 0;
      stopHelp();
    };
    const onLeave = () => {
      if (gHoldTimerRef.current != null) window.clearTimeout(gHoldTimerRef.current);
      gHoldTimerRef.current = null;
      gDownAtRef.current = 0;
      stopHelp();
    };
    window.addEventListener("keyup", onUp);
    window.addEventListener("blur", onLeave);
    return () => {
      window.removeEventListener("keyup", onUp);
      window.removeEventListener("blur", onLeave);
    };
  }, [stopHelp, toggleWatch]);

  const navIdx = detail?.index ?? appState?.current_index ?? -1;
  const navChallenge = navIdx >= 0 ? appState?.challenges[navIdx] : detail?.challenge;

  // Active miner scored against left-pane labels-as-GT. p comes from that miner's platform metrics.
  const topVsLeftScore = useMemo(() => {
    if (appState && workspaceFromState(appState) !== "platform") return null;
    if (!refBoxes.length || !boxes.length) return null;
    const variants = effectiveReferenceVariants(
      detail?.reference_variants,
      navChallenge,
      detail?.challenge,
      refBoxes.length ? refBoxes : normalizeBoxes(detail?.reference_boxes),
    );
    const clamped = variants.length ? Math.max(0, Math.min(refVariantIdx, variants.length - 1)) : 0;
    const variant = variants[clamped];
    const baseRecord =
      detail?.record && Object.keys(detail.record).length
        ? detail.record
        : navIdx >= 0 && appState?.challenges[navIdx]
          ? appState.challenges[navIdx].record || {}
          : {};
    const rec = variant
      ? {
          total_weighted: variant.total_weighted,
          map50: variant.map50,
          false_positive: variant.false_positive,
          score_pillars: (baseRecord as Record<string, unknown>).score_pillars,
        }
      : baseRecord;
    return scoreTopAgainstLeftGt(
      refBoxes,
      boxes,
      rec,
      detail?.width || navChallenge?.image_width || undefined,
      detail?.height || navChallenge?.image_height || undefined,
    );
  }, [appState, boxes, detail, navChallenge, navIdx, refBoxes, refVariantIdx]);

  const topVsLeftEmpty = useMemo(() => {
    if (!boxes.length) return "draw GT on the left to score miners";
    if (!refBoxes.length) return "waiting for top detections";
    return null;
  }, [boxes.length, refBoxes.length]);

  const refCompare = useMemo(() => {
    if (!boxes.length || !refBoxes.length || !detail) {
      return {
        roles: undefined,
        missedGt: undefined,
        missedGtIndices: undefined,
        predToGt: undefined,
        gtToPred: undefined,
        wrongCount: 0,
      };
    }
    // Never compare across images (common race when pressing N before paint settles).
    const liveKey = appState?.challenges[detail.index]?.item_key;
    if (
      !liveKey ||
      !itemKeysMatch(detail.challenge?.item_key, liveKey) ||
      detailIndexRef.current !== detail.index
    ) {
      return {
        roles: undefined,
        missedGt: undefined,
        missedGtIndices: undefined,
        predToGt: undefined,
        gtToPred: undefined,
        wrongCount: 0,
      };
    }
    const imgW = detail.width || detail.challenge?.image_width;
    const imgH = detail.height || detail.challenge?.image_height;
    const { roles, missedGt, missedGtIndices, predToGt, gtToPred } = comparePredToGt(
      refBoxes,
      boxes,
      imgW,
      imgH,
    );
    const wrongCount = roles.filter((r) => r !== "tp").length + missedGt.length;
    return { roles, missedGt, missedGtIndices, predToGt, gtToPred, wrongCount };
  }, [
    appState?.challenges,
    boxes,
    refBoxes,
    detail,
    detail?.width,
    detail?.height,
    detail?.challenge?.item_key,
    detail?.challenge?.image_width,
    detail?.challenge?.image_height,
    detail?.index,
    refVariantIdx,
  ]);

  const onGtHoverFocus = useCallback(
    (payload: { boxIdx: number | null; ghostIdx: number | null }) => {
      // Keyboard / click explore owns focus — ignore hover so C/V stays stable.
      if (boxInspectIdxRef.current != null) return;
      const gi = payload.boxIdx;
      if (gi == null) {
        setLinkedBoxFocus(null);
        return;
      }
      const gtIdxs = [gi];
      const predIdxs: number[] = [];
      const ghostIdxs: number[] = [];
      const predMatch = refCompare.gtToPred?.[gi];
      if (predMatch != null) predIdxs.push(predMatch);
      // Wrong-class preds that overlap this GT.
      refCompare.predToGt?.forEach((mappedGi, pi) => {
        if (mappedGi === gi && refCompare.roles?.[pi] === "wrong_class" && !predIdxs.includes(pi)) {
          predIdxs.push(pi);
        }
      });
      // Missed ghost for this GT (no miner match).
      const missAt = refCompare.missedGtIndices?.indexOf(gi) ?? -1;
      if (missAt >= 0) ghostIdxs.push(missAt);
      setLinkedBoxFocus({ gtIdxs, predIdxs, ghostIdxs });
    },
    [refCompare],
  );

  const applyGtLinkedFocus = useCallback(
    (gi: number) => {
      const gtIdxs = [gi];
      const predIdxs: number[] = [];
      const ghostIdxs: number[] = [];
      const predMatch = refCompare.gtToPred?.[gi];
      if (predMatch != null) predIdxs.push(predMatch);
      refCompare.predToGt?.forEach((mappedGi, pi) => {
        if (mappedGi === gi && refCompare.roles?.[pi] === "wrong_class" && !predIdxs.includes(pi)) {
          predIdxs.push(pi);
        }
      });
      const missAt = refCompare.missedGtIndices?.indexOf(gi) ?? -1;
      if (missAt >= 0) ghostIdxs.push(missAt);
      setLinkedBoxFocus({ gtIdxs, predIdxs, ghostIdxs });
    },
    [refCompare],
  );

  const beginBoxInspect = useCallback(
    (idx: number, opts?: { zoom?: boolean }) => {
      if (idx < 0 || idx >= boxes.length) return;
      boxInspectIdxRef.current = idx;
      applyGtLinkedFocus(idx);
      setSelected(idx);
      if (opts?.zoom === false) return;
      setBoxInspect((prev) => ({ idx, seq: (prev?.seq ?? 0) + 1 }));
    },
    [applyGtLinkedFocus, boxes.length, setSelected],
  );

  const onGtSelect = useCallback(
    (idx: number | null) => {
      setSelected(idx);
      if (idx == null) {
        // Click empty space — end explore (same as Esc).
        boxInspectIdxRef.current = null;
        setBoxInspect(null);
        setLinkedBoxFocus(null);
        return;
      }
      // Click only anchors C/V — do not auto-zoom (scroll zoom must keep working).
      beginBoxInspect(idx, { zoom: false });
    },
    [beginBoxInspect, setSelected],
  );

  const cycleBoxInspect = useCallback(
    (dir: -1 | 1) => {
      if (!boxes.length) return;
      const center = (b: (typeof boxes)[number]) => ({
        cx: (b.x1 + b.x2) / 2,
        cy: (b.y1 + b.y2) / 2,
      });
      const order = boxes
        .map((_, i) => i)
        .sort((a, b) => {
          const ca = center(boxes[a]);
          const cb = center(boxes[b]);
          if (Math.abs(ca.cx - cb.cx) > 0.5) return ca.cx - cb.cx;
          return ca.cy - cb.cy;
        });
      const sel = useLabeler.getState().selected;
      const anchor = boxInspectIdxRef.current ?? sel ?? null;
      let pos = anchor == null ? (dir > 0 ? -1 : 0) : order.indexOf(anchor);
      if (pos < 0) pos = dir > 0 ? -1 : 0;
      let nextPos = pos + dir;
      if (nextPos < 0) nextPos = order.length - 1;
      if (nextPos >= order.length) nextPos = 0;
      beginBoxInspect(order[nextPos]);
    },
    [beginBoxInspect, boxes],
  );

  const clearBoxInspect = useCallback(() => {
    boxInspectIdxRef.current = null;
    setBoxInspect(null);
    setLinkedBoxFocus(null);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      if (e.key === "c" || e.key === "C") {
        e.preventDefault();
        cycleBoxInspect(-1);
        return;
      }
      if (e.key === "v" || e.key === "V") {
        e.preventDefault();
        cycleBoxInspect(1);
        return;
      }
      const app = useLabeler.getState().appState;
      const inOwn = Boolean(app && workspaceFromState(app) === "own");
      if (inOwn) {
        if (e.key === "e" || e.key === "E") {
          e.preventDefault();
          setOwnDualView((prev) => !prev);
          return;
        }
        if (e.key === "r" || e.key === "R") {
          e.preventDefault();
          setOwnGuides((prev) => !prev);
          return;
        }
      } else {
        if (e.key === "w" || e.key === "W") {
          e.preventDefault();
          setMinerAlertFilter((prev) => (prev === "wrong" ? null : "wrong"));
          return;
        }
        if (e.key === "e" || e.key === "E") {
          e.preventDefault();
          setMinerAlertFilter((prev) => (prev === "extra" ? null : "extra"));
          return;
        }
        if (e.key === "r" || e.key === "R") {
          e.preventDefault();
          setMinerAlertFilter((prev) => (prev === "missed" ? null : "missed"));
          return;
        }
      }
      if (e.key === "Escape") {
        if (!inOwn && minerAlertFilter != null) {
          e.preventDefault();
          setMinerAlertFilter(null);
          return;
        }
        if (boxInspectIdxRef.current == null) return;
        e.preventDefault();
        clearBoxInspect();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [clearBoxInspect, cycleBoxInspect, minerAlertFilter]);

  const onMinerHoverFocus = useCallback(
    (payload: { boxIdx: number | null; ghostIdx: number | null }) => {
      // Same as GT: don't let hover steal explore cursor / clear inspect.
      if (boxInspectIdxRef.current != null) return;
      if (payload.boxIdx == null && payload.ghostIdx == null) {
        setLinkedBoxFocus(null);
        return;
      }
      const gtIdxs: number[] = [];
      const predIdxs: number[] = [];
      const ghostIdxs: number[] = [];
      if (payload.ghostIdx != null) {
        ghostIdxs.push(payload.ghostIdx);
        const gi = refCompare.missedGtIndices?.[payload.ghostIdx];
        if (gi != null) gtIdxs.push(gi);
      }
      if (payload.boxIdx != null) {
        predIdxs.push(payload.boxIdx);
        const gi = refCompare.predToGt?.[payload.boxIdx];
        if (gi != null && !gtIdxs.includes(gi)) gtIdxs.push(gi);
      }
      setLinkedBoxFocus({ gtIdxs, predIdxs, ghostIdxs });
    },
    [refCompare],
  );

  function setReferenceVariant(nextIdx: number) {
    const current = useLabeler.getState().detail;
    const navI = current?.index ?? useLabeler.getState().appState?.current_index ?? -1;
    const ch = navI >= 0 ? useLabeler.getState().appState?.challenges[navI] : current?.challenge;
    const variants = effectiveReferenceVariants(
      current?.reference_variants,
      ch,
      current?.challenge,
      useLabeler.getState().refBoxes,
    );
    if (!variants.length) return;
    const clamped = Math.max(0, Math.min(nextIdx, variants.length - 1));
    const selectedVariant = variants[clamped];
    if (!selectedVariant.boxes.length) return;
    setRefVariantIdx(clamped);
    activeRefVariantIdxGate.current = clamped;
    setRefBoxes(selectedVariant.boxes);
    if (current) {
      useLabeler.getState().setDetail({
        ...current,
        reference_boxes: selectedVariant.boxes as unknown as Record<string, unknown>[],
      });
    }
  }

  useEffect(() => {
    const current = useLabeler.getState().detail;
    if (!current) return;
    const navI = current.index ?? useLabeler.getState().appState?.current_index ?? -1;
    const ch = navI >= 0 ? useLabeler.getState().appState?.challenges[navI] : current.challenge;
    const variants = effectiveReferenceVariants(
      current.reference_variants,
      ch,
      current.challenge,
      useLabeler.getState().refBoxes,
    );
    if (!variants.length) return;
    const clamped = Math.max(0, Math.min(refVariantIdx, variants.length - 1));
    const selectedVariant = variants[clamped];
    const currentRef = useLabeler.getState().refBoxes;
    if (!boxesEqual(currentRef, selectedVariant.boxes)) {
      setRefBoxes(selectedVariant.boxes);
      useLabeler.getState().setDetail({
        ...current,
        reference_boxes: selectedVariant.boxes as unknown as Record<string, unknown>[],
      });
    }
  }, [detail?.challenge?.item_key, detail?.reference_variants, refVariantIdx, setRefBoxes]);

  const referenceVariants = useMemo(
    () =>
      effectiveReferenceVariants(
        detail?.reference_variants,
        navChallenge,
        detail?.challenge,
        refBoxes.length ? refBoxes : normalizeBoxes(detail?.reference_boxes),
      ),
    [detail?.reference_variants, detail?.reference_boxes, detail?.challenge, navChallenge, refBoxes],
  );
  const activeRefVariantIdx =
    referenceVariants.length > 0
      ? Math.max(0, Math.min(refVariantIdx, referenceVariants.length - 1))
      : 0;
  const activeRefVariant = referenceVariants[activeRefVariantIdx] || null;
  // ---- Own Review (label audit): live findings for the open image, highlighted on the canvas -------------
  const isReviewDay = appState?.review_mode === "local" && appState?.own_day === "review";
  // The server's list changed (an edit made another image risky, or images left): reload it, unless a save is in flight.
  const reloadReviewList = useCallback(() => {
    if (hasPendingSaves() || dirtyItemKeyRef.current || saveSnapshotRef.current) return;
    void api
      .getState()
      .then((s) => {
        if (s.review_mode !== "local" || s.own_day !== "review") return;
        workspacePoolsRef.current.own = s;
        applyState(s);
        if (!s.challenges.length) reviewFinishRef.current?.();
      })
      .catch(() => {});
  }, []);
  useReviewSync(Boolean(isReviewDay), reloadReviewList);
  const reviewItems = useReview((s) => s.items);
  const [reviewHover, setReviewHover] = useState<ReviewIssue | null>(null);
  const reviewKey = isReviewDay ? detail?.challenge?.item_key ?? null : null;
  const reviewStatus = reviewKey ? reviewItems[reviewKey] : undefined;
  useEffect(() => setReviewHover(null), [reviewKey]);
  const reviewFocusIdxs =
    reviewHover && reviewHover.i >= 0 ? [reviewHover.i, ...(reviewHover.other != null ? [reviewHover.other] : [])] : null;
  const focusReviewIssue = useCallback(
    (issue: ReviewIssue) => {
      if (issue.i >= 0 && issue.i < boxes.length) {
        beginBoxInspect(issue.i);
      } else if (issue.box) {
        const [x1, y1, x2, y2] = issue.box;
        setBoxInspect((prev) => ({ idx: -1, seq: (prev?.seq ?? 0) + 1, box: { cls: issue.model_cls ?? 0, x1, y1, x2, y2 } }));
      }
    },
    [beginBoxInspect, boxes.length],
  );
  const inspectTarget = boxInspect ? boxInspect.box ?? boxes[boxInspect.idx] ?? null : null;

  // ` / Shift+` walk through the reasons of the open image; the one in focus is drawn strongest.
  const [reviewCursor, setReviewCursor] = useState(-1);
  const prevReviewKeyRef = useRef<string | null>(null);
  useEffect(() => {
    setReviewCursor(-1);
    if (lookRightTimerRef.current != null) {
      window.clearTimeout(lookRightTimerRef.current);
      lookRightTimerRef.current = null;
    }
    const pendingLookRight = useReview.getState().lookRight;
    useReview.getState().setLookRight(null);
    // We just moved on (the next image is already on screen): an image marked "looks right" or fixed leaves the
    // list right now, without waiting for the 5 s or the next poll.
    const prev = prevReviewKeyRef.current;
    prevReviewKeyRef.current = reviewKey;
    if (!prev || !reviewKey || prev === reviewKey) return;
    const st = useReview.getState().items[prev];
    const checked = pendingLookRight?.key === prev || (st !== undefined && !reviewIsOpen(st));
    if (!checked || prev === dirtyItemKeyRef.current || prev === saveSnapshotRef.current?.itemKey) return;
    const app = useLabeler.getState().appState;
    if (app?.own_day === "review" && app.challenges.some((c) => c.item_key === prev)) {
      applyState({ ...app, challenges: app.challenges.filter((c) => c.item_key !== prev) });
    }
    void Promise.resolve(lookRightInflightRef.current.get(prev))
      .then(() => api.ownReviewDrop([prev]))
      .then((s) => {
        if (s.own_day !== "review") return;
        workspacePoolsRef.current.own = s;
        applyState(s);
        if (!s.challenges.length) reviewFinishRef.current?.();
      })
      .catch(() => {});
  }, [reviewKey]);
  const reviewOpenIssues = useMemo(() => (reviewStatus?.issues ?? []).filter((i) => i.risk > 0), [reviewStatus]);
  reviewCycleRef.current = (dir) => {
    const n = reviewOpenIssues.length;
    if (!n) return;
    const next = reviewCursor < 0 ? (dir > 0 ? 0 : n - 1) : (reviewCursor + dir + n) % n;
    setReviewCursor(next);
    focusReviewIssue(reviewOpenIssues[next]);
  };
  // What looks wrong, written next to the box itself.
  const reviewCallouts = useMemo<Callout[]>(() => {
    if (!reviewStatus?.open) return [];
    const out: Callout[] = [];
    reviewOpenIssues.forEach((issue, k) => {
      const strong = k === reviewCursor || reviewHover === issue;
      const at = (i: number | undefined) => (i != null && i >= 0 ? boxes[i] : undefined);
      const b = at(issue.i);
      // Icon + colour only (the bar above says it in words): ✕ big box, no drink · • tiny · ⧉ stacked · □ box in box ·
      // ? other class · + unlabeled drink.
      if (issue.kind === "missed" && issue.box) {
        const [x1, y1, x2, y2] = issue.box;
        out.push({ x1, y1, x2, y2, text: "+", color: REVIEW_COLOR.missed, strong, outline: true });
        return;
      }
      if (!b) return;
      const icon = REVIEW_ICON[issue.kind] ?? "!";
      const color = REVIEW_COLOR[issue.kind] ?? "#e879f9";
      out.push({ x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2, text: icon, color, strong });
      const o = at(issue.other);
      if ((issue.kind === "stacked" || issue.kind === "nested") && o) {
        out.push({ x1: o.x1, y1: o.y1, x2: o.x2, y2: o.y2, text: icon, color, strong });
      }
    });
    return out;
  }, [reviewStatus, reviewOpenIssues, reviewCursor, reviewHover, boxes]);

  // Checked images (fixed, or "looks right") leave the list once you move on — the list is worked until it is empty.
  const saveAt = useTeam((s) => s.save.at);
  useEffect(() => {
    if (!isReviewDay) return;
    const t = window.setTimeout(() => {
      const app = useLabeler.getState().appState;
      if (!app || app.own_day !== "review" || hasPendingSaves()) return;
      const items = useReview.getState().items;
      const cur = useLabeler.getState().detail?.challenge?.item_key;
      const busy = new Set([dirtyItemKeyRef.current, saveSnapshotRef.current?.itemKey].filter(Boolean) as string[]);
      const drop = app.challenges
        .filter((c) => c.item_key !== cur && !busy.has(c.item_key) && items[c.item_key] !== undefined && !reviewIsOpen(items[c.item_key]))
        .map((c) => c.item_key);
      if (!drop.length) return;
      void api
        .ownReviewDrop(drop)
        .then((s) => {
          workspacePoolsRef.current.own = s;
          applyState(s);
          clearDetailCache();
          if (!s.challenges.length) reviewFinishRef.current?.();
        })
        .catch(() => {});
    }, 150);
    return () => window.clearTimeout(t);
  }, [isReviewDay, reviewItems, reviewKey, saveAt]);
  reviewFinishRef.current = () => {
    const items = useReview.getState().items;
    if ((useLabeler.getState().appState?.challenges ?? []).some((c) => reviewIsOpen(items[c.item_key]))) return;
    useTeam.getState().setNotice("Review list is empty — nothing left to check");
    void leaveOwnDayRef.current?.();
  };

  if (!appState) {
    return <BrandLoader label="Opening workspace" hint={`Loading ${topicSlug}`} accent="emerald" />;
  }

  const idx = viewIndex ?? detail?.index ?? appState.current_index;
  void stripTick;
  const isPlatform = workspaceFromState(appState) === "platform";
  const ownDayListMode = !isPlatform && !appState.own_day;
  // Hard isolation: never paint a detail that isn't in the active workspace pool.
  // Own: accept local OR missing source_type (instant paint / thin summaries).
  const inActivePool = Boolean(
    detail &&
      (openItemKeyRef.current
        ? appState.challenges.some(
            (c) => c.item_key === openItemKeyRef.current || itemKeysMatch(c.item_key, openItemKeyRef.current),
          )
        : appState.challenges.some((c) => itemKeysMatch(c.item_key, detail.challenge?.item_key))),
  );
  const activeDetail =
    inActivePool && detail
      ? isPlatform
        ? detail.challenge?.source_type === "local"
          ? null
          : detail
        : // Own: always accept the open detail. Never leave "Loading image…" because of a
          // stale/missing source_type on thin summaries or instant paint.
          {
            ...detail,
            challenge: {
              ...detail.challenge,
              source_type: "local",
            },
          }
      : null;
  if (detail && !activeDetail && detailIndexRef.current != null) {
    // Sync clear on the next tick if a stale Platform detail survived the switch.
    queueMicrotask(() => {
      if (switchingWorkspaceRef.current) return;
      const live = useLabeler.getState();
      const key = live.detail?.challenge?.item_key;
      if (!key) return;
      const app = live.appState;
      if (!app?.challenges.some((c) => itemKeysMatch(c.item_key, key))) {
        clearLabelerPanes();
      }
    });
  }
  if (!isPlatform && !activeDetail && (boxes.length > 0 || refBoxes.length > 0)) {
    queueMicrotask(() => {
      if (switchingWorkspaceRef.current) return;
      if (workspaceFromState(useLabeler.getState().appState || appState) !== "own") return;
      if (useLabeler.getState().detail) return;
      useLabeler.getState().setBoxes([]);
      useLabeler.getState().setRefBoxes([]);
    });
  }
  const platformMetrics = appState.platform_metrics ?? defaultWorkspaceMetrics();
  const ownMetrics = appState.own_metrics ?? defaultWorkspaceMetrics();
  // Progress of what is on screen (the open Own day, or the Platform pool) with per-person counts. The SHARED team log
  // decides who labeled what and whether an image counts as labeled, so every browser derives the same numbers, live
  // (your own save lands at once, a partner's within one 1.5 s beat). The local `reviewed` flag is only the fallback for
  // images labeled before the log existed: a partner's new Done marks reach this browser's flags much later, which made
  // each person see the other's count as 0 while their own showed the real number.
  const dayScope = (() => {
    if (!isPlatform && !appState.own_day) return null;
    let total = 0;
    let done = 0;
    const perUser: Record<string, number> = {};
    for (const c of appState.challenges) {
      if (!c.loaded) continue;
      total += 1;
      const t = teamLabels[c.item_key];
      if (!(t ? t.u != null : Boolean(c.reviewed))) continue;
      done += 1;
      if (t?.u) perUser[t.u] = (perUser[t.u] ?? 0) + 1;
    }
    return { day: isPlatform ? null : appState.own_day, total, done, perUser };
  })();
  const loadedChallenges = appState.challenges.filter((c) => c.loaded);
  const ownCounts = {
    all: loadedChallenges.length,
    new: loadedChallenges.filter((c) => !c.reviewed && !c.has_label).length,
    sam3: loadedChallenges.filter((c) => !c.reviewed && Boolean(c.has_label)).length,
    done: loadedChallenges.filter((c) => c.reviewed).length,
  };
  const viewItem = (c: (typeof appState.challenges)[number], i: number) =>
    isPlatform ? challengeForStrip(c, i) : c;
  // Per-image status for the thin overview bar — same list/filter as the filmstrip.
  const computeStripCodes = (): StripCodes => {
    const codes: number[] = [];
    const idxs: number[] = [];
    appState.challenges.forEach((raw, i) => {
      const c = viewItem(raw, i);
      if (!matchesFilmstripFilter(appState, c, stripFilter, ownFilter)) return;
      let code: number;
      if (!isPlatform) {
        code = c.reviewed ? STRIP_CODE.PASS : c.has_label ? STRIP_CODE.SAM3 : STRIP_CODE.NEW;
      } else {
        const b = stripBucket(c);
        code = b === "new" ? STRIP_CODE.NEW : b === "pass" ? STRIP_CODE.PASS : b === "fail" ? STRIP_CODE.FAIL : STRIP_CODE.MID;
      }
      codes.push(code);
      idxs.push(i);
    });
    return { codes: Int8Array.from(codes), idxs: Int32Array.from(idxs) };
  };
  const guidesToggle = (
    <button
      type="button"
      onClick={() => setOwnGuides((v) => !v)}
      title="Dotted guide lines that follow the pointer and snap to nearby box edges (R in Own)"
      className={cn(
        "shrink-0 rounded-md border px-1.5 py-0.5 text-[10px] font-bold",
        ownGuides ? "border-sky-500 bg-sky-500/15 text-sky-700 dark:text-sky-300" : "border-slate-300 text-slate-400 dark:border-slate-600",
      )}
    >
      Guides {ownGuides ? "on" : "off"}
    </button>
  );

  const statusTagFor = (layout: "rail" | "inline") =>
    activeDetail ? (
      <ImageStatusBar
        layout={layout}
        itemKey={activeDetail.challenge?.item_key ?? null}
        reviewed={Boolean(activeDetail.challenge?.reviewed)}
        onToggleWatch={toggleWatch}
      />
    ) : null;
  const names =
    activeDetail?.class_names?.length
      ? activeDetail.class_names
      : appState.model_class_names;
  const refNames =
    activeDetail?.reference_class_names?.length
      ? activeDetail.reference_class_names
      : names.length
        ? names
        : appState.model_class_names.length
          ? appState.model_class_names
          : appState.miner_class_names;
  const reviewedCount = appState.challenges.filter((c) => c.reviewed).length;
  const stripCounts = { all: 0, new: 0, fail: 0, mid: 0, pass: 0 };
  const genCounts = { all: 0, openai: 0, gemini: 0 };
  appState.challenges.forEach((raw, i) => {
    const c = viewItem(raw, i);
    if (!c.loaded) return;
    genCounts.all += 1;
    if (c.generator === "openai" || c.generator === "gemini") genCounts[c.generator] += 1;
    if (genFilter !== "all" && c.generator !== genFilter) return;
    stripCounts.all += 1;
    stripCounts[stripBucket(c)] += 1;
  });
  const showGenFilter = genCounts.openai + genCounts.gemini > 0 || genFilter !== "all";
  const showBeverageHfImport = !isPlatform && /beverage/i.test(topicSlug || "");
  const showBeverageSam3 = showBeverageHfImport;
  const showBeverageSendGt = isPlatform && /beverage/i.test(topicSlug || "");
  const renderFooterActions = () => (
    <FooterActions
      downloadBusy={downloadBusy}
      reviewedCount={reviewedCount}
      collectRunning={isPlatform && Boolean(appState.collect.running)}
      collectMessage={appState.collect.message || ""}
      stopCollectBusy={stopCollectBusy}
      onStopCollect={async () => {
        setStopCollectBusy(true);
        setError("");
        try {
          const next = await api.stopCollect();
          applyState(next);
        } catch (e) {
          setError(String(e));
        } finally {
          setStopCollectBusy(false);
        }
      }}
      onDownload={() => setDownloadOpen(true)}
      onSendGt={showBeverageSendGt ? () => setSendGtOpen(true) : undefined}
      onUploadLabels={() => setUploadLabelsOpen(true)}
      onUploadUltralytics={() => setUltralyticsOpen(true)}
      onDeepPostProc={() => setDppOpen(true)}
      onCompareHfModels={() => setHfCompareOpen(true)}
      onSelectBaseModel={() => setBaseModelOpen(true)}
      onSubmit={() => setSubmitModeOpen(true)}
      theme={theme}
      onToggleTheme={toggleTheme}
      ownMode={!isPlatform}
      uploadImagesBusy={false}
      onUploadImages={!isPlatform ? () => setOwnUploadOpen(true) : undefined}
      onHfImport={showBeverageHfImport ? () => setOwnHfImportOpen(true) : undefined}
      onSam3={showBeverageSam3 ? () => openSam3Modal() : undefined}
      sam3Busy={sam3Busy}
      dayListMode={ownDayListMode}
      onDeleteRange={!isPlatform && !ownDayListMode ? () => setOwnDeleteOpen(true) : undefined}
      onClearNew={!isPlatform && appState.own_day && appState.own_day !== "review" ? () => void clearOwnDayNew() : undefined}
      clearNewCount={!isPlatform && appState.own_day ? ownCounts.new : 0}
      clearNewBusy={modeBusy}
    />
  );

  return (
    <div className="rt-in flex h-full min-h-0 flex-col gap-1.5 bg-slate-50 p-1.5 dark:bg-transparent">
      {helpOn && (
        <div className="pointer-events-none fixed bottom-16 left-1/2 z-[60] flex -translate-x-1/2 items-center gap-2 rounded-full bg-rose-600 px-4 py-1.5 text-[12px] font-extrabold text-white shadow-xl ring-2 ring-white/30">
          <span className="h-2 w-2 animate-ping rounded-full bg-white" />
          Asking the team for help on this spot · release G to stop
        </div>
      )}
      <TeamBar
        total={dayScope ? dayScope.total : (isPlatform ? platformMetrics : ownMetrics).total}
        reviewed={dayScope ? dayScope.done : (isPlatform ? platformMetrics : ownMetrics).done}
        scopeLabel={
          dayScope?.day === "review"
            ? "Review"
            : dayScope?.day
            ? `Day ${dayScope.day}${appState.own_field ? ` · ${appState.own_field === "openai" ? "OpenAI" : "Gemini"}` : ""}`
            : !isPlatform
              ? ownField
                ? `${ownField === "openai" ? "OpenAI" : "Gemini"} Own days`
                : "All Own days"
              : undefined
        }
        perUser={dayScope?.perUser}
        onPeek={startWatch}
        onLogout={() => {
          void api.logout().finally(() => window.location.reload());
        }}
      />
      {ownDayListMode ? (
        <OwnDayList
          field={ownField}
          onFieldChange={setOwnField}
          onEnterDay={enterOwnDay}
          busy={modeBusy}
          selected={daySel}
          onSelectedChange={setDaySel}
          onSendDays={(days) => {
            setDaySel(days);
            setUltralyticsOpen(true);
          }}
          onDeleteDays={deleteOwnDaysFromList}
        />
      ) : (
      <>
      <div className="flex min-h-[84px] w-full min-w-0 items-stretch gap-1.5">
        {showGenFilter && (
          <div className="flex h-[80px] w-[4.4rem] shrink-0 flex-col justify-center gap-1 py-px" title="Which image generator made the image">
            {(
              [
                ["all", "Any", genCounts.all, "border-slate-300 text-slate-700 dark:border-slate-600 dark:text-slate-200", "border-slate-700 bg-slate-700 text-white dark:border-slate-200 dark:bg-slate-200 dark:text-slate-900"],
                ["openai", "OpenAI", genCounts.openai, "border-sky-300 text-sky-700 dark:border-sky-800 dark:text-sky-200", "border-sky-500 bg-sky-500 text-white"],
                ["gemini", "Gemini", genCounts.gemini, "border-violet-300 text-violet-700 dark:border-violet-800 dark:text-violet-200", "border-violet-500 bg-violet-500 text-white"],
              ] as const
            ).map(([id, label, count, idle, active]) => (
              <button
                key={id}
                type="button"
                onClick={() => setGenFilter(id)}
                title={id === "all" ? "Every image" : id === "openai" ? "Images made by OpenAI (gpt-image)" : "Images made by Gemini"}
                className={cn(
                  "flex h-[20px] items-center justify-between rounded border px-1 text-[8px] font-extrabold uppercase leading-none tracking-wide",
                  genFilter === id ? active : idle,
                )}
              >
                <span>{label}</span>
                <span className="tabular-nums">{count}</span>
              </button>
            ))}
          </div>
        )}
        {isPlatform ? (
        <div className="flex h-[80px] w-[4.7rem] shrink-0 flex-col justify-between py-px">
          {(
            [
              ["all", "All", stripCounts.all, "border-slate-300 text-slate-700 dark:border-slate-600 dark:text-slate-200", "border-sky-500 bg-sky-500 text-white"],
              ["new", "New", stripCounts.new, "border-amber-300 text-amber-800 dark:border-amber-700 dark:text-amber-200", "border-amber-400 bg-amber-400 text-amber-950"],
              ["fail", "Fail", stripCounts.fail, "border-rose-300 text-rose-700 dark:border-rose-800 dark:text-rose-200", "border-rose-500 bg-rose-500 text-white"],
              ["mid", "Mid", stripCounts.mid, "border-lime-300 text-lime-800 dark:border-lime-800 dark:text-lime-200", "border-lime-500 bg-lime-500 text-lime-950"],
              ["pass", "Pass", stripCounts.pass, "border-emerald-300 text-emerald-700 dark:border-emerald-800 dark:text-emerald-200", "border-emerald-500 bg-emerald-500 text-white"],
            ] as const
          ).map(([id, label, count, idle, active]) => (
            <button
              key={id}
              type="button"
              onClick={() => setStripFilter(id)}
              title={
                id === "all"
                  ? `All loaded images (${stripCounts.new + stripCounts.fail + stripCounts.mid + stripCounts.pass})`
                  : id === "new"
                    ? "Unlabeled (new)"
                    : id === "fail"
                      ? "Labeled · 0/3 miners close to p"
                      : id === "mid"
                        ? "Labeled · 1/3 or 2/3 miners close to p"
                        : "Labeled · 3/3 miners close to p"
              }
              className={cn(
                "flex h-[14px] items-center justify-between rounded border px-1 text-[8px] font-extrabold uppercase leading-none tracking-wide",
                stripFilter === id ? active : idle,
              )}
            >
              <span>{label}</span>
              <span className="tabular-nums">{count}</span>
            </button>
          ))}
        </div>
        ) : (
        <div className="flex h-[80px] w-[4.7rem] shrink-0 flex-col justify-between py-px">
          <button
            type="button"
            onClick={() => void leaveOwnDay()}
            disabled={modeBusy}
            title="Back to Own day list"
            className="flex h-[12px] items-center justify-center rounded border border-violet-400 bg-violet-50 px-1 text-[7px] font-extrabold uppercase leading-none tracking-wide text-violet-800 hover:bg-violet-100 disabled:opacity-50 dark:border-violet-700 dark:bg-violet-950/40 dark:text-violet-200"
          >
            Days
          </button>
          {(
            [
              ["all", "All", ownCounts.all, "border-violet-300 text-violet-800 dark:border-violet-700 dark:text-violet-200", "border-violet-500 bg-violet-500 text-white"],
              ["new", "New", ownCounts.new, "border-amber-300 text-amber-800 dark:border-amber-700 dark:text-amber-200", "border-amber-400 bg-amber-400 text-amber-950"],
              ["sam3", "SAM3", ownCounts.sam3, "border-sky-300 text-sky-800 dark:border-sky-700 dark:text-sky-200", "border-sky-500 bg-sky-500 text-white"],
              ["done", "Done", ownCounts.done, "border-emerald-300 text-emerald-700 dark:border-emerald-800 dark:text-emerald-200", "border-emerald-500 bg-emerald-500 text-white"],
            ] as const
          ).map(([id, label, count, idle, active]) => (
            <button
              key={id}
              type="button"
              onClick={() => setOwnFilter(id)}
              title={
                id === "all"
                  ? `All Own images (${ownCounts.all}) · ${appState.own_day || ""}`
                  : id === "new"
                    ? "Unlabeled (new)"
                    : id === "sam3"
                      ? "SAM3 draft — review then Done"
                      : "Labeled Done"
              }
              className={cn(
                "flex h-[12px] items-center justify-between rounded border px-1 text-[7px] font-extrabold uppercase leading-none tracking-wide",
                ownFilter === id ? active : idle,
              )}
            >
              <span>{label}</span>
              <span className="tabular-nums">{count}</span>
            </button>
          ))}
        </div>
        )}
      <div
        ref={filmstripRef}
        onWheel={onFilmstripWheel}
        onScroll={onFilmstripScroll}
        className="filmstrip-scroll flex min-h-[84px] min-w-0 flex-1 gap-1.5 overflow-x-auto pb-1"
      >
        {(() => {
          const loaded: { c: (typeof appState.challenges)[number]; i: number }[] = [];
          appState.challenges.forEach((raw, i) => {
            const c = viewItem(raw, i);
            if (matchesFilmstripFilter(appState, c, stripFilter, ownFilter)) loaded.push({ c, i });
          });
          // Keep the open image visible only when it matches the active filter.
          if (idx >= 0 && appState.challenges[idx]?.loaded) {
            const curC = viewItem(appState.challenges[idx], idx);
            if (
              !loaded.some((x) => x.i === idx) &&
              matchesFilmstripFilter(appState, curC, stripFilter, ownFilter)
            ) {
              loaded.push({ c: curC, i: idx });
              loaded.sort((a, b) => a.i - b.i);
            }
          }
          const totalSlots = loaded.length;
          if (totalSlots <= 0) {
            return (
              <div className="flex h-[80px] min-w-0 flex-1 items-center rounded-lg border border-dashed border-slate-300 px-3 text-[11px] font-semibold text-slate-500 dark:border-slate-700 dark:text-slate-400">
                {!isPlatform
                  ? ownFilter === "all"
                    ? "Upload images to start labeling"
                    : ownFilter === "sam3"
                      ? "No SAM3 drafts — run SAM3 on New images"
                      : `No ${ownFilter} images`
                  : stripFilter === "new"
                    ? "No unlabeled images"
                    : stripFilter === "fail"
                      ? "No labeled images with 0/3 miners"
                      : stripFilter === "mid"
                        ? "No labeled images with 1/3 or 2/3 miners"
                        : stripFilter === "pass"
                          ? "No labeled images with 3/3 miners"
                          : "No images"}
              </div>
            );
          }
          const visFirst = Math.max(0, stripRange.first);
          const visLast = Math.min(totalSlots - 1, stripRange.last);
          const leftPad = visFirst * THUMB_SLOT;
          const rightPad = Math.max(0, (totalSlots - visLast - 1) * THUMB_SLOT);
          const loadedStart = visFirst;
          const loadedEnd = Math.min(loaded.length - 1, visLast);
          return (
            <>
              {leftPad > 0 && <div className="shrink-0" style={{ width: leftPad }} aria-hidden />}
              {loaded.slice(loadedStart, loadedEnd + 1).map(({ c, i }) => {
                const selected = i === idx;
                const reviewed = Boolean(c.reviewed);
                const badge = isPlatform
                  ? stripBadge(c)
                  : reviewed
                    ? { text: "done", className: "bg-emerald-500 text-white", title: `${c.display_name} · Done` }
                    : c.has_label
                      ? { text: "sam3", className: "bg-sky-500 text-white", title: `${c.display_name} · SAM3 draft` }
                      : { text: "new", className: "bg-amber-400 text-amber-950", title: `${c.display_name} · New` };
                const passed = c.checks_passed;
                const passComplete = checksPassComplete(c);
                return (
          <button
            key={`${imageCacheToken}:${c.item_key}`}
            data-filmstrip-index={i}
            type="button"
            onPointerDown={(e) => {
              if (e.button !== 0) return;
              if (i === (detailIndexRef.current ?? idx)) return;
              // Keep focus on the strip so ←/→ keep working after a mouse pick.
              e.preventDefault();
              filmstripPickAtRef.current = performance.now();
              primeDecodedThumbnail(challengeThumbnailUrl(i, appState.review_mode, c.block, true, imageCacheToken, c.item_key, c.media_rev));
              preloadChallengeUrgent(i, appState.review_mode, c.block, imageCacheToken, c.item_key, c.media_rev, { full: true });
              void selectChallenge(i, "filmstrip").catch((err) => {
                if (!isAbortish(err) && !isStaleChallengeError(err)) setError(String(err));
              });
            }}
            onMouseEnter={(e) => {
              // Thumb right away (full getChallenge/image on hover saturated the server)...
              preloadChallengeUrgent(i, appState.review_mode, c.block, imageCacheToken, c.item_key, c.media_rev);
              // ...and the small sharp preview once the pointer rests on the tile: the click that usually follows
              // then lands on a ready preview instead of a blurry thumbnail.
              const el = e.currentTarget as HTMLElement & { __previewTimer?: number };
              window.clearTimeout(el.__previewTimer);
              el.__previewTimer = window.setTimeout(() => {
                preloadImage(challengePreviewUrl(i, appState.review_mode, c.block, imageCacheToken, c.item_key, c.media_rev), 70);
              }, 140);
            }}
            onMouseLeave={(e) => {
              window.clearTimeout((e.currentTarget as HTMLElement & { __previewTimer?: number }).__previewTimer);
            }}
            title={badge.title}
            className={cn(
              "relative h-[80px] w-[112px] shrink-0 overflow-hidden rounded-lg border p-0.5 transition-all duration-200 ease-out",
              selected
                ? "scale-[1.03] border-sky-400 ring-2 ring-sky-200 dark:border-sky-500 dark:ring-sky-900"
                : !reviewed
                  ? "border-amber-300 bg-amber-50/70 hover:border-amber-400 dark:border-amber-700/80 dark:bg-amber-950/20"
                  : passComplete
                    ? "border-emerald-400/80 bg-emerald-50/80 hover:border-emerald-500 dark:border-emerald-700 dark:bg-emerald-950/30"
                    : passed === 0
                      ? "border-rose-400/80 bg-rose-50/80 hover:border-rose-500 dark:border-rose-800 dark:bg-rose-950/30"
                      : "border-lime-400/80 bg-lime-50/70 hover:border-lime-500 dark:border-lime-800 dark:bg-lime-950/20",
            )}
          >
            <div className="relative">
              <FilmstripThumb
                url={challengeThumbnailUrl(i, appState.review_mode, c.block, true, imageCacheToken, c.item_key, c.media_rev)}
                className={cn(
                  "h-[56px] w-full rounded-md object-cover",
                  reviewed && "opacity-90",
                )}
              />
              <span
                className={cn(
                  "absolute right-1 top-1 inline-flex items-center gap-0.5 rounded px-1 py-0.5 text-[8px] font-extrabold uppercase tracking-wide shadow-sm",
                  badge.className,
                )}
              >
                {badge.text}
              </span>
              {(() => {
                const claimedBy = teamClaims[c.item_key];
                const lb = teamLabels[c.item_key]?.u;
                if (claimedBy) {
                  return (
                    <span
                      className="absolute left-1 top-1 inline-flex max-w-[70px] animate-pulse items-center gap-0.5 truncate rounded px-1 py-0.5 text-[8px] font-extrabold text-slate-900 shadow-sm"
                      style={{ backgroundColor: teamColors[claimedBy] ?? "#94a3b8" }}
                      title={`${claimedBy} is working on this image`}
                    >
                      ✎ {claimedBy}
                    </span>
                  );
                }
                if (lb) {
                  return (
                    <span
                      className="absolute left-1 top-1 inline-flex max-w-[70px] items-center truncate rounded px-1 py-0.5 text-[8px] font-extrabold text-slate-900 shadow-sm"
                      style={{ backgroundColor: teamColors[lb] ?? "#94a3b8" }}
                      title={`Labeled by ${lb}`}
                    >
                      {lb}
                    </span>
                  );
                }
                return null;
              })()}
              {isReviewDay &&
                (() => {
                  const st = reviewItems[c.item_key];
                  if (st === undefined) return null;
                  const open = reviewIsOpen(st);
                  const top = st?.issues.find((i) => i.risk > 0);
                  return (
                    <span
                      className={cn(
                        "absolute bottom-3 left-1 inline-flex max-w-[100px] items-center truncate rounded px-1 py-0.5 text-[8px] font-extrabold uppercase shadow-sm",
                        open ? "bg-amber-400 text-amber-950" : "bg-emerald-500 text-white",
                      )}
                      title={top ? top.text : undefined}
                    >
                      {open ? `⚠ ${top?.kind ?? "check"}` : st?.dismissed ? "✓ ok" : "✓ fixed"}
                    </span>
                  );
                })()}
            </div>
            <span
              className={cn(
                "absolute bottom-0 left-1 right-1 truncate pb-0.5 pt-2 text-[9px] font-bold leading-none",
                !reviewed
                  ? "text-amber-900 dark:text-amber-200"
                  : passComplete
                    ? "text-emerald-800 dark:text-emerald-300"
                    : passed === 0
                      ? "text-rose-800 dark:text-rose-300"
                      : "text-lime-800 dark:text-lime-300",
              )}
            >
              {c.display_name}
            </span>
          </button>
                );
              })}
              {rightPad > 0 && <div className="shrink-0" style={{ width: rightPad }} aria-hidden />}
            </>
          );
        })()}
      </div>
      </div>

      <StripStatusBar
        challenges={appState.challenges}
        compute={computeStripCodes}
        version={`${stripFilter}:${ownFilter}:${stripTick}:${appState.review_mode}:${appState.own_day ?? ""}:${appState.challenges.length}`}
        currentIdx={idx}
        visible={stripRange}
        onJump={(i) => {
          void selectChallenge(i, "overview-bar")
            .then(() => scrollThumbIntoView(i))
            .catch((err) => {
              if (!isAbortish(err) && !isStaleChallengeError(err)) setError(String(err));
            });
        }}
      />
      {activeDetail ? (
        isPlatform ? (
        <div className="flex min-h-0 flex-1 flex-col gap-2 lg:flex-row lg:items-stretch">
          <Pane title={peekView?.title ?? `Your GT · ${boxes.length} boxes`} headerExtra={guidesToggle} hint={`Left pane is ground truth · draw · Space/right/middle-drag to pan · right-click a box = delete · scroll / Z·X zoom · C/V inspect · T ${showBoxLabels ? "hides" : "shows"} labels`}>
            <AnnotationCanvas
              sceneId={`gt:${activeDetail.challenge.item_key}`}
              fullUrl={challengeImageUrl(idx, appState!.review_mode, activeDetail.challenge.block, imageCacheToken, activeDetail.challenge.item_key, activeDetail.challenge.media_rev)}
              previewUrl={challengePreviewUrl(idx, appState!.review_mode, activeDetail.challenge.block, imageCacheToken, activeDetail.challenge.item_key, activeDetail.challenge.media_rev)}
              thumbUrl={challengeThumbnailUrl(idx, appState!.review_mode, activeDetail.challenge.block, true, imageCacheToken, activeDetail.challenge.item_key, activeDetail.challenge.media_rev)}
              width={activeDetail.width}
              height={activeDetail.height}
              boxes={boxes}
              classNames={names}
              currentClass={currentClass}
              editable={!teamReadOnly}
              selectedIndex={selected}
              onBoxesChange={persistBoxes}
              onSelect={onGtSelect}
              showGuides={ownGuides}
              syncTransform={syncTransform}
              onTransformChange={setSyncTransform}
              canvasBg={theme === "dark" ? "#0b1220" : "#f1f5f9"}
              showLabels={showBoxLabels}
              linkedFocusIdxs={linkedBoxFocus?.gtIdxs ?? null}
              onHoverFocus={onGtHoverFocus}
              inspectBox={inspectTarget}
              inspectSeq={boxInspect?.seq ?? 0}
              onViewChange={onMainView}
              {...peekProps}
            />
          </Pane>
          <MiddleRail
            topSlot={statusTagFor("rail")}
            classNames={names.length ? names : refNames}
            leftBoxes={boxes}
            rightBoxes={refBoxes}
            referenceVariants={referenceVariants}
            activeRefVariantIdx={activeRefVariantIdx}
            onSelectRefVariant={setReferenceVariant}
            currentClass={currentClass}
            onSelectClass={setCurrentClass}
            imageWidth={activeDetail.width || activeDetail.challenge?.image_width || undefined}
            imageHeight={activeDetail.height || activeDetail.challenge?.image_height || undefined}
            scoreRecord={(activeDetail.record || activeDetail.challenge?.record || {}) as Record<string, unknown>}
          />
          <Pane
            title={`Miner R${activeRefVariantIdx + 1} · ${refBoxes.length} boxes${refCompare.wrongCount ? ` · ${refCompare.wrongCount} wrong` : ""}${
              minerAlertFilter === "wrong"
                ? " · W only"
                : minerAlertFilter === "extra"
                  ? " · E only"
                  : minerAlertFilter === "missed"
                    ? " · R only"
                    : ""
            }`}
            hint={`Scored vs your GT · W wrong · E extra · R missed (toggle) · rose EXTRA · amber WRONG · cyan MISSED`}
          >
            <AnnotationCanvas
              sceneId={`ref:${activeDetail.challenge.item_key}:${activeRefVariantIdx}`}
              fullUrl={challengeImageUrl(idx, appState!.review_mode, activeDetail.challenge.block, imageCacheToken, activeDetail.challenge.item_key, activeDetail.challenge.media_rev)}
              previewUrl={challengePreviewUrl(idx, appState!.review_mode, activeDetail.challenge.block, imageCacheToken, activeDetail.challenge.item_key, activeDetail.challenge.media_rev)}
              thumbUrl={challengeThumbnailUrl(idx, appState!.review_mode, activeDetail.challenge.block, true, imageCacheToken, activeDetail.challenge.item_key, activeDetail.challenge.media_rev)}
              width={activeDetail.width}
              height={activeDetail.height}
              boxes={refBoxes}
              classNames={refNames}
              currentClass={0}
              editable={false}
              syncTransform={syncTransform}
              onTransformChange={setSyncTransform}
              canvasBg={theme === "dark" ? "#0b1220" : "#f1f5f9"}
              boxCompare={boxes.length ? refCompare.roles : undefined}
              ghostBoxes={boxes.length ? refCompare.missedGt : undefined}
              showLabels={showBoxLabels}
              linkedFocusIdxs={linkedBoxFocus?.predIdxs ?? null}
              linkedGhostIdxs={linkedBoxFocus?.ghostIdxs ?? null}
              onHoverFocus={onMinerHoverFocus}
              alertFilter={minerAlertFilter}
            />
          </Pane>
        </div>
        ) : (
        <div className="flex min-h-0 flex-1 flex-col gap-2">
          {isReviewDay && !peekView && (
            <ReviewPanel
              itemKey={activeDetail.challenge.item_key}
              boxes={boxes}
              classNames={names}
              openCount={appState.challenges.filter((c) => c.loaded && reviewIsOpen(reviewItems[c.item_key])).length}
              totalCount={appState.challenges.length}
              cursor={reviewCursor}
              onFocus={(issue) => {
                setReviewCursor(reviewOpenIssues.indexOf(issue));
                focusReviewIssue(issue);
              }}
              onHover={setReviewHover}
              onLooksRight={setReviewLooksRight}
            />
          )}
          {(() => {
            const ownReviewed = Boolean(activeDetail.challenge.reviewed);
            const ownSam3 = Boolean(activeDetail.challenge.has_label) && !ownReviewed;
            // Dual: GT/SAM3 labels (left) + winner base detection (right). NEW stays single-pane.
            const showDual = ownDualView && (ownReviewed || ownSam3);
            const itemKey = activeDetail.challenge.item_key;
            const fromDetail = normalizeBoxes(activeDetail.prediction_boxes || []);
            if (itemKey && fromDetail.length) {
              rememberWinnerPreds(
                itemKey,
                fromDetail,
                normalizeBoxes(boxes),
                ownReviewed,
              );
            }
            const ownBaseBoxes =
              (itemKey && winnerPredCache.get(itemKey)) || fromDetail;
            const ownHint =
              `C/V inspect · R guides${ownGuides ? " on" : ""} · E dual${ownDualView ? " on" : " off"} · Z/X zoom · N next / B back (NEW) · Tab/Q = →/← · M Done${isReviewDay ? " · \` reason · Enter looks right" : ""}`;
            const ownUrls = {
              fullUrl: challengeImageUrl(
                idx,
                appState!.review_mode,
                activeDetail.challenge.block,
                imageCacheToken,
                activeDetail.challenge.item_key,
                activeDetail.challenge.media_rev,
              ),
              previewUrl: challengePreviewUrl(
                idx,
                appState!.review_mode,
                activeDetail.challenge.block,
                imageCacheToken,
                activeDetail.challenge.item_key,
                activeDetail.challenge.media_rev,
              ),
              thumbUrl: challengeThumbnailUrl(
                idx,
                appState!.review_mode,
                activeDetail.challenge.block,
                true,
                imageCacheToken,
                activeDetail.challenge.item_key,
                activeDetail.challenge.media_rev,
              ),
            };
            const canvasBg = theme === "dark" ? "#0b1220" : "#f1f5f9";
            if (showDual) {
              const leftTitle = ownSam3
                ? `SAM3 · ${boxes.length} boxes`
                : `Own labeled · ${boxes.length} boxes`;
              return (
                <div className="flex min-h-0 flex-1 flex-col gap-2 lg:flex-row lg:items-stretch">
                  <Pane title={peekView?.title ?? leftTitle} hint={ownHint} headerExtra={<>{statusTagFor("inline")}{guidesToggle}</>}>
                    <AnnotationCanvas
                      sceneId={`own:${activeDetail.challenge.item_key}`}
                      {...ownUrls}
                      width={activeDetail.width}
                      height={activeDetail.height}
                      boxes={boxes}
                      classNames={names}
                      currentClass={currentClass}
                      editable={!teamReadOnly}
                      selectedIndex={selected}
                      onBoxesChange={persistBoxes}
                      onSelect={onGtSelect}
                              syncTransform={syncTransform}
                      onTransformChange={setSyncTransform}
                      canvasBg={canvasBg}
                      showLabels={showBoxLabels}
                      inspectBox={inspectTarget}
                      inspectSeq={boxInspect?.seq ?? 0}
                      showGuides={ownGuides}
                      onViewChange={onMainView}
                      linkedFocusIdxs={reviewFocusIdxs}
                      linkedGhostIdxs={null}
                      callouts={isReviewDay ? reviewCallouts : undefined}
                      {...peekProps}
                    />
                  </Pane>
                  <Pane
                    title={`Base detection · ${ownBaseBoxes.length} boxes`}
                    hint="Winner-model detections (read-only) · Space/right/middle-drag to pan · synced zoom · Z/X toward cursor"
                  >
                    <AnnotationCanvas
                      sceneId={`own-base:${activeDetail.challenge.item_key}`}
                      {...ownUrls}
                      width={activeDetail.width}
                      height={activeDetail.height}
                      boxes={ownBaseBoxes}
                      classNames={names}
                      currentClass={0}
                      editable={false}
                      syncTransform={syncTransform}
                      onTransformChange={setSyncTransform}
                      canvasBg={canvasBg}
                      showLabels={showBoxLabels}
                    />
                  </Pane>
                </div>
              );
            }
            return (
              <Pane
                title={
                  peekView?.title ??
                  (ownReviewed
                    ? `Own · ${boxes.length} boxes`
                    : ownSam3
                      ? `SAM3 · ${boxes.length} boxes`
                      : `Own · base detection · ${boxes.length} boxes`)
                }
                hint={ownHint}
                headerExtra={<>{statusTagFor("inline")}{guidesToggle}</>}
              >
                <AnnotationCanvas
                  sceneId={`own:${activeDetail.challenge.item_key}`}
                  {...ownUrls}
                  width={activeDetail.width}
                  height={activeDetail.height}
                  boxes={boxes}
                  classNames={names}
                  currentClass={currentClass}
                  editable={!teamReadOnly}
                  selectedIndex={selected}
                  onBoxesChange={persistBoxes}
                  onSelect={onGtSelect}
                      syncTransform={syncTransform}
                  onTransformChange={setSyncTransform}
                  canvasBg={canvasBg}
                  showLabels={showBoxLabels}
                  inspectBox={inspectTarget}
                  inspectSeq={boxInspect?.seq ?? 0}
                  showGuides={ownGuides}
                  onViewChange={onMainView}
                  linkedFocusIdxs={reviewFocusIdxs}
                  linkedGhostIdxs={null}
                  callouts={isReviewDay ? reviewCallouts : undefined}
                  {...peekProps}
                />
              </Pane>
            );
          })()}
          <div className="flex flex-wrap items-center gap-1.5 px-1">
            {names.map((name, classId) => (
              <button
                key={`${classId}:${name}`}
                type="button"
                onClick={() => setCurrentClass(classId)}
                className={cn(
                  "rounded border px-2 py-0.5 text-[10px] font-bold",
                  currentClass === classId
                    ? "border-violet-500 bg-violet-500 text-white"
                    : "border-slate-300 bg-white text-slate-700 dark:border-slate-600 dark:bg-ink-900 dark:text-slate-200",
                )}
              >
                {classId + 1}. {name}
              </button>
            ))}
          </div>
        </div>
        )
      ) : (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-sky-200 bg-white text-sm text-slate-500 dark:border-sky-900 dark:bg-ink-900 dark:text-slate-400">
          {appState.challenges.some((c) => c.loaded) ? (
            <>
              <span className="h-8 w-8 animate-spin rounded-full border-2 border-sky-200 border-t-sky-500" />
              <p>
                {isPlatform && appState.collect.running
                  ? appState.collect.message || "Loading first image…"
                  : "Loading image…"}
              </p>
            </>
          ) : !isPlatform ? (
            topicReady ? (
              <>
                <span className="h-8 w-8 animate-spin rounded-full border-2 border-sky-200 border-t-sky-500" />
                <p>Loading Own pool…</p>
              </>
            ) : (
              <>
                <p>Upload images to start labeling in Own mode.</p>
                <p className="text-xs text-slate-400">Click Upload in the footer to open the Own upload window.</p>
              </>
            )
          ) : appState.collect.running ? (
            <>
              <span className="h-8 w-8 animate-spin rounded-full border-2 border-sky-200 border-t-sky-500" />
              <p>{appState.collect.message || "Fetching images from Score subnet…"}</p>
              <button
                type="button"
                disabled={stopCollectBusy}
                onClick={async () => {
                  setStopCollectBusy(true);
                  setError("");
                  try {
                    applyState(await api.stopCollect());
                  } catch (e) {
                    setError(String(e));
                  } finally {
                    setStopCollectBusy(false);
                  }
                }}
                className="mt-1 inline-flex items-center gap-1.5 rounded-lg border border-rose-300 bg-rose-50 px-3 py-1.5 text-xs font-bold text-rose-700 hover:bg-rose-100 disabled:opacity-50 dark:border-rose-800 dark:bg-rose-950/40 dark:text-rose-300"
              >
                <Square className="h-3 w-3 fill-current" />
                {stopCollectBusy ? "Stopping…" : "Stop downloading"}
              </button>
            </>
          ) : (
            <>
              <p>No images loaded yet.</p>
              <p className="text-xs text-slate-400">Images appear automatically once the server cache is ready.</p>
            </>
          )}
        </div>
      )}
      </>
      )}

      <footer className="mt-auto shrink-0 rounded-lg border border-slate-200 bg-white/95 px-1.5 py-0.5 shadow-soft dark:border-slate-700/60 dark:bg-ink-900/90">
        <div className="flex flex-wrap items-center gap-1.5 lg:flex-nowrap">
          <div className="flex min-w-0 flex-1 items-center gap-1.5 lg:max-w-[36%]">
            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-sky-500 text-white">
              <SquareDashedMousePointer className="h-3.5 w-3.5" />
            </div>
            <WorkspaceSwitch
              isPlatform={isPlatform}
              platformMetrics={platformMetrics}
              ownMetrics={ownMetrics}
              modeBusy={modeBusy}
              onSwitchPlatform={() => void switchWorkspace("platform")}
              onSwitchOwn={() => void switchWorkspace("own")}
            />
            <LabeledProgress left={appState.metrics.not_reviewed} labeled={appState.metrics.ready_to_tune} />
            <div className="lg:hidden">{renderFooterActions()}</div>
          </div>

          <div className="min-w-0 shrink-0 lg:mx-auto">
            {isPlatform ? (
              <TopVsLeftGtCard
                score={topVsLeftScore}
                emptyMessage={topVsLeftEmpty}
                minerLabel={`R${activeRefVariantIdx + 1}`}
                fallbackP={{
                  s: activeRefVariant?.total_weighted ?? null,
                  m: activeRefVariant?.map50 ?? null,
                  f: activeRefVariant?.false_positive ?? null,
                }}
              />
            ) : null}
          </div>

          <div className="flex min-w-0 flex-1 items-center gap-1.5 lg:max-w-[42%] lg:justify-end">
            <div className="no-scrollbar flex min-w-0 flex-1 items-center gap-1 overflow-x-auto">
              <Link
                to="/"
                className="shrink-0 rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] font-semibold text-slate-500 hover:border-sky-300 hover:text-sky-700 dark:border-slate-700 dark:bg-ink-800 dark:text-slate-300"
                title="Back to topics"
              >
                Topics
              </Link>
              {!isPlatform && appState.own_day ? (
                <button
                  type="button"
                  onClick={() => void leaveOwnDay()}
                  disabled={modeBusy}
                  title="Back to Own day list"
                  className="shrink-0 rounded-full border border-violet-300 bg-violet-50 px-2 py-0.5 text-[10px] font-bold tabular-nums text-violet-800 hover:bg-violet-100 disabled:opacity-50 dark:border-violet-700 dark:bg-violet-950/40 dark:text-violet-200"
                >
                  {appState.own_day}
                </button>
              ) : null}
              {!isPlatform && appState.own_day && appState.own_day !== "review" && ownCounts.new > 0 ? (
                <button
                  type="button"
                  onClick={() => void clearOwnDayNew()}
                  disabled={modeBusy}
                  title={`Delete all ${ownCounts.new} New images in ${appState.own_day} (keeps Done)`}
                  className="shrink-0 rounded-full border border-rose-300 bg-rose-50 px-2 py-0.5 text-[10px] font-bold text-rose-700 hover:bg-rose-100 disabled:opacity-50 dark:border-rose-800 dark:bg-rose-950/40 dark:text-rose-300"
                >
                  Clear new ({ownCounts.new})
                </button>
              ) : null}
              {appState.skills.map((s) => {
                const slug = topicSlugFromSkill(s.element_path);
                if (!slug) return null;
                const active = slug === topicSlug;
                return (
                  <Link
                    key={s.element_path}
                    to={`/t/${slug}`}
                    className={cn(
                      "shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-semibold",
                      active
                        ? "border-sky-400 bg-sky-50 text-sky-700 dark:border-sky-600 dark:bg-sky-950/40 dark:text-sky-300"
                        : "border-slate-200 bg-white text-slate-600 hover:border-sky-300 hover:text-sky-700 dark:border-slate-700 dark:bg-ink-800 dark:text-slate-300",
                    )}
                    title={s.element_id}
                  >
                    {slug}
                  </Link>
                );
              })}
              {appState.hf_model_id && (
                <a
                  href={`https://huggingface.co/${appState.hf_model_id}`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex shrink-0 items-center gap-1 rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] font-semibold text-emerald-700 hover:bg-emerald-100 dark:border-emerald-800/60 dark:bg-emerald-950/40 dark:text-emerald-300"
                  title={appState.hf_model_id}
                >
                  <Zap className="h-3 w-3 shrink-0" />
                  <span className="max-w-[7.5rem] truncate">{appState.hf_model_id.split("/").pop()}</span>
                </a>
              )}
            </div>
            <div className="hidden lg:block">{renderFooterActions()}</div>
          </div>
        </div>
        {error && <div className="mt-1 text-[10px] font-semibold text-rose-500 dark:text-rose-400">{error}</div>}
      </footer>
      <SubmitModeModal
        open={submitModeOpen}
        onClose={() => setSubmitModeOpen(false)}
        onAuto={() => setFindBestOpen(true)}
        onManual={() => setManualMergeOpen(true)}
      />
      <MountOnOpen open={findBestOpen}>
        <FindBestModal
          open={findBestOpen}
          onClose={() => setFindBestOpen(false)}
          reviewedCount={reviewedCount}
          defaultElement={(appState.active_skill?.element_id || topicSlug || "")
            .replace(/^manak0\/Detect-/i, "")
            .replace(/^manak0\//i, "")
            .replace(/-/g, " ")
            .trim()}
          onMinerLoaded={(state) => {
            applyState(state);
            const idx = state.current_index ?? 0;
            void selectChallenge(idx);
          }}
          challenges={appState.challenges}
          ownMode={!isPlatform}
        />
      </MountOnOpen>
      <MountOnOpen open={manualMergeOpen}>
        <ManualMergeModal
          open={manualMergeOpen}
          onClose={() => setManualMergeOpen(false)}
          challenges={appState.challenges}
          ownMode={!isPlatform}
        />
      </MountOnOpen>
      <MountOnOpen open={ownUploadOpen}>
        <OwnUploadModal
          open={ownUploadOpen}
          onClose={() => setOwnUploadOpen(false)}
          onDone={onUploadOwnDone}
          imageCacheToken={imageCacheToken}
        />
      </MountOnOpen>
      <MountOnOpen open={ownHfImportOpen}>
        <OwnHfImportModal
          open={ownHfImportOpen}
          onClose={() => setOwnHfImportOpen(false)}
          onDone={async (state) => {
            setOwnHfImportOpen(false);
            await onUploadOwnDone(state);
          }}
        />
      </MountOnOpen>
      <MountOnOpen open={sam3Open}>
        <OwnSam3Modal
          open={sam3Open}
          onClose={() => setSam3Open(false)}
          challenges={appState.challenges}
          day={appState.own_day}
          onLabelOne={async (opts) => {
            setSam3Busy(true);
            await runSam3LabelOne(opts);
          }}
          onRunComplete={onSam3RunComplete}
        />
      </MountOnOpen>
      {deleteImageAsk ? (
        <DeleteImageConfirm
          name={deleteImageAsk.name}
          hasLabel={deleteImageAsk.hasLabel}
          onCancel={() => setDeleteImageAsk(null)}
          onConfirm={async () => {
            const ask = deleteImageAsk;
            clearPendingSaveFor(ask.index, ask.key);
            const next = await api.deleteOwnRange({ itemKeys: [ask.key] });
            workspacePoolsRef.current.own = next;
            applyState(next);
            // One image gone: the details we hold carry the old block numbers, so those go. Decoded images and thumbnails
            // are keyed by URL (stem + block): the ones before the deleted image stay valid, the shifted ones simply
            // get new URLs. Wiping them all made every thumbnail and the next image download again.
            clearDetailCache();
            setDeleteImageAsk(null);
            setError("");
            const idx = Math.min(ask.index, next.challenges.length - 1);
            if (idx >= 0 && next.challenges[idx]?.loaded) await loadChallenge(idx);
            else clearLabelerPanes();
          }}
        />
      ) : null}
      <MountOnOpen open={ownDeleteOpen}>
        <OwnDeleteRangeModal
          open={ownDeleteOpen}
          onClose={() => setOwnDeleteOpen(false)}
          challenges={appState.challenges}
          onDelete={async (opts) => {
            setError("");
            const prevKey = useLabeler.getState().detail?.challenge?.item_key || null;
            const prevIdx = detailIndexRef.current ?? appState.current_index ?? -1;
            const next = await api.deleteOwnRange({ blockMin: opts.blockMin, blockMax: opts.blockMax, itemKeys: opts.itemKeys });
            workspacePoolsRef.current.own = next;
            applyState(next);
            clearDetailCache();
            clearDecodedImages();
            clearThumbnailBlobCache();
            const keepIdx =
              prevKey != null ? next.challenges.findIndex((c) => itemKeysMatch(c.item_key, prevKey)) : -1;
            const idx = keepIdx >= 0 ? keepIdx : Math.min(prevIdx, next.challenges.length - 1);
            if (idx >= 0 && next.challenges[idx]?.loaded) await loadChallenge(idx);
            else clearLabelerPanes();
          }}
        />
      </MountOnOpen>
      <MountOnOpen open={uploadLabelsOpen}>
        <UploadLabelsModal
          open={uploadLabelsOpen}
          onClose={() => setUploadLabelsOpen(false)}
          classCount={names.length || appState.model_class_names.length}
          reviewMode={appState.review_mode}
          onApplied={(state) => {
            for (const c of state.challenges) {
              if (c.reviewed) notePendingDone(c.item_key);
            }
            applyState(state);
            void api
              .getState()
              .then((fresh) => applyState(fresh))
              .catch(() => undefined);
            const idx = detailIndexRef.current ?? state.current_index ?? 0;
            void selectChallenge(idx);
          }}
        />
      </MountOnOpen>
      <MountOnOpen open={downloadOpen}>
        <DownloadGtModal
          open={downloadOpen}
          onClose={() => setDownloadOpen(false)}
          challenges={appState.challenges}
          ownMode={!isPlatform}
          onDownload={async (opts) => {
            setDownloadBusy(true);
            setError("");
            try {
              const downloadIdx = detailIndexRef.current;
              if (downloadIdx != null) await flushSave(downloadIdx);
              await api.downloadReviewedGt(
                {
                  blockMin: opts.blockMin,
                  blockMax: opts.blockMax,
                  quality: opts.quality,
                  itemKeys: opts.itemKeys,
                },
                opts.onProgress,
              );
            } catch (e) {
              setError(String(e));
              throw e;
            } finally {
              setDownloadBusy(false);
            }
          }}
        />
      </MountOnOpen>
      <MountOnOpen open={sendGtOpen}>
        <SendGtModal
          open={sendGtOpen}
          onClose={() => setSendGtOpen(false)}
          challenges={appState.challenges}
          onSend={async (opts) => {
            setError("");
            const downloadIdx = detailIndexRef.current;
            if (downloadIdx != null) await flushSave(downloadIdx);
            return api.sendReviewedGt({
              blockMin: opts.blockMin,
              blockMax: opts.blockMax,
              quality: opts.quality,
              itemKeys: opts.itemKeys,
            });
          }}
        />
      </MountOnOpen>
      <MountOnOpen open={ultralyticsOpen}>
        <UploadUltralyticsModal
          open={ultralyticsOpen}
          onClose={() => setUltralyticsOpen(false)}
          challenges={appState.challenges}
          ownMode={!isPlatform}
          dayMode={ownDayListMode ? { preselect: daySel, field: ownField } : undefined}
        />
      </MountOnOpen>
      <MountOnOpen open={dppOpen}>
        <DeepPostProcModal
          open={dppOpen}
          onClose={() => setDppOpen(false)}
          challenges={appState.challenges}
          ownMode={!isPlatform}
        />
      </MountOnOpen>
      <MountOnOpen open={hfCompareOpen}>
        <CompareHfModelsModal
          open={hfCompareOpen}
          onClose={() => setHfCompareOpen(false)}
          challenges={appState.challenges}
          ownMode={!isPlatform}
        />
      </MountOnOpen>
      <MountOnOpen open={baseModelOpen}>
        <SelectBaseModelModal
          open={baseModelOpen}
          onClose={() => setBaseModelOpen(false)}
          onApplied={async () => {
            winnerPredCache.clear();
            try {
              applyState(await api.getState());
            } catch {
              /* ignore */
            }
          }}
        />
      </MountOnOpen>
    </div>
  );
}

/** Mount a lazily-loaded modal on first open, then keep it mounted so its state survives close. */
function MountOnOpen({ open, children }: { open: boolean; children: ReactNode }) {
  const [seen, setSeen] = useState(open);
  if (open && !seen) setSeen(true);
  return seen ? <Suspense fallback={null}>{children}</Suspense> : null;
}

function plausibilityOf(
  score: {
    pS: number | null;
    pM: number | null;
    pF: number | null;
    tS: number | null;
    tM: number | null;
    tF: number | null;
  },
  opts?: { hasPredBoxes?: boolean },
): { label: string; tone: string } {
  if (score.tS == null || score.tM == null || score.tF == null) {
    if (
      opts?.hasPredBoxes === false &&
      (score.pS != null || score.pM != null || score.pF != null)
    ) {
      return { label: "no preds", tone: "text-slate-500" };
    }
    return { label: "label GT", tone: "text-slate-500" };
  }
  if (score.pS == null || score.pM == null || score.pF == null) {
    return { label: "no p", tone: "text-slate-500" };
  }
  const mean =
    (Math.abs(score.tS - score.pS) + Math.abs(score.tM - score.pM) + Math.abs(score.tF - score.pF)) / 3;
  if (mean <= 0.03) return { label: "plausible", tone: "text-emerald-500" };
  if (mean <= 0.06) return { label: "close", tone: "text-lime-500" };
  if (mean <= 0.1) return { label: "near", tone: "text-amber-500" };
  if (mean <= 0.16) return { label: "so-so", tone: "text-orange-500" };
  return { label: "far", tone: "text-rose-500" };
}

function scorePct(v: number | null | undefined): string {
  return v == null ? "—" : `${Math.round(v * 100)}`;
}

function pct(v: number | null | undefined): string {
  return scorePct(v);
}

function TweenNum({
  value,
  format,
  className,
}: {
  value: number | null;
  format: (n: number) => string;
  className?: string;
}) {
  const [shown, setShown] = useState(value);
  const curRef = useRef(value);
  const tgtRef = useRef(value);
  const rafRef = useRef(0);
  tgtRef.current = value;

  useEffect(() => {
    const tick = () => {
      const tgt = tgtRef.current;
      const cur = curRef.current;
      if (tgt == null) {
        curRef.current = null;
        setShown(null);
        rafRef.current = 0;
        return;
      }
      const from = cur == null ? tgt : cur;
      const next = from + (tgt - from) * 0.28;
      const done = Math.abs(next - tgt) < (Math.abs(tgt) >= 1 ? 0.35 : 0.0015);
      const val = done ? tgt : next;
      curRef.current = val;
      setShown(val);
      if (!done) rafRef.current = requestAnimationFrame(tick);
      else rafRef.current = 0;
    };
    if (!rafRef.current) rafRef.current = requestAnimationFrame(tick);
  }, [value]);

  useEffect(() => () => cancelAnimationFrame(rafRef.current), []);

  if (shown == null) return <span className={className}>—</span>;
  return <span className={className}>{format(shown)}</span>;
}

function MetricFlashCell({
  value,
  toneClass,
  title,
}: {
  value: number | null;
  toneClass?: string;
  title?: string;
}) {
  const prevRef = useRef(value);
  const [flash, setFlash] = useState(false);
  useEffect(() => {
    if (prevRef.current === value) return;
    const changed = prevRef.current != null || value != null;
    prevRef.current = value;
    if (!changed) return;
    setFlash(true);
    const t = window.setTimeout(() => setFlash(false), 480);
    return () => window.clearTimeout(t);
  }, [value]);
  return (
    <span
      title={title}
      className={cn(
        "rounded px-0.5 py-px text-center tabular-nums transition-[background-color,box-shadow,transform] duration-300 ease-out",
        "bg-black/5 dark:bg-white/5",
        toneClass,
        flash && "scale-105 bg-sky-500/20 shadow-[0_0_0_1px_rgba(56,189,248,0.55)] dark:bg-sky-400/20",
      )}
    >
      <TweenNum value={value} format={(n) => `${Math.round(n * 100)}`} />
    </span>
  );
}

function metricGapClass(p: number | null, t: number | null): string {
  if (p == null || t == null) return "text-slate-400";
  const d = Math.abs(p - t);
  if (d <= 0.03) return "text-emerald-600 dark:text-emerald-400";
  if (d <= 0.06) return "text-lime-600 dark:text-lime-400";
  if (d <= 0.1) return "text-amber-600 dark:text-amber-400";
  if (d <= 0.16) return "text-orange-600 dark:text-orange-400";
  return "text-rose-600 dark:text-rose-400";
}

function GlyphS({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" className={cn("shrink-0", className)} fill="none" aria-hidden>
      <path d="M8 1.4 14.6 8 8 14.6 1.4 8Z" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

function GlyphM({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" className={cn("shrink-0", className)} fill="none" aria-hidden>
      <circle cx="8" cy="8" r="6.1" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="8" cy="8" r="2.1" fill="currentColor" />
    </svg>
  );
}

function GlyphF({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" className={cn("shrink-0", className)} fill="none" aria-hidden>
      <path d="M8 2.2 14.4 13.4H1.6Z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

function PtCompareCard({
  pS,
  pM,
  pF,
  tS,
  tM,
  tF,
  title,
  compact,
  inline,
}: {
  pS: number | null;
  pM: number | null;
  pF: number | null;
  tS: number | null;
  tM: number | null;
  tF: number | null;
  title?: string;
  compact?: boolean;
  inline?: boolean;
}) {
  const grade = plausibilityOf({ pS, pM, pF, tS, tM, tF });
  const metrics = [
    { key: "s", title: "s = mean total_weighted", Glyph: GlyphS, p: pS, t: tS },
    { key: "m", title: "m = map50", Glyph: GlyphM, p: pM, t: tM },
    { key: "f", title: "f = false positive", Glyph: GlyphF, p: pF, t: tF },
  ] as const;
  if (inline) {
    return (
      <div className="flex items-center gap-1.5 text-[10px] font-black leading-none">
        {title ? <span className="shrink-0 text-[11px] font-extrabold tracking-tight">{title}</span> : null}
        <span
          key={grade.label}
          className={cn(
            "shrink-0 uppercase transition-colors duration-300 animate-[metricGradeIn_0.35s_ease-out]",
            grade.tone,
          )}
        >
          {grade.label}
        </span>
        {metrics.map((metric) => (
          <span
            key={metric.key}
            className="inline-flex items-center gap-0.5 rounded bg-black/5 px-1 py-0.5 tabular-nums dark:bg-white/5"
            title={metric.title}
          >
            <metric.Glyph className="h-2.5 w-2.5 text-slate-500" />
            <TweenNum value={metric.p} format={(n) => `${Math.round(n * 100)}`} className="text-slate-600 dark:text-slate-300" />
            <span className="text-slate-400">/</span>
            <TweenNum value={metric.t} format={(n) => `${Math.round(n * 100)}`} className={metricGapClass(metric.p, metric.t)} />
          </span>
        ))}
      </div>
    );
  }
  const icon = compact ? "h-2.5 w-2.5" : "h-3 w-3";
  return (
    <div className={cn("w-full", compact ? "text-[9px]" : "text-[10px]")}>
      {compact ? (
        // Narrow rail: title left, grade right, no filler lines — nothing gets clipped.
        <div className="flex items-baseline justify-between gap-1 leading-none">
          {title ? <span className="shrink-0 text-[10px] font-extrabold tracking-tight">{title}</span> : <span />}
          <span
            key={grade.label}
            className={cn(
              "min-w-0 truncate text-[8px] font-black uppercase tracking-tight transition-colors duration-300 animate-[metricGradeIn_0.35s_ease-out]",
              grade.tone,
            )}
            title={grade.label}
          >
            {grade.label}
          </span>
        </div>
      ) : (
      <div className="flex items-center gap-1">
        {title ? <span className="shrink-0 text-[10px] font-extrabold tracking-tight">{title}</span> : null}
        <span className="h-px min-w-1 flex-1 bg-slate-300 dark:bg-slate-600" />
        <span
          key={grade.label}
          className={cn(
            "shrink-0 font-black uppercase tracking-wide transition-colors duration-300 animate-[metricGradeIn_0.35s_ease-out]",
            grade.tone,
          )}
        >
          {grade.label}
        </span>
        <span className="h-px min-w-1 flex-1 bg-slate-300 dark:bg-slate-600" />
      </div>
      )}
      <div className="mt-0.5 grid grid-cols-4 items-center gap-x-0.5 gap-y-px font-black tabular-nums">
        <span />
        {metrics.map((metric) => (
          <span key={metric.key} className="inline-flex items-center justify-center text-slate-500" title={metric.title}>
            <metric.Glyph className={icon} />
          </span>
        ))}
        <span className="text-center text-slate-500" title="platform scores">p</span>
        <MetricFlashCell value={pS} />
        <MetricFlashCell value={pM} />
        <MetricFlashCell value={pF} />
        <span className="text-center text-slate-500" title="scored using your labeling as GT">t</span>
        <MetricFlashCell value={tS} toneClass={metricGapClass(pS, tS)} />
        <MetricFlashCell value={tM} toneClass={metricGapClass(pM, tM)} />
        <MetricFlashCell value={tF} toneClass={metricGapClass(pF, tF)} />
      </div>
      <div className="mt-0.5 grid grid-cols-4 gap-x-0.5">
        <span />
        {metrics.map((metric) => {
          const gap =
            metric.p != null && metric.t != null ? Math.min(1, Math.abs(metric.p - metric.t) / 0.2) : 0;
          return (
            <span key={`gap-${metric.key}`} className="h-0.5 overflow-hidden rounded-full bg-slate-200/80 dark:bg-slate-700/80">
              <span
                className={cn(
                  "block h-full rounded-full transition-[width,background-color] duration-500 ease-out",
                  gap <= 0.15
                    ? "bg-emerald-400"
                    : gap <= 0.35
                      ? "bg-lime-400"
                      : gap <= 0.55
                        ? "bg-amber-400"
                        : gap <= 0.8
                          ? "bg-orange-400"
                          : "bg-rose-400",
                )}
                style={{ width: `${Math.max(gap * 100, metric.t == null ? 0 : 8)}%` }}
              />
            </span>
          );
        })}
      </div>
    </div>
  );
}

function countByClass(boxes: DetectionBox[], classCount: number): number[] {
  const counts = Array.from({ length: classCount }, () => 0);
  for (const b of boxes) {
    const cls = Number(b.cls);
    if (Number.isInteger(cls) && cls >= 0 && cls < classCount) counts[cls] += 1;
    else if (Number.isInteger(cls) && cls >= classCount) {
      // Keep overflow classes visible by extending.
      while (counts.length <= cls) counts.push(0);
      counts[cls] += 1;
    }
  }
  return counts;
}

function MiddleRail({
  topSlot,
  classNames,
  leftBoxes,
  rightBoxes,
  referenceVariants,
  activeRefVariantIdx,
  onSelectRefVariant,
  currentClass,
  onSelectClass,
  imageWidth,
  imageHeight,
  scoreRecord,
}: {
  /** Labeled-by tag / read-only banner / save status, pinned at the very top of the rail. */
  topSlot?: ReactNode;
  classNames: string[];
  leftBoxes: DetectionBox[];
  rightBoxes: DetectionBox[];
  referenceVariants: RefVariant[];
  activeRefVariantIdx: number;
  onSelectRefVariant: (idx: number) => void;
  currentClass: number;
  onSelectClass: (cls: number) => void;
  imageWidth?: number;
  imageHeight?: number;
  scoreRecord?: Record<string, unknown>;
}) {
  const variantSummaries = useMemo(
    () =>
      referenceVariants.map((variant, i) => {
        const pS = variant.total_weighted ?? null;
        const pM = variant.map50 ?? null;
        const pF = variant.false_positive ?? null;
        const scored = leftBoxes.length
          ? scoreTopAgainstLeftGt(
              variant.boxes,
              leftBoxes,
              {
                total_weighted: variant.total_weighted,
                map50: variant.map50,
                false_positive: variant.false_positive,
                score_pillars: scoreRecord?.score_pillars,
              },
              imageWidth,
              imageHeight,
            )
          : null;
        const tS = scored?.composite ?? null;
        const tM = scored?.map50 ?? null;
        const tF = scored?.false_positive ?? null;
        const grade = plausibilityOf({ pS, pM, pF, tS, tM, tF }, { hasPredBoxes: variant.boxes.length > 0 });
        const passed = minerPasses(pS, pM, pF, tS, tM, tF);
        const cls =
          grade.label === "plausible" || grade.label === "close"
            ? "text-emerald-800 bg-emerald-50 dark:bg-emerald-950/40 dark:text-emerald-300"
            : grade.label === "near"
              ? "text-amber-800 bg-amber-50 dark:bg-amber-950/40 dark:text-amber-300"
              : grade.label === "far"
                ? "text-rose-800 bg-rose-50 dark:bg-rose-950/40 dark:text-rose-300"
                : "text-slate-600 bg-slate-50 dark:bg-ink-950 dark:text-slate-300";
        return {
          idx: i,
          pS,
          pM,
          pF,
          tS,
          tM,
          tF,
          cls,
          passed,
          source: variant.source,
          hotkey: variant.hotkey || "",
          isWinner: Boolean(variant.is_winner),
        };
      }),
    [imageHeight, imageWidth, leftBoxes, referenceVariants, scoreRecord],
  );
  return (
    <aside className="flex w-full shrink-0 flex-col overflow-hidden rounded-xl border border-slate-200/80 bg-white shadow-soft lg:h-full lg:min-h-0 lg:w-[7.2rem] lg:max-w-[7.2rem] dark:border-slate-800 dark:bg-ink-900">
      <div className="flex min-h-0 flex-1 flex-col justify-around px-1.5 py-2">
        {topSlot && <section className="shrink-0">{topSlot}</section>}
        <section className="shrink-0 space-y-1">
          <div className="text-center text-[9px] font-black uppercase tracking-wide text-sky-600 dark:text-sky-400">
            p vs t
          </div>
          <div className="flex flex-col gap-1">
            {variantSummaries.map((variant) => (
              <button
                key={`${variant.idx}:${variant.source}`}
                type="button"
                onClick={() => onSelectRefVariant(variant.idx)}
                className={cn(
                  "rounded-lg border px-1.5 py-1.5 text-left transition-all duration-300 ease-out",
                  variant.cls,
                  variant.idx === activeRefVariantIdx
                    ? "border-sky-400 shadow-[0_0_0_1px_rgba(56,189,248,0.35)] ring-1 ring-inset ring-sky-400"
                    : "border-transparent hover:brightness-110",
                )}
                title={[
                  "If your left labels match real GT, t should match p.",
                  `p(s,m,f)=(${pct(variant.pS)}, ${pct(variant.pM)}, ${pct(variant.pF)})`,
                  variant.tS != null
                    ? `t(s,m,f)=(${pct(variant.tS)}, ${pct(variant.tM)}, ${pct(variant.tF)})`
                    : "draw GT on the left to fill t",
                  variant.hotkey ? `hotkey ${variant.hotkey}` : variant.source,
                ].join("\n")}
              >
                <PtCompareCard
                  compact
                  title={`R${variant.idx + 1}${variant.isWinner ? " ★" : ""}${variant.passed ? " ✓" : ""}`}
                  pS={variant.pS}
                  pM={variant.pM}
                  pF={variant.pF}
                  tS={variant.tS}
                  tM={variant.tM}
                  tF={variant.tF}
                />
              </button>
            ))}
          </div>
        </section>
        <section className="flex min-h-0 max-h-[46%] shrink flex-col overflow-hidden">
          <div className="shrink-0 pb-1 text-center text-[9px] font-black uppercase tracking-wide text-slate-400">
            GT → miner
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden">
            <ClassCompareRail
              classNames={classNames}
              leftBoxes={leftBoxes}
              rightBoxes={rightBoxes}
              currentClass={currentClass}
              onSelectClass={onSelectClass}
              embedded
              hideHeader
            />
          </div>
        </section>
      </div>
    </aside>
  );
}

function LabeledProgress({ left, labeled }: { left: number; labeled: number }) {
  const total = Math.max(0, left + labeled);
  const pctDone = total ? Math.round((labeled / total) * 100) : 0;
  return (
    <div
      className="flex h-7 w-[6.5rem] shrink-0 flex-col justify-center rounded border border-slate-200 bg-slate-50 px-1.5 py-0.5 dark:border-slate-700 dark:bg-ink-950"
      title={`${labeled} labeled · ${left} left`}
    >
      <div className="flex items-baseline justify-between gap-2 text-[9px] font-bold uppercase tracking-wide">
        <span className="tabular-nums text-slate-800 dark:text-slate-100">{labeled}</span>
        <span className="text-slate-400">{left} left</span>
      </div>
      <div className="mt-0.5 h-1 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
        <div
          className="h-full rounded-full bg-sky-500 transition-[width] duration-300 ease-out"
          style={{ width: `${pctDone}%` }}
        />
      </div>
    </div>
  );
}

function FooterActions({
  downloadBusy,
  reviewedCount,
  collectRunning,
  collectMessage,
  stopCollectBusy,
  onStopCollect,
  onDownload,
  onSendGt,
  onUploadLabels,
  onUploadUltralytics,
  onDeepPostProc,
  onCompareHfModels,
  onSelectBaseModel,
  onSubmit,
  theme,
  onToggleTheme,
  ownMode = false,
  uploadImagesBusy = false,
  onUploadImages,
  onHfImport,
  onSam3,
  sam3Busy = false,
  onDeleteRange,
  dayListMode = false,
  onClearNew,
  clearNewCount = 0,
  clearNewBusy = false,
}: {
  downloadBusy: boolean;
  reviewedCount: number;
  collectRunning: boolean;
  collectMessage: string;
  stopCollectBusy: boolean;
  onStopCollect: () => void | Promise<void>;
  onDownload: () => void;
  onSendGt?: () => void;
  onUploadLabels: () => void;
  onUploadUltralytics: () => void;
  onDeepPostProc: () => void;
  onCompareHfModels: () => void;
  onSelectBaseModel: () => void;
  onSubmit: () => void;
  theme: string;
  onToggleTheme: () => void;
  ownMode?: boolean;
  uploadImagesBusy?: boolean;
  onUploadImages?: () => void;
  onHfImport?: () => void;
  onSam3?: () => void;
  sam3Busy?: boolean;
  /** Own day list page (no day open): the Ultralytics button works on whole days. */
  dayListMode?: boolean;
  onDeleteRange?: () => void;
  onClearNew?: () => void;
  clearNewCount?: number;
  clearNewBusy?: boolean;
}) {
  return (
    <div className="flex h-7 shrink-0 items-center overflow-hidden rounded border border-slate-200 bg-white dark:border-slate-700 dark:bg-ink-800">
      {collectRunning && (
        <button
          type="button"
          disabled={stopCollectBusy}
          onClick={() => void onStopCollect()}
          title={collectMessage || "Stop downloading images from Score"}
          aria-label="Stop downloading"
          className="inline-flex items-center gap-1 border-r border-slate-200 px-1.5 text-[10px] font-bold text-rose-600 hover:bg-rose-50 disabled:opacity-50 dark:border-slate-700 dark:text-rose-300 dark:hover:bg-rose-950/40"
        >
          <Square className={cn("h-3 w-3 fill-current", stopCollectBusy && "animate-pulse")} />
          Stop
        </button>
      )}
      {ownMode && onUploadImages && (
        <button
          type="button"
          disabled={uploadImagesBusy}
          onClick={onUploadImages}
          title="Upload images to Own pool"
          aria-label="Upload images"
          className="inline-flex items-center justify-center border-r border-slate-200 px-1.5 text-violet-600 hover:bg-violet-50 disabled:opacity-50 dark:border-slate-700 dark:text-violet-300 dark:hover:bg-violet-950/40"
        >
          <Upload className={cn("h-3.5 w-3.5", uploadImagesBusy && "animate-pulse")} />
        </button>
      )}
      {ownMode && onHfImport && (
        <button
          type="button"
          onClick={onHfImport}
          title="Import images+labels from Hugging Face .tar.gz"
          aria-label="Import from Hugging Face"
          className="inline-flex items-center justify-center border-r border-slate-200 px-1.5 text-violet-600 hover:bg-violet-50 dark:border-slate-700 dark:text-violet-300 dark:hover:bg-violet-950/40"
        >
          <CloudDownload className="h-3.5 w-3.5" />
        </button>
      )}
      {ownMode && onSam3 && (
        <button
          type="button"
          disabled={sam3Busy}
          onClick={onSam3}
          title="SAM3-label all New images in this day (one Chutes call per image: cup+can+bottle) → SAM3 state"
          aria-label="SAM3 label day"
          className="inline-flex items-center gap-1 border-r border-slate-200 px-1.5 text-[10px] font-black tracking-wide text-emerald-700 hover:bg-emerald-50 disabled:opacity-50 dark:border-slate-700 dark:text-emerald-300 dark:hover:bg-emerald-950/40"
        >
          {sam3Busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
          SAM3
        </button>
      )}
      {ownMode && onClearNew && (
        <button
          type="button"
          disabled={clearNewBusy || clearNewCount <= 0}
          onClick={onClearNew}
          title={
            clearNewCount > 0
              ? `Clear all ${clearNewCount} New images in this day (keeps Done)`
              : "No New images in this day"
          }
          aria-label="Clear New images in day"
          className="inline-flex items-center gap-1 border-r border-slate-200 px-1.5 text-[10px] font-black tracking-wide text-rose-700 hover:bg-rose-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-700 dark:text-rose-300 dark:hover:bg-rose-950/40"
        >
          <Trash2 className={cn("h-3.5 w-3.5", clearNewBusy && "animate-pulse")} />
          Clear new
        </button>
      )}
      {ownMode && onDeleteRange && (
        <button
          type="button"
          onClick={onDeleteRange}
          title="Delete Own images by selected index range"
          aria-label="Delete Own range"
          className="inline-flex items-center justify-center border-r border-slate-200 px-1.5 text-rose-600 hover:bg-rose-50 dark:border-slate-700 dark:text-rose-300 dark:hover:bg-rose-950/40"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      )}
      <button
        type="button"
        disabled={downloadBusy || reviewedCount <= 0}
        onClick={onDownload}
        title={
          ownMode
            ? "Download Own Done labels as a YOLO zip"
            : "Download labeled GT as a YOLO zip — pick block range"
        }
        aria-label={downloadBusy ? "Downloading" : "Download"}
        className="inline-flex items-center justify-center border-r border-slate-200 px-1.5 text-sky-700 hover:bg-sky-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-700 dark:text-sky-300 dark:hover:bg-sky-950/40"
      >
        <Download className={cn("h-3.5 w-3.5", downloadBusy && "animate-pulse")} />
      </button>
      {!ownMode && onSendGt && (
        <button
          type="button"
          disabled={reviewedCount <= 0}
          onClick={onSendGt}
          title="Send Platform Done GT to beverage-studio VM — pick block range"
          aria-label="Send GT"
          className="inline-flex items-center gap-1 border-r border-slate-200 px-1.5 text-[10px] font-black tracking-wide text-emerald-700 hover:bg-emerald-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-700 dark:text-emerald-300 dark:hover:bg-emerald-950/40"
        >
          <Send className="h-3.5 w-3.5" />
          Send
        </button>
      )}
      <button
        type="button"
        onClick={onUploadLabels}
        title={
          ownMode
            ? "Upload YOLO .txt labels for Own images (matched by filename stem)"
            : "Upload a folder of YOLO label .txt files (matched by block number)"
        }
        aria-label="Upload labels"
        className="inline-flex items-center justify-center border-r border-slate-200 px-1.5 text-violet-700 hover:bg-violet-50 dark:border-slate-700 dark:text-violet-300 dark:hover:bg-violet-950/40"
      >
        <FolderUp className="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        disabled={reviewedCount <= 0 && !dayListMode}
        onClick={onUploadUltralytics}
        title={
          ownMode
            ? "Upload Own Done range to Ultralytics Platform"
            : "Upload selected Done range to Ultralytics Platform"
        }
        aria-label="Upload to Ultralytics"
        className="inline-flex items-center justify-center border-r border-slate-200 px-1.5 text-rose-600 hover:bg-rose-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-700 dark:text-rose-300 dark:hover:bg-rose-950/40"
      >
        <CloudUpload className="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        disabled={reviewedCount <= 0}
        onClick={onDeepPostProc}
        title={
          ownMode
            ? "Deep post-process Own Done GT: upload ONNX/PT → tune → ship zip → compare HF"
            : "Deep post-process: upload ONNX/PT → tune → ship zip → compare HF"
        }
        aria-label="Deep post-process"
        className="inline-flex items-center justify-center border-r border-slate-200 px-1.5 text-fuchsia-600 hover:bg-fuchsia-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-700 dark:text-fuchsia-300 dark:hover:bg-fuchsia-950/40"
      >
        <Sparkles className="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        disabled={reviewedCount <= 0}
        onClick={onCompareHfModels}
        title={
          ownMode
            ? "Compare several Hugging Face models on Own Done GT"
            : "Compare several Hugging Face models on selected Done range"
        }
        aria-label="Compare HF models"
        className="inline-flex items-center justify-center border-r border-slate-200 px-1.5 text-indigo-600 hover:bg-indigo-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-700 dark:text-indigo-300 dark:hover:bg-indigo-950/40"
      >
        <GitCompareArrows className="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        onClick={onSelectBaseModel}
        title="Select base detection model from Hugging Face (test, then apply; Default restores manak0 winner)"
        aria-label="Select base model"
        className="inline-flex items-center justify-center border-r border-slate-200 px-1.5 text-teal-600 hover:bg-teal-50 dark:border-slate-700 dark:text-teal-300 dark:hover:bg-teal-950/40"
      >
        <Crosshair className="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        onClick={onSubmit}
        title={
          ownMode
            ? "Submit: find best model from Own Done GT"
            : "Submit: find best model from Done GT"
        }
        className="inline-flex items-center gap-1 border-r border-slate-200 px-1.5 text-[10px] font-semibold text-sky-700 hover:bg-sky-50 dark:border-slate-700 dark:text-sky-300 dark:hover:bg-sky-950/40"
      >
        <Rocket className="h-3 w-3" />
        Submit
      </button>
      <button
        type="button"
        onClick={onToggleTheme}
        aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
        title={theme === "dark" ? "Light mode" : "Dark mode"}
        className="inline-flex items-center justify-center px-1.5 text-slate-700 hover:bg-slate-50 dark:text-slate-200 dark:hover:bg-slate-800"
      >
        {theme === "dark" ? <Sun className="h-3.5 w-3.5 text-amber-400" /> : <Moon className="h-3.5 w-3.5 text-indigo-500" />}
      </button>
    </div>
  );
}

/** Active miner: platform p vs t scored with left-pane labels as GT. */
function TopVsLeftGtCard({
  score,
  emptyMessage,
  minerLabel,
  fallbackP,
}: {
  score: TopVsLeftScore | null;
  emptyMessage: string | null;
  minerLabel?: string;
  fallbackP?: { s: number | null; m: number | null; f: number | null };
}) {
  const pS = score?.platform_score ?? fallbackP?.s ?? null;
  const pM = score?.platform_map50 ?? fallbackP?.m ?? null;
  const pF = score?.platform_fp ?? fallbackP?.f ?? null;
  const tS = score?.composite ?? null;
  const tM = score?.map50 ?? null;
  const tF = score?.false_positive ?? null;
  const grade = plausibilityOf({ pS, pM, pF, tS, tM, tF });
  const tip = [
    "If your left labels match real GT, t should be close to platform p.",
    "p = platform · t = this miner scored against your labeling",
    emptyMessage && !score ? emptyMessage : "",
  ]
    .filter(Boolean)
    .join("\n");

  return (
    <div
      className={cn(
        "w-max max-w-full rounded border px-1.5 py-0.5",
        grade.label === "far"
          ? "border-rose-300 bg-rose-50 dark:border-rose-500/50 dark:bg-[#1a1012]"
          : grade.label === "plausible" || grade.label === "close"
            ? "border-emerald-200 bg-emerald-50/80 dark:border-emerald-800/50 dark:bg-emerald-950/20"
            : "border-slate-200 bg-slate-50 dark:border-slate-700 dark:bg-ink-950",
      )}
      title={tip}
    >
      <PtCompareCard inline compact title={minerLabel} pS={pS} pM={pM} pF={pF} tS={tS} tM={tM} tF={tF} />
    </div>
  );
}

function ClassCompareRail({
  classNames,
  leftBoxes,
  rightBoxes,
  currentClass,
  onSelectClass,
  embedded,
  hideHeader,
}: {
  classNames: string[];
  leftBoxes: DetectionBox[];
  rightBoxes: DetectionBox[];
  currentClass: number;
  onSelectClass: (cls: number) => void;
  embedded?: boolean;
  hideHeader?: boolean;
}) {
  const n = Math.max(classNames.length, 1);
  const leftCounts = useMemo(() => countByClass(leftBoxes, n), [leftBoxes, n]);
  const rightCounts = useMemo(() => countByClass(rightBoxes, n), [rightBoxes, n]);
  const rows = useMemo(() => {
    const maxId = Math.max(n, leftCounts.length, rightCounts.length);
    const out: { id: number; name: string; left: number; right: number }[] = [];
    for (let id = 0; id < maxId; id += 1) {
      const left = leftCounts[id] || 0;
      const right = rightCounts[id] || 0;
      if (id < classNames.length || left > 0 || right > 0) {
        out.push({ id, name: classNames[id] || `class ${id + 1}`, left, right });
      }
    }
    return out;
  }, [classNames, leftCounts, rightCounts, n]);

  return (
    <div
      className={cn(
        "flex flex-row items-stretch justify-center gap-1 lg:flex-col",
        !embedded &&
          "overflow-x-auto rounded-xl border border-slate-200/80 bg-white px-1.5 py-1 shadow-soft lg:w-[4.25rem] lg:overflow-y-auto lg:overflow-x-hidden dark:border-slate-800 dark:bg-ink-900",
      )}
      title="Per class: your GT count vs this miner"
    >
      <div className={cn("hidden px-0.5 pt-0.5 text-center text-[10px] font-black uppercase tracking-wide text-slate-400 lg:block", hideHeader && "!hidden")}>
        GT → miner
      </div>
      <div className="flex flex-row gap-1 lg:flex-col lg:gap-1">
        {rows.map((row) => {
          const mismatch = row.left !== row.right;
          const empty = row.left === 0 && row.right === 0;
          const active = currentClass === row.id;
          const color = colorForClass(row.id);
          return (
            <button
              key={row.id}
              type="button"
              title={`${row.name}: your GT ${row.left} · miner ${row.right}`}
              onClick={() => onSelectClass(row.id)}
              className={cn(
                "flex min-w-[7.25rem] items-center gap-1.5 rounded-md border px-1.5 py-1 text-left transition-colors duration-200 ease-out lg:min-w-0 lg:w-full",
                empty && "opacity-40",
                mismatch && !empty && "border-rose-500/50 bg-rose-500/10",
                !mismatch && !empty && "border-slate-200/80 bg-slate-50 dark:border-slate-700 dark:bg-ink-950",
                empty && "border-transparent bg-transparent",
                active && "ring-1 ring-sky-400/80",
              )}
            >
              <span
                className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="min-w-0 flex-1 truncate text-[10px] font-bold text-slate-800 dark:text-slate-100" title={row.name}>
                {row.name}
              </span>
              <span className="shrink-0 font-mono text-[10px] font-black tabular-nums leading-none">
                <TweenNum
                  value={row.left}
                  format={(n) => `${Math.round(n)}`}
                  className={cn(mismatch && row.left > row.right ? "text-rose-500" : "text-slate-700 dark:text-slate-200")}
                />
                <span className="px-0.5 text-slate-400">{mismatch ? "≠" : "="}</span>
                <TweenNum
                  value={row.right}
                  format={(n) => `${Math.round(n)}`}
                  className={cn(mismatch && row.right > row.left ? "text-rose-500" : "text-slate-700 dark:text-slate-200")}
                />
                <span className="pl-0.5 text-[10px]">{mismatch ? "✗" : "✓"}</span>
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function Pane({
  title,
  hint,
  headerExtra,
  children,
}: {
  title: string;
  hint: string;
  headerExtra?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="flex min-h-[320px] min-w-0 flex-1 flex-col overflow-hidden rounded-xl border border-slate-200/80 bg-white p-2 shadow-soft dark:border-slate-800 dark:bg-ink-900/60 dark:backdrop-blur-[2px]">
      <div className="mb-1 flex shrink-0 items-center gap-2 px-1">
        <h2 className="shrink-0 text-xs font-bold text-slate-800 dark:text-slate-100">{title}</h2>
        {headerExtra}
        <span className="ml-auto truncate text-[10px] text-slate-400">{hint}</span>
      </div>
      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">{children}</div>
    </section>
  );
}
