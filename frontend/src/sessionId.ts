/** Per-browser-tab labeler session. sessionStorage is tab-scoped, unlike cookies. */

const STORAGE_KEY = "labeler_tab_sid";

function randomSid(): string {
  const bytes = new Uint8Array(18);
  crypto.getRandomValues(bytes);
  let out = "";
  for (const b of bytes) out += b.toString(16).padStart(2, "0");
  return out;
}

/** Stable id for this tab only — two topics in two tabs get two backend sessions. */
export function getTabSessionId(): string {
  try {
    const existing = sessionStorage.getItem(STORAGE_KEY);
    if (existing && existing.length >= 16) return existing;
    const sid = randomSid();
    sessionStorage.setItem(STORAGE_KEY, sid);
    return sid;
  } catch {
    // Private mode / blocked storage — fall back to a memory id for this page life.
    const g = globalThis as { __labelerTabSid?: string };
    if (!g.__labelerTabSid) g.__labelerTabSid = randomSid();
    return g.__labelerTabSid;
  }
}

export function withTabSessionParam(url: string): string {
  if (!url || url.startsWith("blob:") || url.startsWith("data:")) return url;
  const sid = getTabSessionId();
  const join = url.includes("?") ? "&" : "?";
  // Avoid duplicating if already present.
  if (/[?&]sid=/.test(url)) return url;
  return `${url}${join}sid=${encodeURIComponent(sid)}`;
}

export function tabSessionHeaders(extra?: HeadersInit): HeadersInit {
  return { "X-Labeler-Sid": getTabSessionId(), ...(extra || {}) };
}
