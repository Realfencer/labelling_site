/** Web Push helpers for Chutes low-balance OS notifications. */

function urlBase64ToUint8Array(base64String: string): Uint8Array {
  const padding = "=".repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding).replace(/-/g, "+").replace(/_/g, "/");
  const raw = window.atob(base64);
  const output = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) output[i] = raw.charCodeAt(i);
  return output;
}

export function pushSupported(): boolean {
  return (
    typeof window !== "undefined" &&
    "serviceWorker" in navigator &&
    "PushManager" in window &&
    "Notification" in window
  );
}

export async function ensureChutesPushSubscription(): Promise<{ ok: boolean; detail?: string }> {
  if (!pushSupported()) {
    return { ok: false, detail: "This browser does not support Web Push" };
  }
  if (!window.isSecureContext && location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
    return {
      ok: false,
      detail: "Web Push needs HTTPS (or localhost). Open the site over HTTPS to enable background alerts.",
    };
  }

  const perm = await Notification.requestPermission();
  if (perm !== "granted") {
    return { ok: false, detail: "Notification permission denied" };
  }

  const reg = await navigator.serviceWorker.register("/sw.js", { scope: "/" });
  await navigator.serviceWorker.ready;

  const keyRes = await fetch("/api/chutes/push/vapid-public-key", { credentials: "include" });
  if (!keyRes.ok) {
    return { ok: false, detail: "Failed to load VAPID public key" };
  }
  const { publicKey } = (await keyRes.json()) as { publicKey: string };
  if (!publicKey) return { ok: false, detail: "Missing VAPID public key" };

  let sub = await reg.pushManager.getSubscription();
  if (!sub) {
    sub = await reg.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: urlBase64ToUint8Array(publicKey) as BufferSource,
    });
  }

  const payload = sub.toJSON();
  const saveRes = await fetch("/api/chutes/push/subscribe", {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      endpoint: payload.endpoint,
      keys: payload.keys,
      expirationTime: payload.expirationTime ?? null,
    }),
  });
  if (!saveRes.ok) {
    const body = await saveRes.json().catch(() => ({ detail: saveRes.statusText }));
    return { ok: false, detail: String((body as { detail?: string }).detail || "Subscribe failed") };
  }
  return { ok: true };
}
