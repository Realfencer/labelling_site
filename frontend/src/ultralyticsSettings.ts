const API_KEY_STORAGE = "ultralytics_platform_api_key";

export type UltralyticsSettings = {
  apiKey: string;
};

export function loadUltralyticsSettings(): UltralyticsSettings {
  if (typeof window === "undefined") {
    return { apiKey: "" };
  }
  return { apiKey: (window.localStorage.getItem(API_KEY_STORAGE) || "").trim() };
}

export function saveUltralyticsSettings(settings: UltralyticsSettings): UltralyticsSettings {
  const apiKey = (settings.apiKey || "").trim();
  window.localStorage.setItem(API_KEY_STORAGE, apiKey);
  window.dispatchEvent(new CustomEvent("ultralytics-settings-changed"));
  return { apiKey };
}

export function maskUltralyticsKey(key: string): string {
  const k = (key || "").trim();
  if (k.length <= 10) return k ? "••••••••" : "";
  return `${k.slice(0, 6)}…${k.slice(-4)}`;
}
