const API_KEY_STORAGE = "chutes_api_key";
const THRESHOLD_STORAGE = "chutes_balance_threshold_usd";
const DEFAULT_THRESHOLD = 20;

export type ChutesSettings = {
  apiKey: string;
  threshold: number;
};

export function loadChutesSettings(): ChutesSettings {
  if (typeof window === "undefined") {
    return { apiKey: "", threshold: DEFAULT_THRESHOLD };
  }
  const apiKey = (window.localStorage.getItem(API_KEY_STORAGE) || "").trim();
  const raw = window.localStorage.getItem(THRESHOLD_STORAGE);
  let threshold = DEFAULT_THRESHOLD;
  if (raw != null && raw !== "") {
    const parsed = Number(raw);
    if (Number.isFinite(parsed) && parsed >= 0) threshold = parsed;
  }
  return { apiKey, threshold };
}

export function saveChutesSettings(settings: ChutesSettings): ChutesSettings {
  const apiKey = (settings.apiKey || "").trim();
  const threshold =
    Number.isFinite(settings.threshold) && settings.threshold >= 0
      ? Number(settings.threshold)
      : DEFAULT_THRESHOLD;
  window.localStorage.setItem(API_KEY_STORAGE, apiKey);
  window.localStorage.setItem(THRESHOLD_STORAGE, String(threshold));
  window.dispatchEvent(new CustomEvent("chutes-settings-changed"));
  return { apiKey, threshold };
}

export function formatChutesUsd(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function formatChutesHeaderLine(snapshot: {
  configured?: boolean;
  balance?: number | null;
  balance_alert_usd?: number;
  low_balance?: boolean;
  username?: string | null;
} | null): string {
  if (!snapshot?.configured) return "Chutes · set API key in Chutes panel";
  const bal = formatChutesUsd(snapshot.balance);
  const user = snapshot.username ? `${snapshot.username} · ` : "";
  if (snapshot.low_balance) {
    return `Chutes · ${user}${bal} · below ${formatChutesUsd(snapshot.balance_alert_usd)}`;
  }
  return `Chutes · ${user}${bal} · OK`;
}
