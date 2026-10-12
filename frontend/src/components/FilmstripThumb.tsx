import { useEffect, useState } from "react";
import { getThumbnailBlobUrl, preloadThumbnail, subscribeThumbnail } from "../store/thumbnailLoader";
import { cn } from "../lib/utils";

interface Props {
  url: string;
  className?: string;
}

/** Lazy strip tile — placeholder until the visible-window batch fills the blob cache. */
export default function FilmstripThumb({ url, className }: Props) {
  const [src, setSrc] = useState(() => getThumbnailBlobUrl(url));

  useEffect(() => {
    if (!url) {
      setSrc(null);
      return;
    }
    const cached = getThumbnailBlobUrl(url);
    if (cached) {
      setSrc(cached);
      return subscribeThumbnail(url, setSrc);
    }
    setSrc(null);
    const unsub = subscribeThumbnail(url, (blobUrl) => setSrc(blobUrl));
    // Give the visible-window batch (~50–200ms) a chance before an individual GET.
    // Immediate fallback caused an N+1 PNG-encode storm on cold Own pools.
    const timer = window.setTimeout(() => {
      if (!getThumbnailBlobUrl(url)) preloadThumbnail(url, 40);
    }, 280);
    return () => {
      window.clearTimeout(timer);
      unsub();
    };
  }, [url]);

  if (!src) {
    return (
      <div className={cn("fx-skeleton relative", className)} aria-hidden />
    );
  }

  return (
    <img
      src={src}
      alt=""
      decoding="async"
      loading="lazy"
      draggable={false}
      className={cn("bg-slate-800 object-cover", className)}
    />
  );
}
