import { useEffect, useState } from "react";
import { getThumbnailBlobUrl, preloadThumbnail, subscribeThumbnail } from "../store/thumbnailLoader";
import { cn } from "../lib/utils";

interface Props {
  url: string;
  className?: string;
}

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
    // Self-fetch if batch strip load was cleared/missed (Own workspace mode switches).
    preloadThumbnail(url, 220);
    return unsub;
  }, [url]);

  if (!src) {
    return <div className={cn("bg-slate-800", className)} aria-hidden />;
  }

  return (
    <img
      src={src}
      alt=""
      decoding="async"
      draggable={false}
      className={cn("bg-slate-800 object-cover", className)}
    />
  );
}
