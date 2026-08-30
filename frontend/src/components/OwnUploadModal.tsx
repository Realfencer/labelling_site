import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FolderOpen, Images, Loader2, Upload, X } from "lucide-react";
import { api } from "../api";
import type { AppState } from "../types";
import { cn } from "../lib/utils";
import { challengeImageUrl, challengeThumbnailUrl } from "../store/imageLoader";

const IMAGE_RE = /\.(jpe?g|png|webp|bmp|gif|tif{1,2})$/i;
const BATCH_SIZE = 80; // keep each multipart request well under Starlette defaults

type OwnExisting = {
  filename: string;
  stem: string;
  block: number;
  size: number;
  digest?: string;
  width: number;
  height: number;
  reviewed: boolean;
};

type UploadStats = { added: number; skipped: number; deduped?: number; replaced?: number; total: number };

function isImageFile(f: File): boolean {
  return f.type.startsWith("image/") || IMAGE_RE.test(f.name);
}

function safeFilename(value: string): string {
  return Array.from(value)
    .map((ch) => (/[a-zA-Z0-9._-]/.test(ch) ? ch : "_"))
    .join("");
}

function destNameFor(file: File): string {
  const base = file.name.replace(/\\/g, "/").split("/").pop() || "image.jpg";
  return safeFilename(base) || "image.jpg";
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

/** Content fingerprint for client-side identical-file dedupe.
 * Uses Web Crypto when available (HTTPS). On plain HTTP, crypto.subtle is
 * undefined — fall back to a local hash so upload never crashes. */
async function contentFingerprint(file: File): Promise<string> {
  const buf = await file.arrayBuffer();
  const bytes = new Uint8Array(buf);
  const subtle = globalThis.crypto?.subtle;
  if (subtle?.digest) {
    try {
      const digest = await subtle.digest("SHA-256", bytes);
      return Array.from(new Uint8Array(digest))
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");
    } catch {
      // fall through
    }
  }
  // FNV-1a 64-bit-ish (two 32-bit words) — fine for identical-byte detection.
  let h1 = 0x811c9dc5;
  let h2 = 0x811c9dc5 ^ (bytes.length * 0x01000193);
  for (let i = 0; i < bytes.length; i += 1) {
    h1 ^= bytes[i];
    h1 = Math.imul(h1, 0x01000193);
    h2 ^= bytes[bytes.length - 1 - i];
    h2 = Math.imul(h2, 0x01000193);
  }
  return `fnv:${(h1 >>> 0).toString(16).padStart(8, "0")}${(h2 >>> 0).toString(16).padStart(8, "0")}:${bytes.length}`;
}

async function readAllDirectoryEntries(dir: FileSystemDirectoryEntry): Promise<FileSystemEntry[]> {
  const reader = dir.createReader();
  const all: FileSystemEntry[] = [];
  for (;;) {
    const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => {
      reader.readEntries(resolve, reject);
    });
    if (!batch.length) break;
    all.push(...batch);
  }
  return all;
}

async function collectImagesFromEntry(entry: FileSystemEntry, out: File[]): Promise<void> {
  if (entry.isFile) {
    const file = await new Promise<File>((resolve, reject) => {
      (entry as FileSystemFileEntry).file(resolve, reject);
    });
    if (isImageFile(file)) {
      const rel = entry.fullPath.replace(/^\//, "") || file.name;
      out.push(new File([file], rel, { type: file.type || "application/octet-stream" }));
    }
    return;
  }
  if (entry.isDirectory) {
    const children = await readAllDirectoryEntries(entry as FileSystemDirectoryEntry);
    for (const child of children) await collectImagesFromEntry(child, out);
  }
}

async function filesFromDataTransfer(dt: DataTransfer, fileSnapshot?: File[]): Promise<File[]> {
  const out: File[] = [];
  const seen = new Set<string>();
  const add = (f: File) => {
    if (!isImageFile(f)) return;
    const key = `${f.name}:${f.size}:${f.lastModified}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push(f);
  };

  // 1) Prefer the synchronous FileList snapshot. Multi-file drops must NOT rely on
  // webkitGetAsEntry + await — after the first await many browsers clear DataTransfer
  // and only the first file survives.
  const files = fileSnapshot ?? Array.from(dt.files || []);
  for (const f of files) add(f);

  // 2) Folder drops: collect directory entries synchronously, then recurse.
  const dirEntries: FileSystemDirectoryEntry[] = [];
  try {
    for (const item of Array.from(dt.items || [])) {
      if (item.kind !== "file") continue;
      const entry = item.webkitGetAsEntry?.() ?? null;
      if (entry?.isDirectory) dirEntries.push(entry as FileSystemDirectoryEntry);
    }
  } catch {
    // Some browsers revoke items after FileList is read — FileList path above is enough.
  }
  for (const dir of dirEntries) {
    const nested: File[] = [];
    await collectImagesFromEntry(dir, nested);
    for (const f of nested) add(f);
  }
  return out;
}

type Phase = "pick" | "scanning" | "ready" | "conflicts" | "uploading" | "done";

type Slot =
  | { kind: "pool"; existing: OwnExisting }
  | { kind: "batch"; file: File };

type Conflict = {
  destName: string;
  incoming: File;
  current: Slot;
};

function ComparePane({
  title,
  badge,
  src,
  name,
  meta,
  labeled,
}: {
  title: string;
  badge: string;
  src: string;
  name: string;
  meta: string;
  labeled?: boolean;
}) {
  return (
    <div className="min-w-0 flex-1 overflow-hidden rounded-xl border border-slate-200 bg-slate-50 dark:border-slate-700 dark:bg-ink-950">
      <div className="flex items-center justify-between gap-2 px-2.5 py-1.5">
        <span className="text-[9px] font-extrabold uppercase tracking-wider text-slate-500">{title}</span>
        <span className="rounded bg-slate-200 px-1.5 py-0.5 text-[9px] font-bold uppercase text-slate-700 dark:bg-slate-800 dark:text-slate-200">
          {badge}
        </span>
      </div>
      <div className="flex h-40 items-center justify-center bg-slate-200/70 px-2 dark:bg-black/40">
        {src ? (
          <img src={src} alt={title} className="max-h-40 max-w-full object-contain" />
        ) : (
          <span className="text-[11px] text-slate-400">No preview</span>
        )}
      </div>
      <div className="space-y-0.5 px-2.5 py-2">
        <div className="truncate font-mono text-[11px] font-semibold text-slate-800 dark:text-slate-100" title={name}>
          {name}
        </div>
        <div className="text-[10px] text-slate-500">{meta}</div>
        {labeled ? (
          <div className="text-[10px] font-bold text-emerald-600 dark:text-emerald-300">Already labeled Done</div>
        ) : null}
      </div>
    </div>
  );
}

export default function OwnUploadModal({
  open,
  onClose,
  onDone,
  imageCacheToken = "",
}: {
  open: boolean;
  onClose: () => void;
  onDone: (state: AppState & { upload?: UploadStats }) => void | Promise<void>;
  imageCacheToken?: string;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const folderRef = useRef<HTMLInputElement>(null);
  const [phase, setPhase] = useState<Phase>("pick");
  const [files, setFiles] = useState<File[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState({
    sentFiles: 0,
    totalFiles: 0,
    sentBytes: 0,
    totalBytes: 0,
    added: 0,
    skipped: 0,
    deduped: 0,
    replaced: 0,
    batch: 0,
    batches: 0,
  });
  const [lastState, setLastState] = useState<(AppState & { upload?: UploadStats }) | null>(null);
  const abortRef = useRef(false);

  const [conflictIdx, setConflictIdx] = useState(0);
  const [conflicts, setConflicts] = useState<Conflict[]>([]);
  const keptRef = useRef<Map<string, { file: File; replace: boolean }>>(new Map());
  const skippedRef = useRef(0);
  const dedupedRef = useRef(0);
  const incomingUrls = useRef<Map<File, string>>(new Map());
  const digestCacheRef = useRef<Map<string, string>>(new Map());
  const [previewTick, setPreviewTick] = useState(0);

  const objectUrlFor = useCallback((file: File) => {
    const hit = incomingUrls.current.get(file);
    if (hit) return hit;
    const url = URL.createObjectURL(file);
    incomingUrls.current.set(file, url);
    return url;
  }, []);

  const revokeObjectUrls = useCallback(() => {
    incomingUrls.current.forEach((url) => URL.revokeObjectURL(url));
    incomingUrls.current.clear();
  }, []);

  useEffect(() => {
    if (!open) return;
    setPhase("pick");
    setFiles([]);
    setError("");
    setDragOver(false);
    setLastState(null);
    abortRef.current = false;
    setConflicts([]);
    setConflictIdx(0);
    keptRef.current = new Map();
    skippedRef.current = 0;
    dedupedRef.current = 0;
    digestCacheRef.current = new Map();
    revokeObjectUrls();
    setProgress({
      sentFiles: 0,
      totalFiles: 0,
      sentBytes: 0,
      totalBytes: 0,
      added: 0,
      skipped: 0,
      deduped: 0,
      replaced: 0,
      batch: 0,
      batches: 0,
    });
  }, [open, revokeObjectUrls]);

  useEffect(() => () => revokeObjectUrls(), [revokeObjectUrls]);

  useEffect(() => {
    const el = folderRef.current;
    if (!el) return;
    el.setAttribute("webkitdirectory", "");
    el.setAttribute("directory", "");
  }, [open]);

  const totalBytes = useMemo(() => files.reduce((s, f) => s + f.size, 0), [files]);

  const ingestFiles = useCallback(async (incoming: File[], { append = false }: { append?: boolean } = {}) => {
    setError("");
    setPhase("scanning");
    try {
      const images = incoming.filter(isImageFile);
      if (!images.length) throw new Error("No image files found (jpg / png / webp / bmp / gif / tiff)");
      setFiles((prev) => {
        const seen = new Set<string>();
        const unique: File[] = [];
        const seed = append ? prev : [];
        for (const f of [...seed, ...images]) {
          // Keep every distinct picker entry (name+size+mtime). Exact-content
          // dupes are removed later in prepareAndUpload / on the server.
          const key = `${f.name}:${f.size}:${f.lastModified}`;
          if (seen.has(key)) continue;
          seen.add(key);
          unique.push(f);
        }
        return unique;
      });
      setPhase("ready");
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
      setPhase((p) => (p === "scanning" ? "pick" : p));
    }
  }, []);

  const onPickFiles = (list: FileList | null) => {
    if (!list?.length) return;
    // Always take the full FileList — browsers can report length>1 correctly
    // even when the OS dialog selected a folder / multi-select.
    void ingestFiles(Array.from(list), { append: files.length > 0 });
  };

  const runUpload = useCallback(
    async (toSend: File[], replaceNames: string[], alreadySkipped: number, alreadyDeduped: number) => {
      if (!toSend.length) {
        setProgress((p) => ({
          ...p,
          skipped: alreadySkipped,
          deduped: alreadyDeduped,
          added: 0,
          replaced: 0,
          sentFiles: 0,
          totalFiles: 0,
        }));
        setPhase("done");
        return;
      }
      abortRef.current = false;
      setError("");
      setPhase("uploading");
      const batches = Math.ceil(toSend.length / BATCH_SIZE);
      const bytes = toSend.reduce((s, f) => s + f.size, 0);
      let sentFiles = 0;
      let sentBytes = 0;
      let added = 0;
      let skipped = alreadySkipped;
      let deduped = alreadyDeduped;
      let replaced = 0;
      let finalState: (AppState & { upload?: UploadStats }) | null = null;
      setProgress({
        sentFiles: 0,
        totalFiles: toSend.length,
        sentBytes: 0,
        totalBytes: bytes,
        added: 0,
        skipped,
        deduped,
        replaced: 0,
        batch: 0,
        batches,
      });
      const replaceLc = new Set(replaceNames.map((n) => n.toLowerCase()));
      for (let i = 0; i < toSend.length; i += BATCH_SIZE) {
        if (abortRef.current) throw new Error("Upload cancelled");
        const batch = toSend.slice(i, i + BATCH_SIZE);
        const batchReplace = batch.map(destNameFor).filter((n) => replaceLc.has(n.toLowerCase()));
        const batchNum = Math.floor(i / BATCH_SIZE) + 1;
        setProgress((p) => ({ ...p, batch: batchNum, batches }));
        const state = await api.uploadOwnImages(batch, {
          replaceNames: batchReplace,
          onProgress: (loaded, total) => {
            const batchBytes = batch.reduce((s, f) => s + f.size, 0);
            const ratio = total > 0 ? loaded / total : 0;
            setProgress((p) => ({
              ...p,
              sentBytes: Math.min(p.totalBytes, sentBytes + Math.round(batchBytes * ratio)),
              sentFiles: sentFiles + Math.min(batch.length, Math.floor(batch.length * ratio)),
              batch: batchNum,
            }));
          },
        });
        const up = state.upload || { added: 0, skipped: 0, deduped: 0, replaced: 0, total: 0 };
        added += up.added;
        skipped += up.skipped;
        deduped += up.deduped || 0;
        replaced += up.replaced || 0;
        sentFiles += batch.length;
        sentBytes += batch.reduce((s, f) => s + f.size, 0);
        finalState = state;
        setProgress({
          sentFiles,
          totalFiles: toSend.length,
          sentBytes,
          totalBytes: bytes,
          added,
          skipped,
          deduped,
          replaced,
          batch: batchNum,
          batches,
        });
      }
      if (!finalState) throw new Error("Upload produced no result");
      finalState = {
        ...finalState,
        upload: {
          added,
          skipped,
          deduped,
          replaced,
          total: finalState.upload?.total ?? finalState.own_metrics?.total ?? 0,
        },
      };
      setLastState(finalState);
      setPhase("done");
      await onDone(finalState);
    },
    [onDone],
  );

  const finishConflicts = useCallback(async () => {
    const kept = [...keptRef.current.values()];
    const toSend = kept.map((k) => k.file);
    const replaceNames = kept.filter((k) => k.replace).map((k) => destNameFor(k.file));
    try {
      await runUpload(toSend, replaceNames, skippedRef.current, dedupedRef.current);
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
      setPhase(files.length ? "ready" : "pick");
    }
  }, [files.length, runUpload]);

  const applyConflict = useCallback(
    (action: "skip" | "skip_all" | "replace" | "replace_all") => {
      const rest = conflicts.slice(conflictIdx);
      const current = rest[0];
      if (!current) {
        void finishConflicts();
        return;
      }
      const applyOne = (c: Conflict, replace: boolean) => {
        const key = c.destName.toLowerCase();
        if (replace) {
          keptRef.current.set(key, { file: c.incoming, replace: c.current.kind === "pool" || keptRef.current.get(key)?.replace === true });
        } else {
          skippedRef.current += 1;
        }
      };
      if (action === "skip") applyOne(current, false);
      else if (action === "replace") applyOne(current, true);
      else if (action === "skip_all") rest.forEach((c) => applyOne(c, false));
      else rest.forEach((c) => applyOne(c, true));

      if (action === "skip_all" || action === "replace_all" || conflictIdx + 1 >= conflicts.length) {
        void finishConflicts();
        return;
      }
      setConflictIdx((n) => n + 1);
      setPreviewTick((n) => n + 1);
    },
    [conflictIdx, conflicts, finishConflicts],
  );

  const prepareAndUpload = async () => {
    if (!files.length) return;
    setError("");
    setPhase("scanning");
    try {
      let existing: OwnExisting[] = [];
      try {
        const listed = await api.listOwnExisting();
        existing = Array.isArray(listed?.files) ? listed.files.filter(Boolean) : [];
      } catch {
        existing = [];
      }
      const poolByName = new Map<string, OwnExisting>();
      const poolByDigest = new Set<string>();
      // Only trust SHA-256 digests from the server (never mix with HTTP fallback hashes).
      const subtleOk = Boolean(globalThis.crypto?.subtle?.digest);
      for (const row of existing) {
        if (!row?.filename) continue;
        poolByName.set(String(row.filename).toLowerCase(), row);
        if (subtleOk && row.digest) poolByDigest.add(row.digest);
      }
      const kept = new Map<string, { file: File; replace: boolean }>();
      const found: Conflict[] = [];
      skippedRef.current = 0;
      dedupedRef.current = 0;
      const seenDigests = new Set<string>();
      const digestOf = async (file: File) => {
        const key = `${file.name}:${file.size}:${file.lastModified}`;
        const hit = digestCacheRef.current.get(key);
        if (hit) return hit;
        const digest = await contentFingerprint(file);
        digestCacheRef.current.set(key, digest);
        return digest;
      };

      for (const file of files) {
        const dest = destNameFor(file);
        const key = dest.toLowerCase();
        const digest = await digestOf(file);
        // Identical bytes already in pool or earlier in this batch → keep one.
        if ((subtleOk && digest && poolByDigest.has(digest)) || (digest && seenDigests.has(digest))) {
          dedupedRef.current += 1;
          continue;
        }
        const pool = poolByName.get(key);
        const already = kept.get(key);
        // Same filename, different content → ask skip/replace.
        if (pool && !already) {
          const sameContent = subtleOk && pool.digest && digest && pool.digest === digest;
          if (sameContent) {
            dedupedRef.current += 1;
            continue;
          }
          found.push({ destName: dest, incoming: file, current: { kind: "pool", existing: pool } });
          continue;
        }
        if (already) {
          const alreadyDigest = await digestOf(already.file);
          if (digest && alreadyDigest && digest === alreadyDigest) {
            dedupedRef.current += 1;
            continue;
          }
          found.push({ destName: dest, incoming: file, current: { kind: "batch", file: already.file } });
          continue;
        }
        kept.set(key, { file, replace: false });
        if (digest) seenDigests.add(digest);
      }

      keptRef.current = kept;
      if (!found.length) {
        await runUpload(
          [...kept.values()].map((k) => k.file),
          [],
          0,
          dedupedRef.current,
        );
        return;
      }
      setConflicts(found);
      setConflictIdx(0);
      setPreviewTick((n) => n + 1);
      setPhase("conflicts");
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/i, ""));
      setPhase("ready");
    }
  };

  if (!open) return null;

  const pct =
    progress.totalBytes > 0
      ? Math.min(100, Math.round((progress.sentBytes / progress.totalBytes) * 100))
      : progress.totalFiles > 0
        ? Math.min(100, Math.round((progress.sentFiles / progress.totalFiles) * 100))
        : 0;

  const conflict = phase === "conflicts" ? conflicts[conflictIdx] : null;
  const incomingSrc = conflict ? objectUrlFor(conflict.incoming) : "";
  const existingSrc = conflict
    ? conflict.current.kind === "pool"
      ? challengeThumbnailUrl(
          0,
          "local",
          conflict.current.existing.block,
          false,
          imageCacheToken,
          conflict.current.existing.stem,
        ) ||
        challengeImageUrl(
          0,
          "local",
          conflict.current.existing.block,
          imageCacheToken,
          conflict.current.existing.stem,
        )
      : objectUrlFor(conflict.current.file)
    : "";
  void previewTick;

  const wide = phase === "conflicts";

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-slate-950/60 p-3 backdrop-blur-[2px] sm:items-center">
      <div
        className={cn(
          "flex max-h-[min(92vh,720px)] w-full flex-col overflow-hidden rounded-2xl border border-violet-500/30 bg-white shadow-2xl dark:border-violet-700/50 dark:bg-ink-900",
          wide ? "max-w-3xl" : "max-w-lg",
        )}
      >
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-700">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 text-white shadow-md shadow-violet-500/30">
              <Upload className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-sm font-extrabold text-slate-900 dark:text-white">
                {phase === "conflicts" ? "Same filename already exists" : "Upload to Own pool"}
              </h2>
              <p className="text-[11px] text-slate-500">
                {phase === "conflicts"
                  ? `Conflict ${conflictIdx + 1} of ${conflicts.length} · skip or replace`
                  : "Images or whole folders · no file-count limit"}
              </p>
            </div>
          </div>
          <button
            type="button"
            disabled={phase === "uploading"}
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 disabled:opacity-40 dark:hover:bg-slate-800"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-4 py-4">
          <input
            ref={fileRef}
            type="file"
            accept="image/*,.jpg,.jpeg,.png,.webp,.bmp,.gif,.tif,.tiff"
            multiple
            className="hidden"
            onChange={(e) => {
              onPickFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <input
            ref={folderRef}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => {
              onPickFiles(e.target.files);
              e.target.value = "";
            }}
          />

          {(phase === "pick" || phase === "scanning" || phase === "ready") && (
            <div
              onDragEnter={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                if (e.currentTarget.contains(e.relatedTarget as Node)) return;
                setDragOver(false);
              }}
              onDrop={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setDragOver(false);
                // Snapshot FileList NOW — DataTransfer.files can empty after the handler returns.
                const snapshot = Array.from(e.dataTransfer.files || []);
                void filesFromDataTransfer(e.dataTransfer, snapshot).then((list) =>
                  ingestFiles(list, { append: files.length > 0 }),
                );
              }}
              className={cn(
                "relative overflow-hidden rounded-2xl border-2 border-dashed px-4 py-8 text-center transition",
                dragOver
                  ? "border-violet-400 bg-violet-50 dark:border-violet-400 dark:bg-violet-950/40"
                  : "border-slate-300 bg-slate-50 dark:border-slate-600 dark:bg-ink-950",
              )}
            >
              <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(139,92,246,0.12),transparent_55%)]" />
              <Images className={cn("mx-auto h-10 w-10", dragOver ? "text-violet-500" : "text-slate-400")} />
              <p className="mt-3 text-sm font-bold text-slate-800 dark:text-slate-100">
                {phase === "scanning" ? "Scanning…" : dragOver ? "Drop to add" : "Drop images or a folder here"}
              </p>
              <p className="mt-1 text-[11px] text-slate-500">jpg · png · webp · bmp · gif · tiff</p>
              <div className="relative mt-4 flex flex-wrap items-center justify-center gap-2">
                <button
                  type="button"
                  disabled={phase === "scanning"}
                  onClick={() => fileRef.current?.click()}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-2 text-[11px] font-bold text-white hover:bg-violet-500 disabled:opacity-50"
                >
                  <Upload className="h-3.5 w-3.5" />
                  Choose images
                </button>
                <button
                  type="button"
                  disabled={phase === "scanning"}
                  onClick={() => folderRef.current?.click()}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-violet-300 bg-white px-3 py-2 text-[11px] font-bold text-violet-800 hover:bg-violet-50 disabled:opacity-50 dark:border-violet-700 dark:bg-ink-800 dark:text-violet-200"
                >
                  <FolderOpen className="h-3.5 w-3.5" />
                  Choose folder
                </button>
              </div>
            </div>
          )}

          {phase === "ready" && files.length > 0 && (
            <div className="rounded-xl border border-violet-200 bg-violet-50/80 px-3 py-3 dark:border-violet-800 dark:bg-violet-950/30">
              <div className="flex items-baseline justify-between gap-2">
                <div className="text-[10px] font-bold uppercase tracking-wider text-violet-600 dark:text-violet-300">Ready to upload</div>
                <button
                  type="button"
                  onClick={() => {
                    setFiles([]);
                    setPhase("pick");
                  }}
                  className="text-[10px] font-semibold text-violet-600 hover:underline dark:text-violet-300"
                >
                  Clear
                </button>
              </div>
              <div className="mt-1 text-2xl font-black tabular-nums text-slate-900 dark:text-white">
                {files.length.toLocaleString()}
                <span className="ml-1.5 text-sm font-bold text-slate-500">images</span>
              </div>
              <div className="text-[11px] text-slate-500">{formatBytes(totalBytes)} total</div>
              <div className="mt-2 max-h-24 overflow-y-auto rounded-lg bg-white/70 px-2 py-1.5 text-[10px] text-slate-600 dark:bg-ink-950/60 dark:text-slate-300">
                {files.slice(0, 40).map((f) => (
                  <div key={`${f.name}:${f.size}`} className="truncate font-mono">
                    {f.name}
                  </div>
                ))}
                {files.length > 40 && (
                  <div className="pt-0.5 font-semibold text-slate-400">…and {(files.length - 40).toLocaleString()} more</div>
                )}
              </div>
            </div>
          )}

          {phase === "conflicts" && conflict && (
            <div className="space-y-3">
              <p className="text-[12px] text-slate-600 dark:text-slate-300">
                <span className="font-mono font-bold text-slate-900 dark:text-white">{conflict.destName}</span>
                {conflict.current.kind === "pool"
                  ? " is already in this Own pool. Compare the two images, then skip or replace."
                  : " appears more than once in this upload. Compare, then keep one."}
              </p>
              <div className="grid grid-cols-2 gap-3">
                <ComparePane
                  title={conflict.current.kind === "pool" ? "In Own pool" : "Already chosen"}
                  badge="existing"
                  src={existingSrc}
                  name={
                    conflict.current.kind === "pool" ? conflict.current.existing.filename : conflict.current.file.name
                  }
                  meta={
                    conflict.current.kind === "pool"
                      ? [
                          conflict.current.existing.width && conflict.current.existing.height
                            ? `${conflict.current.existing.width}×${conflict.current.existing.height}`
                            : "",
                          formatBytes(conflict.current.existing.size),
                        ]
                          .filter(Boolean)
                          .join(" · ")
                      : formatBytes(conflict.current.file.size)
                  }
                  labeled={conflict.current.kind === "pool" ? conflict.current.existing.reviewed : false}
                />
                <ComparePane
                  title="New file"
                  badge="incoming"
                  src={incomingSrc}
                  name={conflict.incoming.name}
                  meta={formatBytes(conflict.incoming.size)}
                />
              </div>
            </div>
          )}

          {(phase === "uploading" || phase === "done") && (
            <div className="space-y-3">
              <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-3 dark:border-slate-700 dark:bg-ink-950">
                <div className="mb-1.5 flex items-center justify-between text-[11px] font-bold">
                  <span className="text-slate-600 dark:text-slate-300">
                    {phase === "done" ? "Upload complete" : `Uploading batch ${progress.batch}/${progress.batches}`}
                  </span>
                  <span className="tabular-nums text-violet-600 dark:text-violet-300">{pct}%</span>
                </div>
                <div className="h-2.5 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                  <div
                    className={cn(
                      "h-full rounded-full transition-[width] duration-200",
                      phase === "done"
                        ? "bg-emerald-500"
                        : "bg-gradient-to-r from-violet-500 via-fuchsia-500 to-violet-400",
                    )}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] sm:grid-cols-5">
                  <div>
                    <div className="text-[9px] font-bold uppercase tracking-wider text-slate-400">Files</div>
                    <div className="font-bold tabular-nums text-slate-800 dark:text-slate-100">
                      {progress.sentFiles.toLocaleString()} / {progress.totalFiles.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-[9px] font-bold uppercase tracking-wider text-slate-400">Added</div>
                    <div className="font-bold tabular-nums text-emerald-600 dark:text-emerald-300">
                      {progress.added.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-[9px] font-bold uppercase tracking-wider text-slate-400">Replaced</div>
                    <div className="font-bold tabular-nums text-sky-600 dark:text-sky-300">
                      {progress.replaced.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-[9px] font-bold uppercase tracking-wider text-slate-400">Deduped</div>
                    <div className="font-bold tabular-nums text-violet-600 dark:text-violet-300">
                      {progress.deduped.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-[9px] font-bold uppercase tracking-wider text-slate-400">Skipped</div>
                    <div className="font-bold tabular-nums text-amber-600 dark:text-amber-300">
                      {progress.skipped.toLocaleString()}
                    </div>
                  </div>
                </div>
              </div>
              {phase === "done" && lastState && (
                <p className="text-center text-[12px] font-semibold text-slate-600 dark:text-slate-300">
                  Own pool now has{" "}
                  <span className="tabular-nums text-violet-600 dark:text-violet-300">
                    {(lastState.upload?.total ?? lastState.own_metrics?.total ?? 0).toLocaleString()}
                  </span>{" "}
                  images
                </p>
              )}
            </div>
          )}

          {error && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-[12px] text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-200">
              {error}
            </div>
          )}
        </div>

        <div className="flex flex-wrap justify-end gap-2 border-t border-slate-200 px-4 py-3 dark:border-slate-700">
          {phase === "conflicts" ? (
            <>
              <button
                type="button"
                onClick={() => applyConflict("skip_all")}
                className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300"
              >
                Skip all
              </button>
              <button
                type="button"
                onClick={() => applyConflict("skip")}
                className="rounded-lg border border-amber-300 px-3 py-2 text-sm font-semibold text-amber-800 hover:bg-amber-50 dark:border-amber-800 dark:text-amber-200"
              >
                Skip
              </button>
              <button
                type="button"
                onClick={() => applyConflict("replace")}
                className="rounded-lg bg-sky-600 px-3 py-2 text-sm font-bold text-white hover:bg-sky-500"
              >
                Replace
              </button>
              <button
                type="button"
                onClick={() => applyConflict("replace_all")}
                className="rounded-lg bg-violet-600 px-3 py-2 text-sm font-bold text-white hover:bg-violet-500"
              >
                Replace all
              </button>
            </>
          ) : phase === "uploading" ? (
            <button
              type="button"
              onClick={() => {
                abortRef.current = true;
              }}
              className="rounded-lg border border-rose-300 px-3 py-2 text-sm font-semibold text-rose-600 hover:bg-rose-50 dark:border-rose-800 dark:text-rose-300"
            >
              Cancel
            </button>
          ) : (
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-600 dark:border-slate-700 dark:text-slate-300"
            >
              {phase === "done" ? "Close" : "Cancel"}
            </button>
          )}
          {phase === "ready" && (
            <button
              type="button"
              disabled={!files.length}
              onClick={() => void prepareAndUpload()}
              className="inline-flex items-center gap-1.5 rounded-lg bg-violet-600 px-4 py-2 text-sm font-bold text-white hover:bg-violet-500 disabled:opacity-40"
            >
              <Upload className="h-4 w-4" />
              Upload {files.length.toLocaleString()}
            </button>
          )}
          {phase === "uploading" && (
            <button
              type="button"
              disabled
              className="inline-flex items-center gap-1.5 rounded-lg bg-violet-600 px-4 py-2 text-sm font-bold text-white opacity-80"
            >
              <Loader2 className="h-4 w-4 animate-spin" />
              Uploading…
            </button>
          )}
          {phase === "done" && (
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-bold text-white hover:bg-emerald-500"
            >
              Done
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
