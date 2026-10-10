import { useEffect, useMemo } from "react";
import { cn } from "../lib/utils";
import { useKeys } from "../store/useKeys";

/**
 * "Push to": pick one of YOUR saved Hugging Face accounts and a repo name (type a new one or pick an
 * existing one scanned from the account). No token is ever typed here.
 * ``value`` is the final "owner/name" the push endpoints expect.
 */
export default function HfPushTarget({
  accountId,
  repoName,
  onChange,
  className,
}: {
  accountId: string;
  repoName: string;
  onChange: (next: { accountId: string; repoName: string; repo: string }) => void;
  className?: string;
}) {
  const view = useKeys((s) => s.view);
  const models = useKeys((s) => s.models);
  const setOpen = useKeys((s) => s.setOpen);
  const accounts = view?.hf ?? [];
  const current = accounts.find((a) => a.id === accountId) ?? accounts[0];

  useEffect(() => {
    if (current && current.id !== accountId) onChange({ accountId: current.id, repoName, repo: repoName ? `${current.account}/${repoName.split("/").pop()}` : "" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current?.id]);

  const existing = useMemo(
    () => models.filter((m) => m.account_id === current?.id).map((m) => m.repo_id.split("/").slice(1).join("/")),
    [models, current?.id],
  );

  if (accounts.length === 0) {
    return (
      <div className={cn("flex items-center gap-2 rounded border border-amber-500/40 bg-amber-500/10 px-2 py-1.5 text-[11px] text-amber-700 dark:text-amber-300", className)}>
        No Hugging Face account saved yet.
        <button type="button" onClick={() => setOpen(true)} className="rounded bg-amber-500 px-2 py-0.5 text-[11px] font-bold text-amber-950">
          Add token
        </button>
      </div>
    );
  }
  const emit = (acc: string, name: string) => {
    const a = accounts.find((x) => x.id === acc) ?? accounts[0];
    const clean = name.trim().replace(/^.*\//, "");
    onChange({ accountId: a.id, repoName: name, repo: clean ? `${a.account}/${clean}` : "" });
  };
  const listId = `hf-repos-${current?.id}`;
  return (
    <div className={cn("flex flex-wrap items-center gap-1.5", className)}>
      <select
        value={current?.id}
        onChange={(e) => emit(e.target.value, repoName)}
        className="h-8 rounded border border-slate-300 bg-transparent px-2 text-[12px] font-bold dark:border-slate-600"
        title="Hugging Face account to push to"
      >
        {accounts.map((a) => (
          <option key={a.id} value={a.id}>
            {a.account}
          </option>
        ))}
      </select>
      <span className="text-slate-400">/</span>
      <input
        list={listId}
        value={repoName}
        onChange={(e) => emit(current?.id ?? "", e.target.value)}
        placeholder="repo name (new or existing)"
        className="h-8 min-w-[160px] flex-1 rounded border border-slate-300 bg-transparent px-2 text-[12px] dark:border-slate-600"
      />
      <datalist id={listId}>
        {existing.map((n) => (
          <option key={n} value={n} />
        ))}
      </datalist>
    </div>
  );
}
