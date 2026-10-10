import { useEffect, useState } from "react";
import { studio } from "./api";
import { Confirm, errText } from "./ui";

export default function ClearConfirm({ onDone, onCancel }: { onDone: () => void; onCancel: () => void }) {
  const [preview, setPreview] = useState<Awaited<ReturnType<typeof studio.clearPreview>> | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    studio.clearPreview().then(setPreview).catch((e) => setErr(errText(e)));
  }, []);
  const busy = preview && (preview.gen_running || preview.analyze_running);
  return (
    <Confirm title="Clear the Studio?" danger typeToConfirm={busy ? "—" : "CLEAR"} confirmLabel="Delete everything" onCancel={onCancel} onConfirm={async () => {
      await studio.clear();
      onDone();
    }}>
      {err && <p className="text-rose-600">{err}</p>}
      {!preview ? (
        <p>Loading…</p>
      ) : busy ? (
        <p className="text-amber-600">A job is running — wait for it to finish first.</p>
      ) : (
        <ul className="list-disc pl-5 text-[12px]">
          <li>
            Eval dataset: {preview.beverage_gt_images} images + {preview.beverage_gt_labels} labels
          </li>
          <li>
            {preview.history_runs} generated runs{preview.history_run_names.length ? ` (${preview.history_run_names.slice(0, 4).join(", ")}…)` : ""}
          </li>
          <li>{preview.analysis_files} analysis files</li>
        </ul>
      )}
      <p>This cannot be undone. Type CLEAR to confirm.</p>
    </Confirm>
  );
}

