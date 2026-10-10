/** Shapes of the Studio API (studio/app/main.py), reached through the labeler at /api/studio/*. */

export type GenMode = "openai" | "gemini";

export type ImageModel = { id: string; label: string; hint: string; provider: GenMode };

export type DetectorInfo = {
  weights: string;
  conf: number;
  imgsz: number;
  exists: boolean;
  enabled: boolean;
  source: "uploaded" | "default";
  sv: { composite: number; map50: number; false_positive: number; fp_per_image: number; recall: number } | null;
  gt_images: number | null;
  uploaded_at: string | null;
};

export type SwitchName = "detector_openai" | "detector_gemini" | "person_detector" | "post_openai" | "post_gemini";

export type Health = {
  status: string;
  element: string;
  dataset: string;
  out_dir: string | null;
  images: number;
  default_model: string;
  has_hf_token: boolean;
  hf_repo: string | null;
  hf_private: boolean;
  image_models: ImageModel[];
  gen_mode: GenMode;
  detectors: Partial<Record<GenMode, DetectorInfo>>;
  checkpoint_switches: Record<SwitchName, boolean>;
  base_model?: { name: string; custom: boolean; applied_at: string | null; exists: boolean; key: string | null };
  analysis?: { at: number | null; detector_key: string | null; detector: string | null; errors_at?: number; package_at: number | null };
  has_openai_key: boolean;
  has_gemini_key: boolean;
  has_openrouter_key: boolean;
  openai_defaults: { quality: string; input_fidelity: string; cctv_degrade: boolean };
};

export type JobStatus = "idle" | "running" | "done" | "error";

export type AnalyzeJob = {
  status: JobStatus;
  message: string;
  progress: number;
  total: number;
  done: number;
  dataset?: string;
  out_dir?: string;
  error: string | null;
  summary_preview?: Record<string, unknown>;
};

export type GenItem = {
  filename: string;
  source_image?: string;
  source_index?: number;
  source_path?: string;
  variant?: number;
  url?: string;
  source_url?: string;
  layout_similarity?: number | null;
  near_copy?: boolean;
  camera_drift?: string[];
};

export type GenJob = {
  status: JobStatus;
  mode: "test" | "final" | "autolabel" | "";
  message: string;
  progress: number;
  done: number;
  total: number;
  run_dir: string;
  error: string | null;
  images: GenItem[];
  hf_upload: null | { ok?: boolean; url?: string; repo_id?: string; path_in_repo?: string; error?: string };
};

export type CkptJob = {
  status: JobStatus;
  mode: string;
  name: string;
  message: string;
  done: number;
  total: number;
  progress: number;
  error: string | null;
  result: null | Record<string, unknown>;
};

export type SendJob = {
  status: JobStatus;
  run: string;
  message: string;
  done: number;
  total: number;
  progress: number;
  error: string | null;
  result: null | { ok: boolean; run: string; host: string; dest: string; images: number; filenames: string[] };
};

export type Gallery = {
  key: string;
  title: string;
  short: string;
  why: string;
  color: string;
  group: "scene" | "class_hard" | "overlap";
  count: number;
  pct: number;
  mean_object_count?: number;
  mean_brightness?: number;
  mean_bbox_area?: number;
};

export type Summary = {
  count: number;
  mean_object_count?: number;
  mean_brightness?: number;
  median_bbox_area?: number;
  class_instance_totals?: Record<string, number>;
  dataset_path?: string;
};

export type MiningReport = {
  recommendations?: { priority: "high" | "medium" | "low"; title: string; detail: string }[];
  label_health?: Record<string, unknown>;
};

export type RecordRow = {
  filename: string;
  object_count: number;
  brightness: number;
  regimes: string[];
  primary_regime?: string;
};

export type DeCaption = {
  object?: string | null;
  placement_detail?: string | null;
  likely_cause?: string | null;
  confusers?: string | null;
  generation_directive?: string | null;
  is_actually_drink?: boolean | null;
  distance?: "near" | "mid" | "far" | "very_far" | null;
  depth_detail?: string | null;
  relative_size?: string | null;
};

export type DeError = {
  id: number;
  status: "missed" | "wrong_class" | "false_positive";
  gt_class?: string | null;
  pred_class?: string | null;
  pred_conf?: number | null;
  size_px?: [number, number];
  frame_pct?: { h: number; y: number; x: number };
  reason?: string;
  caption?: DeCaption;
};

export type DeSource = {
  filename: string;
  counts: { gt: number; hit: number; missed: number; wrong_class: number; false_positive: number; false_alarm_objects: number };
  caption_error?: string | null;
  errors: DeError[];
};

export type ModelErrors =
  | { ready: false }
  | {
      ready: true;
      version: string;
      mode: string;
      detector: string;
      conf: number;
      caption_model?: string;
      summary: {
        gt: number;
        hit: number;
        missed: number;
        wrong_class: number;
        false_positive: number;
        false_alarm_objects: number;
        recall: number;
        sources_with_errors: number;
        by_class: Record<string, { gt: number; missed: number; wrong_class: number; error_rate: number }>;
        confusions: Record<string, number>;
        false_alarm_objects_seen?: Record<string, number>;
        hard_negatives?: number;
      };
      sources: DeSource[];
    };

export type DistRow = {
  key: string;
  title: string;
  group: string;
  color: string;
  eval_count: number;
  eval_pct: number;
  why?: string;
};

export type Distribution = { eval_total: number; galleries: DistRow[]; overlaps: DistRow[] };

export type PackageSource = {
  source_index: number;
  source_image: string;
  source_path: string;
  target_outputs: number;
  regimes: string;
  detector_errors: number;
  resolved: boolean;
};

export type Package = {
  ready: boolean;
  sources: PackageSource[];
  per_source?: number;
  n_sources?: number;
  total_targets?: number;
  estimated_cost_usd?: number;
  model_default?: string;
};

export type Run = {
  name: string;
  count: number;
  mode: "test" | "final" | null;
  batch: boolean;
  model: string | null;
  provider: GenMode | null;
  created_at: string | null;
  pending: boolean;
  chunks: number;
  autolabelled: boolean;
  camera_drift: number | null;
  near_copies: number;
};

export type ClearPreview = {
  beverage_gt_images: number;
  beverage_gt_labels: number;
  history_runs: number;
  history_run_names: string[];
  analysis_files: number;
  gen_running: boolean;
  analyze_running: boolean;
};

export type StudioStatus = {
  up: boolean;
  service: string;
  health: Partial<Health> & { error?: string };
  keys: Record<"openai" | "gemini" | "openrouter" | "ultralytics", { configured: boolean; masked: string }>;
  hf_accounts: string[];
  hf_repo: string;
  allowed_users: string[];
  gt_images: number;
  gt_updated_at: number | null;
  gt_block_min: number | null;
  gt_block_max: number | null;
  me: string;
  can_spend: boolean;
};
