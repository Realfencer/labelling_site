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
  /** Optional short tag and explanation (comparison runs). */
  label?: string | null;
  note?: string | null;
  instant?: boolean;
  kind?: "i2i" | "t2i";
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

/** A Done Platform image the eval set can be built from. */
export type GtCandidate = { item_key: string; block: number; quality: "pass" | "mid" | "fail" | null; in_eval: boolean; generator?: "openai" | "gemini" | null };
export type Generator = "openai" | "gemini";

/** Results page data for one run (GET /gen/insight). */
export type LabelBox = { cls: number; name: string; x1: number; y1: number; x2: number; y2: number };
export type SourceFault = {
  status: "missed" | "wrong_class" | "false_positive";
  cls: string;
  pred_class?: string | null;
  size_px?: [number, number];
  height_pct?: number | null;
  xyxy?: [number, number, number, number];
  reason?: string;
  distance?: string | null;
  what?: string | null;
  where?: string | null;
  orientation?: string | null;
  occlusion?: string | null;
  why?: string | null;
  is_drink?: boolean | null;
};
export type SourceInsight = {
  detector?: string | null;
  current_analysis?: boolean;
  faults: { counts: Record<string, number>; score: number | null; errors: SourceFault[] } | null;
  look?: string | null;
  scene?: { place?: string; time_of_day?: string; light?: string; weather?: string; indoor?: boolean; people?: number } | null;
};
export type ImagePlan = {
  variant?: number;
  of?: number;
  regime?: string;
  view?: string;
  targets?: { distance: string; height_pct: number | null; directive: string; why: string }[];
  negative?: string;
  motion_blur?: boolean;
  deep_scene?: boolean;
  camera_look?: boolean;
  prompt_chars?: number;
};
export type RunInsight = {
  run: string;
  sources: Record<string, SourceInsight | null>;
  plans: Record<string, ImagePlan>;
  labels: Record<string, LabelBox[]>;
  label_source: "detector" | "copied_from_source";
};

/** GET /analyze/needed — is the analysis up to date for the eval set + base model (generation runs it first if not). */
export type AnalyzeNeeded = {
  needed: boolean;
  reasons: string[];
  frames: number;
  new_frames: number;
  /** Likely Gemini cost (measured average per call × the frames that need it) and the worst case. */
  est_usd: number;
  est_max_usd: number;
  /** true once the cost ledger has real measurements */
  measured?: boolean;
  mode: GenMode;
};

/** Text → Image */
export type T2iJob = { status: string; message: string; progress: number; done: number; total: number; error: string | null; usd: number | null };
export type T2iCoverage = {
  images: number;
  drinks: number;
  drinks_per_image: number;
  class_mix: Record<string, number>;
  source_class_mix: Record<string, number>;
  hard_share: number;
  hard_cases_used: number;
  hard_cases_total: number;
  anchors: Record<string, number>;
  look_alikes: number;
  look_alike_bank: number;
  drink_count_range?: [number, number];
  places_mixed?: number;
  crowds_mixed?: number;
  drinks_changed?: number;
  distinct_places?: number;
};
export type T2iPlanMeta = { created_at: number; n: number; provider: GenMode; hard_share: number; frames: number; frames_missing_cards: string[]; coverage: T2iCoverage };
export type T2iDrink = { key: string; cls: string; hard: boolean; status: string | null; h_pct: number; distance?: string; object?: string; holder?: string; spot?: string; why?: string; changed?: string[] };
export type T2iRecipe = { id: string; anchor: string; variation: string | null; place?: string; place_from?: string | null; layout?: string; drinks: T2iDrink[]; look_alikes: { object: string; resembles: string }[]; prompt: string };
export type T2iPlan = T2iPlanMeta & { seed: number; recipes: T2iRecipe[] };
export type T2iStatus = {
  frames: number;
  with_cards: number;
  missing_cards: number;
  base: AnalyzeNeeded;
  est_usd: number;
  est_max_usd: number;
  cards_est: { usd: number; max_usd: number; measured: boolean } | null;
  job: T2iJob;
  plan: T2iPlanMeta | null;
};
