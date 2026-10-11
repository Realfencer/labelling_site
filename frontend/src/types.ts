export interface DetectionBox {
  cls: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  conf?: number | null;
  /** The network's own confidence before the miner's gate / cluster boost (display only). */
  raw_conf?: number | null;
}

export interface ChallengeItem {
  source_type: string;
  record: Record<string, unknown>;
  reviewed: boolean;
  tuned: boolean;
  loaded: boolean;
  error: string;
  block: number;
  display_name: string;
  item_key: string;
  /** File mtime (ns) — busts image URL caches when Own pixels change. */
  media_rev?: number;
  /** Own: YOLO label file exists (SAM3 draft or Done). */
  has_label?: boolean;
  image_width?: number;
  image_height?: number;
  top_boxes?: Record<string, unknown>[];
  cached_left_boxes?: Record<string, unknown>[];
  checks_passed?: number | null;
  checks_total?: number | null;
  checks?: { r1?: boolean | null; r2?: boolean | null; r3?: boolean | null } | null;
  checks_kind?: string;
  /** Which image generator made it (beverage): Platform from the pixels, Own from the OpenAI-days rule. */
  generator?: "openai" | "gemini" | null;
}

export interface Skill {
  element_id: string;
  element_path: string;
  objects: string[];
  first_block: number;
  latest_block: number;
  evaluation_count: number;
}

export interface TopVsLeftScore {
  composite: number;
  map50: number;
  false_positive: number;
  ffpi: number;
  tp: number;
  fp: number;
  fn: number;
  n_gt: number;
  n_pred: number;
  platform_score: number | null;
  platform_map50: number | null;
  platform_fp: number | null;
  delta_vs_platform: number | null;
}

export interface MissingClass {
  class_id: number;
  name: string;
  reason: string;
  confidence: string;
  source: string;
  missing_count: number;
}

export interface WorkspaceMetrics {
  total: number;
  not_reviewed: number;
  ready_to_tune: number;
  tuning_used: number;
  done: number;
  loaded?: number;
}

export interface AppState {
  status: string;
  review_mode: string;
  workspace?: "platform" | "own";
  /** Own calendar-day cluster (YYYY-MM-DD); null/undefined = day-list mode. */
  own_day?: string | null;
  /** Own day opened for one generator's images only. */
  own_field?: "openai" | "gemini" | null;
  project_id: string;
  project_root: string;
  session_id?: string | null;
  miner_folder: string | null;
  hf_model_id: string | null;
  imported_project: string | null;
  local_dataset: string;
  own_pool_dir?: string;
  tune_checkpoint: string | null;
  model_class_names: string[];
  miner_class_names: string[];
  active_skill: Skill | null;
  skills: Skill[];
  metrics: WorkspaceMetrics;
  platform_metrics?: WorkspaceMetrics;
  own_metrics?: WorkspaceMetrics;
  /** Day-scoped metrics when own_day is set. */
  own_day_metrics?: WorkspaceMetrics;
  challenges: ChallengeItem[];
  current_index: number;
  collect: { running: boolean; message: string };
  collect_seq: number;
  boxes_seq: number;
  label_seq?: number;
}

export interface ChallengeDetail {
  index: number;
  challenge: ChallengeItem;
  image_url: string;
  width: number;
  height: number;
  boxes: DetectionBox[];
  reference_boxes: Record<string, unknown>[];
  reference_variants?: {
    source: string;
    boxes: Record<string, unknown>[];
    hotkey?: string;
    total_weighted?: number | null;
    map50?: number | null;
    false_positive?: number | null;
    composite_score?: number | null;
    is_winner?: boolean;
  }[];
  prediction_boxes?: DetectionBox[];
  class_names: string[];
  reference_class_names: string[];
  score: TopVsLeftScore | null;
  missing_classes: MissingClass[];
  record: Record<string, unknown>;
}
