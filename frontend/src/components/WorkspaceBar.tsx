import type { WorkspaceMetrics } from "../types";

export type OwnFilter = "all" | "new" | "done";

export function defaultWorkspaceMetrics(): WorkspaceMetrics {
  return { total: 0, not_reviewed: 0, ready_to_tune: 0, tuning_used: 0, done: 0, loaded: 0 };
}
