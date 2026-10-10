import type { ChallengeItem } from "./types";

export function checksTotal(c: ChallengeItem): number {
  if (c.checks_total != null && c.checks_total > 0) return c.checks_total;
  const slots = [c.checks?.r1, c.checks?.r2, c.checks?.r3];
  const active = slots.filter((v) => v !== null && v !== undefined);
  if (active.length > 0) return active.length;
  return 3;
}

export function checksPassComplete(c: ChallengeItem): boolean {
  const total = checksTotal(c);
  if (total <= 0) return false;
  return (c.checks_passed ?? 0) >= total;
}

export function checksFailComplete(c: ChallengeItem): boolean {
  return c.checks_passed === 0;
}
