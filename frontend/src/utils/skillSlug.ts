/** Match backend core.skill_slug() for URL/topic guards. */
export function topicSlugFromSkill(elementPath: string | undefined | null): string | null {
  if (!elementPath) return null;
  let name = elementPath.trim();
  for (const prefix of ["manak0_Detect-", "manak0_detect-", "manak0_"]) {
    if (name.startsWith(prefix)) {
      name = name.slice(prefix.length);
      break;
    }
  }
  name = name.replace(/_/g, "-").replace(/--+/g, "-").replace(/^-+|-+$/g, "");
  return name.toLowerCase() || null;
}

export function stateMatchesTopic(state: { active_skill?: { element_path?: string } | null }, topicSlug: string): boolean {
  const slug = topicSlugFromSkill(state.active_skill?.element_path);
  return slug === topicSlug;
}

/** Match platform skill:block / block_N keys — never Own stems via shared "v01" tails. */
export function itemKeysMatch(a?: string | null, b?: string | null): boolean {
  if (!a || !b) return false;
  if (a === b) return true;
  const normA = a.replace(/[\/\-_]/g, "").toLowerCase();
  const normB = b.replace(/[\/\-_]/g, "").toLowerCase();
  if (normA === normB) return true;
  // Platform only: trailing pure digits (skill:8773200 / block_8773200).
  // Own "8773200_v01" must not match "8769900_v01" via a shared "v01" suffix.
  const platformBlockId = (key: string): string | null => {
    let tail = key.includes(":") ? key.slice(key.lastIndexOf(":") + 1) : key;
    if (tail.startsWith("block_")) tail = tail.slice(6);
    if (/^\d{3,}$/.test(tail)) return tail;
    return null;
  };
  const idA = platformBlockId(a);
  const idB = platformBlockId(b);
  if (idA && idB && idA === idB) return true;
  return false;
}
