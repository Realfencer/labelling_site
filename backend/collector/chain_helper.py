"""Run with the turbovision venv (has bittensor + scorevision):

    python chain_helper.py <netuid> <element_substring> <min_block>

Prints one JSON line: {"ok":true,"head":N,"events":[{block,hotkey,model,revision,element_id,...}]}.
Read-only chain query.
"""

import asyncio
import json
import sys
from pathlib import Path

TURBOVISION_ROOT = Path("/root/workspace/turbovision")


async def main(netuid: int, element: str, min_block: int) -> dict:
    sys.path.insert(0, str(TURBOVISION_ROOT))
    from scorevision.utils.bittensor_commitments import get_all_revealed_commitments
    from scorevision.utils.bittensor_helpers import get_subtensor, reset_subtensor

    st = await get_subtensor()
    try:
        head = int(await st.get_current_block())
        commits = await get_all_revealed_commitments(st, netuid)
    finally:
        reset_subtensor()
    events = []
    for hotkey, arr in (commits or {}).items():
        for item in arr or []:
            if not item or len(item) < 2:
                continue
            block, payload = int(item[0]), item[1]
            if block < min_block:
                continue
            obj = payload
            if isinstance(payload, str):
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
            if not isinstance(obj, dict):
                continue
            if element not in str(obj.get("element_id") or "").lower():
                continue
            model = str(obj.get("model") or "").strip()
            revision = str(obj.get("revision") or "").strip()
            if "/" not in model or not revision:
                continue
            events.append(
                {
                    "block": block,
                    "hotkey": str(hotkey),
                    "model": model,
                    "revision": revision,
                    "element_id": obj.get("element_id"),
                    "slug": obj.get("slug"),
                    "chute_id": obj.get("chute_id"),
                }
            )
    return {"ok": True, "head": head, "events": events}


if __name__ == "__main__":
    try:
        out = asyncio.run(main(int(sys.argv[1]), sys.argv[2].lower(), int(sys.argv[3])))
    except Exception as exc:  # noqa: BLE001
        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(out))
