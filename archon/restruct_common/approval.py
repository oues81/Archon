from typing import Dict, Any


def run(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    requested_action = (state or {}).get("requested_action") or (config.get("configurable", {}) or {}).get("requested_action")
    approval_token = (state or {}).get("approval_token") or (config.get("configurable", {}) or {}).get("approval_token")

    if requested_action == "part1":
        return {"needs_approval": False, "approved": False}
    if requested_action == "part2" and approval_token:
        return {"needs_approval": False, "approved": True}
    return {"needs_approval": True, "approved": False}
