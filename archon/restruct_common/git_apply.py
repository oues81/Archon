from pathlib import Path
from typing import Dict, Any, List
import os
import shutil
import subprocess
import shlex
import json
from datetime import datetime
import time
import logging

# Phase 2: approval-gated safe application

def run(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    cfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    requested = (cfg or {}).get("requested_action")
    token = (cfg or {}).get("approval_token")
    apply_all = bool((cfg or {}).get("apply_all") or False)
    only_apply_marked = bool((cfg or {}).get("only_apply_marked") or False)
    allow_dir_moves = bool((cfg or {}).get("allow_directory_moves") or False)

    # Quick path: support dry-run/validate preflight without token and without applying
    mode = str(requested).lower() if requested else None
    t0 = time.time()

    # Resolve plan path
    art_dir = Path("generated/restruct")
    plan_path = None
    if isinstance(state, dict):
        plan_path = state.get("move_plan_path")
    if not plan_path:
        cand = art_dir / "rename_move_plan.json"
        plan_path = str(cand) if cand.exists() else None
    if not plan_path or not Path(plan_path).exists():
        # Clarify behavior for dry-run/part1 vs apply
        if mode in {None, "", "dry-run", "validate", "preflight"}:
            logger.info("🛈 [apply.skip] dry-run or not approved: skip apply (no plan) | mode=%s", mode)
            return {
                "moves_applied": [],
                "conflicts": [],
                "skipped": [],
                "note": "Dry-run or Part 1 without approval: apply skipped because move plan was not found.",
            }
        logger.warning("[apply] no plan found | mode=%s", mode)
        return {"moves_applied": [], "conflicts": [], "skipped": [], "error": "move plan not found"}

    try:
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    except Exception as e:
        logger.exception("[apply] invalid plan JSON: %s", e)
        return {"moves_applied": [], "conflicts": [], "skipped": [], "error": f"invalid plan: {e}"}

    raw_moves: List[Dict[str, Any]] = plan.get("moves") or []
    moves: List[Dict[str, Any]] = []
    for m in raw_moves:
        if not isinstance(m, dict):
            continue
        src = (m.get("source") or "").strip()
        tgt = (m.get("target") or "").strip()
        if not src or not tgt:
            continue
        if only_apply_marked and not bool(m.get("apply")):
            continue
        moves.append({"source": src, "target": tgt})
    logger.info(
        "🚚 [apply.start] module=archon.archon.restruct_common.git_apply:apply_moves | mode=%s | moves_in_plan=%s | only_apply_marked=%s | allow_dir_moves=%s",
        mode, len(moves), only_apply_marked, allow_dir_moves
    )
    logger.info("[apply] start | mode=%s moves_in_plan=%s only_apply_marked=%s allow_dir_moves=%s", mode, len(raw_moves), only_apply_marked, allow_dir_moves)

    # Helper: compute conflicts for a file move
    def file_conflict(src_path: Path, tgt_path: Path) -> str | None:
        if not src_path.exists():
            return "source_missing"
        if tgt_path.exists():
            return "target_exists"
        return None

    # Helper: compute conflicts for a directory move (all-or-nothing)
    def dir_conflicts(src_dir: Path, tgt_dir: Path) -> List[Dict[str, str]]:
        problems: List[Dict[str, str]] = []
        if not src_dir.exists():
            problems.append({"source": str(src_dir), "target": str(tgt_dir), "reason": "source_missing"})
            return problems
        # If target dir exists, treat as conflict to avoid merges
        if tgt_dir.exists():
            problems.append({"source": str(src_dir), "target": str(tgt_dir), "reason": "target_exists"})
            return problems
        for root, _, files in os.walk(src_dir):
            for f in files:
                s = Path(root) / f
                rel = s.relative_to(src_dir)
                t = tgt_dir / rel
                if t.exists():
                    problems.append({"source": str(s), "target": str(t), "reason": "target_exists"})
        return problems

    # Preflight dry-run/validate mode
    if mode in {"dry-run", "validate", "preflight"}:
        conflicts: List[Dict[str, Any]] = []
        will_apply: List[Dict[str, Any]] = []
        for m in moves:
            src = Path(m["source"]).resolve()
            tgt = Path(m["target"]).resolve()
            if src.is_dir():
                if not allow_dir_moves:
                    conflicts.append({"source": str(src), "target": str(tgt), "reason": "directory_move_not_allowed"})
                    continue
                probs = dir_conflicts(src, tgt)
                if probs:
                    conflicts.extend(probs)
                else:
                    will_apply.append({"source": str(src), "target": str(tgt), "type": "dir"})
            else:
                r = file_conflict(src, tgt)
                if r:
                    conflicts.append({"source": str(src), "target": str(tgt), "reason": r})
                else:
                    will_apply.append({"source": str(src), "target": str(tgt), "type": "file"})
        dur_ms = int((time.time() - t0) * 1000)
        logger.info(
            "🧪 [apply.preflight] module=archon.archon.restruct_common.git_apply:apply_moves | will_apply=%s | conflicts=%s | dur_ms=%s",
            len(will_apply), len(conflicts), dur_ms
        )
        return {
            "preflight": True,
            "will_apply": will_apply,
            "conflicts": conflicts,
            "skipped": [],
            "note": "Dry-run only. No changes applied.",
        }

    # Gate: require explicit request + token for actual apply
    if mode not in {"part2", "apply"} or not token:
        return {
            "moves_applied": [],
            "conflicts": [],
            "skipped": [],
            "note": "Approval token and requested_action=part2/apply required. No changes applied.",
        }

    # Prepare backups root
    backups_root = Path(state.get("backups_root") or (Path("generated") / "backups"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = backups_root / f"apply_{stamp}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Detect git repo
    repo_root = None
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
        repo_root = Path(r.stdout.strip()) if r.stdout.strip() else None
    except Exception:
        repo_root = None
    use_git = bool(repo_root and repo_root.exists())

    applied: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for m in moves:
        src = Path(m["source"]).resolve()
        tgt = Path(m["target"]).resolve()
        try:
            if not src.exists():
                skipped.append({"source": str(src), "target": str(tgt), "reason": "source_missing"})
                continue
            if tgt.exists():
                conflicts.append({"source": str(src), "target": str(tgt), "reason": "target_exists"})
                continue
            # Directory move support (all-or-nothing). Requires allow_dir_moves
            if src.is_dir():
                if not allow_dir_moves:
                    skipped.append({"source": str(src), "target": str(tgt), "reason": "directory_move_not_allowed"})
                    continue
                probs = dir_conflicts(src, tgt)
                if probs:
                    conflicts.extend(probs)
                    continue

            # Backup source into session_dir with relative path preserved if inside repo
            rel = None
            try:
                if repo_root and str(src).startswith(str(repo_root)):
                    rel = src.relative_to(repo_root)
            except Exception:
                rel = src.name
            bkp_path = session_dir / (str(rel) if rel else src.name)
            bkp_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, bkp_path)

            # Ensure target dir exists
            tgt.parent.mkdir(parents=True, exist_ok=True)

            method = "git" if use_git else "os"
            if src.is_dir():
                method = "git" if use_git else "os"
                if use_git:
                    try:
                        subprocess.run(["git", "mv", str(src), str(tgt)], check=True)
                    except Exception:
                        shutil.move(str(src), str(tgt))
                        method = "os"
                else:
                    shutil.move(str(src), str(tgt))
            else:
                if use_git:
                    try:
                        subprocess.run(["git", "mv", str(src), str(tgt)], check=True)
                    except Exception:
                        os.rename(src, tgt)
                        method = "os"
                else:
                    os.rename(src, tgt)

            applied.append({"source": str(src), "target": str(tgt), "method": method, "backup": str(bkp_path)})
        except Exception as e:
            logger.exception("[apply] error while moving %s -> %s: %s", str(src), str(tgt), e)
            skipped.append({"source": str(src), "target": str(tgt), "reason": f"error:{e}"})

    # Write a small journal in the session dir
    try:
        journal = {
            "plan": plan_path,
            "applied": applied,
            "conflicts": conflicts,
            "skipped": skipped,
        }
        (session_dir / "apply_journal.json").write_text(json.dumps(journal, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("[apply] failed to write journal: %s", e)

    dur_ms_all = int((time.time() - t0) * 1000)
    logger.info(
        "✅ [apply.done] module=archon.archon.restruct_common.git_apply:apply_moves | applied=%s | conflicts=%s | backups=%s | dur_ms=%s",
        len(applied), len(conflicts), str(backups_root), dur_ms_all
    )
    logger.info("[apply] done | applied=%s conflicts=%s skipped=%s dur_ms=%s session=%s", len(applied), len(conflicts), len(skipped), dur_ms_all, str(session_dir))
    return {"moves_applied": applied, "conflicts": conflicts, "skipped": skipped, "backup_session": str(session_dir)}
