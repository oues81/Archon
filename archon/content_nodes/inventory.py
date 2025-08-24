from pathlib import Path
import json
import os
import time
from typing import Dict, Any, List
import logging

ART_DIR = Path("generated/restruct")
logger = logging.getLogger(__name__)


def run(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    # Resolve artifacts root: prefer state.artifacts_root, then config.output_root, else default
    try:
        cfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    except Exception:
        cfg = {}
    root = None
    if isinstance(state, dict):
        root = state.get("artifacts_root") or None
    if not root and isinstance(cfg, dict):
        root = cfg.get("output_root")
    art_root = Path(root) if isinstance(root, str) and str(root).strip() else ART_DIR
    art_root.mkdir(parents=True, exist_ok=True)
    out = art_root / "global_inventory.json"

    cfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    workspace_root = None
    if isinstance(cfg, dict):
        wr = cfg.get("workspace_root")
        if isinstance(wr, str) and wr.strip():
            workspace_root = Path(wr)
    targets: List[str] = []
    if isinstance(cfg, dict) and isinstance(cfg.get("targets"), list):
        targets = [str(x) for x in cfg.get("targets") if isinstance(x, (str, os.PathLike))]

    # Fallback defaults if not provided
    if not targets:
        if workspace_root and workspace_root.exists():
            targets = [str(workspace_root)]
        else:
            defaults = ["docs", "scripts", "services", "data"]
            targets = [d for d in defaults if Path(d).exists()]
            if not targets:
                targets = ["."]

    # Resolve relative targets against workspace_root if provided
    if workspace_root and workspace_root.exists():
        abs_targets: List[str] = []
        for t in targets:
            pt = Path(t)
            abs_targets.append(str(pt if pt.is_absolute() else (workspace_root / pt)))
        targets = abs_targets

    include_ext = set([e.lower().lstrip('.') for e in (cfg.get("include_ext") or []) if isinstance(e, str)])
    # If include_ext empty, use broader defaults oriented for docs/content
    if not include_ext:
        include_ext = {"md", "mdx", "markdown", "yml", "yaml", "toml", "json", "txt", "py", "sh"}
    if not include_ext:
        include_ext = {"md", "mdx", "markdown", "yml", "yaml", "toml", "json"}
    exclude_ext = set([e.lower().lstrip('.') for e in (cfg.get("exclude_ext") or []) if isinstance(e, str)])
    max_files = int(cfg.get("max_files") or 5000)
    follow_symlinks = bool(cfg.get("follow_symlinks") or False)
    # Directories to prune during traversal (configurable + sensible defaults)
    default_excludes = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build", "generated", ".mypy_cache", ".pytest_cache"}
    cfg_exclude_dirs = set([str(d).strip().strip("/") for d in (cfg.get("exclude_dirs") or []) if isinstance(d, str)])
    exclude_dirs = default_excludes.union(cfg_exclude_dirs)

    logger.info(
        "[inventory] start | targets=%s include_ext=%s exclude_ext=%s max_files=%s follow_symlinks=%s exclude_dirs=%s",
        targets, sorted(list(include_ext))[:10], sorted(list(exclude_ext))[:10], max_files, follow_symlinks, sorted(list(exclude_dirs))[:12]
    )

    items: List[Dict[str, Any]] = []
    seen = 0
    start_ts = time.time()
    for root in targets:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root_path, followlinks=follow_symlinks):
            # Skip generated/ to avoid feedback loops
            parts = dirpath.split(os.sep)
            if "generated" in parts:
                continue
            # Prune excluded directories in-place for efficiency
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
            for fname in filenames:
                if seen >= max_files:
                    break
                p = Path(dirpath) / fname
                try:
                    ext = p.suffix.lower().lstrip('.')
                    if include_ext and ext not in include_ext:
                        continue
                    if exclude_ext and ext in exclude_ext:
                        continue
                    st = p.stat()
                    size = int(st.st_size)
                    mtime = int(st.st_mtime)
                    # Peek first line for quick hints (safe, bounded)
                    first_line = None
                    if ext in {"md", "markdown", "py", "sh", "yaml", "yml", "json", "txt"}:
                        try:
                            with p.open('r', encoding='utf-8', errors='ignore') as fh:
                                first_line = fh.readline(500).strip() if fh else None
                        except Exception:
                            first_line = None
                    items.append({
                        "path": str(p),
                        "ext": ext,
                        "size": size,
                        "mtime": mtime,
                        "first_line": first_line,
                    })
                    seen += 1
                except Exception:
                    # Skip unreadable files silently
                    continue

    data = {
        "note": "Content inventory (Part 1, read-only)",
        "targets": targets,
        "count": len(items),
        "generated_at": int(time.time()),
        "items": items,
        "limits": {"max_files": max_files}
    }
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    dur_ms = int((time.time() - start_ts) * 1000)
    logger.info("[inventory] done | count=%s dur_ms=%s out=%s", len(items), dur_ms, str(out))
    return {"inventory_path": str(out), "inventory_count": len(items), "artifacts_root": str(art_root)}
