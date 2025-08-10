import os
from pathlib import Path
from typing import Tuple

DOCS_ART_ROOT = Path("generated/docs_reorg")
DOCS_BKP_ROOT = Path("generated/backups/docs_reorg")
CNT_ART_ROOT = Path("generated/restruct")
CNT_BKP_ROOT = Path("generated/backups/restruct")


def ensure_docs_dirs() -> Tuple[Path, Path]:
    DOCS_ART_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS_BKP_ROOT.mkdir(parents=True, exist_ok=True)
    return DOCS_ART_ROOT, DOCS_BKP_ROOT


def ensure_content_dirs() -> Tuple[Path, Path]:
    CNT_ART_ROOT.mkdir(parents=True, exist_ok=True)
    CNT_BKP_ROOT.mkdir(parents=True, exist_ok=True)
    return CNT_ART_ROOT, CNT_BKP_ROOT
