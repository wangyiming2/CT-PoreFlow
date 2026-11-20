from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "pid": os.getpid(),
    }
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        report.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "gpu_memory_bytes": prop.total_memory,
            }
        )
    try:
        report["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        report["git_commit"] = None
    return report
