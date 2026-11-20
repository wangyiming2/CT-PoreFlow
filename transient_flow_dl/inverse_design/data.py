from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


CONDITION_NAMES = (
    "porosity",
    "connected_porosity",
    "dead_pore_ratio",
    "log1p_permeability",
    "hydraulic_tortuosity",
)


def raw_condition(path: Path) -> np.ndarray:
    with np.load(path) as sample:
        metadata = json.loads(str(sample["metadata_json"].item()))
    porosity = float(metadata["porosity"])
    connected = float(metadata["connected_porosity"])
    return np.asarray(
        [
            porosity,
            connected,
            1.0 - connected / max(porosity, 1e-12),
            np.log1p(float(metadata["permeability_lattice"])),
            float(metadata["hydraulic_tortuosity"]),
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class ConditionStats:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, files: list[Path]) -> "ConditionStats":
        values = np.stack([raw_condition(path) for path in files])
        return cls(values.mean(axis=0), np.maximum(values.std(axis=0), 1e-6))

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "condition_names": CONDITION_NAMES,
                    "mean": self.mean.tolist(),
                    "std": self.std.tolist(),
                    "fit_on": "training split only",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "ConditionStats":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32))


class PorousInverseDesignDataset(Dataset):
    """Binary pore masks paired with standardized hydraulic design targets."""

    def __init__(
        self,
        steady_root: str | Path,
        split: str = "train",
        stats: ConditionStats | None = None,
        max_samples: int = 0,
    ) -> None:
        self.root = Path(steady_root) / "samples"
        self.files = sorted((self.root / split).rglob("*.npz"))
        if max_samples > 0:
            self.files = self.files[:max_samples]
        if not self.files:
            raise RuntimeError(f"No inverse-design samples found for split={split}")
        if stats is None:
            if split != "train":
                raise ValueError("Validation/test datasets require training-only condition statistics")
            stats = ConditionStats.fit(self.files)
        self.stats = stats

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.files[index]
        with np.load(path) as sample:
            pore = np.asarray(sample["pore"], dtype=np.float32)
        raw = raw_condition(path)
        normalized = self.stats.normalize(raw).astype(np.float32)
        return {
            "pore": torch.from_numpy(pore[None] * 2.0 - 1.0),
            "condition": torch.from_numpy(normalized),
            "condition_raw": torch.from_numpy(raw),
            "sample_id": path.relative_to(self.root).as_posix(),
        }
