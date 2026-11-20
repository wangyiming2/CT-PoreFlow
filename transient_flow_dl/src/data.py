from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class TransientFlowDataset(Dataset):
    """Join steady geometry and transient fields by their relative sample path."""

    def __init__(
        self,
        steady_root: str | Path,
        transient_root: str | Path,
        split: str,
        velocity_scale: float,
        rho_in: float,
        rho_out: float,
        max_iteration: int,
        mode: str = "random_time",
        time_indices: Sequence[int] | None = None,
        max_samples: int = 0,
        seed: int = 0,
        topology_root: str | Path | None = None,
        time_transform: str = "log",
    ) -> None:
        self.steady_root = Path(steady_root) / "samples"
        self.transient_root = Path(transient_root)
        self.split = "validation" if split in {"val", "validation"} else split
        self.velocity_scale = float(velocity_scale)
        self.rho_in = float(rho_in)
        self.rho_out = float(rho_out)
        self.delta_rho = self.rho_in - self.rho_out
        self.max_iteration = int(max_iteration)
        self.mode = mode
        self.seed = int(seed)
        self.topology_root = Path(topology_root) if topology_root else None
        self.time_transform = time_transform
        if self.time_transform not in {"log", "linear"}:
            raise ValueError(f"Unsupported time transform: {self.time_transform}")

        steady_files = sorted((self.steady_root / self.split).rglob("*.npz"))
        if max_samples > 0:
            steady_files = steady_files[:max_samples]
        self.pairs: list[tuple[Path, Path, str]] = []
        for steady in steady_files:
            relative = steady.relative_to(self.steady_root)
            transient = self.transient_root / relative
            if not transient.exists():
                raise FileNotFoundError(f"Missing transient pair for {relative}")
            self.pairs.append((steady, transient, relative.as_posix()))
        if not self.pairs:
            raise RuntimeError(f"No samples found for split={self.split}")

        with np.load(self.pairs[0][1]) as sample:
            self.iterations = np.asarray(sample["transient_iterations"], dtype=np.int64)
        self.time_indices = list(time_indices) if time_indices is not None else list(range(len(self.iterations)))
        if mode not in {"random_time", "fixed_times"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self._cached_pair_index: int | None = None
        self._cached_pair: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        if self.mode == "fixed_times":
            return len(self.pairs) * len(self.time_indices)
        return len(self.pairs)

    def _index(self, index: int) -> tuple[int, int]:
        if self.mode == "fixed_times":
            pair_index = index // len(self.time_indices)
            time_index = self.time_indices[index % len(self.time_indices)]
            return pair_index, time_index
        pair_index = index
        worker = torch.utils.data.get_worker_info()
        worker_seed = 0 if worker is None else worker.seed
        rng = np.random.default_rng(self.seed + worker_seed + index + np.random.randint(0, 2**20))
        selected = int(rng.integers(0, len(self.time_indices)))
        return pair_index, int(self.time_indices[selected])

    def _load_pair(self, pair_index: int) -> dict[str, np.ndarray]:
        """Load one geometry and its complete trajectory, caching the last pair.

        Fixed-time evaluation visits all time indices of a geometry consecutively.
        The one-pair cache therefore reduces thirty repeated NPZ decompressions to
        one without retaining the complete dataset in memory.
        """
        if self._cached_pair_index == pair_index and self._cached_pair is not None:
            return self._cached_pair
        steady_path, transient_path, sample_id = self.pairs[pair_index]
        with np.load(steady_path) as steady:
            payload: dict[str, np.ndarray] = {
                "pore": np.asarray(steady["pore"], dtype=np.float32),
                "connected": np.asarray(steady["through_connected_pore"], dtype=np.float32),
                "edt": np.asarray(steady["edt_lattice"], dtype=np.float32)
                / float(steady["pore"].shape[0]),
            }
        with np.load(transient_path) as transient:
            payload.update(
                {
                    "ux": np.asarray(transient["velocity_x_raw_lattice"], dtype=np.float32),
                    "uy": np.asarray(transient["velocity_y_raw_lattice"], dtype=np.float32),
                    "rho": np.asarray(transient["density_raw_lattice"], dtype=np.float32),
                    "iterations": np.asarray(transient["transient_iterations"], dtype=np.int64),
                    "source_iterations": np.asarray(
                        transient["transient_source_iterations"], dtype=np.int64
                    ),
                    "steady_fill": np.asarray(
                        transient["transient_steady_fill_mask"], dtype=np.float32
                    ),
                }
            )
        if self.topology_root is not None:
            topology_path = self.topology_root / Path(sample_id)
            if not topology_path.exists():
                raise FileNotFoundError(f"Missing topology features for {sample_id}")
            with np.load(topology_path) as topology:
                payload["distance_from_inlet"] = np.asarray(
                    topology["distance_from_inlet"], dtype=np.float32
                )
                payload["distance_from_outlet"] = np.asarray(
                    topology["distance_from_outlet"], dtype=np.float32
                )
        self._cached_pair_index = pair_index
        self._cached_pair = payload
        return payload

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair_index, time_index = self._index(index)
        _, _, sample_id = self.pairs[pair_index]
        pair = self._load_pair(pair_index)
        pore = pair["pore"]
        connected = pair["connected"]
        edt = pair["edt"]
        ux = pair["ux"][time_index]
        uy = pair["uy"][time_index]
        rho = pair["rho"][time_index]
        iteration = int(pair["iterations"][time_index])
        source_iteration = int(pair["source_iterations"][time_index])
        steady_fill = int(pair["steady_fill"][time_index])

        pressure = np.zeros_like(rho, dtype=np.float32)
        np.divide(rho - self.rho_out, self.delta_rho, out=pressure, where=pore > 0.5)
        ux = (ux / self.velocity_scale) * pore
        uy = (uy / self.velocity_scale) * pore
        pressure *= pore
        channels = [pore, connected, edt]
        if self.topology_root is not None:
            channels.extend([pair["distance_from_inlet"], pair["distance_from_outlet"]])
        geometry = np.stack(channels, axis=0)
        target = np.stack([ux, uy, pressure], axis=0)
        if self.time_transform == "linear":
            time_value = iteration / float(self.max_iteration)
        else:
            time_value = math.log1p(iteration) / math.log1p(self.max_iteration)
        return {
            "geometry": torch.from_numpy(geometry),
            "time": torch.tensor(time_value, dtype=torch.float32),
            "target": torch.from_numpy(target),
            "pore": torch.from_numpy(pore[None]),
            "iteration": torch.tensor(iteration, dtype=torch.int64),
            "source_iteration": torch.tensor(source_iteration, dtype=torch.int64),
            "steady_fill": torch.tensor(steady_fill, dtype=torch.float32),
            "sample_id": sample_id,
        }
