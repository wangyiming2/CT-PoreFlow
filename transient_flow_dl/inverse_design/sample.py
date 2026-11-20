from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import CONDITION_NAMES, ConditionStats
from .models import ConditionalDiffusionUNet, ConditionalPoreGenerator
from .physics import cosine_beta_schedule


def spans_flow(pore: np.ndarray) -> bool:
    height, width = pore.shape
    queue = deque((0, column) for column in range(width) if pore[0, column])
    visited = set(queue)
    while queue:
        row, column = queue.popleft()
        if row == height - 1:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (row + dr, column + dc)
            if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width and pore[neighbor] and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


@torch.inference_mode()
def diffusion_sample(model, condition, steps: int, guidance: float, device: torch.device) -> torch.Tensor:
    beta = cosine_beta_schedule(steps, device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    sample = torch.randn(condition.shape[0], 1, 100, 100, device=device)
    for index in reversed(range(steps)):
        time = torch.full((condition.shape[0],), index / max(steps - 1, 1), device=device)
        conditional = model(sample, time, condition)
        unconditional = model(sample, time, torch.zeros_like(condition))
        predicted_noise = unconditional + guidance * (conditional - unconditional)
        sample = (sample - beta[index] / (1.0 - alpha_bar[index]).sqrt() * predicted_noise) / alpha[index].sqrt()
        if index > 0:
            sample += beta[index].sqrt() * torch.randn_like(sample)
    return sample.clamp(-1.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--targets", required=True, help="comma-separated raw values in CONDITION_NAMES order")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--guidance", type=float, default=2.0)
    args = parser.parse_args()
    checkpoint = torch.load(args.run_dir / "last.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    stats = ConditionStats.load(args.run_dir / "condition_stats.json")
    raw = np.asarray([float(value) for value in args.targets.split(",")], dtype=np.float32)
    if raw.size != len(CONDITION_NAMES):
        raise ValueError(f"Expected {len(CONDITION_NAMES)} targets: {CONDITION_NAMES}")
    normalized = stats.normalize(raw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    condition = torch.from_numpy(normalized).to(device)[None].repeat(args.count, 1)
    if config["method"] == "cgan":
        model = ConditionalPoreGenerator(latent_dim=int(config["latent_dim"]), base=int(config["base_channels"])).to(device)
        model.load_state_dict(checkpoint["generator"])
        generated = model(torch.randn(args.count, int(config["latent_dim"]), device=device), condition)
    else:
        model = ConditionalDiffusionUNet(base=int(config["base_channels"])).to(device)
        model.load_state_dict(checkpoint["model"])
        generated = diffusion_sample(model, condition, int(config["diffusion_steps"]), args.guidance, device)
    masks = (generated[:, 0].cpu().numpy() > 0).astype(np.uint8)
    qc = [
        {"index": index, "porosity": float(mask.mean()), "spans_flow": spans_flow(mask)}
        for index, mask in enumerate(masks)
    ]
    output = args.run_dir / "samples"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "generated_candidates.npz", pore_masks=masks, target_condition_raw=raw)
    (output / "quality_control.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    rows = int(np.ceil(np.sqrt(args.count)))
    fig, axes = plt.subplots(rows, rows, figsize=(8, 8), constrained_layout=True)
    for index, axis in enumerate(np.ravel(axes)):
        axis.axis("off")
        if index < len(masks):
            axis.imshow(masks[index], cmap="gray_r", vmin=0, vmax=1)
            axis.set_title(f"{index}: $\\phi$={masks[index].mean():.3f}, span={spans_flow(masks[index])}", fontsize=7)
    fig.savefig(output / "generated_candidates.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
