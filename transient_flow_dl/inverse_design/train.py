from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ConditionStats, PorousInverseDesignDataset
from .models import ConditionalDiffusionUNet, ConditionalPatchDiscriminator, ConditionalPoreGenerator
from .physics import connectivity_loss, cosine_beta_schedule, porosity_loss


def append_history(output: Path, payload: dict) -> None:
    with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def train_gan(config: dict, loader: DataLoader, device: torch.device, output: Path) -> None:
    generator = ConditionalPoreGenerator(
        latent_dim=int(config["latent_dim"]), base=int(config["base_channels"])
    ).to(device)
    discriminator = ConditionalPatchDiscriminator(base=int(config["base_channels"])).to(device)
    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=float(config["learning_rate"]), betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=float(config["learning_rate"]), betas=(0.5, 0.999))
    for epoch in range(1, int(config["epochs"]) + 1):
        running = {"generator": 0.0, "discriminator": 0.0}
        for batch in loader:
            real = batch["pore"].to(device)
            condition = batch["condition"].to(device)
            raw = batch["condition_raw"].to(device)
            noise = torch.randn(real.shape[0], int(config["latent_dim"]), device=device)
            fake = generator(noise, condition)
            discriminator_optimizer.zero_grad(set_to_none=True)
            real_score = discriminator(real, condition)
            fake_score = discriminator(fake.detach(), condition)
            discriminator_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score))
                + F.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score))
            )
            discriminator_loss.backward()
            discriminator_optimizer.step()
            generator_optimizer.zero_grad(set_to_none=True)
            adversarial = F.binary_cross_entropy_with_logits(
                discriminator(fake, condition), torch.ones_like(fake_score)
            )
            generator_loss = (
                adversarial
                + float(config["porosity_weight"]) * porosity_loss(fake, raw)
                + float(config["connectivity_weight"]) * connectivity_loss(fake)
            )
            generator_loss.backward()
            generator_optimizer.step()
            running["generator"] += float(generator_loss.detach())
            running["discriminator"] += float(discriminator_loss.detach())
        payload = {"epoch": epoch, **running}
        append_history(output, payload)
        print(json.dumps(payload), flush=True)
        torch.save(
            {"generator": generator.state_dict(), "discriminator": discriminator.state_dict(), "config": config},
            output / "last.pt",
        )


def train_diffusion(config: dict, loader: DataLoader, device: torch.device, output: Path) -> None:
    model = ConditionalDiffusionUNet(base=int(config["base_channels"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=1e-4)
    steps = int(config["diffusion_steps"])
    beta = cosine_beta_schedule(steps, device)
    alpha_bar = torch.cumprod(1.0 - beta, dim=0)
    for epoch in range(1, int(config["epochs"]) + 1):
        running = 0.0
        for batch in loader:
            clean = batch["pore"].to(device)
            condition = batch["condition"].to(device)
            raw = batch["condition_raw"].to(device)
            drop = torch.rand(clean.shape[0], device=device) < float(config.get("condition_dropout", 0.1))
            conditioned = condition.clone()
            conditioned[drop] = 0.0
            indices = torch.randint(0, steps, (clean.shape[0],), device=device)
            noise = torch.randn_like(clean)
            cumulative = alpha_bar[indices][:, None, None, None]
            noisy = cumulative.sqrt() * clean + (1.0 - cumulative).sqrt() * noise
            prediction = model(noisy, indices.float() / max(steps - 1, 1), conditioned)
            reconstructed = (noisy - (1.0 - cumulative).sqrt() * prediction) / cumulative.sqrt().clamp_min(1e-4)
            loss = F.mse_loss(prediction, noise) + float(config["porosity_weight"]) * porosity_loss(
                reconstructed.tanh(), raw
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
        payload = {"epoch": epoch, "diffusion": running}
        append_history(output, payload)
        print(json.dumps(payload), flush=True)
        torch.save({"model": model.state_dict(), "config": config}, output / "last.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = Path(config["output_root"]) / config["experiment_name"]
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 2026))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "history.jsonl").write_text("", encoding="utf-8")
    dataset = PorousInverseDesignDataset(
        config["steady_root"], "train", max_samples=int(config.get("max_train_samples", 0))
    )
    dataset.stats.save(output / "condition_stats.json")
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(config.get("num_workers", 0)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config["method"] == "cgan":
        train_gan(config, loader, device, output)
    elif config["method"] == "diffusion":
        train_diffusion(config, loader, device, output)
    else:
        raise ValueError(f"Unknown inverse-design method: {config['method']}")
    (output / "completion.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "experiment_name": config["experiment_name"],
                "method": config["method"],
                "seed": seed,
                "epochs": int(config["epochs"]),
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
