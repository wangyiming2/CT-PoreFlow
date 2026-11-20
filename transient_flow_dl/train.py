from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import TransientFlowDataset
from src.losses import transient_loss
from src.metrics import batch_metrics
from src.models import build_model
from src.utils import environment_report, load_json, save_json, set_seed


def resolve_velocity_scale(config: dict) -> float:
    path = Path(config["normalization_file"])
    if path.exists():
        payload = load_json(path)
        return float(payload["velocity_scale"])
    return float(config["velocity_scale_fallback"])


def make_loader(config: dict, split: str, velocity_scale: float) -> DataLoader:
    train = split == "train"
    max_key = "max_train_samples" if train else "max_validation_samples"
    dataset = TransientFlowDataset(
        steady_root=config["steady_root"],
        transient_root=config["transient_root"],
        split=split,
        velocity_scale=velocity_scale,
        rho_in=config["rho_in"],
        rho_out=config["rho_out"],
        max_iteration=config["max_iteration"],
        mode="random_time" if train else "fixed_times",
        time_indices=(
            config.get("training_time_indices")
            if train
            else config["validation_time_indices"]
        ),
        max_samples=int(config[max_key]),
        seed=int(config["seed"]),
        topology_root=config.get("topology_root"),
        time_transform=config.get("time_encoding", "log"),
    )
    num_workers = int(config["num_workers"])
    loader_options = {
        "batch_size": int(config["batch_size"]),
        "shuffle": train,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": train,
    }
    if num_workers > 0:
        loader_options.update(
            persistent_workers=True,
            prefetch_factor=int(config.get("prefetch_factor", 2)),
        )
    return DataLoader(
        dataset,
        **loader_options,
    )


def run_epoch(model, loader, device, config, optimizer=None, scaler=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    for batch in loader:
        geometry = batch["geometry"].to(device, non_blocking=True)
        query_time = batch["time"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        pore = batch["pore"].to(device, non_blocking=True)
        steady_fill = batch["steady_fill"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(config["amp"]) and device.type == "cuda"):
            prediction = model(geometry, query_time)
            loss, components = transient_loss(
                prediction,
                target,
                pore,
                steady_fill,
                gradient_weight=float(config["gradient_loss_weight"]),
                steady_fill_weight=float(config["steady_fill_weight"]),
                query_time=query_time,
                transport_weight=float(config.get("transport_loss_weight", 0.0)),
                transport_time_power=float(config.get("transport_time_power", 4.0)),
                transport_flux_floor=float(config.get("transport_flux_floor", 0.05)),
            )
        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        metrics = batch_metrics(prediction.detach(), target, pore)
        batch_size = geometry.shape[0]
        totals["loss"] += float(loss.detach()) * batch_size
        for key, value in components.items():
            totals[key] += float(value) * batch_size
        for key, value in metrics.items():
            totals[key] += value * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    set_seed(int(config["seed"]))
    run_dir = Path(config["output_root"]) / config["experiment_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "environment.json", environment_report())
    velocity_scale = resolve_velocity_scale(config)
    save_json(run_dir / "normalization_used.json", {"velocity_scale": velocity_scale})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(config, "train", velocity_scale)
    val_loader = make_loader(config, "validation", velocity_scale)
    model = build_model(config).to(device)
    initial_checkpoint = config.get("initial_checkpoint")
    if initial_checkpoint:
        checkpoint_path = Path(initial_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        print(f"initialized_from={checkpoint_path}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["amp"]) and device.type == "cuda")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={device} parameters={parameter_count} velocity_scale={velocity_scale:.8g}", flush=True)
    save_json(run_dir / "model.json", {"name": type(model).__name__, "parameters": parameter_count})

    history_path = run_dir / "history.jsonl"
    best = float("inf")
    best_path = run_dir / "best.pt"
    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        best = float(best_checkpoint.get("validation_loss", float("inf")))

    start_epoch = 1
    last_path = run_dir / "last.pt"
    if last_path.exists() and not (run_dir / "completion.json").exists():
        resume = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        if "scheduler" in resume:
            scheduler.load_state_dict(resume["scheduler"])
        else:
            scheduler.last_epoch = int(resume["epoch"])
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        if "scaler" in resume:
            scaler.load_state_dict(resume["scaler"])
        if "python_rng_state" in resume:
            random.setstate(resume["python_rng_state"])
        if "numpy_rng_state" in resume:
            np.random.set_state(resume["numpy_rng_state"])
        if "torch_rng_state" in resume:
            torch.set_rng_state(resume["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state" in resume:
            # Checkpoints are loaded with map_location=device so CUDA RNG
            # tensors can themselves land on CUDA.  PyTorch's RNG restore API
            # requires CPU ByteTensors.
            torch.cuda.set_rng_state_all(
                [state.detach().cpu() for state in resume["cuda_rng_state"]]
            )
        start_epoch = int(resume["epoch"]) + 1
        print(f"resuming_from_epoch={start_epoch}", flush=True)

        # A killed process can leave duplicate or partial history from a prior
        # restart. Retain one valid record per completed checkpoint epoch.
        if history_path.exists():
            records: dict[int, dict] = {}
            for line in history_path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                    record_epoch = int(record["epoch"])
                    if record_epoch < start_epoch:
                        records[record_epoch] = record
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
            history_path.write_text(
                "".join(json.dumps(records[index], ensure_ascii=False) + "\n" for index in sorted(records)),
                encoding="utf-8",
            )

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device, config, optimizer, scaler)
        val_metrics = run_epoch(model, val_loader, device, config)
        scheduler.step()
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - start,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "velocity_scale": velocity_scale,
            "validation_loss": val_metrics["loss"],
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        torch.save(state, run_dir / "last.pt")
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            torch.save(state, run_dir / "best.pt")
        if epoch % int(config["save_every"]) == 0:
            torch.save(state, run_dir / f"epoch_{epoch:03d}.pt")
    save_json(run_dir / "completion.json", {"status": "complete", "best_validation_loss": best})


if __name__ == "__main__":
    main()
