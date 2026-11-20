from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import (
    add_panel_label,
    apply_nature_style,
    close_figure,
    method_style,
    save_publication_figure,
    soften_axis,
)


def load_history(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if "epoch" in row and "train" in row and "validation" in row:
            rows.append(row)
    rows.sort(key=lambda row: int(row["epoch"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-epochs", type=int, default=50)
    args = parser.parse_args()
    if len(args.run) != len(args.label):
        raise ValueError("--run and --label counts must match")

    histories: list[tuple[str, list[dict]]] = []
    long_rows: list[dict] = []
    for run, label in zip(args.run, args.label):
        history = load_history(run / "history.jsonl")
        epochs = [int(row["epoch"]) for row in history]
        expected = list(range(1, args.expected_epochs + 1))
        if epochs != expected:
            raise ValueError(f"{run}: epochs are {epochs[:3]}...{epochs[-3:] if epochs else []}, expected 1..{args.expected_epochs}")
        histories.append((label, history))
        for row in history:
            long_rows.append(
                {
                    "label": label,
                    "run": run.name,
                    "epoch": int(row["epoch"]),
                    "learning_rate": float(row["learning_rate"]),
                    "train_loss": float(row["train"]["loss"]),
                    "validation_loss": float(row["validation"]["loss"]),
                    "train_relative_l2": float(row["train"]["relative_l2_nonzero_targets"]),
                    "validation_relative_l2": float(row["validation"]["relative_l2_nonzero_targets"]),
                    "seconds": float(row["seconds"]),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)

    apply_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    for index, (label, history) in enumerate(histories):
        epochs = [int(row["epoch"]) for row in history]
        style = method_style(label, index)
        color = style["color"]
        axes[0].plot(
            epochs,
            [row["train"]["loss"] for row in history],
            color=color,
            alpha=0.28,
            linewidth=0.95,
            linestyle="--",
        )
        axes[0].plot(
            epochs,
            [row["validation"]["loss"] for row in history],
            color=color,
            linewidth=1.7 if label != "CT-PoreFlow" else 2.0,
            linestyle=style["linestyle"],
            label=label,
        )
        axes[1].plot(
            epochs,
            [row["validation"]["relative_l2_nonzero_targets"] for row in history],
            color=color,
            linewidth=1.7 if label != "CT-PoreFlow" else 2.0,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=5,
            markersize=3.0,
            label=label,
        )

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Smooth L1 + gradient loss (log scale)")
    axes[0].set_title("Training (dashed) and validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation velocity relative $L_2$")
    axes[1].set_title("Fixed seven-time validation trajectory")
    for panel, axis in zip("ab", axes):
        soften_axis(axis)
        add_panel_label(axis, panel)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.075), ncol=4)
    save_publication_figure(fig, args.output)
    close_figure(fig)
    print(args.output)
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".svg"))
    print(csv_path)


if __name__ == "__main__":
    main()
