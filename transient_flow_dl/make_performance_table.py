from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    """Load post-processing metadata without importing the PyTorch runtime."""
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for run in sorted(args.runs_root.iterdir()):
        summary_path = run / "evaluation" / "test" / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        model = load_json(run / "model.json")
        rows.append(
            {
                "run": run.name,
                "parameters": model["parameters"],
                "mae_ux_normalized": summary["mae_ux_normalized"],
                "mae_uy_normalized": summary["mae_uy_normalized"],
                "mae_p_normalized": summary["mae_p_normalized"],
                "relative_l2_ux": summary["relative_l2_ux"],
                "relative_l2_uy": summary["relative_l2_uy"],
                "relative_l2_p": summary["relative_l2_p"],
                "relative_l2_velocity": summary["relative_l2_velocity"],
                "mae_speed_lattice": summary["mae_speed_lattice"],
                "relative_error_mean_flux_final": summary.get("relative_error_mean_flux_final"),
                "relative_error_permeability_final": summary.get("relative_error_permeability_final"),
                "relative_error_hydraulic_tortuosity_final": summary.get(
                    "relative_error_hydraulic_tortuosity_final"
                ),
                "pressure_drop_mae_normalized": summary.get("pressure_drop_mae_normalized"),
                "pred_mass_balance_error_final": summary.get("pred_mass_balance_error_final"),
                "true_mass_balance_error_final": summary.get("true_mass_balance_error_final"),
                "pred_divergence_l1_lattice": summary.get("pred_divergence_l1_lattice"),
                "max_solid_speed_lattice": summary.get("max_solid_speed_lattice"),
                "model_milliseconds_per_frame": summary.get("model_milliseconds_per_frame"),
                "end_to_end_milliseconds_per_frame": summary.get(
                    "end_to_end_milliseconds_per_frame", summary["milliseconds_per_frame"]
                ),
                "records": summary["records"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("No completed test summaries found")
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
