from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from statistical_comparison import _load_sample_metrics


class StatisticalFinalMetricTest(unittest.TestCase):
    def test_final_metrics_use_maximum_iteration_independent_of_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            table = run / "evaluation" / "test" / "per_sample_time_metrics.csv"
            table.parent.mkdir(parents=True)
            fields = [
                "sample_id",
                "iteration",
                "relative_l2_velocity",
                "mae_p_normalized",
                "mae_speed_lattice",
                "relative_error_mean_flux_final",
                "relative_error_permeability_final",
                "relative_error_hydraulic_tortuosity_final",
            ]
            rows = []
            for sample_id, offset in (("a", 0.0), ("b", 1.0)):
                for iteration in (20, 0, 5):
                    is_final = iteration == 20
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "iteration": iteration,
                            "relative_l2_velocity": offset + iteration / 10,
                            "mae_p_normalized": offset + iteration / 100,
                            "mae_speed_lattice": offset + iteration / 1000,
                            # A finite early value is deliberately present. The
                            # loader must select by iteration, not by finiteness
                            # or physical CSV row order.
                            "relative_error_mean_flux_final": offset
                            + (0.2 if is_final else 9.0),
                            "relative_error_permeability_final": offset
                            + (0.2 if is_final else 9.0),
                            "relative_error_hydraulic_tortuosity_final": offset
                            + (0.3 if is_final else 9.0),
                        }
                    )
            with table.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            metrics = _load_sample_metrics(run)
            self.assertAlmostEqual(metrics["a"]["relative_l2_velocity"], 1.25)
            self.assertAlmostEqual(metrics["a"]["relative_error_mean_flux_final"], 0.2)
            self.assertAlmostEqual(metrics["a"]["relative_error_permeability_final"], 0.2)
            self.assertAlmostEqual(
                metrics["b"]["relative_error_hydraulic_tortuosity_final"], 1.3
            )


if __name__ == "__main__":
    unittest.main()
