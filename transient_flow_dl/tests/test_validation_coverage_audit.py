from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation_coverage_audit import bin_counts, parse_run


class ValidationCoverageAuditTest(unittest.TestCase):
    def test_run_spec_and_bin_counts(self) -> None:
        label, path = parse_run("CT-PoreFlow=runs/ct")
        self.assertEqual(label, "CT-PoreFlow")
        self.assertEqual(path, Path("runs/ct"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [
                root / "bin_00" / "a.npz",
                root / "bin_00" / "b.npz",
                root / "bin_01" / "c.npz",
            ]
            for file in files:
                file.parent.mkdir(parents=True, exist_ok=True)
                file.touch()
            self.assertEqual(bin_counts(files, root)["bin_00"], 2)
            self.assertEqual(bin_counts(files, root)["bin_01"], 1)


if __name__ == "__main__":
    unittest.main()
