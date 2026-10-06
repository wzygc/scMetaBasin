"""Small synthetic software demo; not a paper dataset or scientific result."""
import argparse
import json
from pathlib import Path

import numpy as np

from scmetabasin.cli import main


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("demo"), help="New demo directory")
args = parser.parse_args()
if args.output.exists():
    parser.error("Demo directory already exists; choose a new directory")
args.output.mkdir(parents=True)
x = np.vstack([np.full((10, 3), center, dtype=float) for center in (-4, 0, 4)])
bank = args.output / "representations.npz"
np.savez_compressed(bank, setting_a=x, setting_b=x * 2, setting_c=x + 1,
                    cell_ids=np.array([f"demo_cell_{i}" for i in range(len(x))]))
main(["--representations", str(bank), "--k", "3", "--seeds", "0:6",
      "--output", str(args.output / "results")])
summary = json.loads((args.output / "results" / "scmetabasin_summary.json").read_text())
expected = {"n_cells": 30, "realized_K": 3, "n_discovered_basins": 3,
            "n_meta_basins": 1, "selection_mode": "V2"}
for key, value in expected.items():
    if summary[key] != value:
        raise RuntimeError(f"Unexpected demo output: {key}={summary[key]!r}, expected {value!r}")
print("Software demo completed: 30 rows, 3 clusters, 3 local basins, 1 meta-basin.")
print("This synthetic demo verifies execution only; it is not paper evidence.")
