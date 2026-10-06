import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scmetabasin.cli import main, make_parser, parse_seeds
from scmetabasin.io import load_representations_npz
from scmetabasin.clusterer import discover_basins


class InterfaceTests(unittest.TestCase):
    def test_invalid_seed_lists(self):
        for value in ["0,0", "1", "-1,2", "4:2", "0:20000", "0:2:4"]:
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_seeds(value)

    def test_custom_parameter_parsing(self):
        args = make_parser().parse_args(["--representations", "bank.npz", "--k", "3", "--output", "out",
                                        "--seeds", "0,2,4,6,8", "--meta-threshold", "0.8", "--max-basins", "4"])
        self.assertEqual(args.seeds, [0, 2, 4, 6, 8])
        self.assertEqual(args.meta_threshold, 0.8)
        self.assertEqual(args.max_basins, 4)

    def test_identical_partitions_form_one_basin(self):
        result = discover_basins(np.ones((6, 6), dtype=np.float32))
        self.assertEqual(result["k"], 1)
        np.testing.assert_array_equal(result["labels"], np.zeros(6, dtype=int))
        self.assertTrue(np.isnan(result["silhouette"]))

    def test_identical_candidate_bank_runs(self):
        x = np.vstack([np.ones((10, 3))*center for center in [-4, 0, 4]])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"bank.npz"
            output = Path(folder)/"out"
            np.savez(path, a=x, b=x*2, c=x+1)
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--representations", str(path), "--k", "3", "--output", str(output), "--seeds", "0:6"])
            summary = json.loads((output/"scmetabasin_summary.json").read_text())
            self.assertEqual(summary["realized_K"], 3)
            self.assertEqual(summary["n_discovered_basins"], 3)
            self.assertEqual(summary["selection_mode"], "V2")

    def test_invalid_input_row_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"bank.npz"
            np.savez(path, a=np.ones((8, 2)), b=np.ones((7, 3)))
            with self.assertRaisesRegex(ValueError, "n_cells"):
                load_representations_npz(path)
            np.savez(path, a=np.ones((8, 2)), b=np.ones((8, 3)), cell_ids=np.array(["only_one"]))
            with self.assertRaisesRegex(ValueError, "cell_ids"):
                load_representations_npz(path)

    def test_cli_effective_parameter_record(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(30, 4))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"bank.npz"
            output = Path(folder)/"out"
            ids = np.array([f"cell_{i}" for i in range(len(x))])
            np.savez(path, a=x, b=x*2, c=x+1, cell_ids=ids)
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--representations", str(path), "--k", "3", "--output", str(output),
                      "--seeds", "0:6", "--meta-threshold", "0.8"])
            summary = json.loads((output/"scmetabasin_summary.json").read_text())
            self.assertEqual(summary["realized_K"], 3)
            self.assertEqual(summary["parameters"]["seeds"], list(range(6)))
            self.assertEqual(summary["parameters"]["meta_threshold"], 0.8)
            self.assertFalse(summary["frozen_default_profile"])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--representations", str(path), "--k", "3", "--output", str(output)])
            self.assertEqual(json.loads((output/"scmetabasin_summary.json").read_text()), summary)


if __name__ == "__main__":
    unittest.main()
