"""Command-line access to the unchanged scMetaBasin v0.1.0 core."""
import argparse
from pathlib import Path
import time

import numpy as np

from . import __version__, scMetaBasin
from .io import load_representations_npz
from .result import save_scmetabasin_results


def parse_seeds(value):
    """Parse a half-open range (0:100) or an ordered list (0,2,4,6,8)."""
    try:
        if ":" in value:
            start, stop = map(int, value.split(":"))
            if stop - start > 10000:
                raise ValueError("Seed range exceeds 10000 candidates")
            seeds = list(range(start, stop))
        else:
            seeds = [int(item) for item in value.split(",")]
        if len(seeds) < 2 or len(seeds) != len(set(seeds)):
            raise ValueError("Provide at least two distinct seeds")
        if any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
            raise ValueError("Seeds must be integers between 0 and 2**32 - 1")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return seeds


def make_parser():
    parser = argparse.ArgumentParser(
        prog="scmetabasin", description="Cluster a frozen representation bank with a predeclared K."
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--representations", required=True, type=Path, help="Input representation NPZ")
    parser.add_argument("--k", required=True, type=int, help="Predeclared cell-cluster count")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--seeds", type=parse_seeds, default=list(range(100)), help="0:100 or 0,2,4,6,8 (default: 0:100)")
    parser.add_argument("--max-basins", type=int, default=5)
    parser.add_argument("--min-basin-size", type=int, default=5)
    parser.add_argument("--meta-threshold", type=float, default=0.85)
    methods = ["average", "complete", "single", "weighted"]
    parser.add_argument("--basin-linkage", choices=methods, default="average")
    parser.add_argument("--consensus-linkage", choices=methods, default="average", help="Final consensus linkage")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.k < 2:
        parser.error("--k must be at least 2")
    if args.max_basins < 2 or args.min_basin_size < 1:
        parser.error("--max-basins must be at least 2 and --min-basin-size at least 1")
    if not np.isfinite(args.meta_threshold) or not -1 <= args.meta_threshold <= 1:
        parser.error("--meta-threshold must be finite and between -1 and 1")
    input_path = args.representations.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error("Output directory already exists; choose a new directory")
    representations, cell_ids = load_representations_npz(input_path)
    n_cells = next(iter(representations.values())).shape[0]
    if args.k > n_cells:
        parser.error("--k exceeds the number of cells")
    parameters = dict(n_clusters=args.k, seeds=args.seeds, max_basins=args.max_basins,
                      min_basin_size=args.min_basin_size, basin_linkage=args.basin_linkage,
                      meta_threshold=args.meta_threshold, meta_linkage=args.consensus_linkage)
    frozen = (args.seeds == list(range(100)) and args.max_basins == 5 and args.min_basin_size == 5
              and args.basin_linkage == "average" and args.meta_threshold == 0.85
              and args.consensus_linkage == "average")
    model = scMetaBasin(**parameters)
    started = time.perf_counter()
    labels = model.fit_predict(representations)
    if len(np.unique(labels)) != args.k:
        raise RuntimeError(f"Final realized K={len(np.unique(labels))} differs from requested K={args.k}")
    elapsed = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    save_scmetabasin_results(model, output, cell_ids=cell_ids, input_path=input_path,
                            extra_summary={"parameters":parameters, "frozen_default_profile":frozen,
                                           "representation_names":sorted(representations),
                                           "runtime_seconds":elapsed})
    print(f"scMetaBasin {__version__}: {n_cells} cells, K={args.k}, selection={model.selection_mode_}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
