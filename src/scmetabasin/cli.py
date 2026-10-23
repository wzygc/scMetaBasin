"""Command-line access to scMetaBasin with preserved paper defaults."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from . import __version__, scMetaBasin
from .io import load_representations
from .result import save_scmetabasin_results, save_k_diagnostics


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
        prog="scmetabasin", description="Cluster a representation bank; omit --k to estimate K automatically."
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--representations", required=True, type=Path, help="Representation NPZ or directory of NPY matrices")
    parser.add_argument("--k", default=None, type=int, help="Optional cluster count; omitted = automatic K")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--seeds", type=parse_seeds, default=list(range(100)), help="0:100 or 0,2,4,6,8 (default: 0:100)")
    parser.add_argument("--max-basins", type=int, default=5)
    parser.add_argument("--min-basin-size", type=int, default=5)
    parser.add_argument("--meta-threshold", type=float, default=0.85)
    parser.add_argument("--prevalence-weight", type=float, default=0.05, help="Non-negative basin-prevalence weight (paper default: 0.05)")
    parser.add_argument("--auto-k-workers", type=int, default=5, help="Automatic K worker processes (default: 5)")
    methods = ["average", "complete", "single", "weighted"]
    parser.add_argument("--basin-linkage", choices=methods, default="average")
    parser.add_argument("--consensus-linkage", choices=methods, default="average", help="Final consensus linkage")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.k is not None and args.k < 2:
        parser.error("--k must be at least 2")
    if args.max_basins < 2 or args.min_basin_size < 1:
        parser.error("--max-basins must be at least 2 and --min-basin-size at least 1")
    if not np.isfinite(args.meta_threshold) or not -1 <= args.meta_threshold <= 1:
        parser.error("--meta-threshold must be finite and between -1 and 1")
    if not np.isfinite(args.prevalence_weight) or args.prevalence_weight < 0:
        parser.error("--prevalence-weight must be finite and non-negative")
    if args.auto_k_workers < 1:
        parser.error("--auto-k-workers must be positive")
    input_path = args.representations.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error("Output directory already exists; choose a new directory")
    representations, cell_ids = load_representations(input_path)
    n_cells = next(iter(representations.values())).shape[0]
    if args.k is not None and args.k > n_cells:
        parser.error("--k exceeds the number of cells")
    parameters = dict(n_clusters=args.k, seeds=args.seeds, max_basins=args.max_basins,
                      min_basin_size=args.min_basin_size, basin_linkage=args.basin_linkage,
                      meta_threshold=args.meta_threshold, meta_linkage=args.consensus_linkage,
                      prevalence_weight=args.prevalence_weight, auto_k_workers=args.auto_k_workers)
    frozen = (args.seeds == list(range(100)) and args.max_basins == 5 and args.min_basin_size == 5
              and args.basin_linkage == "average" and args.meta_threshold == 0.85
              and args.consensus_linkage == "average" and args.prevalence_weight == 0.05)
    model = scMetaBasin(**parameters)
    started = time.perf_counter()
    if args.k is None:
        print('Estimating K automatically, then running scMetaBasin...', flush=True)
    try:
        labels = model.fit_predict(representations)
    except RuntimeError as error:
        if str(error) != 'No selectable meta-basin with >=2 representations.':
            raise
        labels = None
    elapsed = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    if labels is None:
        summary = dict(version=__version__, status=model.status_, k_mode=model.k_mode_,
                       requested_K=args.k, estimated_K=model.estimated_K_, target_K=model.n_clusters,
                       realized_K=None, n_cells=n_cells, parameters=parameters,
                       runtime_seconds=elapsed)
        (output/"scmetabasin_summary.json").write_text(json.dumps(summary, indent=2), encoding='utf-8')
        save_k_diagnostics(model, output)
        print(f"scMetaBasin {__version__}: {model.status_}; no cell labels emitted. Results: {output}")
        return 3
    save_scmetabasin_results(model, output, cell_ids=cell_ids, input_path=input_path,
                            extra_summary={"parameters":parameters, "frozen_default_profile":frozen and args.k is not None,
                                           "core_default_profile":frozen,
                                           "representation_names":sorted(representations),
                                           "runtime_seconds":elapsed})
    print(f"scMetaBasin {__version__}: {n_cells} cells, {model.k_mode_} target K={model.n_clusters}, realized K={model.realized_K_}, selection={model.selection_mode_}")
    print(f"Results: {output}")


if __name__ == "__main__":
    raise SystemExit(main())
