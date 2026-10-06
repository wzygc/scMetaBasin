# scMetaBasin

scMetaBasin groups recurrent complete partitions into local basins, links their
representatives across computational settings into meta-basins, applies a
label-free selector, and reports an exact co-assignment consensus partition.
The number of cell clusters K must be declared before the run.

This package starts from a representation bank. Encoder training and raw-count
preprocessing are upstream steps. Their recovered source records, checkpoints,
and current figure materials are distributed separately.

## Install

Python 3.10 or later is required; the local verification used Python 3.11.4.
From the extracted package directory:

```sh
python -m pip install .
scmetabasin --help
```

Runtime dependencies are declared in pyproject.toml. requirements-tested.txt
records the versions used for local verification on 2026-10-06. It is a current
test snapshot, not a reconstruction of the historical training environment.
Other dependency combinations have not been validated here.

## Quickstart

From the extracted source directory after installation:

```sh
python examples/quickstart.py --output demo
```

This creates a small synthetic software input and verifies the resulting
30 rows, 3 clusters, 3 local basins, and 1 meta-basin. It is an execution demo,
not paper evidence. The directory must be new. Outputs are listed below.
Paper-specific commands are provided separately in the
[frozen run guide](../scMetaBasin_reproduction_notes/FROZEN_RUNS.md).
Extract the companion reproduction notes beside this source directory.

## Input

Supply an NPZ containing at least two numeric, finite, two-dimensional arrays.
Each array is a computational setting, with rows corresponding to exactly the
same cells in the same order. Feature dimensions may differ across arrays.

An optional cell_ids array must have one entry per row. If omitted, output IDs
are positional row indices. Preserve the row-to-cell-ID sidecar when using the
separately prepared frozen prospective inputs. Barcode duplicates require row
indices or occurrence information when joining external metadata.

## Run

```sh
scmetabasin --representations bank.npz --k 15 --output results
```

The equivalent module entry is python -m scmetabasin. The output directory must
be new. Inputs do not contain reference annotations for method selection.

## Parameters

| Option | Default | Meaning |
|---|---|---|
| --k | Required | Predeclared number of cell clusters |
| --seeds | 0:100 | Half-open seed range or ordered comma-separated list |
| --max-basins | 5 | Maximum local-basin cut considered |
| --min-basin-size | 5 | Minimum recurrent-basin size used by discovery |
| --basin-linkage | average | Linkage used in within-setting basin discovery |
| --meta-threshold | 0.85 | Internal-ARI edge threshold for the cross-setting graph |
| --consensus-linkage | average | Linkage used for final co-assignment consensus |

The default seed order is 0 through 99. Changing seed order can change index-based
tie resolution. These options expose the existing v0.1.0 constructor without
changing the core algorithm. CLI validation checks ranges and duplicate seeds.

For a custom run:

```sh
scmetabasin --representations bank.npz --k 15 --output custom_results --seeds 0:50 --meta-threshold 0.80
```

The paper profile uses the defaults. K-means retains n_init=1 and max_iter=300.
The selector retains primary support of at least three settings, fallback
support of at least two, and FrozenScore weight 0.05. These rules remain fixed.
Changing exposed parameters defines a custom configuration; the output summary
records the actual parameter values and whether the default profile was used.

## Output

- cluster_labels.csv: row index, cell ID, and cluster label.
- basin_table.csv: local basin membership summaries and medoid information.
- meta_basin_table.csv: cross-setting support and selection scores.
- selected_meta.json: selected component and selection mode.
- scmetabasin_summary.json: input identity, actual parameters, and realized K.
- file_hashes.csv: output file integrity records.

The driver checks realized K before saving. The core maxclust consensus cut can
produce fewer than K clusters when merge heights tie; such a run raises an error
instead of silently altering clusters. Exact co-assignment uses quadratic memory
in the number of cells, so large datasets require appropriate resources.

When all candidate partitions in a setting coincide, the fallback cut can
realize a single basin. Its silhouette is undefined. This release preserves the
fallback membership and records NaN (an empty CSV field) for that silhouette,
so the run proceeds without attempting an invalid single-group score.

## Python interface

```python
from scmetabasin import scMetaBasin, load_representations_npz

representations, cell_ids = load_representations_npz("bank.npz")
model = scMetaBasin(n_clusters=15, meta_threshold=0.85)
labels = model.fit_predict(representations)
```

## Verification and provenance

```sh
python -m unittest discover -s tests -v
```

Six interface tests and the quickstart demo have passed in the locally tested
environment. PROVENANCE.json records the core's origin and the single-basin
silhouette guard; selection rules and default parameters are unchanged.
Paper-specific verification scope is documented in the separate
[reproduction notes](../scMetaBasin_reproduction_notes/VERIFICATION_SCOPE.md).
MANIFEST.sha256 records the files in this source archive.

## License

This core release is distributed under the MIT License; see LICENSE.
Separately recovered third-party software and scientific materials retain
their applicable terms; see LICENSE_STATUS.md for the scope.
