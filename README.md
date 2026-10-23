# scMetaBasin 0.2.0

K is optional: omit it for automatic estimation, or specify it to use the existing known-K pipeline. Input is an existing representation bank; encoder training and raw-count preprocessing remain upstream steps.

## Install

Python >=3.10. From the extracted source package:

```sh
python -m pip install .
scmetabasin --help
```

The automatic-K code and its MIT-licensed skstab snapshot are inside the package. No research-project folder, prototype scripts, checkpoints, annotations or separate skstab installation are needed.

## Run

```sh
# Automatic K
scmetabasin --representations bank.npz --output results_auto

# Specified K
scmetabasin --representations bank.npz --k 15 --output results_K15

# A directory of representation NPY files is also accepted
scmetabasin --representations features_directory --output results_auto
```

The equivalent module entry is `python -m scmetabasin`. The output directory must be new and is never overwritten.

Input requires >=2 finite numeric 2-D matrices, with the same cells in the same row order. Feature dimensions may differ. Optional `cell_ids` in NPZ or `cell_ids.npy` in a directory preserves cell IDs; otherwise IDs are positional row indices. One-dimensional annotation vectors are not representation matrices.

## Python

```python
from scmetabasin import scMetaBasin, load_representations

bank, cell_ids = load_representations("bank.npz")
model = scMetaBasin()                 # automatic K
labels = model.fit_predict(bank)
print(model.estimated_K_, model.realized_K_, model.status_)

model = scMetaBasin(n_clusters=15)     # specified K
labels = model.fit_predict(bank)
```

`estimated_K_` is set only in automatic mode. `n_clusters` is the effective target after fitting. `requested_n_clusters` preserves the constructor argument, so fitting an automatic model again estimates K again.

## Parameters

| CLI option | Default | Meaning |
|---|---|---|
| --k | omitted | Automatic K; supply an integer >=2 to bypass estimation |
| --seeds | 0:100 | Ordered range or comma-separated seed list for the core |
| --max-basins | 5 | Maximum local-basin cut considered |
| --min-basin-size | 5 | Minimum recurrent-basin size used by discovery |
| --basin-linkage | average | Within-setting basin linkage |
| --meta-threshold | 0.85 | Cross-setting internal-ARI edge threshold |
| --prevalence-weight | 0.05 | Mean basin fraction weight in FrozenScore |
| --consensus-linkage | average | Co-assignment consensus linkage |
| --auto-k-workers | 5 | Automatic-K worker processes |

The Python worker option is `auto_k_workers=5`. Existing core options retain their meaning. They do not silently retune the separately fixed automatic-K screening rule.

## Automatic K scope

The experimental supported-area v3 estimator screens a common sample of <=1000 cells, using two perturbation batches and per-view standardization. It searches K=1..min(64,floor(sqrt(N)),N_sample-1), retains continuous jointly stable score excess over K=1, and recommends one K. Exact-prefix execution skips only suffix values that cannot meet the necessary stability condition. The chosen K enters the unchanged raw-representation core once.

**Automatic K is experimental.** Enge and Zeisel development runs recommended 4 and 12, with different quality tradeoffs. It does not guarantee the reference cell-type count or highest external ARI. Earlier annotation-assisted upstream configuration selection is not undone by a downstream estimator that does not read annotations. Automatic mode is not the paper's frozen known-K profile. The core primary/fallback, FrozenScore and consensus definitions are retained.

## Output and status

- `cluster_labels.csv`, `selected_labels.npy`: actual cell labels.
- `basin_table.csv`, `meta_basin_table.csv`, `selected_meta.json`: core evidence.
- `scmetabasin_summary.json`: mode, requested/estimated/target/realized K, parameters, status, input identity and runtime.
- Automatic mode adds `automatic_K.json`, `automatic_K_ranking.csv`, `automatic_K_windows.csv`, `automatic_K_paths.csv`, `automatic_K_between.csv`. Only the recommended K has run the core; other path-supported values are not confirmed meta-basin outputs. Uncomputed suffix within paths are not fabricated.
- `file_hashes.csv`: result-file integrity records.

Automatic outputs have `known_K=null`; manual outputs record supplied K. `frozen_default_profile` is true only for the specified-K paper-default profile. `core_default_profile` separately records core parameters.

No supported split returns Python `None` with `status_='no_supported_split'`; automatic mode may also return `no_supported_state`. CLI writes a status summary, no fabricated cell labels, and exits with code 3. The manual Python API retains its original exception when no selectable state exists.

The unchanged `maxclust` consensus can return fewer clusters than its target when merge heights tie. This version saves the actual labels and reports `completed_K_mismatch`; it does not force a different cut. Exact co-assignment still uses quadratic memory in N.

## Software checks

```sh
python examples/quickstart.py --output demo_manual
python examples/automatic_quickstart.py --output demo_auto
python -m unittest discover -s tests -v
```

These are synthetic software checks, not paper evidence. See VERSION.json and PROVENANCE.json for verification scope. The prior known-K scientific routine is retained without a body change.

## License

The package is MIT licensed (LICENSE). The private skstab snapshot is MIT licensed, with its authors' notice, commit, original/bundled hashes and two import-only edits recorded in `scmetabasin/_vendor/skstab/`. See LICENSE_STATUS.md. No private data or upstream checkpoints are included.
