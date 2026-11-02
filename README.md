# scMetaBasin

scMetaBasin groups repeated clustering partitions into within-setting basins and
cross-setting meta-basins, then selects a consensus partition. K can be supplied
or estimated automatically.

## Install

Python 3.10 or later. From the extracted source directory:

```sh
python -m pip install .
```

To construct graph views, include the optional PyTorch dependency:

```sh
python -m pip install ".[fiveview]"
```

## Construct the five views

Supply an already-preprocessed expression matrix and a latent embedding as NPY
files. Both must contain the same cells in the same row order. This step builds
GAT, PF-batch, PF-mutual and PF-union views and retains the latent embedding;
it does not train an encoder or preprocess raw counts.

For new data, initialize fixed attention parameters with an explicit seed:

```sh
scmetabasin-build-fiveview --expression expression.npy --embedding embedding.npy --gat-seed 42 --output views.npz
```

For expression input, supply its 32-component PCA scores as `embedding.npy`.
For learned input, supply the model's latent vectors. Existing attention parameters
can be supplied with `--gat-weights parameters.npz` instead of `--gat-seed`.

Defaults: 128-cell consecutive batches, 30 cosine neighbors including self,
one propagation hop, float32 CPU arithmetic and attention negative slope 0.3.
Optional `--cell-ids cell_ids.npy` carries one identifier per row.

## Cluster

```sh
# Estimate K automatically
scmetabasin --representations views.npz --output results_auto

# Supply K
scmetabasin --representations views.npz --k 15 --output results_K15
```

Existing representation NPZ files or directories of NPY matrices are also
accepted. Inputs require at least two finite, real, two-dimensional matrices
with matching cell rows; feature dimensions may differ. Output paths must be new.
Run `scmetabasin --help` for available parameters.

Python usage:

```python
from scmetabasin import scMetaBasin, load_representations

views, cell_ids = load_representations("views.npz")
model = scMetaBasin()  # or scMetaBasin(n_clusters=15)
labels = model.fit_predict(views)
```

Defaults are 100 K-means runs per setting, meta-basin threshold 0.85 and
prevalence weight 0.05. Automatic K uses at least ten cells and screens up to
1,000 cells. The paper analyses used supplied K.

## Results

- `cluster_labels.csv` and `selected_labels.npy`: cell assignments.
- `basin_table.csv`, `meta_basin_table.csv` and `selected_meta.json`: discovered
  states, support and selection.
- `scmetabasin_summary.json`: parameters, status and requested/estimated/realized K.
- Automatic mode also saves K-screening diagnostics; `file_hashes.csv` lists
  output checksums.

If no split is supported, automatic mode returns `None`; inspect `model.status_`.
The CLI exits with code 3 when no partition is selectable. Consensus can realize
fewer clusters than requested when merge heights tie; actual labels and realized K
are reported. Exact consensus requires memory quadratic in the cell count.

## Checks and license

```sh
python -m unittest discover -s tests -v
```

Runnable examples are in `examples/`. The package is MIT licensed. The bundled
Stadion implementation retains its third-party license under
`src/scmetabasin/_vendor/skstab/`. 
