# Frozen prospective run entries

Install the core package, then extract the separately supplied
Frozen_Prospective_Inputs_with_Cell_Mapping.zip into a directory named
frozen_inputs. Run the following commands from its parent directory.
The input ZIP contains these dataset directories directly, without an
additional frozen_inputs directory inside it.

The K values below come from the saved predeclared run summaries. They are
not inferred by this package from annotations. All commands use the default
100 seeds (0 through 99) and the frozen selector configuration.

```sh
scmetabasin --representations frozen_inputs/MacParland/representations.npz --k 15 --output rerun/MacParland
scmetabasin --representations frozen_inputs/Tosches_turtle/representations.npz --k 15 --output rerun/Tosches_turtle
scmetabasin --representations frozen_inputs/worm_neuron_cell/representations.npz --k 10 --output rerun/worm_neuron_cell
scmetabasin --representations frozen_inputs/muris_mam_spl_T_B/representations.npz --k 2 --output rerun/muris_mam_spl_T_B
scmetabasin --representations frozen_inputs/CHOL_GSE125449_aPD1aPDL1aCTLA4/representations.npz --k 8 --output rerun/CHOL
```

| Dataset directory | Cells | Predeclared K | Saved local basins | Saved meta-basins | Saved selection |
|---|---:|---:|---:|---:|---|
| MacParland | 8444 | 15 | 11 | 5 | M0, V2 |
| Tosches_turtle | 18664 | 15 | 11 | 4 | M1, V2 |
| worm_neuron_cell | 4186 | 10 | 10 | 8 | M1, V1_fallback |
| muris_mam_spl_T_B | 11330 | 2 | 10 | 4 | M0, V2 |
| CHOL_GSE125449_aPD1aPDL1aCTLA4 | 5762 | 8 | 13 | 7 | M1, V2 |

This table describes saved results, not a claim that all five commands have
been freshly rerun in a clean environment. The full worm command has been
run using the existing local dependencies: all output labels match exactly,
basin and meta-basin tables match within 1e-12, and selection matches. The
other four full commands have not been freshly executed in this check.
The archive includes the original
result tables for comparison. Runtime and machine paths in new summaries
will differ from historical records; compare scientific outputs, not those
incidental metadata fields. Exact co-assignment requires quadratic memory;
the larger inputs need substantially more memory than the quickstart demo.

The frozen NPZs preserve their original bytes and omit cell_ids. New output
IDs are therefore positional row indices. Join the supplied row_to_cell_id.csv
using row_index, preserving barcode_occurrence; barcode alone is not a unique
key for muris. Do not reorder or normalize the arrays before a frozen rerun.

These entries start from saved representations. They do not rerun encoder
training, expression preprocessing, UMAP, independent methods, or the paper's
other experiments.
