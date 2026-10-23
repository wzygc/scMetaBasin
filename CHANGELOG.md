# 0.2.0 — optional K (2026-10-23)

- Omit CLI `--k` or Python `n_clusters` for experimental automatic K.
- Preserve the specified-K scientific pipeline and adjustable threshold/prevalence parameters.
- Bundle automatic screening, its worker and the pinned MIT skstab code; no prototype folder is required.
- Accept NPZ or an aligned directory of 2-D NPY representations, with optional cell IDs.
- Record requested, estimated, target and realized K; save the actual labels when consensus realizes fewer clusters.
- Report unsupported automatic runs explicitly without fabricating cell labels (CLI exit 3).
- Save automatic-screening evidence and output checksums.

## Verification scope

19 unit tests; fresh installed-wheel synthetic automatic/manual checks; directory/ID check;
exact real Enge K=6 manual-label regression. The known-K scientific body is AST-identical
to v0.1.1. These checks do not establish broad automatic-K biological performance.
Existing dependencies were reused; no global dependency update or upstream training.

v0.1.1 and historical experiment/manuscript files are retained unchanged.
