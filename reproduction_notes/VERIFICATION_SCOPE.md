# Verification scope

- Six core interface tests and the small synthetic execution demo passed.
  The synthetic demo is not scientific evidence.
- All 25 saved 100-seed settings across the five prospective datasets were
  checked for basin discovery, memberships and medoids. Saved-medoid graph,
  scores and selection also matched for all five datasets. These checks
  did not regenerate the saved K-means candidates or large consensus outputs.
- A separate fresh execution used the full frozen worm representation bank:
  4186 rows, five settings, 100 seeds per setting, predeclared K=10. It ran
  K-means, basin discovery, meta-basin selection and exact consensus. Every
  final label matched the saved output; basin/meta-basin tables matched
  within 1e-12 and selection matched exactly.
- The worm run used Python 3.11.4, numpy 1.26.4, pandas 3.0.5, scipy 1.11.2
  and scikit-learn 1.8.0, with numerical thread counts limited to one.
  Runtime was approximately 49 seconds on that machine. This is not a
  cross-machine performance guarantee.
- The other four full input-to-consensus commands were not freshly rerun in
  that execution check. A new-machine dependency installation, upstream
  expression/encoder pipeline and independent reconstruction of all paper
  figures have not been verified.

Internal audit reports, execution logs and machine-specific paths are kept
outside both release archives. The original scientific inputs and saved
results remain unchanged.
