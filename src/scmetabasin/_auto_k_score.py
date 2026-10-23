"""Development selector: persistent, jointly stable excess over the K=1 path."""
import numpy as np
import pandas as pd


def select_supported_area(paths):
    averaged = paths.groupby(['batch', 'K', 'noise_index'])[['stadion', 'between']].mean()
    batches = sorted(paths.batch.unique())
    ks = sorted(paths.K.unique())
    ts = sorted(paths.noise_index.unique())
    if len(batches) != 2 or 1 not in ks or ts != list(range(len(ts))):
        raise ValueError('Expected two batches, K=1, and a common consecutive noise grid')
    if len(averaged) != 2 * len(ks) * len(ts):
        raise ValueError('Missing path entries')
    values = averaged.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError('Non-finite paths')
    s = averaged.stadion.to_numpy().reshape(2, len(ks), len(ts))
    b = averaged.between.to_numpy().reshape(2, len(ks), len(ts))
    excess = s - s[:, [ks.index(1)], :]
    rows, windows = [], []
    for i, k in enumerate(ks):
        if k == 1:
            continue
        valid = ((b[:, i] >= .85) & (excess[:, i] > 1e-12)).all(axis=0)
        valid[0] = False
        retained = np.zeros(len(ts), dtype=bool)
        start = None
        for t, ok in enumerate(np.r_[valid, False]):
            if ok and start is None:
                start = t
            elif not ok and start is not None:
                if t - start >= 2:
                    retained[start:t] = True
                    windows.append(dict(K=int(k), start_index=start, end_index=t-1,
                                        consecutive_levels=t-start))
                start = None
        # Common grid Riemann sum, not a mean over the surviving positions.
        # Taking the minimum excess across batches is conservative, not a CI.
        area = float(np.minimum(excess[0, i], excess[1, i])[retained].sum() / (len(ts)-1))
        rows.append(dict(K=int(k), supported_area=area, supported_levels=int(retained.sum())))
    ranking = pd.DataFrame(rows).sort_values(['supported_area', 'K'], ascending=[False, True])
    selected = int(ranking.iloc[0].K) if ranking.iloc[0].supported_area > 0 else 1
    return selected, ranking, pd.DataFrame(windows, columns=['K','start_index','end_index','consecutive_levels'])

