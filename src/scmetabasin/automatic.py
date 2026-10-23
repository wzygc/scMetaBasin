"""Supported-area automatic K, with bundled dependencies and exact-prefix execution."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from ._auto_k_score import select_supported_area
from .io import validate_representations


def estimate_k(representations, workers=5):
    """Estimate one K without reference labels; return None when no split is supported."""
    n = validate_representations(representations)
    if n < 10:
        raise ValueError('Automatic K requires >=10 cells; specify n_clusters for smaller inputs.')
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer.')
    with tempfile.TemporaryDirectory(prefix='scmetabasin-auto-k-') as temporary:
        return _compute(representations, Path(temporary), workers)


def _compute(bank, folder, workers):
    names = sorted(bank)
    n = len(bank[names[0]])
    aliases = {f'view{i:04d}': name for i, name in enumerate(names)}
    features = {alias: str(folder / f'{alias}.npy') for alias in aliases}
    for alias, name in aliases.items():
        np.save(features[alias], bank[name], allow_pickle=False)
    manifest = folder / 'worker_inputs.json'
    job = dict(features=features, view_names=aliases)
    manifest.write_text(json.dumps(job), encoding='utf-8')
    tasks = [(batch, alias) for batch in [0, 1] for alias in aliases]
    package_root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env['PYTHONPATH'] = package_root + os.pathsep + env.get('PYTHONPATH', '')

    def launch(stage, task):
        batch, alias = task
        log_path = folder / f'{stage}_b{batch}_{alias}.log'
        with log_path.open('w', encoding='utf-8') as log:
            done = subprocess.run([sys.executable, '-m', 'scmetabasin._auto_worker', stage,
                                   str(batch), alias, str(manifest)], env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=1800)
        if done.returncode:
            detail = log_path.read_text(encoding='utf-8', errors='replace')[-4000:]
            raise RuntimeError(f'Automatic K {stage} worker failed: {detail}')

    with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        list(pool.map(lambda task: launch('between', task), tasks))
    arrays = {}
    between_rows = []
    for batch, alias in tasks:
        with np.load(folder / f'b{batch}_{alias}_between.npz', allow_pickle=False) as archive:
            b = archive['between'][:, :, 0].copy()
            ks = archive['K'].tolist()
        arrays[batch, alias] = b
        between_rows.extend(dict(batch=batch, representation=aliases[alias], K=k,
                                 noise_index=t, between=float(b[i, t]))
                            for i, k in enumerate(ks) for t in range(10))
    between = pd.DataFrame(between_rows)
    averaged = np.stack([np.mean([arrays[batch, alias] for alias in aliases], axis=0) for batch in [0, 1]])
    valid = (averaged >= .85).all(axis=0)
    valid[:, 0] = False
    possible = [int(k) for i, k in enumerate(ks) if k > 1 and (valid[i, 1:] & valid[i, :-1]).any()]
    indices = np.sort(np.random.default_rng(41).choice(n, min(n, 1000), replace=False))
    info = dict(method='supported_area_v3', experimental=True, search_cap=max(ks),
                screening_cells=len(indices), screening_seed=41, batches=[101, 211],
                perturbations_per_level=5, noise_levels=10, min_between_stability=0.85,
                min_consecutive_levels=2, omega=[2, 3, 4, 5],
                representation_order=names, within_prefix_K=0, path_supported_Ks=[],
                screening_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                status='no_supported_split', search_boundary_hit=False)
    ranking = pd.DataFrame(columns=['K', 'supported_area', 'supported_levels'])
    windows = pd.DataFrame(columns=['K', 'start_index', 'end_index', 'consecutive_levels'])
    paths = pd.DataFrame(columns=['batch', 'representation', 'K', 'noise_index', 'between', 'within', 'stadion'])
    if not possible:
        return dict(target_K=None, info=info, ranking=ranking, windows=windows, paths=paths, between=between)
    job['prefix_max'] = max(possible)
    manifest.write_text(json.dumps(job), encoding='utf-8')
    with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        list(pool.map(lambda task: launch('within', task), tasks))
    rows = []
    for batch, alias in tasks:
        with np.load(folder / f'b{batch}_{alias}_prefix.npz', allow_pickle=False) as archive:
            for i, k in enumerate(archive['K']):
                for t in range(10):
                    rows.append(dict(batch=batch, representation=aliases[alias], K=int(k), noise_index=t,
                        between=float(archive['between'][i, t, 0]), within=float(archive['within'][i, t, 0]),
                        stadion=float(archive['stadion'][i, t, 0])))
    paths = pd.DataFrame(rows)
    k, ranking, windows = select_supported_area(paths)
    target = None if k == 1 else k
    info.update(within_prefix_K=job['prefix_max'],
                path_supported_Ks=sorted(int(v) for v in ranking.loc[ranking.supported_area > 0, 'K']),
                status='estimated' if target is not None else 'no_supported_split',
                search_boundary_hit=target == max(ks))
    return dict(target_K=target, info=info, ranking=ranking, windows=windows, paths=paths, between=between)
