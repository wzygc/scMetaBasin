"""Exact prefix execution of the pinned author estimator; no source edits."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import hashlib
import json
import sys
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ._vendor.skstab import StadionEstimator


def data_and_estimator(features, rep, batch):
    names = sorted(features)
    seed = [101,211][batch] + names.index(rep)
    raw = np.load(features[rep],allow_pickle=False)
    indices = np.sort(np.random.default_rng(41).choice(len(raw),min(len(raw),1000),replace=False))
    x = StandardScaler().fit_transform(raw[indices]).astype(np.float64)
    ks = list(range(1,min(64,int(np.sqrt(len(raw))),len(indices)-1)+1))
    np.random.seed(seed)
    estimator = StadionEstimator(x,KMeans,param_name='n_clusters',param_values=ks,
        omega=list(range(2,6)),extended=True,runs=5,perturbation='uniform',
        perturbation_kwargs='auto',algo_kwargs=dict(init='k-means++',n_init=10,random_state=seed),n_jobs=1)
    return x, indices, ks, seed, estimator


def save_between(features, rep, batch, folder):
    tick = time.monotonic()
    with threadpool_limits(limits=1):
        x, indices, ks, seed, e = data_and_estimator(features,rep,batch)
        b = e.between_cluster_stability_paths
        state = np.random.get_state()
    np.savez(folder/f'b{batch}_{rep}_between.npz',K=ks,between=b,
        rng_keys=state[1],rng_position=state[2],rng_has_gauss=state[3],rng_cached_gauss=state[4],
        seed=seed,indices=indices,x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        feature_sha256=hashlib.sha256(Path(features[rep]).read_bytes()).hexdigest(),
        seconds=time.monotonic()-tick)
    return time.monotonic()-tick


def save_within_prefix(features,rep,batch,folder,prefix_max):
    tick = time.monotonic()
    with np.load(folder/f'b{batch}_{rep}_between.npz',allow_pickle=False) as a:
        b = a['between'].copy()
        ks = a['K'].tolist()
        state = ('MT19937',a['rng_keys'].copy(),int(a['rng_position']),
                 int(a['rng_has_gauss']),float(a['rng_cached_gauss']))
        expected_x = str(a['x_sha256'])
        expected_input = str(a['feature_sha256'])
    if hashlib.sha256(Path(features[rep]).read_bytes()).hexdigest()!=expected_input:
        raise RuntimeError('Feature changed between stages')
    with threadpool_limits(limits=1):
        x, indices, actual_ks, seed, e = data_and_estimator(features,rep,batch)
        if actual_ks!=ks or hashlib.sha256(x.tobytes()).hexdigest()!=expected_x:
            raise RuntimeError('Input or grid mismatch between stages')
        # Recreate ALL reference partitions, then restore the exact RNG position
        # after ALL between paths. Only the within suffix is omitted.
        np.random.set_state(state)
        e.param_values = ks[:prefix_max]
        e.between_cluster_stability_paths_ = b[:prefix_max].copy()
        w = e.within_cluster_stability_paths
        s = e.stadion_paths
    np.savez(folder/f'b{batch}_{rep}_prefix.npz',K=ks[:prefix_max],
             between=b[:prefix_max],within=w,stadion=s,seconds=time.monotonic()-tick)
    return time.monotonic()-tick


if __name__ == '__main__':
    stage, batch, rep, manifest = sys.argv[1], int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    job = json.loads(manifest.read_text(encoding='utf-8'))
    if stage == 'between':
        save_between(job['features'], rep, batch, manifest.parent)
    elif stage == 'within':
        save_within_prefix(job['features'], rep, batch, manifest.parent, job['prefix_max'])
    else:
        raise ValueError('Unknown automatic K worker stage')
