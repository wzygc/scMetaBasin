from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

@dataclass
class BasinNode:
    node_id: int
    representation: str
    basin_id: int
    basin_fraction: float
    medoid_seed: int
    medoid_stability: float
    labels: np.ndarray

def pairwise_partition_ari(partitions):
    """
    Pairwise ARI between clustering partitions.

    float32 is intentional:
    historical MetaBasin code stored the ARI matrix as float32.
    """
    partitions = np.asarray(partitions)
    n = len(partitions)
    M = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            v = np.float32(adjusted_rand_score(partitions[i], partitions[j]))
            M[i, j] = v
            M[j, i] = v
    return M

def discover_basins(pairwise_ari, max_basins=5, min_basin_size=5, method='average'):
    """
    Historical frozen basin rule:

        distance = 1 - partition ARI
        hierarchical clustering
        K_basin in [2, max_basins]
        min basin size >= 5
        choose maximum silhouette
        exact tie -> smaller K_basin

    If no candidate satisfies min_basin_size:
        fallback to K_basin=2.
    """
    D = 1.0 - pairwise_ari.astype(np.float64)
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=method)
    n_runs = len(pairwise_ari)
    max_k = min(int(max_basins), n_runs - 1)
    candidates = []
    for k in range(2, max_k + 1):
        labels = fcluster(Z, t=k, criterion='maxclust') - 1
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2:
            continue
        if counts.min() < min_basin_size:
            continue
        sil = float(silhouette_score(D, labels, metric='precomputed'))
        candidates.append({'k': int(len(unique)), 'silhouette': sil, 'min_size': int(counts.min()), 'max_size': int(counts.max()), 'labels': labels.copy()})
    if not candidates:
        labels = fcluster(Z, t=2, criterion='maxclust') - 1
        counts = np.bincount(labels)
        sil = float(silhouette_score(D, labels, metric='precomputed')) if len(np.unique(labels)) > 1 else float('nan')
        candidates = [{'k': int(len(np.unique(labels))), 'silhouette': sil, 'min_size': int(counts.min()), 'max_size': int(counts.max()), 'labels': labels.copy()}]
    best = sorted(candidates, key=lambda x: (-x['silhouette'], x['k']))[0]
    return best

def basin_medoid(M, members):
    """
    Select partition with maximum mean ARI
    to other members of the same basin.
    """
    members = np.asarray(members, dtype=int)
    sub = M[np.ix_(members, members)]
    n = len(members)
    if n == 1:
        return (int(members[0]), 1.0)
    stability = (sub.sum(axis=1) - 1.0) / (n - 1)
    local = int(np.argmax(stability))
    return (int(members[local]), float(stability[local]))

def consensus_partition(partitions, known_k, method='average'):
    """
    Exact co-assignment consensus.

    known_k is predeclared and does not come
    from ground truth during clustering.
    """
    partitions = np.asarray(partitions)
    n_runs, n_cells = partitions.shape
    realized = [len(np.unique(p)) for p in partitions]
    if any((k != known_k for k in realized)):
        raise RuntimeError('A source partition does not match predeclared known K.')
    if n_runs <= 255:
        dtype = np.uint8
    elif n_runs <= 65535:
        dtype = np.uint16
    else:
        dtype = np.uint32
    co = np.zeros((n_cells, n_cells), dtype=dtype)
    for pred in partitions:
        for c in np.unique(pred):
            idx = np.flatnonzero(pred == c)
            co[np.ix_(idx, idx)] += 1
    condensed = squareform(co, checks=False)
    del co
    dist = 1.0 - condensed.astype(np.float32) / float(n_runs)
    del condensed
    Z = linkage(dist, method=method, optimal_ordering=False)
    labels = fcluster(Z, t=known_k, criterion='maxclust') - 1
    return labels.astype(np.int32)

class DSU:

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0 for _ in range(n)]

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = (rb, ra)
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

def make_meta_components(nodes, M, threshold=0.85):
    """
    High-ARI edges are permitted only between
    different representations.
    """
    n = len(nodes)
    dsu = DSU(n)
    for i in range(n):
        for j in range(i + 1, n):
            if nodes[i].representation == nodes[j].representation:
                continue
            if M[i, j] >= threshold:
                dsu.union(i, j)
    groups = {}
    for i in range(n):
        r = dsu.find(i)
        groups.setdefault(r, []).append(i)
    return sorted(groups.values(), key=lambda x: (-len(x), min(x)))

def balanced_representatives(component, nodes, M):
    """
    One basin node per representation.

    If one representation contributes multiple basins
    to a connected component, retain the basin with the
    largest mean ARI to nodes from other representations.
    """
    by_rep = {}
    for i in component:
        rep = nodes[i].representation
        by_rep.setdefault(rep, []).append(i)
    selected = []
    for rep, idxs in by_rep.items():
        other = [j for j in component if nodes[j].representation != rep]
        if not other:
            selected.append(idxs[0])
            continue
        best_i = None
        best_score = -np.inf
        for i in idxs:
            score = float(M[i, other].mean())
            if score > best_score:
                best_score = score
                best_i = i
        selected.append(best_i)
    return sorted(selected)

class scMetaBasin:
    """
    scMetaBasin: solution-landscape meta-basin consensus
    for model-agnostic known-K single-cell clustering.

    Input
    -----
    representations:
        {
            representation_name:
                ndarray [n_cells, n_features]
        }

    Frozen selector
    ---------------
    V2 eligible:
        candidate_count >= 3

    score:
        CrossRepMean
        + 0.05 * MeanBasinFraction

    If no V2 eligible meta-basin:
        Frozen V1 fallback:
        candidate_count >= 2

    Ground truth is never used.
    """

    def __init__(self, n_clusters, seeds=range(100), max_basins=5, min_basin_size=5, basin_linkage='average', meta_threshold=0.85, meta_linkage='average'):
        self.n_clusters = int(n_clusters)
        self.seeds = np.asarray(list(seeds), dtype=int)
        self.max_basins = int(max_basins)
        self.min_basin_size = int(min_basin_size)
        self.basin_linkage = basin_linkage
        self.meta_threshold = float(meta_threshold)
        self.meta_linkage = meta_linkage

    def _solution_landscape(self, X):
        preds = []
        for seed in self.seeds:
            p = KMeans(n_clusters=self.n_clusters, n_init=1, init='k-means++', random_state=int(seed), max_iter=300).fit_predict(X)
            preds.append(p.astype(np.int32))
        return np.stack(preds)

    def fit_predict(self, representations: Dict[str, np.ndarray]):
        if len(representations) < 2:
            raise ValueError('MetaBasin requires at least two representations.')
        rep_names = sorted(representations)
        n_cells = None
        nodes: List[BasinNode] = []
        basin_rows = []
        for rep in rep_names:
            X = np.asarray(representations[rep])
            if X.ndim != 2:
                raise ValueError(f'{rep}: representation must be 2-D.')
            if n_cells is None:
                n_cells = X.shape[0]
            elif X.shape[0] != n_cells:
                raise ValueError('All representations must contain the same cells.')
            partitions = self._solution_landscape(X)
            M = pairwise_partition_ari(partitions)
            basin_solution = discover_basins(M, max_basins=self.max_basins, min_basin_size=self.min_basin_size, method=self.basin_linkage)
            assignment = basin_solution['labels']
            for basin_id in sorted(np.unique(assignment)):
                members = np.flatnonzero(assignment == basin_id)
                medoid_index, medoid_stability = basin_medoid(M, members)
                labels = partitions[medoid_index].copy()
                medoid_seed = int(self.seeds[medoid_index])
                fraction = float(len(members) / len(self.seeds))
                node_id = len(nodes)
                nodes.append(BasinNode(node_id=node_id, representation=rep, basin_id=int(basin_id), basin_fraction=fraction, medoid_seed=medoid_seed, medoid_stability=medoid_stability, labels=labels))
                basin_rows.append({'node_id': node_id, 'representation': rep, 'basin_id': int(basin_id), 'basin_size': int(len(members)), 'basin_fraction': fraction, 'medoid_seed': medoid_seed, 'medoid_stability': medoid_stability, 'selected_n_basins': int(basin_solution['k']), 'partition_silhouette': float(basin_solution['silhouette'])})
        node_partitions = np.stack([x.labels for x in nodes])
        node_M = pairwise_partition_ari(node_partitions)
        components = make_meta_components(nodes, node_M, threshold=self.meta_threshold)
        meta_rows = []
        meta_labels = {}
        for meta_id, component in enumerate(components):
            balanced = balanced_representatives(component, nodes, node_M)
            reps = sorted({nodes[i].representation for i in component})
            candidate_count = len(reps)
            candidate_fraction = float(candidate_count / len(rep_names))
            pair_values = []
            for a in range(len(balanced)):
                for b in range(a + 1, len(balanced)):
                    i = balanced[a]
                    j = balanced[b]
                    pair_values.append(float(node_M[i, j]))
            crossrep_mean = float(np.mean(pair_values)) if pair_values else np.nan
            mean_basin_fraction = float(np.mean([nodes[i].basin_fraction for i in balanced]))
            frozen_score = crossrep_mean + 0.05 * mean_basin_fraction if np.isfinite(crossrep_mean) else np.nan
            if candidate_count >= 2:
                parts = np.stack([nodes[i].labels for i in balanced])
                meta_labels[meta_id] = consensus_partition(parts, known_k=self.n_clusters, method=self.meta_linkage)
            meta_rows.append({'meta_id': meta_id, 'node_count': len(component), 'candidate_count': candidate_count, 'candidate_fraction': candidate_fraction, 'representations': '|'.join(reps), 'balanced_nodes': '|'.join([f'{nodes[i].representation}::basin{nodes[i].basin_id}' for i in balanced]), 'crossrep_mean_ARI': crossrep_mean, 'mean_basin_fraction': mean_basin_fraction, 'frozen_score': frozen_score})
        meta_df = pd.DataFrame(meta_rows)
        eligible = meta_df[(meta_df['candidate_count'] >= 3) & np.isfinite(meta_df['crossrep_mean_ARI'])].copy()
        if len(eligible):
            selection_mode = 'V2'
        else:
            eligible = meta_df[(meta_df['candidate_count'] >= 2) & np.isfinite(meta_df['crossrep_mean_ARI'])].copy()
            selection_mode = 'V1_fallback'
        if not len(eligible):
            raise RuntimeError('No selectable meta-basin with >=2 representations.')
        eligible = eligible.sort_values(['frozen_score', 'crossrep_mean_ARI', 'candidate_count', 'meta_id'], ascending=[False, False, False, True])
        selected_meta_id = int(eligible.iloc[0]['meta_id'])
        if selected_meta_id not in meta_labels:
            raise RuntimeError('Selected meta-basin has no consensus.')
        self.basin_table_ = pd.DataFrame(basin_rows)
        self.meta_table_ = meta_df
        self.selection_mode_ = selection_mode
        self.selected_meta_id_ = selected_meta_id
        self.selected_meta_ = eligible.iloc[0].to_dict()
        self.meta_labels_ = meta_labels
        self.labels_ = meta_labels[selected_meta_id]
        return self.labels_
