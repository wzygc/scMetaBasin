"""Five graph views from aligned expression and embedding matrices; no encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.neighbors import kneighbors_graph

from .io import save_representations_npz

VIEW_NAMES = (
    'pure_latent', 'gat_batch_current_k1_b1_ext',
    'pf_batch_symmetric_k1_b1_ext', 'pf_global_union_k1_b1_ext',
    'pf_global_mutual_k1_b1_ext',
)


def _torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError('Five-view construction requires scMetaBasin[fiveview].') from error
    return torch


def initialize_gat_weights(n_features, *, seed):
    """Explicit seeded initialization for a new configuration, not paper replay."""
    if isinstance(n_features, bool) or not isinstance(n_features, (int, np.integer)) or n_features < 1:
        raise ValueError('n_features must be a positive integer.')
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32).')
    torch = _torch()
    weights = {}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        for key, shape in [('W', (n_features, n_features)),
                           ('a_self', (n_features, 1)), ('a_neighs', (n_features, 1))]:
            tensor = torch.empty(shape, dtype=torch.float32, device='cpu')
            torch.nn.init.xavier_uniform_(tensor, gain=1.414)
            weights[key] = tensor.numpy().copy()
    return weights


def load_gat_weights(path):
    """Load the three attention arrays without loading an encoder/checkpoint."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in ('W', 'a_self', 'a_neighs')}


def _matrix(value, name):
    value = np.asarray(value)
    if value.ndim != 2 or min(value.shape) < 1 or not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
        raise ValueError(f'{name} must be a nonempty real numeric matrix.')
    if not np.isfinite(value).all():
        raise ValueError(f'{name} contains NaN or Inf.')
    with np.errstate(over='ignore'):
        value = value.astype(np.float32, copy=False)
    if not np.isfinite(value).all():
        raise ValueError(f'{name} is outside the float32 range.')
    return value


def _symmetric_normalize(graph):
    degree = np.asarray(graph.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree, dtype=np.float32)
    np.divide(1.0, np.sqrt(degree), out=inverse, where=degree > 0)
    diagonal = sparse.diags(inverse)
    return diagonal.dot(graph).dot(diagonal).tocsr().astype(np.float32)


def _knn(expression, neighbors):
    return kneighbors_graph(expression, min(neighbors, len(expression)), mode='connectivity',
                            metric='cosine', include_self=True).tocsr().astype(np.float32)


def build_fiveview(expression, embedding, *, gat_weights, batch_size=128, n_neighbors=30):
    """Construct the fixed one-hop views on CPU, preserving input row order.

    ``expression`` is the already-preprocessed matrix used for cosine graphs;
    no normalization, gene filtering or model training is performed here.
    GAT parameters must be supplied explicitly. Paper replay uses the archived
    parameters and inputs; a new seeded initialization defines a new configuration.
    """
    expression = _matrix(expression, 'expression')
    embedding = _matrix(embedding, 'embedding')
    if len(expression) != len(embedding) or len(embedding) < 2:
        raise ValueError('Expression and embedding must have the same >=2 cells in the same order.')
    for name, value in [('batch_size', batch_size), ('n_neighbors', n_neighbors)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    dimension = embedding.shape[1]
    weights = {}
    for key, shape in [('W', (dimension, dimension)),
                       ('a_self', (dimension, 1)), ('a_neighs', (dimension, 1))]:
        if key not in gat_weights:
            raise ValueError(f'Missing GAT parameter {key}.')
        value = _matrix(gat_weights[key], key)
        if value.shape != shape:
            raise ValueError(f'{key} must have shape {shape}; got {value.shape}.')
        weights[key] = value
    torch = _torch()
    tensors = {key: torch.from_numpy(value.copy()) for key, value in weights.items()}
    gat_parts, batch_parts = [], []
    with torch.inference_mode():
        for start in range(0, len(embedding), batch_size):
            x = expression[start:start + batch_size]
            z = embedding[start:start + batch_size]
            graph = _knn(x, n_neighbors)
            z_tensor = torch.from_numpy(z)
            h = torch.mm(z_tensor, tensors['W'])
            logits = torch.mm(h, tensors['a_self']) + torch.mm(h, tensors['a_neighs']).T
            logits = torch.nn.functional.leaky_relu(logits, negative_slope=0.3)
            mask = torch.from_numpy(graph.toarray())
            masked = torch.where(mask > 0, logits, -9e15 * torch.ones_like(mask))
            attended = torch.nn.functional.elu(torch.matmul(torch.softmax(masked, dim=1), h))
            # Keep the two residual additions in their historical order.
            gat_parts.append((attended + z_tensor).numpy())
            symmetric = ((graph + graph.T) * 0.5).tocsr()
            batch_parts.append(_symmetric_normalize(symmetric).dot(z) + z)
    graph = _knn(expression, n_neighbors)
    union = graph.maximum(graph.T).tocsr()
    mutual = graph.minimum(graph.T).tocsr()
    bank = {
        VIEW_NAMES[0]: embedding.copy(),
        VIEW_NAMES[1]: (np.concatenate(gat_parts, axis=0) + embedding).astype(np.float32),
        VIEW_NAMES[2]: np.concatenate(batch_parts, axis=0).astype(np.float32),
        VIEW_NAMES[3]: (_symmetric_normalize(union).dot(embedding) + embedding).astype(np.float32),
        VIEW_NAMES[4]: (_symmetric_normalize(mutual).dot(embedding) + embedding).astype(np.float32),
    }
    if not all(np.isfinite(value).all() for value in bank.values()):
        raise ValueError('Five-view construction produced NaN or Inf.')
    return bank


def _digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build five views from aligned expression and embedding NPY arrays.')
    parser.add_argument('--expression', type=Path, required=True)
    parser.add_argument('--embedding', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--gat-weights', type=Path, help='NPZ with W, a_self and a_neighs; use archived parameters for replay.')
    source.add_argument('--gat-seed', type=int, help='Explicit seeded parameters for a new configuration; not paper replay.')
    parser.add_argument('--cell-ids', type=Path, help='Optional aligned one-dimensional NPY array.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    metadata = output.with_suffix('.fiveview.json')
    if output.suffix.lower() != '.npz' or output.exists() or metadata.exists():
        parser.error('--output must be a new .npz path with no existing sidecar.')
    expression = np.load(args.expression, allow_pickle=False)
    embedding = np.load(args.embedding, allow_pickle=False)
    weights = (load_gat_weights(args.gat_weights) if args.gat_weights else
               initialize_gat_weights(embedding.shape[1], seed=args.gat_seed))
    cell_ids = np.load(args.cell_ids, allow_pickle=False) if args.cell_ids else None
    if cell_ids is not None and (cell_ids.ndim != 1 or len(cell_ids) != len(embedding)):
        parser.error('--cell-ids must contain one entry per row.')
    bank = build_fiveview(expression, embedding, gat_weights=weights)
    save_representations_npz(output, bank, cell_ids=cell_ids)
    info = dict(n_cells=len(embedding), embedding_dimension=embedding.shape[1],
                batch_size=128, n_neighbors=30, metric='cosine', include_self=True,
                hops=1, residual_weight=1.0, gat_negative_slope=0.3, dtype='float32',
                gat_parameter_source='supplied' if args.gat_weights else 'explicit_seed',
                gat_seed=args.gat_seed, gat_parameter_hashes={k: _digest(v) for k, v in weights.items()},
                expression_hash=_digest(np.asarray(expression, dtype=np.float32)),
                embedding_hash=_digest(np.asarray(embedding, dtype=np.float32)),
                view_hashes={k: _digest(v) for k, v in bank.items()},
                row_order='preserved; input alignment is supplied by the caller',
                encoder_training=False)
    metadata.write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    print(f'Built {len(bank)} views for {len(embedding)} cells: {output}')


if __name__ == '__main__':
    main()
