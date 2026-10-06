from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
RESERVED_KEYS = {'cell_ids', '__scmetabasin_format__'}

def validate_representations(representations: Dict[str, np.ndarray]) -> int:
    """
    Validate representation bank.

    Returns
    -------
    n_cells : int
    """
    if not isinstance(representations, dict):
        raise TypeError('representations must be a dict: {name: ndarray}')
    if len(representations) < 2:
        raise ValueError('scMetaBasin requires at least two representations.')
    n_cells = None
    for name, X in representations.items():
        if not isinstance(name, str):
            raise TypeError('representation names must be strings')
        if not name.strip():
            raise ValueError('representation name cannot be empty')
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f'{name}: expected 2-D array, got shape {X.shape}')
        if X.shape[0] < 2:
            raise ValueError(f'{name}: too few cells')
        if X.shape[1] < 1:
            raise ValueError(f'{name}: zero feature dimension')
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError(f'{name}: representation must be numeric, got {X.dtype}')
        if not np.isfinite(X).all():
            raise ValueError(f'{name}: contains NaN or Inf')
        if n_cells is None:
            n_cells = int(X.shape[0])
        elif X.shape[0] != n_cells:
            raise ValueError(f'{name}: n_cells={X.shape[0]} != expected {n_cells}')
    return int(n_cells)

def load_representations_npz(path) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    """
    Standard scMetaBasin NPZ input.

    Required:
        >=2 2-D representation arrays.

    Optional reserved key:
        cell_ids : shape [n_cells]

    Example keys:
        pure_latent
        gat_batch_current_k1_b1_ext
        pf_batch_symmetric_k1_b1_ext
        pf_global_union_k1_b1_ext
        pf_global_mutual_k1_b1_ext
        cell_ids
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != '.npz':
        raise ValueError('Input must be a .npz file')
    representations = {}
    cell_ids = None
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)
        if 'cell_ids' in keys:
            cell_ids = np.asarray(z['cell_ids'])
        for key in keys:
            if key in RESERVED_KEYS:
                continue
            X = np.asarray(z[key])
            representations[str(key)] = X
    n_cells = validate_representations(representations)
    if cell_ids is not None:
        cell_ids = np.asarray(cell_ids).reshape(-1)
        if len(cell_ids) != n_cells:
            raise ValueError('cell_ids length does not match representation rows')
        cell_ids = cell_ids.astype(str)
    return (representations, cell_ids)

def save_representations_npz(path, representations, cell_ids=None):
    """
    Save a standard scMetaBasin input NPZ.
    """
    path = Path(path)
    n_cells = validate_representations(representations)
    payload = {str(k): np.asarray(v) for k, v in representations.items()}
    if cell_ids is not None:
        cell_ids = np.asarray(cell_ids).reshape(-1)
        if len(cell_ids) != n_cells:
            raise ValueError('cell_ids length mismatch')
        payload['cell_ids'] = cell_ids.astype(str)
    payload['__scmetabasin_format__'] = np.asarray(['scMetaBasin_npz_v1'])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path
