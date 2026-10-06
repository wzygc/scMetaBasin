from __future__ import annotations
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

def sha256_file(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def json_safe(x):
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return [json_safe(v) for v in x.tolist()]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        if not np.isfinite(x):
            return None
        return float(x)
    if isinstance(x, float):
        if not np.isfinite(x):
            return None
        return x
    if isinstance(x, np.bool_):
        return bool(x)
    return x

def save_scmetabasin_results(clusterer, output_dir, cell_ids=None, input_path=None, extra_summary=None):
    """
    Save standard scMetaBasin result bundle.
    """
    output_dir = Path(output_dir)
    labels = np.asarray(clusterer.labels_).reshape(-1)
    n = len(labels)
    if cell_ids is None:
        cell_ids = np.arange(n).astype(str)
    else:
        cell_ids = np.asarray(cell_ids).reshape(-1)
        if len(cell_ids) != n:
            raise ValueError('cell_ids length mismatch during result saving')
        cell_ids = cell_ids.astype(str)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_df = pd.DataFrame({'cell_index': np.arange(n, dtype=int), 'cell_id': cell_ids, 'cluster': labels.astype(int)})
    label_path = output_dir / 'cluster_labels.csv'
    label_df.to_csv(label_path, index=False, encoding='utf-8-sig')
    basin_path = output_dir / 'basin_table.csv'
    clusterer.basin_table_.to_csv(basin_path, index=False, encoding='utf-8-sig')
    meta_path = output_dir / 'meta_basin_table.csv'
    clusterer.meta_table_.to_csv(meta_path, index=False, encoding='utf-8-sig')
    selected_path = output_dir / 'selected_meta.json'
    selected_obj = {'selection_mode': clusterer.selection_mode_, 'selected_meta_id': int(clusterer.selected_meta_id_), 'selected_meta': json_safe(clusterer.selected_meta_)}
    selected_path.write_text(json.dumps(selected_obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    unique, counts = np.unique(labels, return_counts=True)
    summary = {'algorithm': 'scMetaBasin', 'version': '0.1.0', 'known_K': int(clusterer.n_clusters), 'n_cells': int(n), 'realized_K': int(len(unique)), 'selection_mode': str(clusterer.selection_mode_), 'selected_meta_id': int(clusterer.selected_meta_id_), 'n_discovered_basins': int(len(clusterer.basin_table_)), 'n_meta_basins': int(len(clusterer.meta_table_)), 'cluster_sizes': {str(int(k)): int(v) for k, v in zip(unique, counts)}, 'ground_truth_used': False}
    if input_path is not None:
        input_path = Path(input_path)
        summary['input_file'] = str(input_path.resolve())
        summary['input_sha256'] = sha256_file(input_path)
    if extra_summary:
        summary.update(json_safe(extra_summary))
    summary_path = output_dir / 'scmetabasin_summary.json'
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    files = [label_path, basin_path, meta_path, selected_path, summary_path]
    hashes = []
    for p in files:
        hashes.append({'file': p.name, 'sha256': sha256_file(p), 'size_bytes': int(p.stat().st_size)})
    hash_path = output_dir / 'file_hashes.csv'
    pd.DataFrame(hashes).to_csv(hash_path, index=False, encoding='utf-8-sig')
    return {'cluster_labels': label_path, 'basin_table': basin_path, 'meta_basin_table': meta_path, 'selected_meta': selected_path, 'summary': summary_path, 'hashes': hash_path}
