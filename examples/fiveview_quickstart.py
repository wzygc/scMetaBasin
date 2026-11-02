"""Synthetic execution example; no encoder, checkpoints or project scripts."""
import argparse
from pathlib import Path

import numpy as np

from scmetabasin import scMetaBasin, save_representations_npz
from scmetabasin.fiveview import build_fiveview, initialize_gat_weights

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    parser.error('Choose a new output directory.')
rng = np.random.RandomState(42)
centers = np.eye(3, 8, dtype=np.float32) * 5
embedding = np.repeat(centers, 30, axis=0) + rng.normal(0, 0.03, size=(90, 8)).astype(np.float32)
expression = np.abs(embedding) + 0.1
bank = build_fiveview(expression, embedding,
                      gat_weights=initialize_gat_weights(8, seed=42))
labels = scMetaBasin(n_clusters=3).fit_predict(bank)
assert len(labels) == 90 and len(np.unique(labels)) == 3
args.output.mkdir(parents=True)
save_representations_npz(args.output / 'bank.npz', bank)
np.save(args.output / 'cluster_labels.npy', labels)
print('Independent five-view example: 90 cells, five views, three output clusters.')
