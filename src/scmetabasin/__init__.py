"""
scMetaBasin
===========

Solution-landscape meta-basin consensus for
known-K single-cell clustering.
"""
from .clusterer import scMetaBasin
from .io import load_representations_npz, save_representations_npz
__all__ = ['scMetaBasin', 'load_representations_npz', 'save_representations_npz']
__version__ = '0.1.0'
