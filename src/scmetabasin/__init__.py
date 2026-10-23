"""
scMetaBasin
===========

Solution-landscape meta-basin consensus for
single-cell clustering with optional automatic K.
"""
from .clusterer import scMetaBasin
from .io import load_representations, load_representations_npz, save_representations_npz
__all__ = ['scMetaBasin', 'load_representations', 'load_representations_npz', 'save_representations_npz']
__version__ = '0.2.0'
