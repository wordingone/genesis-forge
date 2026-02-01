from . import config
from .conv import SparseConv3d, SparseInverseConv3d

# For backwards compatibility - print algo
print(f"[SPARSE][CONV] spconv algo: {config.SPCONV_ALGO}")

__all__ = ['SparseConv3d', 'SparseInverseConv3d']
