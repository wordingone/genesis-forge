"""
Pure PyTorch sparse 3D convolution implementations.

These replace spconv when the algorithm tuner fails (common on Windows/RTX 4090).
Uses scatter/gather operations which are slower but always work.
"""
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import SparseTensor


def sparse_conv3d_pytorch(
    x: SparseTensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> SparseTensor:
    """
    Pure PyTorch implementation of sparse 3D submanifold convolution.

    This is a fallback when spconv's algorithm tuner fails.
    Uses gather-matmul-scatter pattern.

    Args:
        x: Input SparseTensor with coords [N, 4] (batch, x, y, z) and feats [N, C_in]
        weight: Conv weights [C_out, C_in, K, K, K]
        bias: Optional bias [C_out]
        kernel_size: Must be odd (typically 3)
        padding: Typically (kernel_size - 1) // 2

    Returns:
        SparseTensor with same coords, feats [N, C_out]
    """
    device = x.device
    dtype = x.feats.dtype
    N = x.feats.shape[0]
    C_in = x.feats.shape[1]
    C_out = weight.shape[0]
    K = kernel_size

    # For submanifold convolution, output coords = input coords
    coords = x.coords  # [N, 4]
    feats = x.feats    # [N, C_in]

    # Build spatial hash for neighbor lookup
    # Hash: batch * (max_coord^3) + z * max_coord^2 + y * max_coord + x
    max_coord = 512  # Should be enough for 256^3 grids
    batch_idx = coords[:, 0].long()
    xyz = coords[:, 1:4].long()

    # Create hash for each voxel
    hash_vals = (batch_idx * (max_coord ** 3) +
                 xyz[:, 2] * (max_coord ** 2) +
                 xyz[:, 1] * max_coord +
                 xyz[:, 0])

    # Create hash table (sparse to dense mapping)
    hash_to_idx = torch.full((hash_vals.max().item() + 1,), -1, dtype=torch.long, device=device)
    hash_to_idx[hash_vals] = torch.arange(N, device=device)

    # Generate kernel offsets
    half_k = K // 2
    offsets = []
    for dz in range(-half_k, half_k + 1):
        for dy in range(-half_k, half_k + 1):
            for dx in range(-half_k, half_k + 1):
                offsets.append((dx, dy, dz))
    offsets = torch.tensor(offsets, device=device, dtype=torch.long)  # [K^3, 3]

    # Gather neighbor features for each voxel
    # For each voxel, collect features from K^3 neighbors
    output = torch.zeros(N, C_out, device=device, dtype=dtype)

    # Process in chunks to manage memory
    chunk_size = min(N, 10000)
    weight_flat = weight.view(C_out, C_in, -1)  # [C_out, C_in, K^3]

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_coords = coords[start:end]  # [chunk, 4]
        chunk_batch = chunk_coords[:, 0:1].long()  # [chunk, 1]
        chunk_xyz = chunk_coords[:, 1:4].long()    # [chunk, 3]
        chunk_size_actual = end - start

        # Compute neighbor coords for all voxels in chunk
        # neighbor_xyz: [chunk, K^3, 3]
        neighbor_xyz = chunk_xyz.unsqueeze(1) + offsets.unsqueeze(0)
        neighbor_batch = chunk_batch.expand(-1, K**3)  # [chunk, K^3]

        # Clamp to valid range (handles boundary)
        neighbor_xyz = neighbor_xyz.clamp(0, max_coord - 1)

        # Compute hash for neighbors
        neighbor_hash = (neighbor_batch * (max_coord ** 3) +
                        neighbor_xyz[:, :, 2] * (max_coord ** 2) +
                        neighbor_xyz[:, :, 1] * max_coord +
                        neighbor_xyz[:, :, 0])  # [chunk, K^3]

        # Clamp hash to valid range
        neighbor_hash = neighbor_hash.clamp(0, hash_to_idx.shape[0] - 1)

        # Look up neighbor indices
        neighbor_idx = hash_to_idx[neighbor_hash]  # [chunk, K^3]

        # Create mask for valid neighbors (idx != -1)
        valid_mask = neighbor_idx >= 0  # [chunk, K^3]

        # Gather neighbor features
        # Replace invalid indices with 0 for gathering, then mask out
        safe_idx = neighbor_idx.clamp(min=0)  # [chunk, K^3]
        neighbor_feats = feats[safe_idx]  # [chunk, K^3, C_in]
        neighbor_feats = neighbor_feats * valid_mask.unsqueeze(-1).float()  # Zero out invalid

        # Apply convolution: sum over neighbors with weights
        # neighbor_feats: [chunk, K^3, C_in]
        # weight_flat: [C_out, C_in, K^3]
        # output: [chunk, C_out]
        chunk_output = torch.einsum('nkc,ock->no', neighbor_feats.float(), weight_flat.float())
        output[start:end] = chunk_output.to(dtype)

    if bias is not None:
        output = output + bias.unsqueeze(0)

    return x.replace(output)


class SparseSubMConv3dPyTorch(nn.Module):
    """
    Pure PyTorch submanifold sparse convolution.

    Drop-in replacement for spconv.SubMConv3d when algorithm tuner fails.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = True,
        indice_key: str = None,  # Ignored (spconv compatibility)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # Initialize weights like PyTorch Conv3d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Either SparseTensor or spconv.SparseConvTensor

        Returns:
            Same type as input with convolved features
        """
        # Handle spconv.SparseConvTensor input (from flexi_decoder)
        if hasattr(x, 'features') and hasattr(x, 'indices'):
            # This is a spconv tensor, convert to our SparseTensor
            sparse_x = SparseTensor(
                feats=x.features,
                coords=x.indices,
                shape=x.spatial_shape,
            )
            result = sparse_conv3d_pytorch(
                sparse_x,
                self.weight,
                self.bias,
                self.kernel_size,
                padding=self.padding
            )
            # Convert back to spconv format
            import spconv.pytorch as spconv
            return spconv.SparseConvTensor(
                result.feats,
                x.indices,
                x.spatial_shape,
                x.batch_size,
            )
        else:
            # Standard SparseTensor
            return sparse_conv3d_pytorch(
                x,
                self.weight,
                self.bias,
                self.kernel_size,
                padding=self.padding
            )


def get_sparse_conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    padding: int = 1,
    bias: bool = True,
    indice_key: str = None,
    force_pytorch: bool = False,
) -> nn.Module:
    """
    Factory function to get appropriate sparse conv implementation.

    Tries spconv first, falls back to PyTorch if it fails.

    Args:
        force_pytorch: If True, always use PyTorch implementation

    Returns:
        Sparse conv module (either spconv.SubMConv3d or SparseSubMConv3dPyTorch)
    """
    if force_pytorch:
        return SparseSubMConv3dPyTorch(
            in_channels, out_channels, kernel_size, padding, bias, indice_key
        )

    try:
        import spconv.pytorch as spconv
        # Test if spconv works with a small tensor
        test_tensor = spconv.SparseConvTensor(
            torch.randn(8, in_channels, device='cuda'),
            torch.tensor([[0,1,1,1], [0,1,1,2], [0,1,2,1], [0,1,2,2],
                         [0,2,1,1], [0,2,1,2], [0,2,2,1], [0,2,2,2]], dtype=torch.int32, device='cuda'),
            [4, 4, 4],
            1,
        )
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, indice_key=indice_key)
        conv = conv.cuda()
        _ = conv(test_tensor)  # This triggers algorithm tuning
        return conv
    except Exception as e:
        print(f"[SPARSE CONV] spconv failed ({e}), using PyTorch fallback")
        return SparseSubMConv3dPyTorch(
            in_channels, out_channels, kernel_size, padding, bias, indice_key
        )
