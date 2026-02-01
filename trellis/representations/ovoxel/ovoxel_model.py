"""
O-Voxel (Ordered Voxel) Model - TRELLIS.2 sparse voxel representation with PBR.

This representation stores per-voxel PBR material attributes directly,
eliminating the need for neural field decoding at inference time.

Attribute layout (7 channels per voxel):
    - RGB (3): Base color in linear space
    - M (1): Metallic factor [0, 1]
    - R (1): Roughness factor [0, 1]
    - O (1): Opacity/alpha [0, 1]
    - SDF (1): Signed distance field value (optional, for mesh extraction)
"""
from typing import Optional, Tuple, List
import torch
import torch.nn.functional as F
import numpy as np


class OVoxel:
    """
    Field-free sparse voxels with per-voxel PBR attributes.

    TRELLIS.2 uses this representation for the material generation stage (Stage 3),
    where each voxel directly encodes PBR properties without neural field queries.

    Attributes:
        coords: [N, 4] tensor of (batch_idx, x, y, z) voxel coordinates (int32)
        features: [N, C] tensor of per-voxel features (float32)
            Default layout: RGB(3) + M(1) + R(1) + O(1) + SDF(1) = 7 channels
        resolution: Voxel grid resolution (typically 64 or 128)
        device: Target device
    """

    # Default feature layout
    FEATURE_CHANNELS = 7
    CHANNEL_LAYOUT = {
        'rgb': (0, 3),      # Base color (linear RGB)
        'metallic': (3, 4),
        'roughness': (4, 5),
        'opacity': (5, 6),
        'sdf': (6, 7),      # Optional SDF for mesh extraction
    }

    def __init__(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        resolution: int = 64,
        device: str = 'cuda'
    ):
        """
        Initialize OVoxel representation.

        Args:
            coords: [N, 4] voxel coordinates (batch_idx, x, y, z)
            features: [N, C] per-voxel features
            resolution: Grid resolution
            device: Target device
        """
        self.device = device
        self.resolution = resolution

        # Ensure proper types and devices
        self.coords = coords.to(device).int()
        self.features = features.to(device).float()

        # Validate shapes
        assert self.coords.shape[1] == 4, f"coords must be [N, 4], got {self.coords.shape}"
        assert self.coords.shape[0] == self.features.shape[0], \
            f"coords and features must have same N, got {self.coords.shape[0]} vs {self.features.shape[0]}"

    @classmethod
    def from_sparse_tensor(cls, sparse_tensor, resolution: int = 64) -> 'OVoxel':
        """
        Create OVoxel from a SparseTensor (from TRELLIS sparse module).

        Args:
            sparse_tensor: SparseTensor with coords and feats attributes
            resolution: Grid resolution

        Returns:
            OVoxel instance
        """
        return cls(
            coords=sparse_tensor.coords,
            features=sparse_tensor.feats,
            resolution=resolution,
            device=str(sparse_tensor.feats.device)
        )

    @property
    def num_voxels(self) -> int:
        """Number of active voxels."""
        return self.coords.shape[0]

    @property
    def batch_size(self) -> int:
        """Number of batches."""
        if self.num_voxels == 0:
            return 0
        return int(self.coords[:, 0].max().item()) + 1

    @property
    def num_channels(self) -> int:
        """Number of feature channels."""
        return self.features.shape[1]

    # ========== PBR Attribute Properties ==========

    @property
    def base_color(self) -> torch.Tensor:
        """
        Get base color (RGB) for all voxels.

        Returns:
            [N, 3] tensor of RGB values in linear space [0, 1]
        """
        start, end = self.CHANNEL_LAYOUT['rgb']
        return torch.sigmoid(self.features[:, start:end])

    @property
    def metallic(self) -> torch.Tensor:
        """
        Get metallic factor for all voxels.

        Returns:
            [N, 1] tensor of metallic values [0, 1]
        """
        start, end = self.CHANNEL_LAYOUT['metallic']
        return torch.sigmoid(self.features[:, start:end])

    @property
    def roughness(self) -> torch.Tensor:
        """
        Get roughness factor for all voxels.

        Returns:
            [N, 1] tensor of roughness values [0, 1]
        """
        start, end = self.CHANNEL_LAYOUT['roughness']
        return torch.sigmoid(self.features[:, start:end])

    @property
    def opacity(self) -> torch.Tensor:
        """
        Get opacity (alpha) for all voxels.

        Returns:
            [N, 1] tensor of opacity values [0, 1]
        """
        start, end = self.CHANNEL_LAYOUT['opacity']
        return torch.sigmoid(self.features[:, start:end])

    @property
    def sdf(self) -> torch.Tensor:
        """
        Get SDF values for all voxels (optional, for mesh extraction).

        Returns:
            [N, 1] tensor of SDF values (raw, not activated)
        """
        start, end = self.CHANNEL_LAYOUT['sdf']
        if end <= self.num_channels:
            return self.features[:, start:end]
        return torch.zeros(self.num_voxels, 1, device=self.device)

    # ========== Voxel Grid Conversion ==========

    def to_voxel_grid(self, batch_idx: int = 0) -> torch.Tensor:
        """
        Convert sparse voxels to dense voxel grid for texture baking.

        Args:
            batch_idx: Which batch to convert (default: 0)

        Returns:
            [6, D, H, W] tensor containing:
                - channels 0-2: base_color (RGB)
                - channel 3: metallic
                - channel 4: roughness
                - channel 5: opacity
        """
        D = H = W = self.resolution

        # Initialize empty grid
        grid = torch.zeros(6, D, H, W, device=self.device, dtype=torch.float32)

        # Get voxels for this batch
        batch_mask = self.coords[:, 0] == batch_idx
        if not batch_mask.any():
            return grid

        batch_coords = self.coords[batch_mask][:, 1:]  # [M, 3] xyz coordinates
        batch_features = self.features[batch_mask]  # [M, C]

        # Get activated PBR values
        rgb = torch.sigmoid(batch_features[:, 0:3])
        metallic = torch.sigmoid(batch_features[:, 3:4])
        roughness = torch.sigmoid(batch_features[:, 4:5])
        opacity = torch.sigmoid(batch_features[:, 5:6])

        # Clamp coordinates to valid range
        x = batch_coords[:, 0].clamp(0, W - 1).long()
        y = batch_coords[:, 1].clamp(0, H - 1).long()
        z = batch_coords[:, 2].clamp(0, D - 1).long()

        # Scatter values into grid
        grid[0:3, z, y, x] = rgb.T  # RGB
        grid[3, z, y, x] = metallic.squeeze(-1)  # Metallic
        grid[4, z, y, x] = roughness.squeeze(-1)  # Roughness
        grid[5, z, y, x] = opacity.squeeze(-1)  # Opacity

        return grid

    def to_voxel_grid_batched(self) -> torch.Tensor:
        """
        Convert all batches to dense voxel grids.

        Returns:
            [B, 6, D, H, W] tensor for all batches
        """
        B = self.batch_size
        if B == 0:
            return torch.zeros(1, 6, self.resolution, self.resolution, self.resolution,
                             device=self.device, dtype=torch.float32)

        grids = []
        for b in range(B):
            grids.append(self.to_voxel_grid(b))

        return torch.stack(grids, dim=0)

    # ========== Index Operations ==========

    def get_batch(self, batch_idx: int) -> 'OVoxel':
        """
        Get voxels for a specific batch.

        Args:
            batch_idx: Batch index

        Returns:
            New OVoxel containing only voxels from the specified batch
        """
        mask = self.coords[:, 0] == batch_idx
        new_coords = self.coords[mask].clone()
        new_coords[:, 0] = 0  # Reset batch index
        new_features = self.features[mask]
        return OVoxel(new_coords, new_features, self.resolution, self.device)

    def __getitem__(self, idx: int) -> 'OVoxel':
        """Index by batch."""
        return self.get_batch(idx)

    def __len__(self) -> int:
        """Return batch size."""
        return self.batch_size

    # ========== Utility Methods ==========

    def to(self, device: str) -> 'OVoxel':
        """Move to device."""
        return OVoxel(
            self.coords.to(device),
            self.features.to(device),
            self.resolution,
            device
        )

    def clone(self) -> 'OVoxel':
        """Create a copy."""
        return OVoxel(
            self.coords.clone(),
            self.features.clone(),
            self.resolution,
            self.device
        )

    def get_pbr_dict(self, batch_idx: int = 0) -> dict:
        """
        Get PBR attributes as a dictionary (for debugging/export).

        Args:
            batch_idx: Which batch to extract

        Returns:
            Dictionary with 'base_color', 'metallic', 'roughness', 'opacity' arrays
        """
        batch_mask = self.coords[:, 0] == batch_idx
        batch_features = self.features[batch_mask]

        return {
            'base_color': torch.sigmoid(batch_features[:, 0:3]).cpu().numpy(),
            'metallic': torch.sigmoid(batch_features[:, 3:4]).cpu().numpy(),
            'roughness': torch.sigmoid(batch_features[:, 4:5]).cpu().numpy(),
            'opacity': torch.sigmoid(batch_features[:, 5:6]).cpu().numpy(),
            'coords': self.coords[batch_mask][:, 1:].cpu().numpy(),
        }

    def __repr__(self) -> str:
        return (
            f"OVoxel(num_voxels={self.num_voxels}, "
            f"batch_size={self.batch_size}, "
            f"channels={self.num_channels}, "
            f"resolution={self.resolution}, "
            f"device='{self.device}')"
        )
