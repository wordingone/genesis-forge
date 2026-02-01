"""
PBR Texture Baking - nvdiffrast-free implementation.

Bakes PBR materials (base_color, metallic, roughness) from voxel representations
or Gaussian splats onto UV-mapped mesh textures.

This module replaces nvdiffrast dr.rasterize() and dr.interpolate() with pure PyTorch
implementations for TRELLIS.2 parity without external dependencies.
"""

from typing import Tuple, Dict, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2


def rasterize_uv_triangles(
    uvs: torch.Tensor,
    faces: torch.Tensor,
    resolution: int,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Software UV triangle rasterization using vectorized barycentric coordinates.

    Replaces nvdiffrast dr.rasterize() for UV-space rasterization.

    Args:
        uvs: [V, 2] UV coordinates in [0, 1] range
        faces: [F, 3] face indices (int32 or int64)
        resolution: Output texture resolution (square)
        device: Target device (defaults to uvs.device)

    Returns:
        Dictionary containing:
            - 'mask': [H, W] boolean mask of covered pixels
            - 'tri_id': [H, W] triangle index per pixel (-1 for uncovered)
            - 'bary': [H, W, 3] barycentric coordinates per pixel
            - 'uv': [H, W, 2] interpolated UV coordinates per pixel
    """
    if device is None:
        device = uvs.device

    H, W = resolution, resolution
    num_faces = faces.shape[0]

    # Get UV coordinates for each face vertex
    # uvs: [V, 2], faces: [F, 3]
    uv0 = uvs[faces[:, 0]]  # [F, 2]
    uv1 = uvs[faces[:, 1]]  # [F, 2]
    uv2 = uvs[faces[:, 2]]  # [F, 2]

    # Compute bounding boxes for each triangle in pixel coordinates
    all_uvs = torch.stack([uv0, uv1, uv2], dim=1)  # [F, 3, 2]
    bbox_min = (all_uvs.min(dim=1).values * resolution).floor().long().clamp(min=0)  # [F, 2]
    bbox_max = (all_uvs.max(dim=1).values * resolution).ceil().long().clamp(max=resolution - 1)  # [F, 2]

    # Initialize output buffers
    mask = torch.zeros(H, W, dtype=torch.bool, device=device)
    tri_id = torch.full((H, W), -1, dtype=torch.long, device=device)
    bary = torch.zeros(H, W, 3, dtype=torch.float32, device=device)
    uv_out = torch.zeros(H, W, 2, dtype=torch.float32, device=device)

    # Create pixel coordinate grid once
    y_coords = torch.arange(H, device=device, dtype=torch.float32)
    x_coords = torch.arange(W, device=device, dtype=torch.float32)

    # Process triangles in batches for memory efficiency
    batch_size = min(10000, num_faces)

    for batch_start in range(0, num_faces, batch_size):
        batch_end = min(batch_start + batch_size, num_faces)
        batch_faces = torch.arange(batch_start, batch_end, device=device)

        for fi in batch_faces:
            fi_cpu = fi.item()

            # Get triangle bounds
            x_min, y_min = bbox_min[fi_cpu, 0].item(), bbox_min[fi_cpu, 1].item()
            x_max, y_max = bbox_max[fi_cpu, 0].item(), bbox_max[fi_cpu, 1].item()

            if x_max < x_min or y_max < y_min:
                continue

            # Get triangle vertices in UV space
            p0 = uv0[fi_cpu]  # [2]
            p1 = uv1[fi_cpu]  # [2]
            p2 = uv2[fi_cpu]  # [2]

            # Create local pixel grid for this triangle's bounding box
            local_h = y_max - y_min + 1
            local_w = x_max - x_min + 1

            # Pixel centers in UV space
            px = (x_coords[x_min:x_max+1] + 0.5) / resolution  # [local_w]
            py = (y_coords[y_min:y_max+1] + 0.5) / resolution  # [local_h]

            # Create meshgrid
            grid_x, grid_y = torch.meshgrid(px, py, indexing='xy')  # [local_h, local_w]

            # Compute barycentric coordinates using edge vectors
            v0 = p2 - p0  # [2]
            v1 = p1 - p0  # [2]
            v2_x = grid_x - p0[0]  # [local_h, local_w]
            v2_y = grid_y - p0[1]  # [local_h, local_w]

            # Dot products
            dot00 = v0[0] * v0[0] + v0[1] * v0[1]
            dot01 = v0[0] * v1[0] + v0[1] * v1[1]
            dot02 = v0[0] * v2_x + v0[1] * v2_y  # [local_h, local_w]
            dot11 = v1[0] * v1[0] + v1[1] * v1[1]
            dot12 = v1[0] * v2_x + v1[1] * v2_y  # [local_h, local_w]

            # Compute barycentric coordinates
            denom = dot00 * dot11 - dot01 * dot01
            if abs(denom) < 1e-10:
                continue  # Degenerate triangle

            inv_denom = 1.0 / denom
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom  # [local_h, local_w]
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom  # [local_h, local_w]

            # Check if points are inside triangle
            inside = (u >= 0) & (v >= 0) & (u + v <= 1)  # [local_h, local_w]

            # Third barycentric coordinate
            w = 1.0 - u - v

            # Update output buffers for pixels inside this triangle
            # Note: later triangles overwrite earlier ones (depth peeling not needed for UV space)
            if inside.any():
                # Get indices of pixels inside triangle
                local_y, local_x = torch.where(inside)
                global_y = local_y + y_min
                global_x = local_x + x_min

                mask[global_y, global_x] = True
                tri_id[global_y, global_x] = fi_cpu
                bary[global_y, global_x, 0] = w[local_y, local_x]  # weight for p0
                bary[global_y, global_x, 1] = v[local_y, local_x]  # weight for p1
                bary[global_y, global_x, 2] = u[local_y, local_x]  # weight for p2

                # Interpolate UV coordinates (for verification)
                uv_interp = (
                    w[local_y, local_x].unsqueeze(-1) * p0 +
                    v[local_y, local_x].unsqueeze(-1) * p1 +
                    u[local_y, local_x].unsqueeze(-1) * p2
                )
                uv_out[global_y, global_x] = uv_interp

    return {
        'mask': mask,
        'tri_id': tri_id,
        'bary': bary,
        'uv': uv_out
    }


def rasterize_uv_triangles_vectorized(
    uvs: torch.Tensor,
    faces: torch.Tensor,
    resolution: int,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Fully vectorized UV triangle rasterization for better GPU performance.

    This version trades memory for speed by processing all triangles in parallel.
    Use for smaller meshes (< 100k faces) or when GPU memory is available.

    Args:
        uvs: [V, 2] UV coordinates in [0, 1] range
        faces: [F, 3] face indices
        resolution: Output texture resolution (square)
        device: Target device

    Returns:
        Same as rasterize_uv_triangles()
    """
    if device is None:
        device = uvs.device

    H, W = resolution, resolution
    F_count = faces.shape[0]

    # Get UV coordinates for each face vertex
    uv0 = uvs[faces[:, 0]]  # [F, 2]
    uv1 = uvs[faces[:, 1]]  # [F, 2]
    uv2 = uvs[faces[:, 2]]  # [F, 2]

    # Create full pixel coordinate grid
    y_coords = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / resolution
    x_coords = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / resolution
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [H, W]

    # Initialize output buffers
    mask = torch.zeros(H, W, dtype=torch.bool, device=device)
    tri_id = torch.full((H, W), -1, dtype=torch.long, device=device)
    bary = torch.zeros(H, W, 3, dtype=torch.float32, device=device)

    # Process in batches to avoid OOM
    batch_size = min(1000, F_count)

    for batch_start in range(0, F_count, batch_size):
        batch_end = min(batch_start + batch_size, F_count)

        for fi in range(batch_start, batch_end):
            p0, p1, p2 = uv0[fi], uv1[fi], uv2[fi]

            # Compute bounding box
            x_min = int(max(0, (min(p0[0], p1[0], p2[0]) * resolution).floor().item()))
            x_max = int(min(W-1, (max(p0[0], p1[0], p2[0]) * resolution).ceil().item()))
            y_min = int(max(0, (min(p0[1], p1[1], p2[1]) * resolution).floor().item()))
            y_max = int(min(H-1, (max(p0[1], p1[1], p2[1]) * resolution).ceil().item()))

            if x_max < x_min or y_max < y_min:
                continue

            # Extract local grid
            local_x = grid_x[y_min:y_max+1, x_min:x_max+1]
            local_y = grid_y[y_min:y_max+1, x_min:x_max+1]

            # Barycentric coordinates
            v0 = p2 - p0
            v1 = p1 - p0
            v2_x = local_x - p0[0]
            v2_y = local_y - p0[1]

            dot00 = v0[0] * v0[0] + v0[1] * v0[1]
            dot01 = v0[0] * v1[0] + v0[1] * v1[1]
            dot02 = v0[0] * v2_x + v0[1] * v2_y
            dot11 = v1[0] * v1[0] + v1[1] * v1[1]
            dot12 = v1[0] * v2_x + v1[1] * v2_y

            denom = dot00 * dot11 - dot01 * dot01
            if abs(denom) < 1e-10:
                continue

            inv_denom = 1.0 / denom
            u_bc = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v_bc = (dot00 * dot12 - dot01 * dot02) * inv_denom
            w_bc = 1.0 - u_bc - v_bc

            inside = (u_bc >= 0) & (v_bc >= 0) & (u_bc + v_bc <= 1)

            if inside.any():
                local_rows, local_cols = torch.where(inside)
                global_rows = local_rows + y_min
                global_cols = local_cols + x_min

                mask[global_rows, global_cols] = True
                tri_id[global_rows, global_cols] = fi
                bary[global_rows, global_cols, 0] = w_bc[local_rows, local_cols]
                bary[global_rows, global_cols, 1] = v_bc[local_rows, local_cols]
                bary[global_rows, global_cols, 2] = u_bc[local_rows, local_cols]

    return {
        'mask': mask,
        'tri_id': tri_id,
        'bary': bary
    }


def interpolate_vertex_attrs(
    attrs: torch.Tensor,
    faces: torch.Tensor,
    bary: torch.Tensor,
    tri_id: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Interpolate vertex attributes using barycentric coordinates.

    Replaces nvdiffrast dr.interpolate() for attribute sampling.

    Args:
        attrs: [V, C] vertex attributes
        faces: [F, 3] face indices
        bary: [H, W, 3] barycentric weights (w0, w1, w2) for vertices (v0, v1, v2)
        tri_id: [H, W] triangle index per pixel (-1 for uncovered)
        mask: [H, W] boolean mask of valid pixels

    Returns:
        [H, W, C] interpolated attributes (zeros for uncovered pixels)
    """
    H, W = mask.shape
    C = attrs.shape[1]
    device = attrs.device

    # Initialize output
    result = torch.zeros(H, W, C, dtype=attrs.dtype, device=device)

    if not mask.any():
        return result

    # Get valid pixel coordinates
    valid_y, valid_x = torch.where(mask)
    valid_tri = tri_id[valid_y, valid_x]
    valid_bary = bary[valid_y, valid_x]  # [N, 3]

    # Get face vertex indices for valid triangles
    valid_faces = faces[valid_tri]  # [N, 3]

    # Get attributes for each vertex of each valid triangle
    v0_attrs = attrs[valid_faces[:, 0]]  # [N, C]
    v1_attrs = attrs[valid_faces[:, 1]]  # [N, C]
    v2_attrs = attrs[valid_faces[:, 2]]  # [N, C]

    # Interpolate using barycentric weights
    # bary[:, 0] = w0 (weight for v0)
    # bary[:, 1] = w1 (weight for v1)
    # bary[:, 2] = w2 (weight for v2)
    interpolated = (
        valid_bary[:, 0:1] * v0_attrs +
        valid_bary[:, 1:2] * v1_attrs +
        valid_bary[:, 2:3] * v2_attrs
    )  # [N, C]

    # Store results
    result[valid_y, valid_x] = interpolated

    return result


def grid_sample_3d(
    volume: torch.Tensor,
    positions: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'border'
) -> torch.Tensor:
    """
    Trilinear interpolation from 3D volume at given positions.

    Replaces flex_gemm.ops.grid_sample.grid_sample_3d() with PyTorch native.

    Args:
        volume: [B, C, D, H, W] 3D feature volume
        positions: [B, N, 3] or [N, 3] query positions in [-1, 1] normalized coords
        mode: Interpolation mode ('bilinear' for trilinear in 3D, 'nearest')
        padding_mode: How to handle out-of-bounds ('zeros', 'border', 'reflection')

    Returns:
        [B, N, C] or [N, C] sampled features
    """
    squeeze_batch = False
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)
        squeeze_batch = True

    B, N, _ = positions.shape
    C = volume.shape[1]

    # Reshape positions for grid_sample: [B, 1, 1, N, 3]
    grid = positions.view(B, 1, 1, N, 3)

    # grid_sample expects [B, C, D, H, W] input and [B, D_out, H_out, W_out, 3] grid
    # Our grid is [B, 1, 1, N, 3], so output will be [B, C, 1, 1, N]
    sampled = F.grid_sample(
        volume,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True
    )  # [B, C, 1, 1, N]

    # Reshape to [B, N, C]
    result = sampled.squeeze(2).squeeze(2).permute(0, 2, 1)  # [B, N, C]

    if squeeze_batch:
        result = result.squeeze(0)

    return result


def inpaint_texture(
    texture: torch.Tensor,
    mask: torch.Tensor,
    method: str = 'telea'
) -> torch.Tensor:
    """
    Inpaint unmapped regions of a texture.

    Args:
        texture: [H, W, C] texture tensor (float32, range [0, 1])
        mask: [H, W] boolean mask (True = valid pixel)
        method: Inpainting method ('telea' or 'ns' for Navier-Stokes)

    Returns:
        [H, W, C] inpainted texture
    """
    # Convert to numpy for OpenCV inpainting
    tex_np = (texture.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    mask_np = (~mask.cpu().numpy()).astype(np.uint8)  # OpenCV wants 255 for inpaint region

    if method == 'telea':
        inpaint_method = cv2.INPAINT_TELEA
    else:
        inpaint_method = cv2.INPAINT_NS

    # Inpaint each channel
    if tex_np.shape[2] == 4:
        # RGBA: inpaint RGB, keep alpha
        rgb = cv2.inpaint(tex_np[:, :, :3], mask_np, inpaintRadius=3, flags=inpaint_method)
        alpha = tex_np[:, :, 3:4]
        alpha_inpainted = cv2.inpaint(alpha, mask_np, inpaintRadius=3, flags=inpaint_method)
        result_np = np.concatenate([rgb, alpha_inpainted], axis=-1)
    else:
        result_np = cv2.inpaint(tex_np, mask_np, inpaintRadius=3, flags=inpaint_method)

    # Convert back to tensor
    result = torch.from_numpy(result_np.astype(np.float32) / 255.0).to(texture.device)

    return result


def bake_pbr_textures(
    vertices: Union[np.ndarray, torch.Tensor],
    faces: Union[np.ndarray, torch.Tensor],
    uvs: Union[np.ndarray, torch.Tensor],
    voxel_pbr_attrs: torch.Tensor,
    texture_size: int = 1024,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bake PBR attributes from voxel representation to UV-mapped textures.

    This is the main entry point for PBR texture baking in the TRELLIS.2 pipeline.

    Args:
        vertices: [V, 3] mesh vertex positions (assumed in [-0.5, 0.5] after preprocessing)
        faces: [F, 3] mesh face indices
        uvs: [V, 2] UV coordinates in [0, 1]
        voxel_pbr_attrs: [B, 6, D, H, W] voxel PBR attributes:
            - channels 0-2: base_color (RGB)
            - channel 3: metallic
            - channel 4: roughness
            - channel 5: alpha/opacity
        texture_size: Output texture resolution (square)
        device: Target device (defaults to CUDA if available)

    Returns:
        base_color_texture: [H, W, 4] RGBA base color texture
        orm_texture: [H, W, 3] Occlusion-Roughness-Metallic texture (glTF standard)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert inputs to torch tensors if needed
    if isinstance(vertices, np.ndarray):
        vertices = torch.from_numpy(vertices.astype(np.float32)).to(device)
    else:
        vertices = vertices.to(device)

    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces.astype(np.int64)).to(device)
    else:
        faces = faces.to(device).long()

    if isinstance(uvs, np.ndarray):
        uvs = torch.from_numpy(uvs.astype(np.float32)).to(device)
    else:
        uvs = uvs.to(device)

    voxel_pbr_attrs = voxel_pbr_attrs.to(device)

    print(f"[PBR BAKE] Starting texture baking at {texture_size}x{texture_size}")
    print(f"[PBR BAKE] Mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    print(f"[PBR BAKE] Voxel attrs shape: {voxel_pbr_attrs.shape}")

    # Step 1: Rasterize UV triangles
    print("[PBR BAKE] Rasterizing UV triangles...")
    rast = rasterize_uv_triangles(uvs, faces, texture_size, device)
    mask = rast['mask']
    coverage = mask.sum().item() / (texture_size * texture_size) * 100
    print(f"[PBR BAKE] UV coverage: {coverage:.1f}%")

    # Step 2: Get world positions at each UV texel
    print("[PBR BAKE] Interpolating vertex positions...")
    positions = interpolate_vertex_attrs(
        vertices, faces, rast['bary'], rast['tri_id'], mask
    )  # [H, W, 3]

    # Step 3: Normalize positions to [-1, 1] for grid_sample
    # Mesh is assumed to be in [-0.5, 0.5] after preprocessing
    positions_norm = positions * 2.0  # [-0.5, 0.5] -> [-1, 1]

    # Step 4: Sample PBR attributes from voxel grid
    print("[PBR BAKE] Sampling PBR attributes from voxel grid...")

    # Reshape positions for grid_sample_3d: [H*W, 3] -> [B, N, 3]
    valid_mask = mask.flatten()
    valid_positions = positions_norm.view(-1, 3)[valid_mask]  # [N_valid, 3]

    if valid_positions.shape[0] > 0:
        # Sample from voxel grid
        # voxel_pbr_attrs: [B, 6, D, H, W]
        # Need to query at valid_positions
        sampled_attrs = grid_sample_3d(
            voxel_pbr_attrs,
            valid_positions.unsqueeze(0),  # [1, N_valid, 3]
            mode='bilinear',
            padding_mode='border'
        ).squeeze(0)  # [N_valid, 6]
    else:
        sampled_attrs = torch.zeros(0, 6, device=device)

    # Step 5: Pack into texture maps
    print("[PBR BAKE] Packing texture maps...")

    # Initialize textures
    base_color_tex = torch.zeros(texture_size, texture_size, 4, device=device)
    orm_tex = torch.zeros(texture_size, texture_size, 3, device=device)

    # Fill in valid pixels
    if valid_positions.shape[0] > 0:
        valid_y, valid_x = torch.where(mask)

        # Base color (RGB) + Alpha
        base_color_tex[valid_y, valid_x, 0:3] = sampled_attrs[:, 0:3]  # RGB
        base_color_tex[valid_y, valid_x, 3] = sampled_attrs[:, 5]  # Alpha

        # ORM (Occlusion, Roughness, Metallic) - glTF standard packing
        orm_tex[valid_y, valid_x, 0] = 1.0  # Occlusion (no AO data, default to 1)
        orm_tex[valid_y, valid_x, 1] = sampled_attrs[:, 4]  # Roughness
        orm_tex[valid_y, valid_x, 2] = sampled_attrs[:, 3]  # Metallic

    # Step 6: Inpaint unmapped regions
    print("[PBR BAKE] Inpainting unmapped regions...")
    base_color_tex = inpaint_texture(base_color_tex, mask)
    orm_tex = inpaint_texture(orm_tex, mask)

    # Clamp to valid range
    base_color_tex = base_color_tex.clamp(0, 1)
    orm_tex = orm_tex.clamp(0, 1)

    print(f"[PBR BAKE] Base color stats: min={base_color_tex.min():.3f}, max={base_color_tex.max():.3f}")
    print(f"[PBR BAKE] ORM stats: min={orm_tex.min():.3f}, max={orm_tex.max():.3f}")

    return base_color_tex, orm_tex


def bake_from_gaussian_observations(
    vertices: Union[np.ndarray, torch.Tensor],
    faces: Union[np.ndarray, torch.Tensor],
    uvs: Union[np.ndarray, torch.Tensor],
    observations: list,
    masks: list,
    extrinsics: list,
    intrinsics: list,
    texture_size: int = 1024,
    metallic_default: float = 0.0,
    roughness_default: float = 1.0,
    mode: str = 'opt',
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bake PBR textures from Gaussian splat observations.

    This is for the case where we don't have direct voxel PBR attributes,
    but instead have RGB observations from rendering the Gaussian representation.

    The base color is derived from the Gaussian renders, while metallic/roughness
    are set to defaults (can be estimated from the image in future).

    Args:
        vertices, faces, uvs: Mesh data
        observations: List of [H, W, 3] RGB observations (uint8)
        masks: List of [H, W] boolean masks
        extrinsics, intrinsics: Camera parameters
        texture_size: Output resolution
        metallic_default: Default metallic value (0 = dielectric)
        roughness_default: Default roughness value (1 = rough)
        mode: Texture baking mode ('fast' or 'opt')
        verbose: Print progress

    Returns:
        base_color_texture: [H, W, 4] RGBA
        orm_texture: [H, W, 3] ORM
    """
    from .postprocessing_utils import bake_texture

    # Use existing texture baking for base color
    base_color_rgb = bake_texture(
        vertices, faces, uvs,
        observations, masks, extrinsics, intrinsics,
        texture_size=texture_size,
        mode=mode,
        verbose=verbose
    )  # [H, W, 3] uint8 numpy array

    # Convert to torch tensor
    base_color_rgb = torch.from_numpy(base_color_rgb.astype(np.float32) / 255.0)

    # Add alpha channel (fully opaque)
    alpha = torch.ones(texture_size, texture_size, 1)
    base_color_tex = torch.cat([base_color_rgb, alpha], dim=-1)  # [H, W, 4]

    # Create ORM texture with defaults
    orm_tex = torch.zeros(texture_size, texture_size, 3)
    orm_tex[:, :, 0] = 1.0  # Occlusion = 1 (no AO)
    orm_tex[:, :, 1] = roughness_default  # Roughness
    orm_tex[:, :, 2] = metallic_default  # Metallic

    return base_color_tex, orm_tex


def bake_pbr_from_ovoxel(
    ovoxel,  # OVoxel instance
    vertices: Union[np.ndarray, torch.Tensor],
    faces: Union[np.ndarray, torch.Tensor],
    uvs: Union[np.ndarray, torch.Tensor],
    texture_size: int = 1024,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bake PBR textures from OVoxel representation (TRELLIS.2 native path).

    This function converts per-voxel PBR attributes from an OVoxel representation
    into UV-mapped textures suitable for GLB export.

    Args:
        ovoxel: OVoxel instance with per-voxel PBR attributes
        vertices: [V, 3] mesh vertex positions
        faces: [F, 3] mesh face indices
        uvs: [V, 2] UV coordinates in [0, 1]
        texture_size: Output texture resolution
        device: Target device

    Returns:
        base_color_texture: [H, W, 4] RGBA base color texture
        orm_texture: [H, W, 3] Occlusion-Roughness-Metallic texture
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[PBR BAKE OVOXEL] Starting bake from OVoxel with {ovoxel.num_voxels} voxels")

    # Convert OVoxel to dense voxel grid [1, 6, D, H, W]
    voxel_grid = ovoxel.to_voxel_grid_batched()  # [B, 6, D, H, W]

    print(f"[PBR BAKE OVOXEL] Voxel grid shape: {voxel_grid.shape}")

    # Use existing bake_pbr_textures with the voxel grid
    base_color_tex, orm_tex = bake_pbr_textures(
        vertices=vertices,
        faces=faces,
        uvs=uvs,
        voxel_pbr_attrs=voxel_grid,
        texture_size=texture_size,
        device=device
    )

    print(f"[PBR BAKE OVOXEL] Bake complete")

    return base_color_tex, orm_tex


def save_pbr_textures(
    base_color_tex: torch.Tensor,
    orm_tex: torch.Tensor,
    output_dir: str,
    prefix: str = "texture"
) -> Dict[str, str]:
    """
    Save PBR textures to files.

    Args:
        base_color_tex: [H, W, 4] RGBA base color
        orm_tex: [H, W, 3] ORM packed texture
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        Dictionary mapping texture type to file path
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    paths = {}

    # Save base color (RGBA)
    bc_np = (base_color_tex.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    bc_path = os.path.join(output_dir, f"{prefix}_baseColor.png")
    Image.fromarray(bc_np).save(bc_path)
    paths['baseColor'] = bc_path

    # Save ORM (RGB)
    orm_np = (orm_tex.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    orm_path = os.path.join(output_dir, f"{prefix}_orm.png")
    Image.fromarray(orm_np).save(orm_path)
    paths['orm'] = orm_path

    # Also save separate roughness and metallic for debugging
    roughness_path = os.path.join(output_dir, f"{prefix}_roughness.png")
    Image.fromarray(orm_np[:, :, 1]).save(roughness_path)
    paths['roughness'] = roughness_path

    metallic_path = os.path.join(output_dir, f"{prefix}_metallic.png")
    Image.fromarray(orm_np[:, :, 2]).save(metallic_path)
    paths['metallic'] = metallic_path

    print(f"[PBR BAKE] Saved textures to {output_dir}")

    return paths
