"""
Split-Sum PBR Rendering Utilities

Pure PyTorch implementation of physically-based rendering for TRELLIS.
Replaces nvdiffrec dependency with custom split-sum approximation.

Based on:
- Epic Games "Real Shading in Unreal Engine 4" (SIGGRAPH 2013)
- LearnOpenGL PBR/IBL tutorials
- Google Filament documentation

Material attributes (O-Voxel format):
- base_color: RGB albedo [0,1]^3
- metallic: metalness ratio [0,1]
- roughness: surface roughness [0,1]
- alpha: opacity [0,1]
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional


# ============================================================================
# BRDF Component Functions
# ============================================================================

def ggx_distribution(n_dot_h: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    """
    GGX/Trowbridge-Reitz Normal Distribution Function.

    D(h) = α² / (π * ((n·h)² * (α² - 1) + 1)²)

    Args:
        n_dot_h: Dot product of normal and half vector [N] or [H, W]
        roughness: Surface roughness [N] or [H, W] or scalar

    Returns:
        Distribution value D [same shape as n_dot_h]
    """
    alpha = roughness * roughness
    alpha2 = alpha * alpha

    n_dot_h2 = n_dot_h * n_dot_h
    denom = n_dot_h2 * (alpha2 - 1.0) + 1.0
    denom = denom * denom * math.pi

    # Avoid division by zero
    denom = torch.clamp(denom, min=1e-7)

    return alpha2 / denom


def smith_ggx_geometry(n_dot_v: torch.Tensor, n_dot_l: torch.Tensor,
                       roughness: torch.Tensor) -> torch.Tensor:
    """
    Smith GGX Geometry Function for IBL.

    G(n,v,l) = G1(n,v) * G1(n,l)
    G1(n,x) = (n·x) / ((n·x)(1-k) + k)

    For IBL: k = α² / 2

    Args:
        n_dot_v: Dot product of normal and view direction
        n_dot_l: Dot product of normal and light direction
        roughness: Surface roughness

    Returns:
        Geometry attenuation G
    """
    alpha = roughness * roughness
    k = alpha / 2.0  # IBL remapping

    def g1(n_dot_x):
        return n_dot_x / (n_dot_x * (1.0 - k) + k + 1e-7)

    return g1(n_dot_v) * g1(n_dot_l)


def schlick_fresnel(v_dot_h: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    """
    Schlick's Fresnel Approximation.

    F(v,h) = F0 + (1 - F0) * (1 - v·h)^5

    Args:
        v_dot_h: Dot product of view and half vector
        f0: Reflectance at normal incidence [N, 3] or [3]

    Returns:
        Fresnel reflectance F [same shape as v_dot_h expanded to 3 channels]
    """
    return f0 + (1.0 - f0) * torch.pow(1.0 - v_dot_h.unsqueeze(-1), 5.0)


def fresnel_schlick_roughness(n_dot_v: torch.Tensor, f0: torch.Tensor,
                              roughness: torch.Tensor) -> torch.Tensor:
    """
    Fresnel-Schlick with roughness for IBL.

    F = F0 + (max(1-roughness, F0) - F0) * (1 - n·v)^5

    Args:
        n_dot_v: Dot product of normal and view direction [H, W] or [N]
        f0: Reflectance at normal incidence [H, W, 3] or [N, 3]
        roughness: Surface roughness [H, W] or [N]

    Returns:
        Fresnel reflectance accounting for roughness [same shape as f0]
    """
    one_minus_rough = 1.0 - roughness

    # Expand roughness to match f0 shape (add channel dimension)
    if one_minus_rough.dim() < f0.dim():
        one_minus_rough = one_minus_rough.unsqueeze(-1)

    # Expand to broadcast with f0's channel dimension
    one_minus_rough = one_minus_rough.expand_as(f0)

    f_max = torch.maximum(one_minus_rough, f0)

    # Expand n_dot_v to match f0 shape
    n_dot_v_expanded = n_dot_v.unsqueeze(-1)

    return f0 + (f_max - f0) * torch.pow(1.0 - n_dot_v_expanded, 5.0)


# ============================================================================
# BRDF LUT Generation
# ============================================================================

def hammersley_2d(i: int, n: int) -> Tuple[float, float]:
    """
    Hammersley low-discrepancy sequence for importance sampling.

    Args:
        i: Sample index
        n: Total number of samples

    Returns:
        (x, y) in [0, 1]^2
    """
    # Van der Corput sequence (radical inverse base 2)
    bits = i
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
    bits = (bits << 16) | (bits >> 16)
    radical_inverse = bits * 2.3283064365386963e-10  # / 0x100000000

    return i / n, radical_inverse


def importance_sample_ggx(xi: Tuple[float, float], roughness: float,
                          n: torch.Tensor) -> torch.Tensor:
    """
    Importance sample the GGX distribution to get a half vector.

    Args:
        xi: Random numbers in [0,1]^2 from Hammersley sequence
        roughness: Surface roughness
        n: Normal vector [3]

    Returns:
        Half vector H in world space [3]
    """
    a = roughness * roughness

    # Spherical coordinates from GGX importance sampling
    phi = 2.0 * math.pi * xi[0]
    cos_theta = math.sqrt((1.0 - xi[1]) / (1.0 + (a * a - 1.0) * xi[1] + 1e-7))
    sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)

    # Tangent space half vector
    h_tangent = torch.tensor([
        sin_theta * math.cos(phi),
        sin_theta * math.sin(phi),
        cos_theta
    ], device=n.device, dtype=n.dtype)

    # Build tangent space basis
    up = torch.tensor([0.0, 1.0, 0.0], device=n.device, dtype=n.dtype)
    if abs(n[1]) > 0.999:
        up = torch.tensor([1.0, 0.0, 0.0], device=n.device, dtype=n.dtype)

    tangent = F.normalize(torch.cross(up, n), dim=0)
    bitangent = torch.cross(n, tangent)

    # Transform to world space
    h_world = tangent * h_tangent[0] + bitangent * h_tangent[1] + n * h_tangent[2]
    return F.normalize(h_world, dim=0)


def generate_brdf_lut(resolution: int = 512, num_samples: int = 1024,
                      device: str = 'cuda') -> torch.Tensor:
    """
    Generate the BRDF integration LUT (DFG term).

    The LUT stores two values for each (NdotV, roughness) pair:
    - Scale (R channel): The F0 multiplier
    - Bias (G channel): The additive term

    Final Fresnel = F0 * scale + bias

    Args:
        resolution: LUT texture resolution (default 512x512)
        num_samples: Number of importance samples per texel
        device: Target device

    Returns:
        BRDF LUT tensor [resolution, resolution, 2] containing (scale, bias)
    """
    lut = torch.zeros((resolution, resolution, 2), device=device, dtype=torch.float32)

    n = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)

    for y in range(resolution):
        roughness = (y + 0.5) / resolution
        roughness = max(roughness, 0.01)  # Avoid zero roughness

        for x in range(resolution):
            n_dot_v = (x + 0.5) / resolution
            n_dot_v = max(n_dot_v, 0.001)  # Avoid grazing angles

            # View vector (in tangent space where N = (0,0,1))
            v = torch.tensor([
                math.sqrt(1.0 - n_dot_v * n_dot_v),  # sin(theta)
                0.0,
                n_dot_v  # cos(theta)
            ], device=device, dtype=torch.float32)

            scale_sum = 0.0
            bias_sum = 0.0

            for i in range(num_samples):
                xi = hammersley_2d(i, num_samples)
                h = importance_sample_ggx(xi, roughness, n)

                # Light direction (reflection of V around H)
                l = 2.0 * torch.dot(v, h) * h - v

                n_dot_l = max(l[2].item(), 0.0)
                n_dot_h = max(h[2].item(), 0.0)
                v_dot_h = max(torch.dot(v, h).item(), 0.0)

                if n_dot_l > 0.0:
                    # Geometry term
                    g = smith_ggx_geometry(
                        torch.tensor(n_dot_v, device=device),
                        torch.tensor(n_dot_l, device=device),
                        torch.tensor(roughness, device=device)
                    ).item()

                    # Visibility term (G * VdotH) / (NdotH * NdotV)
                    g_vis = (g * v_dot_h) / max(n_dot_h * n_dot_v, 1e-7)

                    # Fresnel term split: (1 - (1-VdotH)^5) and (1-VdotH)^5
                    fc = math.pow(1.0 - v_dot_h, 5.0)

                    scale_sum += (1.0 - fc) * g_vis
                    bias_sum += fc * g_vis

            lut[y, x, 0] = scale_sum / num_samples
            lut[y, x, 1] = bias_sum / num_samples

    return lut


def generate_brdf_lut_vectorized(resolution: int = 512, num_samples: int = 1024,
                                  device: str = 'cuda') -> torch.Tensor:
    """
    Vectorized BRDF LUT generation (faster than loop version).

    Args:
        resolution: LUT texture resolution
        num_samples: Number of importance samples per texel
        device: Target device

    Returns:
        BRDF LUT tensor [resolution, resolution, 2]
    """
    # Create coordinate grids
    y_coords = torch.arange(resolution, device=device, dtype=torch.float32)
    x_coords = torch.arange(resolution, device=device, dtype=torch.float32)

    roughness = ((y_coords + 0.5) / resolution).clamp(min=0.01)
    n_dot_v = ((x_coords + 0.5) / resolution).clamp(min=0.001)

    # Expand to grid [R, V]
    roughness_grid = roughness.view(-1, 1).expand(resolution, resolution)
    n_dot_v_grid = n_dot_v.view(1, -1).expand(resolution, resolution)

    # Precompute Hammersley sequence
    xi_samples = torch.tensor(
        [hammersley_2d(i, num_samples) for i in range(num_samples)],
        device=device, dtype=torch.float32
    )  # [num_samples, 2]

    scale_accum = torch.zeros((resolution, resolution), device=device)
    bias_accum = torch.zeros((resolution, resolution), device=device)

    # Process in batches to manage memory
    batch_size = 64
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        xi_batch = xi_samples[batch_start:batch_end]  # [B, 2]

        for idx, xi in enumerate(xi_batch):
            # GGX importance sampling for each roughness level
            a2 = (roughness_grid * roughness_grid) ** 2

            phi = 2.0 * math.pi * xi[0].item()
            xi1 = xi[1].item()

            # cos_theta varies with roughness
            cos_theta = torch.sqrt((1.0 - xi1) / (1.0 + (a2 - 1.0) * xi1 + 1e-7))
            sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)

            # Half vector in tangent space (N = z-axis)
            h_x = sin_theta * math.cos(phi)
            h_y = sin_theta * math.sin(phi)
            h_z = cos_theta

            # View vector
            sin_v = torch.sqrt(1.0 - n_dot_v_grid * n_dot_v_grid)
            v_x = sin_v
            v_z = n_dot_v_grid

            # V dot H
            v_dot_h = (v_x * h_x + v_z * h_z).clamp(min=0.0)

            # Light vector L = 2(V·H)H - V
            l_x = 2.0 * v_dot_h * h_x - v_x
            l_z = 2.0 * v_dot_h * h_z - v_z

            n_dot_l = l_z.clamp(min=0.0)
            n_dot_h = h_z.clamp(min=0.0)

            # Geometry term
            k = (roughness_grid * roughness_grid) / 2.0
            g1_v = n_dot_v_grid / (n_dot_v_grid * (1.0 - k) + k + 1e-7)
            g1_l = n_dot_l / (n_dot_l * (1.0 - k) + k + 1e-7)
            g = g1_v * g1_l

            # Visibility
            g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v_grid + 1e-7)

            # Fresnel split
            fc = torch.pow(1.0 - v_dot_h, 5.0)

            # Mask for valid samples (NdotL > 0)
            valid = (n_dot_l > 0.0).float()

            scale_accum += valid * (1.0 - fc) * g_vis
            bias_accum += valid * fc * g_vis

    lut = torch.stack([
        scale_accum / num_samples,
        bias_accum / num_samples
    ], dim=-1)

    return lut


# ============================================================================
# Environment Map Prefiltering
# ============================================================================

def prefilter_environment_map(env_map: torch.Tensor, roughness: float,
                              num_samples: int = 1024) -> torch.Tensor:
    """
    Prefilter an environment map for a specific roughness level.

    Convolves the environment map with the GGX distribution.

    Args:
        env_map: HDR environment map [H, W, 3]
        roughness: Target roughness level [0, 1]
        num_samples: Number of importance samples

    Returns:
        Prefiltered environment map [H, W, 3]
    """
    h, w, _ = env_map.shape
    device = env_map.device
    dtype = env_map.dtype

    # Output buffer
    filtered = torch.zeros_like(env_map)

    # For each output pixel
    for y in range(h):
        for x in range(w):
            # Convert pixel to direction (equirectangular)
            theta = math.pi * (y + 0.5) / h  # [0, pi]
            phi = 2.0 * math.pi * (x + 0.5) / w  # [0, 2pi]

            # Direction vector (reflection direction R)
            r = torch.tensor([
                math.sin(theta) * math.cos(phi),
                math.cos(theta),
                math.sin(theta) * math.sin(phi)
            ], device=device, dtype=dtype)

            # For split-sum: assume N = V = R
            n = r
            v = r

            color_sum = torch.zeros(3, device=device, dtype=dtype)
            weight_sum = 0.0

            for i in range(num_samples):
                xi = hammersley_2d(i, num_samples)
                h_vec = importance_sample_ggx(xi, roughness, n)

                # Light direction
                l = 2.0 * torch.dot(v, h_vec) * h_vec - v

                n_dot_l = torch.dot(n, l).item()

                if n_dot_l > 0.0:
                    # Sample environment map at L direction
                    # Convert L to equirectangular UV
                    l_theta = math.acos(torch.clamp(l[1], -1.0, 1.0).item())
                    l_phi = math.atan2(l[2].item(), l[0].item())
                    if l_phi < 0:
                        l_phi += 2.0 * math.pi

                    u = l_phi / (2.0 * math.pi)
                    v_coord = l_theta / math.pi

                    # Bilinear sample
                    px = u * w
                    py = v_coord * h

                    x0 = int(px) % w
                    y0 = int(py) % h
                    x1 = (x0 + 1) % w
                    y1 = (y0 + 1) % h

                    fx = px - int(px)
                    fy = py - int(py)

                    c00 = env_map[y0, x0]
                    c10 = env_map[y0, x1]
                    c01 = env_map[y1, x0]
                    c11 = env_map[y1, x1]

                    sampled = (c00 * (1-fx) * (1-fy) + c10 * fx * (1-fy) +
                              c01 * (1-fx) * fy + c11 * fx * fy)

                    color_sum += sampled * n_dot_l
                    weight_sum += n_dot_l

            if weight_sum > 0:
                filtered[y, x] = color_sum / weight_sum

    return filtered


def generate_prefiltered_env_mips(env_map: torch.Tensor,
                                   num_mip_levels: int = 5,
                                   num_samples: int = 512) -> list:
    """
    Generate a mipmap chain of prefiltered environment maps.

    Each mip level corresponds to a different roughness:
    - Mip 0: roughness = 0 (sharp reflections)
    - Mip N: roughness = 1 (diffuse reflections)

    Args:
        env_map: HDR environment map [H, W, 3]
        num_mip_levels: Number of mip levels to generate
        num_samples: Samples per texel for filtering

    Returns:
        List of prefiltered environment maps, one per mip level
    """
    mips = []

    for level in range(num_mip_levels):
        roughness = level / max(num_mip_levels - 1, 1)

        if roughness == 0:
            # Mip 0 is just the original (or slightly blurred)
            mips.append(env_map.clone())
        else:
            filtered = prefilter_environment_map(env_map, roughness, num_samples)
            mips.append(filtered)

    return mips


# ============================================================================
# Split-Sum PBR Evaluation
# ============================================================================

def evaluate_pbr_split_sum(
    base_color: torch.Tensor,
    metallic: torch.Tensor,
    roughness: torch.Tensor,
    normal: torch.Tensor,
    view_dir: torch.Tensor,
    prefiltered_env: torch.Tensor,
    brdf_lut: torch.Tensor,
    irradiance_map: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Evaluate PBR lighting using split-sum approximation.

    Args:
        base_color: Albedo [N, 3] or [H, W, 3]
        metallic: Metalness [N] or [H, W]
        roughness: Roughness [N] or [H, W]
        normal: Surface normal [N, 3] or [H, W, 3]
        view_dir: View direction [N, 3] or [H, W, 3]
        prefiltered_env: Prefiltered environment map [H_env, W_env, 3]
        brdf_lut: BRDF integration LUT [LUT_H, LUT_W, 2]
        irradiance_map: Optional diffuse irradiance map [H_irr, W_irr, 3]

    Returns:
        Final color [same shape as base_color]
    """
    # Ensure proper shapes
    if normal.dim() == 2:
        # [N, 3] point data
        is_point_data = True
    else:
        # [H, W, 3] image data
        is_point_data = False

    # Compute N dot V
    n_dot_v = torch.sum(normal * view_dir, dim=-1).clamp(min=0.0)

    # Compute F0 (reflectance at normal incidence)
    # For dielectrics: F0 = 0.04
    # For metals: F0 = base_color
    f0_dielectric = torch.tensor([0.04, 0.04, 0.04], device=base_color.device, dtype=base_color.dtype)
    if is_point_data:
        f0 = f0_dielectric.unsqueeze(0) * (1.0 - metallic.unsqueeze(-1)) + base_color * metallic.unsqueeze(-1)
    else:
        f0 = f0_dielectric.view(1, 1, 3) * (1.0 - metallic.unsqueeze(-1)) + base_color * metallic.unsqueeze(-1)

    # Fresnel with roughness
    fresnel = fresnel_schlick_roughness(n_dot_v, f0, roughness)

    # Sample BRDF LUT
    # UV = (NdotV, roughness)
    lut_h, lut_w = brdf_lut.shape[:2]

    if is_point_data:
        u = n_dot_v.clamp(0.0, 1.0) * (lut_w - 1)
        v = roughness.clamp(0.0, 1.0) * (lut_h - 1)

        # Bilinear interpolation
        u0 = u.long().clamp(0, lut_w - 2)
        v0 = v.long().clamp(0, lut_h - 2)
        u1 = u0 + 1
        v1 = v0 + 1

        fu = (u - u0.float()).unsqueeze(-1)
        fv = (v - v0.float()).unsqueeze(-1)

        lut_00 = brdf_lut[v0, u0]
        lut_10 = brdf_lut[v0, u1]
        lut_01 = brdf_lut[v1, u0]
        lut_11 = brdf_lut[v1, u1]

        env_brdf = (lut_00 * (1-fu) * (1-fv) + lut_10 * fu * (1-fv) +
                   lut_01 * (1-fu) * fv + lut_11 * fu * fv)
    else:
        # Grid sample for image data
        grid_u = n_dot_v.clamp(0.0, 1.0) * 2.0 - 1.0
        grid_v = roughness.clamp(0.0, 1.0) * 2.0 - 1.0
        grid = torch.stack([grid_u, grid_v], dim=-1).unsqueeze(0)  # [1, H, W, 2]

        env_brdf = F.grid_sample(
            brdf_lut.permute(2, 0, 1).unsqueeze(0),  # [1, 2, LUT_H, LUT_W]
            grid,
            mode='bilinear',
            align_corners=True
        ).squeeze(0).permute(1, 2, 0)  # [H, W, 2]

    scale = env_brdf[..., 0:1]
    bias = env_brdf[..., 1:2]

    # Specular contribution
    # Sample prefiltered environment at reflection direction
    reflect_dir = 2.0 * n_dot_v.unsqueeze(-1) * normal - view_dir

    # TODO: Sample prefiltered environment map at reflect_dir with roughness-based mip
    # For now, use a placeholder (would need cubemap/equirect sampling)
    # In practice, this samples from the prefiltered env map mip chain
    prefilt_color = torch.ones_like(base_color) * 0.3  # Placeholder ambient

    specular = prefilt_color * (fresnel * scale + bias)

    # Diffuse contribution
    kd = (1.0 - fresnel) * (1.0 - metallic.unsqueeze(-1))

    if irradiance_map is not None:
        # Sample irradiance at normal direction
        # TODO: Proper spherical sampling
        irradiance = torch.ones_like(base_color) * 0.2  # Placeholder
    else:
        irradiance = torch.ones_like(base_color) * 0.2

    diffuse = kd * base_color * irradiance

    # Combine
    color = diffuse + specular

    return color


# ============================================================================
# Utility Functions
# ============================================================================

def load_or_generate_brdf_lut(cache_path: Optional[str] = None,
                              resolution: int = 512,
                              device: str = 'cuda') -> torch.Tensor:
    """
    Load BRDF LUT from cache or generate if not found.

    Args:
        cache_path: Path to cached LUT file (.pt)
        resolution: LUT resolution if generating
        device: Target device

    Returns:
        BRDF LUT tensor [resolution, resolution, 2]
    """
    import os

    if cache_path and os.path.exists(cache_path):
        lut = torch.load(cache_path, map_location=device)
        print(f"[PBR] Loaded BRDF LUT from {cache_path}")
        return lut

    print(f"[PBR] Generating BRDF LUT ({resolution}x{resolution})...")
    lut = generate_brdf_lut_vectorized(resolution=resolution, device=device)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(lut, cache_path)
        print(f"[PBR] Saved BRDF LUT to {cache_path}")

    return lut


def compute_f0_from_ior(ior: float) -> float:
    """
    Compute F0 from index of refraction.

    F0 = ((n - 1) / (n + 1))^2

    Args:
        ior: Index of refraction (e.g., 1.5 for glass)

    Returns:
        F0 reflectance value
    """
    return ((ior - 1.0) / (ior + 1.0)) ** 2


def linear_to_srgb(color: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB to sRGB."""
    return torch.where(
        color <= 0.0031308,
        color * 12.92,
        1.055 * torch.pow(color.clamp(min=1e-7), 1.0/2.4) - 0.055
    )


def srgb_to_linear(color: torch.Tensor) -> torch.Tensor:
    """Convert sRGB to linear RGB."""
    return torch.where(
        color <= 0.04045,
        color / 12.92,
        torch.pow((color + 0.055) / 1.055, 2.4)
    )
