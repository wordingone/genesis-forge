"""
PBR Gaussian Renderer

Extends the standard Gaussian renderer with physically-based rendering support.
Uses split-sum approximation for image-based lighting without nvdiffrec.

Material attributes per Gaussian (O-Voxel format):
- base_color: RGB albedo [0,1]^3
- metallic: metalness ratio [0,1]
- roughness: surface roughness [0,1]
- alpha: opacity [0,1]
"""

import torch
import torch.nn.functional as F
import math
import os
from easydict import EasyDict as edict
import numpy as np
from typing import Optional, Tuple

from ..representations.gaussian import Gaussian
from .gaussian_render import intrinsics_to_projection, render
from .pbr_utils import (
    load_or_generate_brdf_lut,
    evaluate_pbr_split_sum,
    fresnel_schlick_roughness,
    linear_to_srgb,
    srgb_to_linear
)


class PBRGaussian(Gaussian):
    """
    Gaussian with PBR material attributes.

    Extends the base Gaussian class with:
    - metallic: Per-Gaussian metalness [N, 1]
    - roughness: Per-Gaussian roughness [N, 1]
    - normal: Per-Gaussian normal direction [N, 3]
    """

    def __init__(self, aabb: list, sh_degree: int = 0, **kwargs):
        super().__init__(aabb=aabb, sh_degree=sh_degree, **kwargs)
        self._metallic = torch.empty(0)
        self._roughness = torch.empty(0)
        self._normal = torch.empty(0)

    @property
    def get_metallic(self) -> torch.Tensor:
        """Get metallic values [N, 1]"""
        return self._metallic

    @property
    def get_roughness(self) -> torch.Tensor:
        """Get roughness values [N, 1]"""
        return self._roughness

    @property
    def get_normal(self) -> torch.Tensor:
        """Get normal vectors [N, 3]"""
        if self._normal.shape[0] == 0:
            # Default: use view-facing normals (computed during rendering)
            return None
        return self._normal

    def set_pbr_attributes(self, metallic: torch.Tensor, roughness: torch.Tensor,
                           normal: Optional[torch.Tensor] = None):
        """
        Set PBR material attributes.

        Args:
            metallic: [N, 1] or [N] metalness values
            roughness: [N, 1] or [N] roughness values
            normal: Optional [N, 3] normal vectors
        """
        if metallic.dim() == 1:
            metallic = metallic.unsqueeze(-1)
        if roughness.dim() == 1:
            roughness = roughness.unsqueeze(-1)

        self._metallic = metallic.clamp(0.0, 1.0)
        self._roughness = roughness.clamp(0.04, 1.0)  # Avoid zero roughness

        if normal is not None:
            self._normal = F.normalize(normal, dim=-1)


class PBRGaussianRenderer:
    """
    Renderer for Gaussians with PBR materials.

    Supports:
    - Split-sum IBL approximation
    - Metallic-roughness workflow
    - Environment map lighting
    - Optional normal mapping

    Args:
        rendering_options: Rendering configuration dict
        brdf_lut_path: Optional path to cached BRDF LUT
        env_map: Optional HDR environment map for IBL
    """

    def __init__(self,
                 rendering_options: dict = {},
                 brdf_lut_path: Optional[str] = None,
                 env_map: Optional[torch.Tensor] = None):

        self.pipe = edict({
            "kernel_size": 0.1,
            "convert_SHs_python": False,
            "compute_cov3D_python": False,
            "scale_modifier": 1.0,
            "debug": False
        })

        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1,
            "bg_color": 'random',
            "pbr_enabled": True,
            "ambient_intensity": 0.3,
            "env_rotation": 0.0,  # Environment map rotation in radians
        })
        self.rendering_options.update(rendering_options)

        self.bg_color = None

        # Initialize BRDF LUT
        default_lut_path = brdf_lut_path or os.path.join(
            os.path.dirname(__file__), '..', '..', 'assets', 'brdf_lut.pt'
        )
        self.brdf_lut = load_or_generate_brdf_lut(
            cache_path=default_lut_path,
            resolution=256,  # 256x256 is sufficient for most cases
            device='cuda'
        )

        # Environment map for IBL
        self.env_map = env_map
        self.prefiltered_env_mips = None

        if env_map is not None:
            self._setup_environment_map(env_map)

    def _setup_environment_map(self, env_map: torch.Tensor):
        """
        Precompute environment map mip chain for IBL.

        Args:
            env_map: HDR environment map [H, W, 3]
        """
        from .pbr_utils import generate_prefiltered_env_mips

        self.env_map = env_map.to('cuda')

        # Generate prefiltered mip chain (expensive, do once)
        print("[PBR] Prefiltering environment map...")
        self.prefiltered_env_mips = generate_prefiltered_env_mips(
            self.env_map,
            num_mip_levels=5,
            num_samples=256  # Reduce for speed, increase for quality
        )
        print("[PBR] Environment map prefiltering complete")

    def set_environment_map(self, env_map: torch.Tensor):
        """Update the environment map used for IBL."""
        self._setup_environment_map(env_map)

    def render(
        self,
        gaussian: Gaussian,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        metallic: Optional[torch.Tensor] = None,
        roughness: Optional[torch.Tensor] = None,
        colors_overwrite: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> edict:
        """
        Render Gaussians with PBR materials.

        Args:
            gaussian: Gaussian representation (or PBRGaussian)
            extrinsics: [4, 4] camera extrinsics (world to camera)
            intrinsics: [3, 3] camera intrinsics
            metallic: Optional [N, 1] metalness per Gaussian (default: 0 = dielectric)
            roughness: Optional [N, 1] roughness per Gaussian (default: 0.5)
            colors_overwrite: Optional [N, 3] base color override
            return_components: If True, return diffuse/specular separately

        Returns:
            edict containing:
                color: [3, H, W] final rendered image
                diffuse: [3, H, W] diffuse component (if return_components)
                specular: [3, H, W] specular component (if return_components)
                depth: [H, W] depth buffer (if available)
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]

        # Background color handling
        if self.rendering_options["bg_color"] == 'random':
            self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
            if np.random.rand() < 0.5:
                self.bg_color += 1
        else:
            self.bg_color = torch.tensor(
                self.rendering_options["bg_color"],
                dtype=torch.float32, device="cuda"
            )

        # Get base color (albedo) from Gaussian
        # First, render the base color pass
        base_color_render = self._render_base_pass(
            gaussian, extrinsics, intrinsics,
            resolution * ssaa, near, far,
            colors_overwrite
        )

        if not self.rendering_options["pbr_enabled"]:
            # Non-PBR path: just return base color
            result = self._apply_ssaa(base_color_render, resolution, ssaa)
            return edict({'color': result['color']})

        # Get PBR attributes
        num_gaussians = gaussian.get_xyz.shape[0]

        if metallic is None:
            if isinstance(gaussian, PBRGaussian) and gaussian.get_metallic.shape[0] > 0:
                metallic = gaussian.get_metallic
            else:
                metallic = torch.zeros(num_gaussians, 1, device='cuda')

        if roughness is None:
            if isinstance(gaussian, PBRGaussian) and gaussian.get_roughness.shape[0] > 0:
                roughness = gaussian.get_roughness
            else:
                roughness = torch.full((num_gaussians, 1), 0.5, device='cuda')

        # Render material attribute passes (metallic, roughness as colors)
        metallic_render = self._render_base_pass(
            gaussian, extrinsics, intrinsics,
            resolution * ssaa, near, far,
            metallic.expand(-1, 3)  # Expand to RGB
        )

        roughness_render = self._render_base_pass(
            gaussian, extrinsics, intrinsics,
            resolution * ssaa, near, far,
            roughness.expand(-1, 3)
        )

        # Extract single-channel material maps from rendered images
        metallic_map = metallic_render['color'][0]  # [H, W] - use R channel
        roughness_map = roughness_render['color'][0]  # [H, W]
        base_color_img = base_color_render['color']  # [3, H, W]

        # Compute view direction per pixel
        view_dir = self._compute_view_directions(
            resolution * ssaa, intrinsics, extrinsics
        )

        # Compute normals (for now, use view-facing normals based on depth)
        # In a full implementation, this would use per-Gaussian normals
        normal = self._estimate_normals_from_depth(base_color_render, view_dir)

        # Apply PBR shading
        pbr_result = self._apply_pbr_shading(
            base_color_img,
            metallic_map,
            roughness_map,
            normal,
            view_dir,
            return_components
        )

        # Apply SSAA downsampling
        result = self._apply_ssaa(pbr_result, resolution, ssaa)

        return result

    def _render_base_pass(
        self,
        gaussian: Gaussian,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        render_resolution: int,
        near: float,
        far: float,
        colors_overwrite: Optional[torch.Tensor] = None
    ) -> edict:
        """Render a single pass with the standard Gaussian rasterizer."""

        view = extrinsics
        perspective = intrinsics_to_projection(intrinsics, near, far)
        camera = torch.inverse(view)[:3, 3]
        focalx = intrinsics[0, 0]
        focaly = intrinsics[1, 1]
        fovx = 2 * torch.atan(0.5 / focalx)
        fovy = 2 * torch.atan(0.5 / focaly)

        camera_dict = edict({
            "image_height": render_resolution,
            "image_width": render_resolution,
            "FoVx": fovx,
            "FoVy": fovy,
            "znear": near,
            "zfar": far,
            "world_view_transform": view.T.contiguous(),
            "projection_matrix": perspective.T.contiguous(),
            "full_proj_transform": (perspective @ view).T.contiguous(),
            "camera_center": camera
        })

        render_ret = render(
            camera_dict, gaussian, self.pipe, self.bg_color,
            override_color=colors_overwrite,
            scaling_modifier=self.pipe.scale_modifier
        )

        return edict({
            'color': render_ret['render'],
            'viewspace_points': render_ret.get('viewspace_points'),
            'visibility_filter': render_ret.get('visibility_filter'),
            'radii': render_ret.get('radii')
        })

    def _compute_view_directions(
        self,
        resolution: int,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute view direction for each pixel.

        Args:
            resolution: Image resolution
            intrinsics: [3, 3] camera intrinsics
            extrinsics: [4, 4] camera extrinsics

        Returns:
            view_dir: [H, W, 3] normalized view directions (pointing towards camera)
        """
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Create pixel grid
        y_coords = torch.arange(resolution, device='cuda', dtype=torch.float32)
        x_coords = torch.arange(resolution, device='cuda', dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Pixel centers to camera space directions
        x_cam = (xx + 0.5 - cx * resolution) / (fx * resolution)
        y_cam = (yy + 0.5 - cy * resolution) / (fy * resolution)
        z_cam = torch.ones_like(x_cam)

        # Stack and normalize to get ray directions in camera space
        ray_dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
        ray_dirs_cam = F.normalize(ray_dirs_cam, dim=-1)

        # Transform to world space
        cam_to_world = torch.inverse(extrinsics)
        rotation = cam_to_world[:3, :3]

        # Apply rotation to get world-space ray directions
        ray_dirs_world = torch.einsum('ij,hwj->hwi', rotation, ray_dirs_cam)

        # View direction is opposite of ray direction (pointing towards camera)
        view_dir = -F.normalize(ray_dirs_world, dim=-1)

        return view_dir

    def _estimate_normals_from_depth(
        self,
        render_result: edict,
        view_dir: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate surface normals from rendered result.

        For Gaussian splatting without explicit normals, we approximate
        by using view-facing normals (perpendicular to view direction).

        Args:
            render_result: Render pass result containing color
            view_dir: [H, W, 3] view directions

        Returns:
            normal: [H, W, 3] estimated surface normals
        """
        # Simple approximation: normals face the camera
        # A more sophisticated approach would use depth gradients
        normal = view_dir  # View-facing normals

        return normal

    def _apply_pbr_shading(
        self,
        base_color: torch.Tensor,
        metallic: torch.Tensor,
        roughness: torch.Tensor,
        normal: torch.Tensor,
        view_dir: torch.Tensor,
        return_components: bool = False
    ) -> edict:
        """
        Apply PBR shading to the rendered image.

        Args:
            base_color: [3, H, W] albedo
            metallic: [H, W] metalness
            roughness: [H, W] roughness
            normal: [H, W, 3] surface normals
            view_dir: [H, W, 3] view directions
            return_components: Whether to return diffuse/specular separately

        Returns:
            edict with 'color' and optionally 'diffuse', 'specular'
        """
        # Convert to [H, W, 3] for processing
        base_color = base_color.permute(1, 2, 0)  # [H, W, 3]

        # Compute N dot V
        n_dot_v = torch.sum(normal * view_dir, dim=-1).clamp(min=0.001)  # [H, W]

        # Compute F0 (reflectance at normal incidence)
        f0_dielectric = torch.tensor([0.04, 0.04, 0.04], device='cuda')
        f0 = f0_dielectric.view(1, 1, 3) * (1.0 - metallic.unsqueeze(-1)) + \
             base_color * metallic.unsqueeze(-1)

        # Fresnel with roughness correction
        fresnel = fresnel_schlick_roughness(n_dot_v, f0, roughness)

        # Sample BRDF LUT
        env_brdf = self._sample_brdf_lut(n_dot_v, roughness)
        scale = env_brdf[..., 0:1]
        bias = env_brdf[..., 1:2]

        # Specular term
        # For now, use ambient light instead of environment map sampling
        ambient_intensity = self.rendering_options.get("ambient_intensity", 0.3)

        if self.prefiltered_env_mips is not None:
            # Sample environment map at reflection direction with roughness-based mip
            reflect_dir = 2.0 * n_dot_v.unsqueeze(-1) * normal - view_dir
            prefilt_color = self._sample_environment(reflect_dir, roughness)
        else:
            # Fallback: constant ambient
            prefilt_color = torch.ones_like(base_color) * ambient_intensity

        specular = prefilt_color * (fresnel * scale + bias)

        # Diffuse term
        kd = (1.0 - fresnel) * (1.0 - metallic.unsqueeze(-1))

        if self.prefiltered_env_mips is not None:
            # Sample irradiance map at normal direction (lowest mip = diffuse)
            irradiance = self._sample_environment(normal, torch.ones_like(roughness))
        else:
            irradiance = torch.ones_like(base_color) * ambient_intensity

        diffuse = kd * base_color * irradiance

        # Combine diffuse and specular
        color = diffuse + specular

        # Tone mapping (simple Reinhard)
        color = color / (color + 1.0)

        # Convert to [3, H, W]
        color = color.permute(2, 0, 1)

        result = edict({'color': color})

        if return_components:
            result['diffuse'] = diffuse.permute(2, 0, 1)
            result['specular'] = specular.permute(2, 0, 1)

        return result

    def _sample_brdf_lut(
        self,
        n_dot_v: torch.Tensor,
        roughness: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample the BRDF integration LUT.

        Args:
            n_dot_v: [H, W] N dot V values
            roughness: [H, W] roughness values

        Returns:
            env_brdf: [H, W, 2] containing (scale, bias)
        """
        # Prepare grid for sampling
        # LUT is indexed by (NdotV, roughness) in [0, 1] range
        grid_u = n_dot_v.clamp(0.0, 1.0) * 2.0 - 1.0  # Map to [-1, 1]
        grid_v = roughness.clamp(0.0, 1.0) * 2.0 - 1.0

        grid = torch.stack([grid_u, grid_v], dim=-1).unsqueeze(0)  # [1, H, W, 2]

        # Sample LUT using grid_sample
        lut_input = self.brdf_lut.permute(2, 0, 1).unsqueeze(0)  # [1, 2, LUT_H, LUT_W]

        sampled = F.grid_sample(
            lut_input,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # [1, 2, H, W]

        return sampled.squeeze(0).permute(1, 2, 0)  # [H, W, 2]

    def _sample_environment(
        self,
        direction: torch.Tensor,
        roughness: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample the prefiltered environment map.

        Args:
            direction: [H, W, 3] sample direction (reflection or normal)
            roughness: [H, W] roughness (determines mip level)

        Returns:
            color: [H, W, 3] sampled environment color
        """
        if self.prefiltered_env_mips is None:
            return torch.ones_like(direction) * 0.3

        num_mips = len(self.prefiltered_env_mips)

        # Compute mip level from roughness
        mip_level = (roughness * (num_mips - 1)).clamp(0, num_mips - 1)

        # Direction to equirectangular UV
        # theta = arccos(y), phi = atan2(z, x)
        theta = torch.acos(direction[..., 1].clamp(-1.0, 1.0))
        phi = torch.atan2(direction[..., 2], direction[..., 0])
        phi = torch.where(phi < 0, phi + 2 * math.pi, phi)

        u = phi / (2 * math.pi)
        v = theta / math.pi

        # Sample each mip level and blend
        h, w = direction.shape[:2]
        color = torch.zeros(h, w, 3, device='cuda')

        for mip_idx in range(num_mips):
            env_map = self.prefiltered_env_mips[mip_idx]
            env_h, env_w = env_map.shape[:2]

            # Convert UV to grid coordinates
            grid_u = u * 2.0 - 1.0
            grid_v = v * 2.0 - 1.0
            grid = torch.stack([grid_u, grid_v], dim=-1).unsqueeze(0)

            # Sample
            sampled = F.grid_sample(
                env_map.permute(2, 0, 1).unsqueeze(0),
                grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            ).squeeze(0).permute(1, 2, 0)

            # Compute blend weight
            weight = 1.0 - torch.abs(mip_level - mip_idx).clamp(0, 1)

            color += sampled * weight.unsqueeze(-1)

        return color

    def _apply_ssaa(
        self,
        result: edict,
        target_resolution: int,
        ssaa: int
    ) -> edict:
        """Apply SSAA downsampling to render result."""

        if ssaa <= 1:
            return result

        output = edict()

        for key, value in result.items():
            if value is None:
                output[key] = None
                continue

            if isinstance(value, torch.Tensor):
                if value.dim() == 3 and value.shape[0] == 3:  # [3, H, W]
                    value = value.unsqueeze(0)  # [1, 3, H, W]
                    value = F.interpolate(
                        value,
                        (target_resolution, target_resolution),
                        mode='bilinear',
                        align_corners=False,
                        antialias=True
                    )
                    value = value.squeeze(0)  # [3, H, W]
                elif value.dim() == 2:  # [H, W]
                    value = value.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    value = F.interpolate(
                        value,
                        (target_resolution, target_resolution),
                        mode='bilinear',
                        align_corners=False,
                        antialias=True
                    )
                    value = value.squeeze()  # [H, W]

            output[key] = value

        return output
