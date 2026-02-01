"""
gsplat wrapper to provide diff_gaussian_rasterization compatible interface
This allows TRELLIS to use gsplat for Gaussian Splatting rendering without modification

CRITICAL: gsplat does NOT add the +0.5 bias that diff-gaussian-rasterization adds
during SH-to-RGB conversion. TRELLIS was trained with diff-gaussian-rasterization,
so we must handle this bias ourselves. See: https://github.com/nerfstudio-project/gsplat/issues/147
"""
import torch
import math
from dataclasses import dataclass
from typing import Optional
import gsplat
from .sh_utils import eval_sh, C0


@dataclass
class GaussianRasterizationSettings:
    """Settings for Gaussian rasterization - compatible with diff_gaussian_rasterization API"""
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    kernel_size: float = 0.1
    subpixel_offset: Optional[torch.Tensor] = None


class GaussianRasterizer:
    """
    Wrapper around gsplat.rasterization to provide diff_gaussian_rasterization compatible API
    """
    def __init__(self, raster_settings: GaussianRasterizationSettings):
        self.settings = raster_settings

    def __call__(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        shs: Optional[torch.Tensor] = None,
        colors_precomp: Optional[torch.Tensor] = None,
        opacities: torch.Tensor = None,
        scales: Optional[torch.Tensor] = None,
        rotations: Optional[torch.Tensor] = None,
        cov3D_precomp: Optional[torch.Tensor] = None
    ):
        """
        Rasterize Gaussians to image

        Returns:
            rendered_image: [3, H, W] tensor
            radii: [N] tensor of radii
        """
        # Convert viewmatrix to camera parameters gsplat expects
        # TRELLIS passes viewmatrix = extrinsics.T where extrinsics is W2C from utils3d.extrinsics_look_at
        # diff-gaussian-rasterization uses column-major (transposed), gsplat uses row-major
        # So we need to UN-TRANSPOSE to get the row-major W2C that gsplat expects
        viewmat_w2c_colmajor = self.settings.viewmatrix  # [4, 4] W2C transposed (column-major)
        viewmat_w2c = viewmat_w2c_colmajor.T  # [4, 4] W2C row-major (un-transpose)

        # Debug: log camera matrix transformation (first render only)
        if not hasattr(self, '_logged_viewmat'):
            self._logged_viewmat = True
            # For W2C, camera position = -R^T * t where R is rotation, t is translation
            R = viewmat_w2c[:3, :3]
            t = viewmat_w2c[:3, 3]
            cam_pos = -R.T @ t
            print(f"[GSPLAT DEBUG] Camera position (from W2C): {cam_pos.tolist()}")
            print(f"[GSPLAT DEBUG] viewmat_w2c[0,0:3]: {viewmat_w2c[0, :3].tolist()}")

        # gsplat expects viewmats in batch format [1, 4, 4]
        viewmats = viewmat_w2c.unsqueeze(0)

        # Construct intrinsics matrix K from FOV
        fx = self.settings.image_width / (2.0 * self.settings.tanfovx)
        fy = self.settings.image_height / (2.0 * self.settings.tanfovy)
        cx = self.settings.image_width / 2.0
        cy = self.settings.image_height / 2.0

        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=means3D.device).unsqueeze(0)  # [1, 3, 3]

        # Convert rotations from quaternion to rotation matrix if needed
        if rotations is not None:
            quats = rotations  # Assuming already in quaternion format [N, 4]
            # Normalize quaternions
            quats = quats / quats.norm(dim=-1, keepdim=True)
        else:
            # Identity quaternions if not provided
            quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=means3D.device).repeat(means3D.shape[0], 1)

        # Use precomputed colors or SH coefficients
        # CRITICAL FIX: gsplat does NOT add the +0.5 bias that diff-gaussian-rasterization adds.
        # We must pre-compute RGB colors ourselves using the TRELLIS formula: RGB = C0 * SH + 0.5
        # See: https://github.com/nerfstudio-project/gsplat/issues/147

        if colors_precomp is not None:
            colors = colors_precomp
            sh_degree_for_gsplat = None
            print(f"[GSPLAT DEBUG] Using precomputed colors: shape={colors.shape}")
        elif shs is not None:
            # TRELLIS decoder outputs _features_dc as [N, K, 3] where K=1 for sh_degree=0
            # Log raw SH statistics for diagnostics (first render only)
            if not hasattr(self, '_logged_color_stats'):
                self._logged_color_stats = True
                print(f"[GSPLAT DEBUG] SH coefficients stats (raw): min={shs.min().item():.4f}, max={shs.max().item():.4f}, mean={shs.mean().item():.4f}")
                sample_colors = shs[:5, 0, :]  # First 5 Gaussians, DC coefficients
                print(f"[GSPLAT DEBUG] Sample SH DC values (first 5): {sample_colors.tolist()}")

            # Pre-compute RGB colors from SH coefficients using diff-gaussian-rasterization formula
            # For sh_degree=0 (DC only): RGB = C0 * SH_DC + 0.5
            # This matches what diff_gaussian_rasterization does internally
            if self.settings.sh_degree == 0:
                # DC-only: simple formula RGB = C0 * SH + 0.5
                sh_dc = shs[:, 0, :]  # [N, 3] - extract DC coefficients
                rgb_colors = C0 * sh_dc + 0.5
                # Clamp to valid RGB range [0, 1]
                colors = torch.clamp(rgb_colors, 0.0, 1.0)
                sh_degree_for_gsplat = None  # Pass as precomputed colors, not SH

                if not hasattr(self, '_logged_rgb_conversion'):
                    self._logged_rgb_conversion = True
                    print(f"[GSPLAT DEBUG] Converted SH DC to RGB: min={colors.min().item():.4f}, max={colors.max().item():.4f}, mean={colors.mean().item():.4f}")
            else:
                # Higher degree SH: need to evaluate with view direction
                # For now, fall back to gsplat's internal handling (may have color issues)
                # TODO: Implement full eval_sh with view directions for higher degrees
                print(f"[GSPLAT WARNING] sh_degree={self.settings.sh_degree} > 0: using gsplat internal SH (may have color bias)")
                colors = shs
                sh_degree_for_gsplat = self.settings.sh_degree

            print(f"[GSPLAT DEBUG] Final colors: shape={colors.shape}, sh_degree_for_gsplat={sh_degree_for_gsplat}")
        else:
            # Default white color
            colors = torch.ones(means3D.shape[0], 3, device=means3D.device)
            sh_degree_for_gsplat = None
            print(f"[GSPLAT DEBUG] Using default white color")

        # Prepare scales
        if scales is None:
            scales = torch.ones(means3D.shape[0], 3, device=means3D.device) * 0.01

        # Prepare opacities
        if opacities is None:
            opacities = torch.ones(means3D.shape[0], 1, device=means3D.device)

        # Background color - gsplat expects [C] for flat color, not [1, C]
        backgrounds = self.settings.bg  # [3] - keep as-is, gsplat handles it

        # Call gsplat rasterization
        try:
            # Debug: log color channel statistics before gsplat call
            if not hasattr(self, '_logged_pre_gsplat'):
                self._logged_pre_gsplat = True
                r_ch = colors[:, 0] if len(colors.shape) == 2 else colors[:, :, 0].flatten()
                g_ch = colors[:, 1] if len(colors.shape) == 2 else colors[:, :, 1].flatten()
                b_ch = colors[:, 2] if len(colors.shape) == 2 else colors[:, :, 2].flatten()
                print(f"[GSPLAT DEBUG] PRE-GSPLAT Color channels - R: mean={r_ch.mean():.4f}, G: mean={g_ch.mean():.4f}, B: mean={b_ch.mean():.4f}")
                print(f"[GSPLAT DEBUG] PRE-GSPLAT Scales: shape={scales.shape}, mean={scales.mean():.6f}, min={scales.min():.6f}, max={scales.max():.6f}")
                # Log Gaussian positions to check if they're in view
                print(f"[GSPLAT DEBUG] PRE-GSPLAT means3D: shape={means3D.shape}, X=[{means3D[:,0].min():.3f}, {means3D[:,0].max():.3f}], Y=[{means3D[:,1].min():.3f}, {means3D[:,1].max():.3f}], Z=[{means3D[:,2].min():.3f}, {means3D[:,2].max():.3f}]")
                print(f"[GSPLAT DEBUG] PRE-GSPLAT viewmats: shape={viewmats.shape}")
                print(f"[GSPLAT DEBUG] PRE-GSPLAT K: {K[0].tolist()}")
            render_colors, render_alphas, meta = gsplat.rasterization(
                means=means3D,
                quats=quats,
                scales=scales,
                opacities=opacities.squeeze(-1),
                colors=colors,
                viewmats=viewmats,
                Ks=K,
                width=self.settings.image_width,
                height=self.settings.image_height,
                sh_degree=sh_degree_for_gsplat,
                backgrounds=backgrounds,
            )

            # render_colors is [1, H, W, 3], need [3, H, W]
            rendered_image = render_colors[0].permute(2, 0, 1)  # [3, H, W]

            # Log rendered image and alpha statistics (first render only)
            if not hasattr(self, '_logged_render_stats'):
                # Log per-channel rendered output for diagnosis
                r_out = rendered_image[0]
                g_out = rendered_image[1]
                b_out = rendered_image[2]
                print(f"[GSPLAT DEBUG] POST-GSPLAT Rendered channels - R: mean={r_out.mean():.4f}, G: mean={g_out.mean():.4f}, B: mean={b_out.mean():.4f}")
                print(f"[GSPLAT DEBUG] POST-GSPLAT render_colors raw shape: {render_colors.shape}, dtype: {render_colors.dtype}")

            if not hasattr(self, '_logged_render_stats'):
                self._logged_render_stats = True
                alpha_stats = f"min={render_alphas.min().item():.4f}, max={render_alphas.max().item():.4f}, mean={render_alphas.mean().item():.4f}"
                print(f"[GSPLAT DEBUG] Alpha channel stats: {alpha_stats}")
                print(f"[GSPLAT DEBUG] Background color: {backgrounds.tolist()}")
                print(f"[GSPLAT DEBUG] Opacity input stats: min={opacities.min().item():.4f}, max={opacities.max().item():.4f}, mean={opacities.mean().item():.4f}")
                print(f"[GSPLAT DEBUG] Rendered image stats: min={rendered_image.min().item():.4f}, max={rendered_image.max().item():.4f}, mean={rendered_image.mean().item():.4f}")

            # Extract radii from meta
            # gsplat returns radii as [N_visible, 2] with (x_radius, y_radius) for visible Gaussians
            # diff_gaussian_rasterization expects [N] with max(x,y) radius for ALL Gaussians
            # We need to reconstruct the full [N] array with zeros for invisible Gaussians
            radii = torch.zeros(means3D.shape[0], dtype=torch.int32, device=means3D.device)

            if 'radii' in meta and 'gaussian_ids' in meta:
                # meta['gaussian_ids'] tells us which Gaussians are visible
                # meta['radii'] is [N_visible, 2] with (x_radius, y_radius)
                visible_ids = meta['gaussian_ids']
                visible_radii = meta['radii']  # [N_visible, 2]

                # Take max of x and y radii for each visible Gaussian
                max_radii = visible_radii.max(dim=1).values  # [N_visible]

                # Assign to the corresponding positions in the full array
                radii[visible_ids] = max_radii

            return rendered_image, radii

        except Exception as e:
            import traceback
            print(f"[GSPLAT ERROR] gsplat rasterization failed: {e}")
            print(f"[GSPLAT ERROR] Full traceback:")
            traceback.print_exc()
            print(f"[GSPLAT ERROR] Input shapes: means3D={means3D.shape}, colors={colors.shape}, sh_degree_for_gsplat={sh_degree_for_gsplat}")
            print(f"[GSPLAT ERROR] Falling back to black image...")
            # Fallback: return black image and zero radii
            rendered_image = torch.zeros(3, self.settings.image_height, self.settings.image_width,
                                        dtype=torch.float32, device=means3D.device)
            radii = torch.zeros(means3D.shape[0], dtype=torch.int32, device=means3D.device)
            return rendered_image, radii
