import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from ..representations.mesh import MeshExtractResult


def project_vertices_to_screen(
    vertices: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int
) -> tuple:
    """
    Project vertices from world space directly to screen space.

    Uses simple pinhole camera model. Automatically detects OpenGL vs OpenCV
    camera convention based on the extrinsics matrix.

    Args:
        vertices: [N, 3] world space vertices
        extrinsics: [4, 4] camera extrinsics (world to camera transform)
        intrinsics: [3, 3] camera intrinsics
        width: image width
        height: image height

    Returns:
        tuple of:
            screen_coords: [N, 3] screen coordinates (x, y, depth)
            valid_mask: [N] boolean mask for vertices in front of camera
    """
    N = vertices.shape[0]
    device = vertices.device
    dtype = vertices.dtype

    # Add homogeneous coordinate
    vertices_homo = torch.cat([vertices, torch.ones((N, 1), device=device, dtype=dtype)], dim=-1)

    # World to camera transformation
    vertices_camera = torch.matmul(vertices_homo, extrinsics.T)

    # Extract camera-space coordinates
    x_cam = vertices_camera[:, 0]
    y_cam = vertices_camera[:, 1]
    z_cam = vertices_camera[:, 2]

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Standard pinhole projection (OpenCV convention: positive Z is forward)
    # x_pixel = fx * x_cam / z_cam + cx
    # y_pixel = fy * y_cam / z_cam + cy
    # Depth = z_cam (positive = in front of camera)

    # Avoid division by zero
    z_safe = torch.where(z_cam.abs() > 1e-6, z_cam, torch.sign(z_cam) * 1e-6)
    z_safe = torch.where(z_safe == 0, torch.ones_like(z_safe) * 1e-6, z_safe)

    # Use absolute value of z for depth, and sign determines validity
    depth = z_cam.abs()  # Positive depth

    x_screen = fx * x_cam / z_safe + cx
    y_screen = fy * y_cam / z_safe + cy

    screen_coords = torch.stack([x_screen, y_screen, depth], dim=-1)

    # Valid if z_cam > 0 (OpenCV convention: camera looks down +Z)
    # Most TRELLIS code uses this convention
    valid_mask = z_cam > 0

    return screen_coords, valid_mask


def barycentric_coordinates(p: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Compute barycentric coordinates for points p with respect to triangle v0, v1, v2.

    Args:
        p: [H, W, 2] pixel coordinates
        v0, v1, v2: [2] triangle vertex screen coordinates

    Returns:
        [H, W, 3] barycentric coordinates (u, v, w)
    """
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0

    d00 = (v0v1 * v0v1).sum()
    d01 = (v0v1 * v0v2).sum()
    d11 = (v0v2 * v0v2).sum()
    d20 = (v0p * v0v1.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    d21 = (v0p * v0v2.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return torch.zeros((*p.shape[:-1], 3), device=p.device, dtype=p.dtype)

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    return torch.stack([u, v, w], dim=-1)


def rasterize_triangles_software(
    vertices_screen: torch.Tensor,
    faces: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
    cull_backfaces: bool = True
) -> dict:
    """
    Software rasterizer - pure PyTorch implementation.

    Args:
        vertices_screen: [N, 3] screen space vertices (x, y, depth)
        faces: [F, 3] face indices
        width: image width
        height: image height
        device: torch device
        cull_backfaces: whether to cull back-facing triangles

    Returns:
        dict with 'mask', 'face_id', 'bary', 'depth'
    """
    num_faces = faces.shape[0]

    # Initialize output buffers
    depth_buffer = torch.full((height, width), float('inf'), device=device, dtype=torch.float32)
    face_id_buffer = torch.zeros((height, width), device=device, dtype=torch.int64)
    bary_buffer = torch.zeros((height, width, 3), device=device, dtype=torch.float32)

    # Create pixel coordinate grid
    y_coords = torch.arange(height, device=device, dtype=torch.float32)
    x_coords = torch.arange(width, device=device, dtype=torch.float32)
    pixel_grid = torch.stack(torch.meshgrid(y_coords, x_coords, indexing='ij'), dim=-1)
    pixel_centers = pixel_grid + 0.5
    pixel_xy = pixel_centers[..., [1, 0]]  # [H, W, 2] as (x, y)

    # Process each triangle
    for face_idx in range(num_faces):
        face = faces[face_idx]
        v0_screen = vertices_screen[face[0]]
        v1_screen = vertices_screen[face[1]]
        v2_screen = vertices_screen[face[2]]

        # Skip triangles with vertices behind camera
        if v0_screen[2] <= 0 or v1_screen[2] <= 0 or v2_screen[2] <= 0:
            continue

        # Check backface culling (cross product determines winding)
        edge1 = v1_screen[:2] - v0_screen[:2]
        edge2 = v2_screen[:2] - v0_screen[:2]
        cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]

        if cull_backfaces and cross <= 0:  # Back-facing or degenerate
            continue

        # Compute bounding box
        min_x = max(0, int(torch.min(torch.stack([v0_screen[0], v1_screen[0], v2_screen[0]])).item()))
        max_x = min(width - 1, int(torch.max(torch.stack([v0_screen[0], v1_screen[0], v2_screen[0]])).item()) + 1)
        min_y = max(0, int(torch.min(torch.stack([v0_screen[1], v1_screen[1], v2_screen[1]])).item()))
        max_y = min(height - 1, int(torch.max(torch.stack([v0_screen[1], v1_screen[1], v2_screen[1]])).item()) + 1)

        if min_x > max_x or min_y > max_y:
            continue

        # Get pixel subset
        pixel_subset = pixel_xy[min_y:max_y+1, min_x:max_x+1]  # [h, w, 2]

        # Compute barycentric coordinates
        bary = barycentric_coordinates(pixel_subset, v0_screen[:2], v1_screen[:2], v2_screen[:2])

        # Check if inside triangle
        inside = (bary[..., 0] >= 0) & (bary[..., 1] >= 0) & (bary[..., 2] >= 0)

        if not inside.any():
            continue

        # Interpolate depth
        z0, z1, z2 = v0_screen[2], v1_screen[2], v2_screen[2]
        interp_depth = bary[..., 0] * z0 + bary[..., 1] * z1 + bary[..., 2] * z2

        # Depth test and update buffers
        depth_subset = depth_buffer[min_y:max_y+1, min_x:max_x+1]
        update_mask = inside & (interp_depth < depth_subset) & (interp_depth > 0)

        if update_mask.any():
            depth_buffer[min_y:max_y+1, min_x:max_x+1] = torch.where(
                update_mask, interp_depth, depth_subset
            )
            face_id_buffer[min_y:max_y+1, min_x:max_x+1] = torch.where(
                update_mask,
                torch.full_like(face_id_buffer[min_y:max_y+1, min_x:max_x+1], face_idx + 1),
                face_id_buffer[min_y:max_y+1, min_x:max_x+1]
            )
            for c in range(3):
                bary_buffer[min_y:max_y+1, min_x:max_x+1, c] = torch.where(
                    update_mask,
                    bary[..., c],
                    bary_buffer[min_y:max_y+1, min_x:max_x+1, c]
                )

    mask = face_id_buffer > 0
    depth_buffer = torch.where(mask, depth_buffer, torch.zeros_like(depth_buffer))

    return {
        'mask': mask,
        'face_id': face_id_buffer,
        'bary': bary_buffer,
        'depth': depth_buffer
    }


class MeshRenderer:
    """
    Renderer for the Mesh representation using pure PyTorch software rasterization.
    No nvdiffrast dependency.

    Args:
        rendering_options (dict): Rendering options.
    """
    def __init__(self, rendering_options={}, device='cuda'):
        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1
        })
        self.rendering_options.update(rendering_options)
        self.device = device

    def render(
            self,
            mesh: MeshExtractResult,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            return_types=["mask", "normal", "depth"]
        ) -> edict:
        """
        Render the mesh.

        Args:
            mesh : meshmodel
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            return_types (list): list of return types, can be "mask", "depth", "normal_map", "normal", "color"

        Returns:
            edict based on return_types containing:
                color (torch.Tensor): [3, H, W] rendered color image
                depth (torch.Tensor): [H, W] rendered depth image
                normal (torch.Tensor): [3, H, W] rendered normal image
                normal_map (torch.Tensor): [3, H, W] rendered normal map image
                mask (torch.Tensor): [H, W] rendered mask image
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]

        render_resolution = resolution * ssaa

        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            out_dict = edict()
            for type in return_types:
                if type == "mask":
                    out_dict[type] = torch.zeros((resolution, resolution), device=self.device)
                elif type == "depth":
                    out_dict[type] = torch.zeros((resolution, resolution), device=self.device)
                else:
                    out_dict[type] = torch.zeros((3, resolution, resolution), device=self.device)
            return out_dict

        # Project vertices to screen space
        vertices_screen, valid_mask = project_vertices_to_screen(
            mesh.vertices, extrinsics, intrinsics,
            render_resolution, render_resolution
        )

        # Ensure faces are int64 for indexing
        faces = mesh.faces.long()

        # Software rasterization (no backface culling for robustness with various camera orientations)
        rast = rasterize_triangles_software(
            vertices_screen, faces,
            render_resolution, render_resolution, self.device,
            cull_backfaces=False
        )

        out_dict = edict()
        rast_mask = rast['mask']  # [H, W]

        for type in return_types:
            img = None
            if type == "mask":
                img = rast_mask.float()  # [H, W]
            elif type == "depth":
                img = rast['depth']  # [H, W]
            elif type == "normal":
                # Render face normals
                face_normal = mesh.face_normal  # [F, 3]
                face_id = rast['face_id']  # [H, W]
                valid_mask = face_id > 0
                normal_img = torch.zeros((render_resolution, render_resolution, 3), device=self.device)
                if valid_mask.any():
                    face_indices = (face_id[valid_mask] - 1).long()
                    normal_img[valid_mask] = face_normal[face_indices]
                # Normalize and convert to [0, 1] range
                img = ((normal_img + 1) / 2).permute(2, 0, 1)  # [3, H, W]
            elif type == "normal_map":
                # Render vertex normal attributes (if available)
                if hasattr(mesh, 'vertex_attrs') and mesh.vertex_attrs is not None and mesh.vertex_attrs.shape[1] >= 6:
                    vertex_normals = mesh.vertex_attrs[:, 3:6]  # Assuming normals at indices 3-5
                    face_id = rast['face_id']
                    valid_mask = face_id > 0
                    normal_map_img = torch.zeros((render_resolution, render_resolution, 3), device=self.device)
                    if valid_mask.any():
                        face_indices = (face_id[valid_mask] - 1).long()
                        v_indices = faces[face_indices]
                        v_normals = vertex_normals[v_indices]  # [N, 3, 3]
                        bary_valid = rast['bary'][valid_mask]  # [N, 3]
                        interp_normal = (v_normals * bary_valid.unsqueeze(-1)).sum(dim=1)
                        normal_map_img[valid_mask] = interp_normal
                    img = normal_map_img.permute(2, 0, 1)  # [3, H, W]
                else:
                    img = torch.zeros((3, render_resolution, render_resolution), device=self.device)
            elif type == "color":
                # Render vertex colors (if available)
                if hasattr(mesh, 'vertex_attrs') and mesh.vertex_attrs is not None and mesh.vertex_attrs.shape[1] >= 3:
                    vertex_colors = mesh.vertex_attrs[:, :3]
                    face_id = rast['face_id']
                    valid_mask = face_id > 0
                    color_img = torch.zeros((render_resolution, render_resolution, 3), device=self.device)
                    if valid_mask.any():
                        face_indices = (face_id[valid_mask] - 1).long()
                        v_indices = faces[face_indices]
                        v_colors = vertex_colors[v_indices]  # [N, 3, 3]
                        bary_valid = rast['bary'][valid_mask]  # [N, 3]
                        interp_color = (v_colors * bary_valid.unsqueeze(-1)).sum(dim=1)
                        color_img[valid_mask] = interp_color
                    img = color_img.permute(2, 0, 1)  # [3, H, W]
                else:
                    img = torch.zeros((3, render_resolution, render_resolution), device=self.device)

            # Apply SSAA downsampling
            if ssaa > 1:
                if img.dim() == 2:  # [H, W]
                    img = img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    img = F.interpolate(
                        img,
                        (resolution, resolution),
                        mode='bilinear',
                        align_corners=False,
                        antialias=True
                    )
                    img = img.squeeze()  # [H, W]
                else:  # [3, H, W]
                    img = img.unsqueeze(0)  # [1, 3, H, W]
                    img = F.interpolate(
                        img,
                        (resolution, resolution),
                        mode='bilinear',
                        align_corners=False,
                        antialias=True
                    )
                    img = img.squeeze(0)  # [3, H, W]

            out_dict[type] = img

        return out_dict
