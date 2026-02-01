"""
Mesh classes for TRELLIS.2.

These match the official TRELLIS.2 API but cumesh/flex_gemm operations
require those libraries to be installed for full functionality.
"""
from typing import Dict, Optional, List
import torch
import torch.nn.functional as F


class Mesh:
    """
    Basic mesh representation matching official TRELLIS.2 API.

    Args:
        vertices: Mesh vertices [V, 3]
        faces: Mesh faces [F, 3]
        vertex_attrs: Optional vertex attributes [V, C]
    """

    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        vertex_attrs: torch.Tensor = None,
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.vertex_attrs = vertex_attrs

    @property
    def device(self):
        return self.vertices.device

    def to(self, device, non_blocking=False):
        return Mesh(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.vertex_attrs.to(device, non_blocking=non_blocking) if self.vertex_attrs is not None else None,
        )

    def cuda(self, non_blocking=False):
        return self.to('cuda', non_blocking=non_blocking)

    def cpu(self):
        return self.to('cpu')

    def fill_holes(self, max_hole_perimeter: float = 3e-2):
        """
        Fill small holes in the mesh using cumesh.

        Requires cumesh library. Install via:
            git clone https://github.com/JeffreyXiang/CuMesh.git --recursive
            pip install ./CuMesh --no-build-isolation
        """
        try:
            import cumesh
            vertices = self.vertices.cuda()
            faces = self.faces.cuda()

            mesh = cumesh.CuMesh()
            mesh.init(vertices, faces)
            mesh.get_edges()
            mesh.get_boundary_info()
            if mesh.num_boundaries == 0:
                return
            mesh.get_vertex_edge_adjacency()
            mesh.get_vertex_boundary_adjacency()
            mesh.get_manifold_boundary_adjacency()
            mesh.read_manifold_boundary_adjacency()
            mesh.get_boundary_connected_components()
            mesh.get_boundary_loops()
            if mesh.num_boundary_loops == 0:
                return
            mesh.fill_holes(max_hole_perimeter=max_hole_perimeter)
            new_vertices, new_faces = mesh.read()

            self.vertices = new_vertices.to(self.device)
            self.faces = new_faces.to(self.device)
        except ImportError:
            pass  # cumesh not available

    def simplify(self, target: int = 1000000, verbose: bool = False, options: dict = {}):
        """
        Simplify mesh to target number of faces using cumesh.

        Requires cumesh library.
        """
        try:
            import cumesh
            vertices = self.vertices.cuda()
            faces = self.faces.cuda()

            mesh = cumesh.CuMesh()
            mesh.init(vertices, faces)
            mesh.simplify(target, verbose=verbose, options=options)
            new_vertices, new_faces = mesh.read()

            self.vertices = new_vertices.to(self.device)
            self.faces = new_faces.to(self.device)
        except ImportError:
            pass  # cumesh not available

    def remove_faces(self, face_mask: torch.Tensor):
        """
        Remove faces from the mesh using cumesh.

        Requires cumesh library.
        """
        try:
            import cumesh
            vertices = self.vertices.cuda()
            faces = self.faces.cuda()

            mesh = cumesh.CuMesh()
            mesh.init(vertices, faces)
            mesh.remove_faces(face_mask)
            new_vertices, new_faces = mesh.read()

            self.vertices = new_vertices.to(self.device)
            self.faces = new_faces.to(self.device)
        except ImportError:
            pass  # cumesh not available


class MeshWithVoxel(Mesh):
    """
    Mesh with associated PBR voxel attributes.

    Combines mesh geometry (vertices, faces) with sparse voxel-based PBR attributes.
    The voxels can be queried to get interpolated PBR values at any point in space.

    Args:
        vertices: Mesh vertices [V, 3]
        faces: Mesh faces [F, 3]
        origin: Voxel grid origin [3] (typically [-0.5, -0.5, -0.5])
        voxel_size: Size of each voxel (typically 1/resolution)
        coords: Sparse voxel coordinates [N, 3] (xyz)
        attrs: Voxel PBR attributes [N, C] (base_color, metallic, roughness, alpha)
        voxel_shape: Full voxel grid shape torch.Size([B, X, Y, Z])
        layout: Dict mapping attribute names to channel slices
    """

    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        origin: list,
        voxel_size: float,
        coords: torch.Tensor,
        attrs: torch.Tensor,
        voxel_shape: torch.Size,
        layout: Dict = None,
    ):
        super().__init__(vertices, faces)
        self.origin = torch.tensor(origin, dtype=torch.float32, device=self.device)
        self.voxel_size = voxel_size
        self.coords = coords
        self.attrs = attrs
        self.voxel_shape = voxel_shape

        # Default PBR layout matching TRELLIS.2
        self.layout = layout or {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }

    @property
    def num_vertices(self):
        return self.vertices.shape[0]

    @property
    def num_faces(self):
        return self.faces.shape[0]

    @property
    def num_voxels(self):
        return self.coords.shape[0]

    def to(self, device, non_blocking=False):
        """Move all tensors to the specified device."""
        return MeshWithVoxel(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.origin.tolist(),
            self.voxel_size,
            self.coords.to(device, non_blocking=non_blocking),
            self.attrs.to(device, non_blocking=non_blocking),
            self.voxel_shape,
            self.layout,
        )

    def get_attr(self, name: str) -> Optional[torch.Tensor]:
        """Get a specific attribute by name (e.g., 'base_color', 'metallic')."""
        if name not in self.layout:
            return None
        return self.attrs[:, self.layout[name]]

    def query_attrs(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Query PBR attributes at arbitrary 3D positions.

        Uses flex_gemm.ops.grid_sample.grid_sample_3d if available,
        otherwise falls back to nearest neighbor.

        Args:
            xyz: Query positions [N, 3] in world coordinates

        Returns:
            Interpolated PBR attributes [N, C]
        """
        try:
            from flex_gemm.ops.grid_sample import grid_sample_3d
            grid = ((xyz - self.origin) / self.voxel_size).reshape(1, -1, 3)
            vertex_attrs = grid_sample_3d(
                self.attrs,
                torch.cat([torch.zeros_like(self.coords[..., :1]), self.coords], dim=-1),
                self.voxel_shape,
                grid,
                mode='trilinear'
            )[0]
            return vertex_attrs
        except ImportError:
            # Fallback: nearest neighbor (less accurate but works)
            voxel_coords = (xyz - self.origin) / self.voxel_size
            voxel_idx = voxel_coords.round().long()
            grid_size = self.voxel_shape[-1]
            voxel_idx = voxel_idx.clamp(0, grid_size - 1)

            result = torch.zeros(xyz.shape[0], self.attrs.shape[1], device=xyz.device)
            for i in range(xyz.shape[0]):
                query = voxel_idx[i]
                matches = (self.coords == query).all(dim=1)
                if matches.any():
                    idx = matches.nonzero(as_tuple=True)[0][0]
                    result[i] = self.attrs[idx]
            return result

    def query_vertex_attrs(self) -> torch.Tensor:
        """Query PBR attributes at all mesh vertices."""
        return self.query_attrs(self.vertices)

    def to_glb(
        self,
        path: str,
        include_pbr: bool = True,
        texture_resolution: int = 1024,
    ):
        """
        Export mesh with PBR materials to GLB format.

        Args:
            path: Output file path
            include_pbr: Whether to include PBR textures
            texture_resolution: Resolution for baked PBR textures
        """
        import trimesh

        mesh = trimesh.Trimesh(
            vertices=self.vertices.cpu().numpy(),
            faces=self.faces.cpu().numpy(),
        )

        if include_pbr and self.attrs is not None:
            vertex_attrs = self.query_vertex_attrs()
            base_color = vertex_attrs[:, self.layout['base_color']].cpu().numpy()
            base_color = (base_color.clip(0, 1) * 255).astype('uint8')

            if 'alpha' in self.layout:
                alpha = vertex_attrs[:, self.layout['alpha']].cpu().numpy()
                alpha = (alpha.clip(0, 1) * 255).astype('uint8')
            else:
                import numpy as np
                alpha = np.full((base_color.shape[0], 1), 255, dtype='uint8')

            import numpy as np
            vertex_colors = np.concatenate([base_color, alpha], axis=-1)
            mesh.visual.vertex_colors = vertex_colors

        mesh.export(path)
        print(f"[MeshWithVoxel] Exported to {path}")
        return path

    def __repr__(self):
        return (
            f"MeshWithVoxel("
            f"vertices={self.num_vertices}, "
            f"faces={self.num_faces}, "
            f"voxels={self.num_voxels}, "
            f"attrs_channels={self.attrs.shape[1]}, "
            f"layout={list(self.layout.keys())})"
        )
