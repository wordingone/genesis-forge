from typing import *
import torch
from ..voxel import Voxel
import cumesh
from flex_gemm.ops.grid_sample import grid_sample_3d


class Mesh:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None
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
    
    def fill_holes(self, max_hole_perimeter=3e-2):
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
        
    def remove_faces(self, face_mask: torch.Tensor):
        vertices = self.vertices.cuda()
        faces = self.faces.cuda()

        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.remove_faces(face_mask)
        new_vertices, new_faces = mesh.read()

        self.vertices = new_vertices.to(self.device)
        self.faces = new_faces.to(self.device)
        
    def simplify(self, target=1000000, verbose: bool=False, options: dict={}):
        vertices = self.vertices.cuda()
        faces = self.faces.cuda()

        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.simplify(target, verbose=verbose, options=options)
        new_vertices, new_faces = mesh.read()

        self.vertices = new_vertices.to(self.device)
        self.faces = new_faces.to(self.device)


class TextureFilterMode:
    CLOSEST = 0
    LINEAR = 1


class TextureWrapMode:
    CLAMP_TO_EDGE = 0
    REPEAT = 1
    MIRRORED_REPEAT = 2


class AlphaMode:
    OPAQUE = 0
    MASK = 1
    BLEND = 2


class Texture:
    def __init__(
        self,
        image: torch.Tensor,
        filter_mode: TextureFilterMode = TextureFilterMode.LINEAR,
        wrap_mode: TextureWrapMode = TextureWrapMode.REPEAT
    ):
        self.image = image
        self.filter_mode = filter_mode
        self.wrap_mode = wrap_mode

    def to(self, device, non_blocking=False):
        return Texture(
            self.image.to(device, non_blocking=non_blocking),
            self.filter_mode,
            self.wrap_mode,
        )


class PbrMaterial:
    def __init__(
        self,
        base_color_texture: Optional[Texture] = None,
        base_color_factor: Union[torch.Tensor, List[float]] = [1.0, 1.0, 1.0],
        metallic_texture: Optional[Texture] = None,
        metallic_factor: float = 1.0,
        roughness_texture: Optional[Texture] = None,
        roughness_factor: float = 1.0,
        alpha_texture: Optional[Texture] = None,
        alpha_factor: float = 1.0,
        alpha_mode: AlphaMode = AlphaMode.OPAQUE,
        alpha_cutoff: float = 0.5,
    ):
        self.base_color_texture = base_color_texture
        self.base_color_factor = torch.tensor(base_color_factor, dtype=torch.float32)[:3]
        self.metallic_texture = metallic_texture
        self.metallic_factor = metallic_factor
        self.roughness_texture = roughness_texture
        self.roughness_factor = roughness_factor
        self.alpha_texture = alpha_texture
        self.alpha_factor = alpha_factor
        self.alpha_mode = alpha_mode
        self.alpha_cutoff = alpha_cutoff

    def to(self, device, non_blocking=False):
        return PbrMaterial(
            base_color_texture=self.base_color_texture.to(device, non_blocking=non_blocking) if self.base_color_texture is not None else None,
            base_color_factor=self.base_color_factor.to(device, non_blocking=non_blocking),
            metallic_texture=self.metallic_texture.to(device, non_blocking=non_blocking) if self.metallic_texture is not None else None,
            metallic_factor=self.metallic_factor,
            roughness_texture=self.roughness_texture.to(device, non_blocking=non_blocking) if self.roughness_texture is not None else None,
            roughness_factor=self.roughness_factor,
            alpha_texture=self.alpha_texture.to(device, non_blocking=non_blocking) if self.alpha_texture is not None else None,
            alpha_factor=self.alpha_factor,
            alpha_mode=self.alpha_mode,
            alpha_cutoff=self.alpha_cutoff,
        )


class MeshWithPbrMaterial(Mesh):
    def __init__(self,
        vertices,
        faces,
        material_ids,
        uv_coords,
        materials: List[PbrMaterial],
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.material_ids = material_ids    # [M]
        self.uv_coords = uv_coords          # [M, 3, 2]
        self.materials = materials
        self.layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }

    def to(self, device, non_blocking=False):
        return MeshWithPbrMaterial(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.material_ids.to(device, non_blocking=non_blocking),
            self.uv_coords.to(device, non_blocking=non_blocking),
            [material.to(device, non_blocking=non_blocking) for material in self.materials],
        )


class MeshWithVoxel(Mesh, Voxel):
    def __init__(self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        origin: list,
        voxel_size: float,
        coords: torch.Tensor,
        attrs: torch.Tensor,
        voxel_shape: torch.Size,
        layout: Dict = {},
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.origin = torch.tensor(origin, dtype=torch.float32, device=self.device)
        self.voxel_size = voxel_size
        self.coords = coords
        self.attrs = attrs
        self.voxel_shape = voxel_shape
        self.layout = layout

    def to(self, device, non_blocking=False):
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
        
    def query_attrs(self, xyz):
        grid = ((xyz - self.origin) / self.voxel_size).reshape(1, -1, 3)
        vertex_attrs = grid_sample_3d(
            self.attrs,
            torch.cat([torch.zeros_like(self.coords[..., :1]), self.coords], dim=-1),
            self.voxel_shape,
            grid,
            mode='trilinear'
        )[0]
        return vertex_attrs
        
    def query_vertex_attrs(self):
        return self.query_attrs(self.vertices)

    def to_dense_voxel_grid(self) -> torch.Tensor:
        """
        Convert sparse voxel representation to dense 3D grid for trilinear sampling.

        Returns:
            [1, C, X, Y, Z] dense voxel grid with PBR attributes

        Note: PyTorch grid_sample expects input [N, C, D, H, W] where last 3 dims are depth/height/width.
        We treat these as X, Y, Z respectively to match coords ordering.
        """
        # Get grid dimensions from voxel_shape
        # voxel_shape from SparseTensor is [batch, channels, X_max+1, Y_max+1, Z_max+1]
        if len(self.voxel_shape) >= 4:
            X_size, Y_size, Z_size = self.voxel_shape[-3], self.voxel_shape[-2], self.voxel_shape[-1]
        else:
            # Fallback: infer from max coords
            X_size = Y_size = Z_size = int(self.coords.max().item()) + 1

        C = self.attrs.shape[1]

        # Create dense grid with shape [1, C, X, Y, Z]
        # PyTorch grid_sample will treat dims as [N, C, D, H, W] = [N, C, X, Y, Z]
        grid = torch.zeros(1, C, X_size, Y_size, Z_size, device=self.attrs.device, dtype=self.attrs.dtype)

        # Fill in sparse voxels
        # coords: [N, 3] with (x, y, z) ordering
        x = self.coords[:, 0].long()
        y = self.coords[:, 1].long()
        z = self.coords[:, 2].long()

        # Clamp to valid range
        x = x.clamp(0, X_size - 1)
        y = y.clamp(0, Y_size - 1)
        z = z.clamp(0, Z_size - 1)

        # attrs: [N, C]
        # Index as grid[:, :, x, y, z] to match the coordinate ordering
        grid[0, :, x, y, z] = self.attrs.T

        return grid

    def export(self, path: str, simplify: float = 0.95, texture_size: int = 2048, decimation_target: int = 300000,
               remesh: bool = True, remesh_band: float = 1.0, remesh_project: float = 0.0, verbose: bool = True):
        """
        Export mesh with PBR materials to GLB format.
        Uses official TRELLIS.2 pattern: nvdiffrast + cuBVH + trilinear sampling.

        Official HuggingFace demo parameters: decimation_target=300000, texture_size=2048

        This matches official TRELLIS.2 quality by using:
        - cumesh.CuMesh for CUDA-accelerated mesh simplification
        - cumesh.cuBVH for projecting simplified vertices back to original mesh
        - cumesh.remeshing.remesh_narrow_band_dc for Dual Contouring remeshing (fixes fragmentation)
        - nvdiffrast for CUDA-accelerated UV rasterization
        - xatlas for UV unwrapping
        - PyTorch trilinear sampling for PBR attribute lookup
        - OpenCV inpainting for texture completion

        Args:
            path: Output file path (.glb)
            simplify: Ratio of faces to REMOVE (0.95 = keep 5% of faces). Ignored if decimation_target set.
            texture_size: Output texture resolution
            decimation_target: Absolute number of target faces (official TRELLIS.2 uses 1000000)
            remesh: Use Dual Contouring remeshing to fix fragmentation (default True, matches official)
            remesh_band: Size of the remeshing band (default 1.0)
            remesh_project: Projection factor for snapping to original surface (0.0 = no projection, 0.9 = strong)
            verbose: Print progress information
        """
        import numpy as np
        import torch
        import torch.nn.functional as F
        import trimesh
        import trimesh.visual
        from PIL import Image
        import cv2
        import nvdiffrast.torch as dr
        import cumesh
        # Note: cumesh.uv_unwrap uses xatlas internally, no need to import directly

        if verbose:
            print(f"[TRELLIS.2 EXPORT] Starting GLB export")
            print(f"[TRELLIS.2 EXPORT] Original: {self.vertices.shape[0]} vertices, {self.faces.shape[0]} faces")
            print(f"[TRELLIS.2 EXPORT] Voxels: {self.coords.shape[0]}, attrs: {self.attrs.shape}")

        # Free stale GPU allocations before mesh processing.
        # Move voxel data (coords/attrs) to CPU — they're only needed later
        # for query_attrs() texture baking, not for mesh geometry operations.
        # This frees ~750MB of GPU memory for CuMesh operations.
        import gc
        self.coords = self.coords.cpu()
        self.attrs = self.attrs.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        if verbose:
            alloc = torch.cuda.memory_allocated() / 1024**3
            free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
            print(f"[TRELLIS.2 EXPORT] GPU: alloc={alloc:.2f}GB free={free:.2f}GB")

        # Keep original mesh on GPU for BVH
        orig_vertices = self.vertices.cuda()
        orig_faces = self.faces.cuda()

        # Step 1: Pre-simplify large meshes before fill_holes.
        # CuMesh fill_holes builds full edge/boundary adjacency structures that
        # require several GB of raw CUDA memory for 29M+ face meshes.  Pre-simplify
        # to a manageable size first; the geometry is preserved since QEM simplification
        # keeps the visual shape.  The BVH is built on the pre-simplified mesh.
        fill_holes_limit = 5_000_000  # fill_holes comfortably handles 5M faces
        if orig_faces.shape[0] > fill_holes_limit:
            if verbose:
                print(f"[TRELLIS.2 EXPORT] Pre-simplifying for fill_holes: {orig_faces.shape[0]} -> {fill_holes_limit} faces...")
                import sys; sys.stdout.flush()
            pre_mesh = cumesh.CuMesh()
            pre_mesh.init(orig_vertices, orig_faces)
            pre_mesh.simplify(fill_holes_limit)
            orig_vertices, orig_faces = pre_mesh.read()
            del pre_mesh
            torch.cuda.empty_cache()
            if verbose:
                print(f"[TRELLIS.2 EXPORT] Pre-simplified: {orig_vertices.shape[0]} vertices, {orig_faces.shape[0]} faces")

        # Step 2: Build BVH for mesh
        if verbose:
            print("[TRELLIS.2 EXPORT] Building BVH...")
        bvh = cumesh.cuBVH(orig_vertices, orig_faces)

        # Step 3: Initial mesh cleaning (official pipeline step)
        if verbose:
            print("[TRELLIS.2 EXPORT] Filling holes...")
        mesh = cumesh.CuMesh()
        mesh.init(orig_vertices, orig_faces)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"[TRELLIS.2 EXPORT] After filling holes: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        orig_vertices, orig_faces = mesh.read()
        del mesh
        torch.cuda.empty_cache()

        # Rebuild BVH after cleaning
        bvh = cumesh.cuBVH(orig_vertices, orig_faces)

        # Step 3: Mesh processing - either remesh (fixes fragmentation) or simplify only
        import sys
        original_faces = orig_faces.shape[0]

        # Use decimation_target if provided (official TRELLIS.2 uses 1000000)
        # Otherwise fall back to percentage-based calculation
        if decimation_target is not None:
            target_faces = decimation_target
        elif simplify > 0:
            target_faces = int(original_faces * (1 - simplify))
        else:
            target_faces = original_faces  # No simplification

        if remesh:
            # === REMESH BRANCH (official TRELLIS.2 default) ===
            # Uses Dual Contouring to rebuild mesh topology, eliminating fragmentation
            if verbose:
                print(f"[TRELLIS.2 EXPORT] Using Dual Contouring remeshing to fix fragmentation...")
                sys.stdout.flush()

            # Pre-simplify if input mesh is too large for integer overflow safety.
            # At resolution 1024, remeshing roughly doubles face count.
            pre_remesh_limit = 4_000_000
            if orig_faces.shape[0] > pre_remesh_limit:
                if verbose:
                    print(f"[TRELLIS.2 EXPORT] Pre-remesh simplification: {orig_faces.shape[0]} -> {pre_remesh_limit} faces...")
                    sys.stdout.flush()
                pre_mesh = cumesh.CuMesh()
                pre_mesh.init(orig_vertices, orig_faces)
                pre_mesh.simplify(pre_remesh_limit, verbose=verbose)
                orig_vertices, orig_faces = pre_mesh.read()
                del pre_mesh
                torch.cuda.empty_cache()
                # Rebuild BVH with simplified mesh
                bvh = cumesh.cuBVH(orig_vertices, orig_faces)
                if verbose:
                    print(f"[TRELLIS.2 EXPORT] Pre-remesh result: {orig_vertices.shape[0]} vertices, {orig_faces.shape[0]} faces")
                    sys.stdout.flush()

            # Use official TRELLIS.2 fixed AABB: [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
            # The mesh is always generated in normalized [-0.5, 0.5] space
            official_aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device='cuda')
            center = official_aabb.mean(dim=0)  # [0.0, 0.0, 0.0]
            scale = (official_aabb[1] - official_aabb[0]).max().item()  # 1.0

            # Get grid resolution from voxel_shape
            if len(self.voxel_shape) >= 5:
                W, H, D = self.voxel_shape[-3], self.voxel_shape[-2], self.voxel_shape[-1]
            elif len(self.voxel_shape) >= 3:
                W, H, D = self.voxel_shape[-3], self.voxel_shape[-2], self.voxel_shape[-1]
            else:
                W = H = D = int(self.coords.max().item()) + 1
            grid_size = max(W, H, D)

            if verbose:
                print(f"[TRELLIS.2 EXPORT] Remesh params (official): center={center.tolist()}, scale={scale:.4f}, resolution={grid_size}, band={remesh_band}")
                sys.stdout.flush()

            # Perform Dual Contouring remeshing
            # This rebuilds the mesh with clean topology, eliminating fragments
            # Free the external BVH before calling remeshing — the function builds
            # its own BVH internally, and we need maximum free GPU memory for the
            # narrow-band voxel operations (~37M voxels at resolution 1024).
            del bvh
            gc.collect()
            torch.cuda.empty_cache()

            if verbose:
                alloc = torch.cuda.memory_allocated() / 1024**3
                free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
                print(f"[TRELLIS.2 EXPORT] Before remeshing: GPU alloc={alloc:.2f}GB free={free_mem:.2f}GB")
                sys.stdout.flush()

            import cumesh.remeshing
            remeshed_verts, remeshed_faces = cumesh.remeshing.remesh_narrow_band_dc(
                orig_vertices, orig_faces,
                center=center,
                scale=(grid_size + 3 * remesh_band) / grid_size * scale,
                resolution=grid_size,
                band=remesh_band,
                project_back=remesh_project,
                verbose=verbose,
                bvh=None,  # Let remeshing build its own BVH
            )

            mesh = cumesh.CuMesh()
            mesh.init(remeshed_verts, remeshed_faces)
            if verbose:
                print(f"[TRELLIS.2 EXPORT] After remeshing: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
                sys.stdout.flush()

            # Simplify the remeshed result using staged reduction (prevents integer overflow)
            if mesh.num_faces > target_faces:
                current_faces = mesh.num_faces

                # For very large meshes (>5M faces), use staged simplification
                # cumesh.simplify can overflow with large face counts in a single call
                if current_faces > 5_000_000:
                    # Define stages: each stage reduces by ~50% to avoid overflow
                    stages = []
                    face_count = current_faces
                    while face_count > max(target_faces * 2, 3_000_000):
                        next_target = int(face_count * 0.5)
                        if next_target < target_faces * 2:
                            break
                        stages.append(next_target)
                        face_count = next_target

                    if verbose and stages:
                        print(f"[TRELLIS.2 EXPORT] Planning {len(stages)} staged simplification: {current_faces} -> {' -> '.join(str(s) for s in stages)} -> {target_faces}")
                        sys.stdout.flush()

                    for stage_idx, stage_target in enumerate(stages):
                        if verbose:
                            print(f"[TRELLIS.2 EXPORT] Stage {stage_idx + 1}/{len(stages)}: {mesh.num_faces} -> {stage_target} faces...")
                            sys.stdout.flush()
                        mesh.simplify(stage_target, verbose=verbose)
                        mesh.remove_duplicate_faces()
                        torch.cuda.empty_cache()
                        if verbose:
                            print(f"[TRELLIS.2 EXPORT] Stage {stage_idx + 1} complete: {mesh.num_faces} faces")
                            sys.stdout.flush()

                # Final simplification to target
                if mesh.num_faces > target_faces:
                    if verbose:
                        print(f"[TRELLIS.2 EXPORT] Final simplification: {mesh.num_faces} -> {target_faces} faces...")
                        sys.stdout.flush()
                    mesh.simplify(target_faces, verbose=verbose)
                    if verbose:
                        print(f"[TRELLIS.2 EXPORT] After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
                        sys.stdout.flush()

            simp_vertices, simp_faces = mesh.read()

        else:
            # === STANDARD BRANCH (no remeshing, preserves original topology) ===
            simp_vertices = orig_vertices
            simp_faces = orig_faces
            mesh = cumesh.CuMesh()
            mesh.init(orig_vertices, orig_faces)

            if target_faces > 100 and target_faces < original_faces:
                if verbose:
                    print(f"[TRELLIS.2 EXPORT] Simplifying mesh with CuMesh (target: {target_faces} faces)...")

                # For very large meshes, use gradual staged simplification
                # to avoid memory allocation failures in cumesh
                current_faces = original_faces

                # Define stages: each stage reduces by at most ~35%
                stages = []
                face_count = original_faces
                while face_count > max(target_faces * 2, 2_000_000):
                    next_target = int(face_count * 0.65)
                    if next_target < target_faces * 2:
                        break
                    stages.append(next_target)
                    face_count = next_target

                if verbose and stages:
                    print(f"[TRELLIS.2 EXPORT] Planning {len(stages)} gradual reduction stages: {original_faces} -> {' -> '.join(str(s) for s in stages)} -> {target_faces}")
                    sys.stdout.flush()

                for stage_idx, stage_target in enumerate(stages):
                    if verbose:
                        print(f"[TRELLIS.2 EXPORT] Stage {stage_idx + 1}/{len(stages)}: Reducing from {current_faces} to {stage_target} faces...")
                        sys.stdout.flush()
                    mesh.simplify(stage_target, verbose=verbose)
                    mesh.remove_duplicate_faces()
                    mesh.remove_small_connected_components(1e-5)
                    current_faces = mesh.num_faces
                    if verbose:
                        print(f"[TRELLIS.2 EXPORT] Stage {stage_idx + 1} complete: {mesh.num_vertices} vertices, {current_faces} faces")
                        sys.stdout.flush()
                    torch.cuda.empty_cache()

                # Initial aggressive simplification (3x target or current)
                initial_target = min(target_faces * 3, current_faces - 1)
                if current_faces > initial_target:
                    if verbose:
                        print(f"[TRELLIS.2 EXPORT] Stage 3: Reducing from {current_faces} to {initial_target} faces...")
                        sys.stdout.flush()
                    mesh.simplify(initial_target, verbose=verbose)
                    if verbose:
                        print(f"[TRELLIS.2 EXPORT] After initial simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
                        sys.stdout.flush()

                # Clean up topology
                mesh.remove_duplicate_faces()
                mesh.repair_non_manifold_edges()
                mesh.remove_small_connected_components(1e-5)
                mesh.fill_holes(max_hole_perimeter=3e-2)
                if verbose:
                    print(f"[TRELLIS.2 EXPORT] After cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
                    sys.stdout.flush()

                # Final simplification to target count
                if mesh.num_faces > target_faces:
                    mesh.simplify(target_faces, verbose=verbose)
                    if verbose:
                        print(f"[TRELLIS.2 EXPORT] After final simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
                        sys.stdout.flush()

                # Final cleanup
                mesh.remove_duplicate_faces()
                mesh.repair_non_manifold_edges()
                mesh.remove_small_connected_components(1e-5)
                mesh.fill_holes(max_hole_perimeter=3e-2)

                # Unify face orientations
                mesh.unify_face_orientations()

                simp_vertices, simp_faces = mesh.read()
                if verbose:
                    print(f"[TRELLIS.2 EXPORT] After final cleanup: {simp_vertices.shape[0]} vertices, {simp_faces.shape[0]} faces")
                    sys.stdout.flush()

        # Step 4: UV unwrapping using cumesh (uses accelerated clustering + xatlas)
        # This is the official pipeline method - better quality than direct xatlas
        if verbose:
            print("[TRELLIS.2 EXPORT] UV unwrapping with cumesh (accelerated clustering + xatlas)...")
            sys.stdout.flush()

        # Clear GPU memory before UV unwrapping (heavy operation)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Need to reinitialize mesh with cleaned vertices for uv_unwrap
        mesh.init(simp_vertices, simp_faces)

        try:
            out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
                compute_charts_kwargs={
                    "threshold_cone_half_angle_rad": np.radians(90.0),
                    "refine_iterations": 0,
                    "global_iterations": 1,
                    "smooth_strength": 1,
                },
                return_vmaps=True,
                verbose=verbose,
            )
        except Exception as e:
            if verbose:
                print(f"[TRELLIS.2 EXPORT] UV unwrap failed: {e}")
                print("[TRELLIS.2 EXPORT] Retrying with smaller mesh...")
                sys.stdout.flush()

            # If UV unwrap fails, try with even smaller mesh
            torch.cuda.empty_cache()
            reduced_target = min(500000, simp_faces.shape[0] // 2)
            mesh.simplify(reduced_target, verbose=verbose)
            mesh.remove_duplicate_faces()
            mesh.repair_non_manifold_edges()
            simp_vertices, simp_faces = mesh.read()
            if verbose:
                print(f"[TRELLIS.2 EXPORT] Reduced mesh: {simp_vertices.shape[0]} vertices, {simp_faces.shape[0]} faces")
                sys.stdout.flush()

            mesh.init(simp_vertices, simp_faces)
            out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
                compute_charts_kwargs={
                    "threshold_cone_half_angle_rad": np.radians(90.0),
                    "refine_iterations": 0,
                    "global_iterations": 1,
                    "smooth_strength": 1,
                },
                return_vmaps=True,
                verbose=verbose,
            )
        out_vertices = out_vertices.cuda()
        out_faces = out_faces.cuda()
        out_uvs = out_uvs.cuda()

        # Compute vertex normals for export
        mesh.compute_vertex_normals()
        out_normals = mesh.read_vertex_normals()[out_vmaps.cuda()]

        if verbose:
            print(f"[TRELLIS.2 EXPORT] UV unwrap: {out_uvs.shape[0]} UV coords, {out_vertices.shape[0]} vertices")
            sys.stdout.flush()

        # Step 4.5: Remove degenerate faces AFTER UV unwrapping
        # xatlas can create zero-area triangles at UV seams - remove them to prevent black bar artifacts
        v0 = out_vertices[out_faces[:, 0]]
        v1 = out_vertices[out_faces[:, 1]]
        v2 = out_vertices[out_faces[:, 2]]
        face_areas = torch.linalg.norm(torch.cross(v1 - v0, v2 - v0), dim=-1) / 2
        valid_face_mask = face_areas > 1e-10
        if verbose:
            degenerate_count = (~valid_face_mask).sum().item()
            if degenerate_count > 0:
                print(f"[TRELLIS.2 EXPORT] Removed {degenerate_count} degenerate faces after UV unwrap")
        out_faces = out_faces[valid_face_mask]

        # Step 5: nvdiffrast UV rasterization (CRITICAL - replaces Python loop)
        if verbose:
            print(f"[TRELLIS.2 EXPORT] Rasterizing UV space at {texture_size}x{texture_size} with nvdiffrast...")

        ctx = dr.RasterizeCudaContext()
        # Prepare UV coords for rasterization: [u, v] -> [u*2-1, v*2-1, 0, 1]
        uvs_rast = torch.cat([
            out_uvs * 2 - 1,
            torch.zeros_like(out_uvs[:, :1]),
            torch.ones_like(out_uvs[:, :1])
        ], dim=-1).unsqueeze(0)

        # Rasterize in chunks (100k faces at a time for memory efficiency)
        rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)
        for i in range(0, out_faces.shape[0], 100000):
            rast_chunk, _ = dr.rasterize(
                ctx, uvs_rast, out_faces[i:i+100000],
                resolution=[texture_size, texture_size]
            )
            mask_chunk = rast_chunk[..., 3:4] > 0
            rast_chunk[..., 3:4] += i  # Store face ID offset
            rast = torch.where(mask_chunk, rast_chunk, rast)

        mask = rast[0, ..., 3] > 0

        if verbose:
            print(f"[TRELLIS.2 EXPORT] Raster mask: {mask.sum().item()} / {mask.numel()} pixels ({mask.sum().item() / mask.numel() * 100:.1f}%)")

        # Check if rasterization produced valid output
        if mask.sum() == 0:
            if verbose:
                print("[TRELLIS.2 EXPORT] WARNING: No valid pixels in UV rasterization!")
                print(f"[TRELLIS.2 EXPORT] UV stats: min={out_uvs.min(dim=0).values.tolist()}, max={out_uvs.max(dim=0).values.tolist()}")
                print(f"[TRELLIS.2 EXPORT] Face stats: min={out_faces.min().item()}, max={out_faces.max().item()}, verts={out_vertices.shape[0]}")
            # Try to debug by checking if UVs are in expected range [0, 1]
            # If not, normalize them
            uv_min = out_uvs.min()
            uv_max = out_uvs.max()
            if uv_min < 0 or uv_max > 1:
                if verbose:
                    print(f"[TRELLIS.2 EXPORT] Normalizing UVs from [{uv_min:.4f}, {uv_max:.4f}] to [0, 1]")
                out_uvs = (out_uvs - uv_min) / (uv_max - uv_min + 1e-8)
                # Re-rasterize with normalized UVs
                uvs_rast = torch.cat([
                    out_uvs * 2 - 1,
                    torch.zeros_like(out_uvs[:, :1]),
                    torch.ones_like(out_uvs[:, :1])
                ], dim=-1).unsqueeze(0)
                rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)
                for i in range(0, out_faces.shape[0], 100000):
                    rast_chunk, _ = dr.rasterize(
                        ctx, uvs_rast, out_faces[i:i+100000],
                        resolution=[texture_size, texture_size]
                    )
                    mask_chunk = rast_chunk[..., 3:4] > 0
                    rast_chunk[..., 3:4] += i
                    rast = torch.where(mask_chunk, rast_chunk, rast)
                mask = rast[0, ..., 3] > 0
                if verbose:
                    print(f"[TRELLIS.2 EXPORT] After UV normalization: {mask.sum().item()} pixels")

        # Step 5: Interpolate 3D positions in UV space
        if verbose:
            print("[TRELLIS.2 EXPORT] Interpolating 3D positions...")
        pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
        valid_pos = pos[mask]

        # Step 6: BVH projection to original mesh for accurate attribute sampling
        # Rebuild BVH if it was deleted during remeshing (remesh branch deletes it for memory)
        try:
            bvh
        except NameError:
            if verbose:
                print("[TRELLIS.2 EXPORT] Rebuilding BVH for texture sampling...")
            bvh = cumesh.cuBVH(orig_vertices, orig_faces)
        if verbose:
            print("[TRELLIS.2 EXPORT] Projecting to original mesh via BVH...")
        _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
        orig_tri_verts = orig_vertices[orig_faces[face_id.long()]]  # [N, 3, 3]
        valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

        # Step 7: Sample PBR attributes using TRILINEAR interpolation
        # Use flex_gemm.grid_sample_3d directly (compiled for Windows) - NOT the PyTorch fallback
        # PyTorch fallback causes moire/contour artifacts due to subtle differences in interpolation
        if verbose:
            print("[TRELLIS.2 EXPORT] Sampling PBR attributes via trilinear interpolation...")

        from flex_gemm.ops.grid_sample import grid_sample_3d

        C = self.attrs.shape[1]  # 6 channels: RGB, metallic, roughness, alpha

        # Grid dimensions from voxel_shape
        # voxel_shape format: [B, C, W, H, D]
        if len(self.voxel_shape) >= 5:
            W, H, D = self.voxel_shape[-3], self.voxel_shape[-2], self.voxel_shape[-1]
        elif len(self.voxel_shape) >= 3:
            W, H, D = self.voxel_shape[-3], self.voxel_shape[-2], self.voxel_shape[-1]
        else:
            W = H = D = int(self.coords.max().item()) + 1

        if verbose:
            print(f"[TRELLIS.2 EXPORT] Voxel grid size: {W}x{H}x{D}, channels: {C}")
            grid_mem_mb = (W * H * D * C * 4) / (1024 * 1024)
            print(f"[TRELLIS.2 EXPORT] Dense grid memory estimate: {grid_mem_mb:.1f} MB")

        # Convert world positions to voxel coordinates
        # CRITICAL FIX: The official example passes SCALAR voxel_size to to_glb()
        # In postprocess.py, grid_size is computed as: ((aabb[1] - aabb[0]) / voxel_size).round()
        # This gives grid_size = [512, 512, 512] when voxel_size = 0.001953125 and aabb = 1.0
        #
        # The voxel_shape [1, 6, 478, 384, 512] shows the ACTUAL occupied extent,
        # but the coordinate system is based on the SCALAR voxel_size!
        #
        # So we must use: pos_voxel = (pos - origin) / scalar_voxel_size
        # NOT: pos_voxel = (pos - aabb[0]) / per_axis_voxel_size

        if verbose:
            print(f"[TRELLIS.2 EXPORT] Using mesh.origin={self.origin.tolist()} and scalar voxel_size={self.voxel_size}")

        # Compute voxel coordinates using SCALAR voxel size (matching official)
        pos_voxel = (valid_pos - self.origin.cuda()) / self.voxel_size

        if verbose:
            print(f"[TRELLIS.2 EXPORT] Trilinear sampling {valid_pos.shape[0]} positions...")
            if valid_pos.shape[0] > 0:
                print(f"[TRELLIS.2 EXPORT] Voxel coord range: [{pos_voxel.min(dim=0).values.tolist()}] to [{pos_voxel.max(dim=0).values.tolist()}]")
            else:
                print(f"[TRELLIS.2 EXPORT] WARNING: No valid positions to sample!")

        # Handle empty valid_pos gracefully
        if valid_pos.shape[0] == 0:
            print("[TRELLIS.2 EXPORT] ERROR: No valid texture positions. Export will have no texture data.")
            sampled = torch.zeros(0, C, device='cuda')
        else:
            # Prepare inputs for grid_sample_3d (matches flex_gemm interface exactly)
            # feats: [N, C] sparse features
            # coords: [N, 4] with batch index at column 0
            # shape: [B, C, W, H, D]
            # grid: [B, L, 3] query positions in VOXEL coordinates
            coords_cuda = self.coords.cuda()
            coords_4d = torch.cat([
                torch.zeros_like(coords_cuda[:, :1]),  # batch index = 0
                coords_cuda
            ], dim=-1)  # [N, 4]

            shape = torch.Size([1, C, W, H, D])
            grid = pos_voxel.reshape(1, -1, 3)  # [1, L, 3]

            # Sample using our flex_gemm-compatible implementation
            sampled = grid_sample_3d(
                self.attrs.cuda(),  # [N, C]
                coords_4d,          # [N, 4]
                shape,              # [1, C, W, H, D]
                grid,               # [1, L, 3]
                mode='trilinear'
            )  # Output: [1, L, C]

            # Reshape to [N, C]
            sampled = sampled.squeeze(0)  # [L, C]

            if verbose:
                print(f"[TRELLIS.2 EXPORT] Trilinear sampling complete")

        # Build output textures
        attrs = torch.zeros(texture_size, texture_size, C, device='cuda')
        attrs[mask] = sampled

        coverage = mask.sum().item() / (texture_size * texture_size) * 100
        if verbose:
            print(f"[TRELLIS.2 EXPORT] UV coverage: {coverage:.1f}%")

        # Step 8: Inpainting and material construction
        if verbose:
            print("[TRELLIS.2 EXPORT] Inpainting unmapped regions...")
        mask_np = mask.cpu().numpy()
        mask_inv = (~mask_np).astype(np.uint8)

        # Extract channels (layout: 0:3=RGB, 3=metallic, 4=roughness, 5=alpha)
        base_color = np.clip(attrs[..., 0:3].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., 3:4].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., 4:5].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., 5:6].cpu().numpy() * 255, 0, 255).astype(np.uint8)

        base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
        metallic = cv2.inpaint(metallic.squeeze(-1), mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        roughness = cv2.inpaint(roughness.squeeze(-1), mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        alpha = cv2.inpaint(alpha.squeeze(-1), mask_inv, 1, cv2.INPAINT_TELEA)[..., None]

        if verbose:
            print(f"[TRELLIS.2 EXPORT] Base color: min={base_color.min()}, max={base_color.max()}")
            print(f"[TRELLIS.2 EXPORT] Metallic: min={metallic.min()}, max={metallic.max()}")
            print(f"[TRELLIS.2 EXPORT] Roughness: min={roughness.min()}, max={roughness.max()}")
            # Analyze color channels
            print(f"[TRELLIS.2 EXPORT] RGB Channel means: R={base_color[mask_np][..., 0].mean():.1f}, G={base_color[mask_np][..., 1].mean():.1f}, B={base_color[mask_np][..., 2].mean():.1f}")
            print(f"[TRELLIS.2 EXPORT] RGB Channel maxes: R={base_color[mask_np][..., 0].max()}, G={base_color[mask_np][..., 1].max()}, B={base_color[mask_np][..., 2].max()}")

        # Step 9: Create GLB with PBR material
        # Coordinate conversion (Z-up to Y-up for GLB) - matches official exactly
        vertices_out = out_vertices.cpu().numpy()
        faces_out = out_faces.cpu().numpy()
        uvs_out = out_uvs.cpu().numpy()
        normals_out = out_normals.cpu().numpy()

        # Swap Y and Z axes, invert Y (official conversion for GLB compatibility)
        vertices_out[:, 1], vertices_out[:, 2] = vertices_out[:, 2].copy(), -vertices_out[:, 1].copy()
        normals_out[:, 1], normals_out[:, 2] = normals_out[:, 2].copy(), -normals_out[:, 1].copy()
        uvs_out[:, 1] = 1 - uvs_out[:, 1]

        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode='OPAQUE',
            doubleSided=not remesh,
        )

        final_mesh = trimesh.Trimesh(
            vertices=vertices_out,
            faces=faces_out,
            vertex_normals=normals_out,
            process=False,
            visual=trimesh.visual.TextureVisuals(uv=uvs_out, material=material)
        )

        if verbose:
            print(f"[TRELLIS.2 EXPORT] Writing {path}...")
        final_mesh.export(path)

        import os
        size = os.path.getsize(path)
        if verbose:
            print(f"[TRELLIS.2 EXPORT] Complete! File size: {size/1024:.1f} KB")
