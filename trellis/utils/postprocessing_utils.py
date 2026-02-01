from typing import *
import numpy as np
import torch
import torch.nn.functional as F
import utils3d
from tqdm import tqdm
import trimesh
import trimesh.visual
import xatlas
import pyvista as pv
from pymeshfix import _meshfix
import igraph
import cv2
from PIL import Image
from .random_utils import sphere_hammersley_sequence
from .render_utils import render_multiview
from ..renderers import GaussianRenderer
from ..representations import Strivec, Gaussian, MeshExtractResult


def differentiable_texture_sample(texture: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """
    Differentiable texture sampling using PyTorch grid_sample.
    Replaces nvdiffrast dr.texture for optimization mode.

    Args:
        texture: [1, H, W, C] texture tensor (stored top-to-bottom in memory)
        uv: [1, H, W, 2] UV coordinates in [0, 1] range (OpenGL convention: (0,0)=bottom-left)

    Returns:
        [1, H, W, C] sampled colors

    Note on coordinate systems:
        - nvdiffrast dr.texture: UV (0,0) = bottom-left of texture, (1,1) = top-right
        - PyTorch grid_sample: grid (-1,-1) = top-left of tensor, (1,1) = bottom-right
        - Texture tensor: index [0,0] = top-left in memory (standard image convention)

        For dr.texture compatibility:
        - UV.x (horizontal): 0->1 maps to left->right, same as grid.x -1->1
        - UV.y (vertical): 0->1 maps to bottom->top in texture
        - In memory, bottom = last row (index H-1), top = first row (index 0)
        - grid.y -1->1 maps to index 0->H-1 (top->bottom)
        - So UV.y=0 (bottom) needs grid.y=1, UV.y=1 (top) needs grid.y=-1
        - Therefore: grid.y = 1 - 2*UV.y = -(UV.y*2 - 1)
    """
    # Convert UV from [0,1] to [-1,1] for grid_sample
    # UV.x: 0->1 maps to grid.x: -1->1 (left to right) - standard conversion
    # UV.y: 0->1 maps to grid.y: 1->-1 (bottom to top, which is high index to low index)
    grid_x = uv[..., 0] * 2 - 1  # [0,1] -> [-1,1]
    grid_y = 1 - uv[..., 1] * 2  # [0,1] -> [1,-1] (flipped)
    grid = torch.stack([grid_x, grid_y], dim=-1)

    # texture: [1, H, W, C] -> [1, C, H, W] for grid_sample
    tex_nchw = texture.permute(0, 3, 1, 2)

    # Sample using bilinear interpolation
    sampled = F.grid_sample(
        tex_nchw,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    )

    # [1, C, H, W] -> [1, H, W, C]
    return sampled.permute(0, 2, 3, 1)


@torch.no_grad()
def _fill_holes(
    verts,
    faces,
    max_hole_size=0.04,
    max_hole_nbe=32,
    resolution=128,
    num_views=500,
    debug=False,
    verbose=False
):
    """
    Rasterize a mesh from multiple views and remove invisible faces.
    Also includes postprocessing to:
        1. Remove connected components that are have low visibility.
        2. Mincut to remove faces at the inner side of the mesh connected to the outer side with a small hole.

    Args:
        verts (torch.Tensor): Vertices of the mesh. Shape (V, 3).
        faces (torch.Tensor): Faces of the mesh. Shape (F, 3).
        max_hole_size (float): Maximum area of a hole to fill.
        resolution (int): Resolution of the rasterization.
        num_views (int): Number of views to rasterize the mesh.
        verbose (bool): Whether to print progress.
    """
    # Construct cameras
    yaws = []
    pitchs = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views)
        yaws.append(y)
        pitchs.append(p)
    yaws = torch.tensor(yaws).cuda()
    pitchs = torch.tensor(pitchs).cuda()
    radius = 2.0
    fov = torch.deg2rad(torch.tensor(40)).cuda()
    projection = utils3d.torch.perspective_from_fov_xy(fov, fov, 1, 3)
    views = []
    for (yaw, pitch) in zip(yaws, pitchs):
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda().float() * radius
        view = utils3d.torch.view_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        views.append(view)
    views = torch.stack(views, dim=0)

    # Rasterize
    visblity = torch.zeros(faces.shape[0], dtype=torch.int32, device=verts.device)
    rastctx = utils3d.torch.RastContext(backend='cuda')
    for i in tqdm(range(views.shape[0]), total=views.shape[0], disable=not verbose, desc='Rasterizing'):
        view = views[i]
        buffers = utils3d.torch.rasterize_triangle_faces(
            rastctx, verts[None], faces, resolution, resolution, view=view, projection=projection
        )
        face_id = buffers['face_id'][0][buffers['mask'][0] > 0.95] - 1
        face_id = torch.unique(face_id).long()
        visblity[face_id] += 1
    visblity = visblity.float() / num_views

    # Mincut
    ## construct outer faces
    edges, face2edge, edge_degrees = utils3d.torch.compute_edges(faces)
    boundary_edge_indices = torch.nonzero(edge_degrees == 1).reshape(-1)
    connected_components = utils3d.torch.compute_connected_components(faces, edges, face2edge)
    outer_face_indices = torch.zeros(faces.shape[0], dtype=torch.bool, device=faces.device)
    for i in range(len(connected_components)):
        outer_face_indices[connected_components[i]] = visblity[connected_components[i]] > min(max(visblity[connected_components[i]].quantile(0.75).item(), 0.25), 0.5)
    outer_face_indices = outer_face_indices.nonzero().reshape(-1)

    ## construct inner faces
    inner_face_indices = torch.nonzero(visblity == 0).reshape(-1)
    if verbose:
        tqdm.write(f'Found {inner_face_indices.shape[0]} invisible faces')
    if inner_face_indices.shape[0] == 0:
        return verts, faces

    ## Construct dual graph (faces as nodes, edges as edges)
    dual_edges, dual_edge2edge = utils3d.torch.compute_dual_graph(face2edge)
    dual_edge2edge = edges[dual_edge2edge]
    dual_edges_weights = torch.norm(verts[dual_edge2edge[:, 0]] - verts[dual_edge2edge[:, 1]], dim=1)
    if verbose:
        tqdm.write(f'Dual graph: {dual_edges.shape[0]} edges')

    ## solve mincut problem
    ### construct main graph
    g = igraph.Graph()
    g.add_vertices(faces.shape[0])
    g.add_edges(dual_edges.cpu().numpy())
    g.es['weight'] = dual_edges_weights.cpu().numpy()

    ### source and target
    g.add_vertex('s')
    g.add_vertex('t')

    ### connect invisible faces to source
    g.add_edges([(f, 's') for f in inner_face_indices], attributes={'weight': torch.ones(inner_face_indices.shape[0], dtype=torch.float32).cpu().numpy()})

    ### connect outer faces to target
    g.add_edges([(f, 't') for f in outer_face_indices], attributes={'weight': torch.ones(outer_face_indices.shape[0], dtype=torch.float32).cpu().numpy()})

    ### solve mincut
    cut = g.mincut('s', 't', (np.array(g.es['weight']) * 1000).tolist())
    remove_face_indices = torch.tensor([v for v in cut.partition[0] if v < faces.shape[0]], dtype=torch.long, device=faces.device)
    if verbose:
        tqdm.write(f'Mincut solved, start checking the cut')

    ### check if the cut is valid with each connected component
    to_remove_cc = utils3d.torch.compute_connected_components(faces[remove_face_indices])
    if debug:
        tqdm.write(f'Number of connected components of the cut: {len(to_remove_cc)}')
    valid_remove_cc = []
    cutting_edges = []
    for cc in to_remove_cc:
        #### check if the connected component has low visibility
        visblity_median = visblity[remove_face_indices[cc]].median()
        if debug:
            tqdm.write(f'visblity_median: {visblity_median}')
        if visblity_median > 0.25:
            continue

        #### check if the cuting loop is small enough
        cc_edge_indices, cc_edges_degree = torch.unique(face2edge[remove_face_indices[cc]], return_counts=True)
        cc_boundary_edge_indices = cc_edge_indices[cc_edges_degree == 1]
        cc_new_boundary_edge_indices = cc_boundary_edge_indices[~torch.isin(cc_boundary_edge_indices, boundary_edge_indices)]
        if len(cc_new_boundary_edge_indices) > 0:
            cc_new_boundary_edge_cc = utils3d.torch.compute_edge_connected_components(edges[cc_new_boundary_edge_indices])
            cc_new_boundary_edges_cc_center = [verts[edges[cc_new_boundary_edge_indices[edge_cc]]].mean(dim=1).mean(dim=0) for edge_cc in cc_new_boundary_edge_cc]
            cc_new_boundary_edges_cc_area = []
            for i, edge_cc in enumerate(cc_new_boundary_edge_cc):
                _e1 = verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 0]] - cc_new_boundary_edges_cc_center[i]
                _e2 = verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 1]] - cc_new_boundary_edges_cc_center[i]
                cc_new_boundary_edges_cc_area.append(torch.norm(torch.cross(_e1, _e2, dim=-1), dim=1).sum() * 0.5)
            if debug:
                cutting_edges.append(cc_new_boundary_edge_indices)
                tqdm.write(f'Area of the cutting loop: {cc_new_boundary_edges_cc_area}')
            if any([l > max_hole_size for l in cc_new_boundary_edges_cc_area]):
                continue

        valid_remove_cc.append(cc)

    if debug:
        face_v = verts[faces].mean(dim=1).cpu().numpy()
        vis_dual_edges = dual_edges.cpu().numpy()
        vis_colors = np.zeros((faces.shape[0], 3), dtype=np.uint8)
        vis_colors[inner_face_indices.cpu().numpy()] = [0, 0, 255]
        vis_colors[outer_face_indices.cpu().numpy()] = [0, 255, 0]
        vis_colors[remove_face_indices.cpu().numpy()] = [255, 0, 255]
        if len(valid_remove_cc) > 0:
            vis_colors[remove_face_indices[torch.cat(valid_remove_cc)].cpu().numpy()] = [255, 0, 0]
        utils3d.io.write_ply('dbg_dual.ply', face_v, edges=vis_dual_edges, vertex_colors=vis_colors)

        vis_verts = verts.cpu().numpy()
        vis_edges = edges[torch.cat(cutting_edges)].cpu().numpy()
        utils3d.io.write_ply('dbg_cut.ply', vis_verts, edges=vis_edges)


    if len(valid_remove_cc) > 0:
        remove_face_indices = remove_face_indices[torch.cat(valid_remove_cc)]
        mask = torch.ones(faces.shape[0], dtype=torch.bool, device=faces.device)
        mask[remove_face_indices] = 0
        faces = faces[mask]
        faces, verts = utils3d.torch.remove_unreferenced_vertices(faces, verts)
        if verbose:
            tqdm.write(f'Removed {(~mask).sum()} faces by mincut')
    else:
        if verbose:
            tqdm.write(f'Removed 0 faces by mincut')

    mesh = _meshfix.PyTMesh()
    mesh.load_array(verts.cpu().numpy(), faces.cpu().numpy())
    mesh.fill_small_boundaries(nbe=max_hole_nbe, refine=True)
    verts, faces = mesh.return_arrays()
    verts, faces = torch.tensor(verts, device='cuda', dtype=torch.float32), torch.tensor(faces, device='cuda', dtype=torch.int32)

    return verts, faces


def postprocess_mesh(
    vertices: np.array,
    faces: np.array,
    simplify: bool = True,
    simplify_ratio: float = 0.9,
    fill_holes: bool = True,
    fill_holes_max_hole_size: float = 0.04,
    fill_holes_max_hole_nbe: int = 32,
    fill_holes_resolution: int = 1024,
    fill_holes_num_views: int = 1000,
    debug: bool = False,
    verbose: bool = False,
):
    """
    Postprocess a mesh by simplifying, removing invisible faces, and removing isolated pieces.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        simplify (bool): Whether to simplify the mesh, using quadric edge collapse.
        simplify_ratio (float): Ratio of faces to keep after simplification.
        fill_holes (bool): Whether to fill holes in the mesh.
        fill_holes_max_hole_size (float): Maximum area of a hole to fill.
        fill_holes_max_hole_nbe (int): Maximum number of boundary edges of a hole to fill.
        fill_holes_resolution (int): Resolution of the rasterization.
        fill_holes_num_views (int): Number of views to rasterize the mesh.
        verbose (bool): Whether to print progress.
    """

    if verbose:
        tqdm.write(f'Before postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    # Simplify
    if simplify and simplify_ratio > 0:
        mesh = pv.PolyData(vertices, np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=1))
        mesh = mesh.decimate(simplify_ratio, progress_bar=verbose)
        vertices, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
        if verbose:
            tqdm.write(f'After decimate: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    # Remove invisible faces
    if fill_holes:
        vertices, faces = torch.tensor(vertices).cuda(), torch.tensor(faces.astype(np.int32)).cuda()
        vertices, faces = _fill_holes(
            vertices, faces,
            max_hole_size=fill_holes_max_hole_size,
            max_hole_nbe=fill_holes_max_hole_nbe,
            resolution=fill_holes_resolution,
            num_views=fill_holes_num_views,
            debug=debug,
            verbose=verbose,
        )
        vertices, faces = vertices.cpu().numpy(), faces.cpu().numpy()
        if verbose:
            tqdm.write(f'After remove invisible faces: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    return vertices, faces


def parametrize_mesh(vertices: np.array, faces: np.array):
    """
    Parametrize a mesh to a texture space, using xatlas.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
    """

    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

    vertices = vertices[vmapping]
    faces = indices

    return vertices, faces, uvs


def bake_texture(
    vertices: np.array,
    faces: np.array,
    uvs: np.array,
    observations: List[np.array],
    masks: List[np.array],
    extrinsics: List[np.array],
    intrinsics: List[np.array],
    texture_size: int = 2048,
    near: float = 0.1,
    far: float = 10.0,
    mode: Literal['fast', 'opt'] = 'opt',
    lambda_tv: float = 1e-2,
    verbose: bool = False,
):
    """
    Bake texture to a mesh from multiple observations.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        uvs (np.array): UV coordinates of the mesh. Shape (V, 2).
        observations (List[np.array]): List of observations. Each observation is a 2D image. Shape (H, W, 3).
        masks (List[np.array]): List of masks. Each mask is a 2D image. Shape (H, W).
        extrinsics (List[np.array]): List of extrinsics. Shape (4, 4).
        intrinsics (List[np.array]): List of intrinsics. Shape (3, 3).
        texture_size (int): Size of the texture.
        near (float): Near plane of the camera.
        far (float): Far plane of the camera.
        mode (Literal['fast', 'opt']): Mode of texture baking.
        lambda_tv (float): Weight of total variation loss in optimization.
        verbose (bool): Whether to print progress.
    """
    vertices = torch.tensor(vertices).cuda()
    faces = torch.tensor(faces.astype(np.int32)).cuda()
    uvs = torch.tensor(uvs).cuda()
    observations = [torch.tensor(obs / 255.0).float().cuda() for obs in observations]
    masks = [torch.tensor(m>0).bool().cuda() for m in masks]
    views = [utils3d.torch.extrinsics_to_view(torch.tensor(extr).cuda()) for extr in extrinsics]
    projections = [utils3d.torch.intrinsics_to_perspective(torch.tensor(intr).cuda(), near, far) for intr in intrinsics]

    # DEBUG: Log input statistics for texture baking
    print(f"[TEXTURE BAKE DEBUG] Mode: {mode}, texture_size: {texture_size}")
    print(f"[TEXTURE BAKE DEBUG] Vertices range: min={vertices.min().item():.4f}, max={vertices.max().item():.4f}")
    print(f"[TEXTURE BAKE DEBUG] UVs range: min={uvs.min().item():.4f}, max={uvs.max().item():.4f}")
    obs_tensor_stats = [f"view{i}: mean={obs.mean().item():.4f}" for i, obs in enumerate(observations[:3])]
    print(f"[TEXTURE BAKE DEBUG] First 3 observation means: {obs_tensor_stats}")
    mask_coverage = [m.sum().item() / m.numel() for m in masks[:3]]
    print(f"[TEXTURE BAKE DEBUG] First 3 mask coverage: {[f'{c*100:.1f}%' for c in mask_coverage]}")

    if mode == 'fast':
        texture = torch.zeros((texture_size * texture_size, 3), dtype=torch.float32).cuda()
        texture_weights = torch.zeros((texture_size * texture_size), dtype=torch.float32).cuda()
        rastctx = utils3d.torch.RastContext(backend='cuda')
        for observation, view, projection in tqdm(zip(observations, views, projections), total=len(observations), disable=not verbose, desc='Texture baking (fast)'):
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx, vertices[None], faces, observation.shape[1], observation.shape[0], uv=uvs[None], view=view, projection=projection
                )
                uv_map = rast['uv'][0].detach().flip(0)
                mask = rast['mask'][0].detach().bool() & masks[0]

            # nearest neighbor interpolation
            uv_map = (uv_map * texture_size).floor().long()
            obs = observation[mask]
            uv_map = uv_map[mask]
            idx = uv_map[:, 0] + (texture_size - uv_map[:, 1] - 1) * texture_size
            texture = texture.scatter_add(0, idx.view(-1, 1).expand(-1, 3), obs)
            texture_weights = texture_weights.scatter_add(0, idx, torch.ones((obs.shape[0]), dtype=torch.float32, device=texture.device))

        mask = texture_weights > 0
        texture[mask] /= texture_weights[mask][:, None]
        texture = np.clip(texture.reshape(texture_size, texture_size, 3).cpu().numpy() * 255, 0, 255).astype(np.uint8)

        # inpaint
        mask = (texture_weights == 0).cpu().numpy().astype(np.uint8).reshape(texture_size, texture_size)
        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)

    elif mode == 'opt':
        rastctx = utils3d.torch.RastContext(backend='cuda')
        # NOTE: Official TRELLIS flips observations here for dr.texture compatibility.
        # However, our differentiable_texture_sample uses different conventions.
        # Let's NOT flip and work in consistent coordinate space.
        _uv = []
        _masks_rast = []  # Use rasterization mask instead of flipped observation mask
        for observation, mask, view, projection in tqdm(zip(observations, masks, views, projections), total=len(views), disable=not verbose, desc='Texture baking (opt): UV'):
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx, vertices[None], faces, observation.shape[1], observation.shape[0], uv=uvs[None], view=view, projection=projection
                )
                _uv.append(rast['uv'].detach())
                # Use the rasterization mask which is aligned with the UV map
                rast_mask = rast['mask'][0].detach().bool() & mask
                _masks_rast.append(rast_mask)

        texture = torch.nn.Parameter(torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32).cuda())
        optimizer = torch.optim.Adam([texture], betas=(0.5, 0.9), lr=1e-2)

        # Log UV stats for diagnostics
        uv_mins = [uv.min().item() for uv in _uv]
        uv_maxs = [uv.max().item() for uv in _uv]
        print(f"[TEXTURE OPT] UV range: [{min(uv_mins):.4f}, {max(uv_maxs):.4f}]")

        def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
            return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))

        def tv_loss(texture):
            return torch.nn.functional.l1_loss(texture[:, :-1, :, :], texture[:, 1:, :, :]) + \
                   torch.nn.functional.l1_loss(texture[:, :, :-1, :], texture[:, :, 1:, :])

        # Debug: Check observations for NaN/Inf before optimization
        for i, obs in enumerate(observations):
            if torch.isnan(obs).any() or torch.isinf(obs).any():
                print(f"[TEXTURE DEBUG] WARNING: observation {i} contains NaN/Inf!")
            if masks[i].sum() == 0:
                print(f"[TEXTURE DEBUG] WARNING: mask {i} is empty!")

        total_steps = 2500
        nan_count = 0
        with tqdm(total=total_steps, disable=not verbose, desc='Texture baking (opt): optimizing') as pbar:
            for step in range(total_steps):
                optimizer.zero_grad()
                selected = np.random.randint(0, len(views))
                # Use rasterization-aligned mask instead of flipped observation mask
                uv, observation, mask = _uv[selected], observations[selected], _masks_rast[selected]

                # Skip views with empty masks to avoid NaN loss
                if mask.sum() == 0:
                    pbar.update()
                    continue

                # Use pure PyTorch texture sampling instead of nvdiffrast
                render = differentiable_texture_sample(texture, uv)[0]
                loss = torch.nn.functional.l1_loss(render[mask], observation[mask])
                if lambda_tv > 0:
                    loss += lambda_tv * tv_loss(texture)

                # Skip backward if loss is NaN to prevent texture corruption
                if torch.isnan(loss):
                    nan_count += 1
                    if nan_count <= 5:
                        print(f"[TEXTURE DEBUG] NaN loss at step {step}, view {selected}, mask_sum={mask.sum().item()}")
                    pbar.update()
                    continue

                loss.backward()
                optimizer.step()
                # annealing
                optimizer.param_groups[0]['lr'] = cosine_anealing(optimizer, step, total_steps, 1e-2, 1e-5)
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()


        if nan_count > 0:
            print(f"[TEXTURE DEBUG] Total NaN losses skipped: {nan_count}/{total_steps}")

        # DEBUG: Log texture stats before post-processing
        texture_raw = texture[0].detach()
        print(f"[TEXTURE OPT DEBUG] Final texture (before flip): min={texture_raw.min().item():.4f}, max={texture_raw.max().item():.4f}, mean={texture_raw.mean().item():.4f}")

        texture = np.clip(texture[0].flip(0).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)

        # DEBUG: Log texture stats after conversion
        print(f"[TEXTURE OPT DEBUG] Final texture (uint8): min={texture.min()}, max={texture.max()}, mean={texture.mean():.1f}")

        mask = 1 - utils3d.torch.rasterize_triangle_faces(
            rastctx, (uvs * 2 - 1)[None], faces, texture_size, texture_size
        )['mask'][0].detach().cpu().numpy().astype(np.uint8)

        # DEBUG: Log inpaint mask coverage
        inpaint_coverage = mask.sum() / mask.size
        print(f"[TEXTURE OPT DEBUG] Inpaint mask coverage: {inpaint_coverage*100:.1f}% of texture needs inpainting")

        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)

        # DEBUG: Final texture after inpainting
        print(f"[TEXTURE OPT DEBUG] After inpainting: min={texture.min()}, max={texture.max()}, mean={texture.mean():.1f}")
    else:
        raise ValueError(f'Unknown mode: {mode}')

    # DEBUG: Log final texture color distribution
    print(f"[TEXTURE DEBUG] Final texture shape: {texture.shape}")
    r_mean, g_mean, b_mean = texture[:,:,0].mean(), texture[:,:,1].mean(), texture[:,:,2].mean()
    print(f"[TEXTURE DEBUG] Final texture RGB means: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")

    return texture


def to_glb(
    app_rep: Union[Strivec, Gaussian],
    mesh: MeshExtractResult,
    simplify: float = 0.95,
    fill_holes: bool = True,
    fill_holes_max_size: float = 0.04,
    texture_size: int = 1024,
    debug: bool = False,
    verbose: bool = True,
) -> trimesh.Trimesh:
    """
    Convert a generated asset to a glb file.

    Args:
        app_rep (Union[Strivec, Gaussian]): Appearance representation.
        mesh (MeshExtractResult): Extracted mesh.
        simplify (float): Ratio of faces to remove in simplification.
        fill_holes (bool): Whether to fill holes in the mesh.
        fill_holes_max_size (float): Maximum area of a hole to fill.
        texture_size (int): Size of the texture.
        debug (bool): Whether to print debug information.
        verbose (bool): Whether to print progress.
    """
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()

    # mesh postprocess
    max_hole_nbe = int(500 * np.sqrt(1-simplify))  # Increased from 250 to fill larger holes
    print(f"[MESH DEBUG] Postprocess: simplify={simplify:.2f} ({(1-simplify)*100:.0f}% faces kept), max_hole_nbe={max_hole_nbe}")
    print(f"[MESH DEBUG] Before postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    vertices, faces = postprocess_mesh(
        vertices, faces,
        simplify=simplify > 0,
        simplify_ratio=simplify,
        fill_holes=fill_holes,
        fill_holes_max_hole_size=fill_holes_max_size,
        fill_holes_max_hole_nbe=max_hole_nbe,
        fill_holes_resolution=1024,
        fill_holes_num_views=1000,
        debug=debug,
        verbose=verbose,
    )
    print(f"[MESH DEBUG] After postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # parametrize mesh
    vertices, faces, uvs = parametrize_mesh(vertices, faces)

    # DEBUG: Log mesh and UV statistics
    print(f"[TEXTURE DEBUG] Mesh stats: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    print(f"[TEXTURE DEBUG] UV stats: min={uvs.min():.4f}, max={uvs.max():.4f}, shape={uvs.shape}")
    # Check UV is in [0,1] range
    if uvs.min() < 0 or uvs.max() > 1:
        print(f"[TEXTURE WARNING] UV coordinates outside [0,1] range! This will cause texture mapping issues.")

    # bake texture
    # Use 1024 for render_multiview - higher resolutions cause significant slowdown
    # The texture baking optimization handles upscaling adequately
    observations, extrinsics, intrinsics = render_multiview(app_rep, resolution=1024, nviews=100)

    # DEBUG: Log observation statistics
    obs_colors = [obs.mean() for obs in observations]
    obs_min = min([obs.min() for obs in observations])
    obs_max = max([obs.max() for obs in observations])
    obs_mean = np.mean(obs_colors)
    print(f"[TEXTURE DEBUG] Rendered {len(observations)} views at 1024x1024")
    print(f"[TEXTURE DEBUG] Observation pixel stats: min={obs_min}, max={obs_max}, mean={obs_mean:.2f}")
    # DEBUG: Per-channel analysis of first observation
    obs0 = observations[0]
    print(f"[TEXTURE DEBUG] Observation 0 RGB means: R={obs0[:,:,0].mean():.1f}, G={obs0[:,:,1].mean():.1f}, B={obs0[:,:,2].mean():.1f}")
    # Always save first observation to verify rendering colors
    import os
    debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs", "debug")
    os.makedirs(debug_dir, exist_ok=True)
    debug_obs_path = os.path.join(debug_dir, "observation_0.png")
    Image.fromarray(observations[0]).save(debug_obs_path)
    print(f"[TEXTURE DEBUG] Saved observation to: {debug_obs_path}")
    masks = [np.any(observation > 0, axis=-1) for observation in observations]
    extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
    intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]
    texture = bake_texture(
        vertices, faces, uvs,
        observations, masks, extrinsics, intrinsics,
        texture_size=texture_size, mode='opt',
        lambda_tv=0.01,
        verbose=verbose
    )
    texture = Image.fromarray(texture)

    # rotate mesh (from z-up to y-up)
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    material = trimesh.visual.material.PBRMaterial(
        roughnessFactor=1.0,
        baseColorTexture=texture,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8)
    )
    mesh = trimesh.Trimesh(vertices, faces, visual=trimesh.visual.TextureVisuals(uv=uvs, material=material))
    return mesh


def to_glb_pbr(
    app_rep: Union[Strivec, Gaussian],
    mesh: MeshExtractResult,
    simplify: float = 0.95,
    fill_holes: bool = True,
    fill_holes_max_size: float = 0.04,
    texture_size: int = 1024,
    metallic: float = 0.0,
    roughness: float = 1.0,
    debug: bool = False,
    verbose: bool = True,
) -> trimesh.Trimesh:
    """
    Convert a generated asset to a GLB file with full PBR materials.

    This is an enhanced version of to_glb() that exports with proper PBR
    metallic/roughness textures for TRELLIS.2 parity.

    Args:
        app_rep (Union[Strivec, Gaussian]): Appearance representation.
        mesh (MeshExtractResult): Extracted mesh.
        simplify (float): Ratio of faces to remove in simplification.
        fill_holes (bool): Whether to fill holes in the mesh.
        fill_holes_max_size (float): Maximum area of a hole to fill.
        texture_size (int): Size of the texture.
        metallic (float): Default metallic value (0-1). 0 = dielectric, 1 = metal.
        roughness (float): Default roughness value (0-1). 0 = smooth, 1 = rough.
        debug (bool): Whether to save debug textures.
        verbose (bool): Whether to print progress.

    Returns:
        trimesh.Trimesh: Mesh with PBR material.
    """
    from .pbr_texture_baking import bake_from_gaussian_observations, save_pbr_textures
    import os

    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()

    # mesh postprocess
    max_hole_nbe = int(500 * np.sqrt(1-simplify))
    print(f"[PBR GLB] Postprocess: simplify={simplify:.2f}, max_hole_nbe={max_hole_nbe}")
    print(f"[PBR GLB] Before postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    vertices, faces = postprocess_mesh(
        vertices, faces,
        simplify=simplify > 0,
        simplify_ratio=simplify,
        fill_holes=fill_holes,
        fill_holes_max_hole_size=fill_holes_max_size,
        fill_holes_max_hole_nbe=max_hole_nbe,
        fill_holes_resolution=1024,
        fill_holes_num_views=1000,
        debug=debug,
        verbose=verbose,
    )
    print(f"[PBR GLB] After postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # parametrize mesh
    vertices, faces, uvs = parametrize_mesh(vertices, faces)
    print(f"[PBR GLB] UV parametrization complete: {uvs.shape}")

    # Render observations from Gaussian representation
    observations, extrinsics, intrinsics = render_multiview(app_rep, resolution=1024, nviews=100)
    masks = [np.any(observation > 0, axis=-1) for observation in observations]
    extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
    intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]

    # Bake PBR textures
    print(f"[PBR GLB] Baking PBR textures at {texture_size}x{texture_size}...")
    base_color_tex, orm_tex = bake_from_gaussian_observations(
        vertices, faces, uvs,
        observations, masks, extrinsics, intrinsics,
        texture_size=texture_size,
        metallic_default=metallic,
        roughness_default=roughness,
        mode='opt',
        verbose=verbose
    )

    # Convert textures to PIL Images
    base_color_np = (base_color_tex.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    orm_np = (orm_tex.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    base_color_img = Image.fromarray(base_color_np)
    orm_img = Image.fromarray(orm_np)

    # Save debug textures if requested
    if debug:
        debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        base_color_img.save(os.path.join(debug_dir, "pbr_baseColor.png"))
        orm_img.save(os.path.join(debug_dir, "pbr_orm.png"))
        print(f"[PBR GLB] Saved debug textures to {debug_dir}")

    # Rotate mesh (from z-up to y-up)
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    # Create PBR material with metallic/roughness texture
    # glTF stores metallic-roughness in a combined texture:
    # - Blue channel: metallic
    # - Green channel: roughness
    # - Red channel: occlusion (optional)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base_color_img,
        metallicRoughnessTexture=orm_img,
        metallicFactor=1.0,  # Use texture values
        roughnessFactor=1.0,  # Use texture values
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8)
    )

    mesh_out = trimesh.Trimesh(
        vertices, faces,
        visual=trimesh.visual.TextureVisuals(uv=uvs, material=material)
    )

    print(f"[PBR GLB] Export complete with PBR materials")
    return mesh_out


def to_glb_from_ovoxel(
    ovoxel,  # OVoxel instance from TRELLIS.2 Stage 3
    mesh: MeshExtractResult,
    simplify: float = 0.95,
    fill_holes: bool = True,
    fill_holes_max_size: float = 0.04,
    texture_size: int = 1024,
    debug: bool = False,
    verbose: bool = True,
) -> trimesh.Trimesh:
    """
    Convert TRELLIS.2 OVoxel representation to GLB with native PBR materials.

    This is the TRELLIS.2 native path that uses per-voxel PBR attributes directly
    from the OVoxel representation, without needing to render and bake from
    Gaussian observations.

    Args:
        ovoxel: OVoxel instance with per-voxel PBR attributes from Stage 3
        mesh (MeshExtractResult): Extracted mesh from Stage 2
        simplify (float): Ratio of faces to remove in simplification
        fill_holes (bool): Whether to fill holes
        fill_holes_max_size (float): Maximum hole area to fill
        texture_size (int): Output texture resolution
        debug (bool): Save debug textures
        verbose (bool): Print progress

    Returns:
        trimesh.Trimesh: Mesh with PBR materials from OVoxel
    """
    from .pbr_texture_baking import bake_pbr_from_ovoxel, save_pbr_textures
    import os

    print(f"[OVOXEL GLB] Starting TRELLIS.2 native export")
    print(f"[OVOXEL GLB] OVoxel: {ovoxel.num_voxels} voxels, resolution={ovoxel.resolution}")

    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()

    # Mesh postprocessing
    max_hole_nbe = int(500 * np.sqrt(1 - simplify))
    print(f"[OVOXEL GLB] Postprocess: simplify={simplify:.2f}, max_hole_nbe={max_hole_nbe}")
    print(f"[OVOXEL GLB] Before postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    vertices, faces = postprocess_mesh(
        vertices, faces,
        simplify=simplify > 0,
        simplify_ratio=simplify,
        fill_holes=fill_holes,
        fill_holes_max_hole_size=fill_holes_max_size,
        fill_holes_max_hole_nbe=max_hole_nbe,
        fill_holes_resolution=1024,
        fill_holes_num_views=1000,
        debug=debug,
        verbose=verbose,
    )
    print(f"[OVOXEL GLB] After postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # UV parametrization
    vertices, faces, uvs = parametrize_mesh(vertices, faces)
    print(f"[OVOXEL GLB] UV parametrization complete: {uvs.shape}")

    # Bake PBR textures directly from OVoxel
    print(f"[OVOXEL GLB] Baking PBR textures from OVoxel at {texture_size}x{texture_size}...")
    base_color_tex, orm_tex = bake_pbr_from_ovoxel(
        ovoxel=ovoxel,
        vertices=vertices,
        faces=faces,
        uvs=uvs,
        texture_size=texture_size,
    )

    # Convert textures to PIL Images
    base_color_np = (base_color_tex.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    orm_np = (orm_tex.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    base_color_img = Image.fromarray(base_color_np)
    orm_img = Image.fromarray(orm_np)

    # Save debug textures
    if debug:
        debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        base_color_img.save(os.path.join(debug_dir, "ovoxel_baseColor.png"))
        orm_img.save(os.path.join(debug_dir, "ovoxel_orm.png"))
        print(f"[OVOXEL GLB] Saved debug textures to {debug_dir}")

    # Rotate mesh (z-up to y-up)
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    # Create PBR material
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base_color_img,
        metallicRoughnessTexture=orm_img,
        metallicFactor=1.0,
        roughnessFactor=1.0,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8)
    )

    mesh_out = trimesh.Trimesh(
        vertices, faces,
        visual=trimesh.visual.TextureVisuals(uv=uvs, material=material)
    )

    print(f"[OVOXEL GLB] Export complete with native PBR from OVoxel")
    return mesh_out


def simplify_gs(
    gs: Gaussian,
    simplify: float = 0.95,
    verbose: bool = True,
):
    """
    Simplify 3D Gaussians
    NOTE: this function is not used in the current implementation for the unsatisfactory performance.

    Args:
        gs (Gaussian): 3D Gaussian.
        simplify (float): Ratio of Gaussians to remove in simplification.
    """
    if simplify <= 0:
        return gs

    # simplify
    observations, extrinsics, intrinsics = render_multiview(gs, resolution=1024, nviews=100)
    observations = [torch.tensor(obs / 255.0).float().cuda().permute(2, 0, 1) for obs in observations]

    # Following https://arxiv.org/pdf/2411.06019
    renderer = GaussianRenderer({
            "resolution": 1024,
            "near": 0.8,
            "far": 1.6,
            "ssaa": 1,
            "bg_color": (0,0,0),
        })
    new_gs = Gaussian(**gs.init_params)
    new_gs._features_dc = gs._features_dc.clone()
    new_gs._features_rest = gs._features_rest.clone() if gs._features_rest is not None else None
    new_gs._opacity = torch.nn.Parameter(gs._opacity.clone())
    new_gs._rotation = torch.nn.Parameter(gs._rotation.clone())
    new_gs._scaling = torch.nn.Parameter(gs._scaling.clone())
    new_gs._xyz = torch.nn.Parameter(gs._xyz.clone())

    start_lr = [1e-4, 1e-3, 5e-3, 0.025]
    end_lr = [1e-6, 1e-5, 5e-5, 0.00025]
    optimizer = torch.optim.Adam([
        {"params": new_gs._xyz, "lr": start_lr[0]},
        {"params": new_gs._rotation, "lr": start_lr[1]},
        {"params": new_gs._scaling, "lr": start_lr[2]},
        {"params": new_gs._opacity, "lr": start_lr[3]},
    ], lr=start_lr[0])

    def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
        return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))

    _zeta = new_gs.get_opacity.clone().detach().squeeze()
    _lambda = torch.zeros_like(_zeta)
    _delta = 1e-7
    _interval = 10
    num_target = int((1 - simplify) * _zeta.shape[0])

    with tqdm(total=2500, disable=not verbose, desc='Simplifying Gaussian') as pbar:
        for i in range(2500):
            # prune
            if i % 100 == 0:
                mask = new_gs.get_opacity.squeeze() > 0.05
                mask = torch.nonzero(mask).squeeze()
                new_gs._xyz = torch.nn.Parameter(new_gs._xyz[mask])
                new_gs._rotation = torch.nn.Parameter(new_gs._rotation[mask])
                new_gs._scaling = torch.nn.Parameter(new_gs._scaling[mask])
                new_gs._opacity = torch.nn.Parameter(new_gs._opacity[mask])
                new_gs._features_dc = new_gs._features_dc[mask]
                new_gs._features_rest = new_gs._features_rest[mask] if new_gs._features_rest is not None else None
                _zeta = _zeta[mask]
                _lambda = _lambda[mask]
                # update optimizer state
                for param_group, new_param in zip(optimizer.param_groups, [new_gs._xyz, new_gs._rotation, new_gs._scaling, new_gs._opacity]):
                    stored_state = optimizer.state[param_group['params'][0]]
                    if 'exp_avg' in stored_state:
                        stored_state['exp_avg'] = stored_state['exp_avg'][mask]
                        stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]
                    del optimizer.state[param_group['params'][0]]
                    param_group['params'][0] = new_param
                    optimizer.state[param_group['params'][0]] = stored_state

            opacity = new_gs.get_opacity.squeeze()

            # sparisfy
            if i % _interval == 0:
                _zeta = _lambda + opacity.detach()
                if opacity.shape[0] > num_target:
                    index = _zeta.topk(num_target)[1]
                    _m = torch.ones_like(_zeta, dtype=torch.bool)
                    _m[index] = 0
                    _zeta[_m] = 0
                _lambda = _lambda + opacity.detach() - _zeta

            # sample a random view
            view_idx = np.random.randint(len(observations))
            observation = observations[view_idx]
            extrinsic = extrinsics[view_idx]
            intrinsic = intrinsics[view_idx]

            color = renderer.render(new_gs, extrinsic, intrinsic)['color']
            rgb_loss = torch.nn.functional.l1_loss(color, observation)
            loss = rgb_loss + \
                   _delta * torch.sum(torch.pow(_lambda + opacity - _zeta, 2))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update lr
            for j in range(len(optimizer.param_groups)):
                optimizer.param_groups[j]['lr'] = cosine_anealing(optimizer, i, 2500, start_lr[j], end_lr[j])

            pbar.set_postfix({'loss': rgb_loss.item(), 'num': opacity.shape[0], 'lambda': _lambda.mean().item()})
            pbar.update()

    new_gs._xyz = new_gs._xyz.data
    new_gs._rotation = new_gs._rotation.data
    new_gs._scaling = new_gs._scaling.data
    new_gs._opacity = new_gs._opacity.data

    return new_gs
