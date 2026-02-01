from typing import *
import torch
import torch.nn as nn
import numpy as np
# NOTE: Delayed import of transformers to allow env vars to be set before import
# from transformers import CLIPTextModel, AutoTokenizer
import open3d as o3d
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp

print("[MODULE LOAD] trellis_text_to_3d.py loaded (transformers import deferred)")
import sys
sys.stdout.flush()


class TrellisTextTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis text-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        text_cond_model (str): The name of the text conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        text_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        # Only init text_cond_model if provided (from_pretrained handles this separately)
        if text_cond_model is not None:
            self._init_text_cond_model(text_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisTextTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisTextTo3DPipeline, TrellisTextTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisTextTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_text_cond_model(args['text_cond_model'])

        return new_pipeline
    
    def _init_text_cond_model(self, name: str):
        """
        Initialize the text conditioning model.
        Uses local cache with local_files_only=True for offline support.
        Imports transformers lazily to ensure env vars are set before import.
        """
        import os
        # Import transformers here (lazy) to ensure env vars are already set
        from transformers import CLIPTextModel, AutoTokenizer

        # Get cache directory from environment or use default
        cache_dir = os.environ.get('HF_HUB_CACHE') or os.environ.get('HF_HOME')
        if cache_dir and not cache_dir.endswith('/hub') and not cache_dir.endswith('\\hub'):
            # HF_HOME needs /hub appended for model cache
            potential_hub = os.path.join(cache_dir, 'hub')
            if os.path.exists(potential_hub):
                cache_dir = potential_hub

        # Load model with local_files_only to use cached models
        try:
            model = CLIPTextModel.from_pretrained(name, local_files_only=True, cache_dir=cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True, cache_dir=cache_dir)
        except Exception as e:
            # If local loading fails, try online (will fail if HF_HUB_OFFLINE=1)
            print(f"[CLIP] Local model not found, trying online: {e}")
            model = CLIPTextModel.from_pretrained(name, cache_dir=cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)

        model.eval()
        model = model.cuda()
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and all(isinstance(t, str) for t in text), "text must be a list of strings"
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].cuda()
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state
        
        return embeddings
        
    def get_cond(self, prompt: List[str]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            prompt (List[str]): The text prompt.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_text(prompt)
        neg_cond = self.text_cond_model['null_cond']
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        occupancy = decoder(z_s)

        # DEBUG: Log occupancy statistics
        print(f"[SPARSE DEBUG] Occupancy shape: {occupancy.shape}")
        print(f"[SPARSE DEBUG] Occupancy min: {occupancy.min().item():.4f}, max: {occupancy.max().item():.4f}")
        print(f"[SPARSE DEBUG] Occupancy mean: {occupancy.mean().item():.4f}, std: {occupancy.std().item():.4f}")
        print(f"[SPARSE DEBUG] Positive values (>0): {(occupancy > 0).sum().item()}")
        print(f"[SPARSE DEBUG] Values >0.5: {(occupancy > 0.5).sum().item()}")
        print(f"[SPARSE DEBUG] Values >1.0: {(occupancy > 1.0).sum().item()}")

        coords = torch.argwhere(occupancy>0)[:, [0, 2, 3, 4]].int()
        print(f"[SPARSE DEBUG] Final sparse voxel count: {coords.shape[0]}")
        import sys
        sys.stdout.flush()

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']

        print(f"[SLAT SAMPLE] Input coords to SLAT sampler: {coords.shape[0]} voxels")
        import sys
        sys.stdout.flush()

        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}

        print(f"[SLAT SAMPLE] Noise SparseTensor coords: {noise.coords.shape}")
        sys.stdout.flush()

        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        print(f"[SLAT DEBUG] SLAT coords shape: {slat.coords.shape}")
        print(f"[SLAT DEBUG] SLAT feats shape: {slat.feats.shape}")
        import sys
        sys.stdout.flush()

        return slat

    @torch.no_grad()
    def run(
        self,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Run the pipeline.

        Args:
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = self.get_cond([prompt])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
    
    def voxelize(self, mesh: o3d.geometry.TriangleMesh) -> torch.Tensor:
        """
        Voxelize a mesh.

        Args:
            mesh (o3d.geometry.TriangleMesh): The mesh to voxelize.
            sha256 (str): The SHA256 hash of the mesh.
            output_dir (str): The output directory.
        """
        vertices = np.asarray(mesh.vertices)
        aabb = np.stack([vertices.min(0), vertices.max(0)])
        center = (aabb[0] + aabb[1]) / 2
        scale = (aabb[1] - aabb[0]).max()
        vertices = (vertices - center) / scale
        vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
        vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        return torch.tensor(vertices).int().cuda()

    @torch.no_grad()
    def run_variant(
        self,
        mesh: o3d.geometry.TriangleMesh,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Run the pipeline for making variants of an asset.

        Args:
            mesh (o3d.geometry.TriangleMesh): The base mesh.
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = self.get_cond([prompt])
        coords = self.voxelize(mesh)
        coords = torch.cat([
            torch.arange(num_samples).repeat_interleave(coords.shape[0], 0)[:, None].int().cuda(),
            coords.repeat(num_samples, 1)
        ], 1)
        torch.manual_seed(seed)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
