from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        # Handle SparseTensor with different in/out channels
        # For texture model: in=64 (32 shape + 32 texture), out=32 (texture velocity only)
        # The model outputs velocity for texture channels only (LAST 32, not first)
        try:
            from ...modules import sparse as sp
            if isinstance(x_t, sp.SparseTensor) and isinstance(v, sp.SparseTensor):
                v_feats = v.feats
                x_feats = x_t.feats
                out_channels = v_feats.shape[-1]
                in_channels = x_feats.shape[-1]

                # If output has fewer channels, use LAST channels from x_t (texture part)
                if out_channels < in_channels:
                    cond_channels = in_channels - out_channels  # Shape channels (first 32)
                    x_target = x_feats[..., cond_channels:]  # Texture channels (last 32)

                    eps_feats = (1 - t) * v_feats + x_target
                    x_0_feats = (1 - self.sigma_min) * x_target - (self.sigma_min + (1 - self.sigma_min) * t) * v_feats
                    return v.replace(x_0_feats), v.replace(eps_feats)

                # Same channel count - standard path
                eps_feats = (1 - t) * v_feats + x_feats
                x_0_feats = (1 - self.sigma_min) * x_feats - (self.sigma_min + (1 - self.sigma_min) * t) * v_feats
                return v.replace(x_0_feats), v.replace(eps_feats)
        except ImportError:
            pass

        # Standard tensor path
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    # Sampler-specific kwargs that should NOT be passed to the model
    _SAMPLER_KWARGS = frozenset({
        'neg_cond', 'cfg_strength', 'cfg_interval', 'guidance_rescale',
        'guidance_strength',  # Alias for cfg_strength used by some callers
        'guidance_interval',  # Alias for cfg_interval used by TRELLIS.2
        'shape_slat',  # Shape conditioning for texture sampler (handled separately)
    })

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        # Filter out sampler-specific kwargs before passing to model
        model_kwargs = {k: v for k, v in kwargs.items() if k not in self._SAMPLER_KWARGS}
        return model(x_t, t, cond, **model_kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.

        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)

        # Handle SparseTensor with different in/out channels
        # For texture model: in=64 (32 shape + 32 texture), out=32 (texture velocity only)
        # Shape channels stay constant, only texture channels get updated
        try:
            from ...modules import sparse as sp
            if isinstance(x_t, sp.SparseTensor) and isinstance(pred_v, sp.SparseTensor):
                out_channels = pred_v.feats.shape[-1]
                in_channels = x_t.feats.shape[-1]
                if out_channels < in_channels:
                    # Split into conditioning (shape) and target (texture) channels
                    cond_channels = in_channels - out_channels  # 32 shape
                    cond_feats = x_t.feats[..., :cond_channels]  # Shape stays constant
                    target_feats = x_t.feats[..., cond_channels:]  # Texture gets updated

                    # Euler step on texture channels only
                    updated_target = target_feats - (t - t_prev) * pred_v.feats

                    # Reconstruct full tensor: [shape, updated_texture]
                    full_feats = torch.cat([cond_feats, updated_target], dim=-1)
                    pred_x_prev = x_t.replace(full_feats)
                    return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})
        except ImportError:
            pass

        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.

    Supports TRELLIS.2 guidance_rescale parameter for preventing CFG oversaturation.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        guidance_rescale: float = 0.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            guidance_rescale: TRELLIS.2 rescale factor (0.0 = off, 0.7 = typical)
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, guidance_rescale=guidance_rescale, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.

    Supports TRELLIS.2 guidance_rescale parameter for preventing CFG oversaturation.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = None,
        guidance_strength: float = None,  # TRELLIS.2 alias for cfg_strength
        cfg_interval: Union[Tuple[float, float], List[float]] = (0.0, 1.0),
        guidance_interval: Union[Tuple[float, float], List[float]] = None,  # TRELLIS.2 alias
        guidance_rescale: float = 0.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            guidance_strength: TRELLIS.2 alias for cfg_strength.
            cfg_interval: The interval for classifier-free guidance.
            guidance_interval: TRELLIS.2 alias for cfg_interval.
            guidance_rescale: TRELLIS.2 rescale factor (0.0 = off, 0.7 = typical)
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        # Handle TRELLIS.2 naming convention (guidance_strength -> cfg_strength)
        if cfg_strength is None:
            cfg_strength = guidance_strength if guidance_strength is not None else 3.0

        # Handle TRELLIS.2 naming convention (guidance_interval -> cfg_interval)
        if guidance_interval is not None:
            cfg_interval = guidance_interval

        # Convert list to tuple if needed
        if isinstance(cfg_interval, list):
            cfg_interval = tuple(cfg_interval)

        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, guidance_rescale=guidance_rescale, **kwargs)
