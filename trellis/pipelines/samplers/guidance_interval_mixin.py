from typing import *
import torch

# Import SparseTensor for type checking
try:
    from ...modules import sparse as sp
    HAS_SPARSE = True
except ImportError:
    HAS_SPARSE = False


def _get_tensor_std(x):
    """Get standard deviation, handling both Tensor and SparseTensor."""
    if HAS_SPARSE and isinstance(x, sp.SparseTensor):
        return x.feats.std()
    return x.std()


def _apply_rescale(pred, pred_cfg, guidance_rescale: float):
    """Apply guidance rescaling, handling both Tensor and SparseTensor."""
    if HAS_SPARSE and isinstance(pred, sp.SparseTensor):
        std_cond = pred.feats.std()
        std_cfg = pred_cfg.feats.std()
        if std_cfg > 1e-8:
            rescale_factor = std_cond / std_cfg
            new_feats = pred_cfg.feats * (1 - guidance_rescale) + pred_cfg.feats * rescale_factor * guidance_rescale
            return pred_cfg.replace(new_feats)
        return pred_cfg
    else:
        std_cond = pred.std()
        std_cfg = pred_cfg.std()
        if std_cfg > 1e-8:
            rescale_factor = std_cond / std_cfg
            return pred_cfg * (1 - guidance_rescale) + pred_cfg * rescale_factor * guidance_rescale
        return pred_cfg


class GuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.

    Supports TRELLIS.2 guidance_rescale parameter for preventing CFG oversaturation.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, guidance_rescale: float = 0.0, **kwargs):
        """
        Apply classifier-free guidance within a specified interval, with optional rescaling.

        Args:
            cfg_interval: Tuple (t_min, t_max) defining when to apply CFG.
            guidance_rescale: Factor for rescaling CFG output to prevent oversaturation.
                             0.0 = no rescaling (TRELLIS.1 default)
        """
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred = super()._inference_model(model, x_t, t, cond, **kwargs)
            neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)

            # Standard CFG - handle both Tensor and SparseTensor
            if HAS_SPARSE and isinstance(pred, sp.SparseTensor):
                pred_cfg = pred.replace((1 + cfg_strength) * pred.feats - cfg_strength * neg_pred.feats)
            else:
                pred_cfg = (1 + cfg_strength) * pred - cfg_strength * neg_pred

            # TRELLIS.2 guidance rescaling
            if guidance_rescale > 0:
                pred_cfg = _apply_rescale(pred, pred_cfg, guidance_rescale)

            return pred_cfg
        else:
            return super()._inference_model(model, x_t, t, cond, **kwargs)
