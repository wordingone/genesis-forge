from typing import *
import torch


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.

    Supports TRELLIS.2 guidance_rescale parameter for preventing CFG oversaturation.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, guidance_rescale: float = 0.0, **kwargs):
        """
        Apply classifier-free guidance with optional rescaling.

        Args:
            guidance_rescale: Factor for rescaling CFG output to prevent oversaturation.
                             0.0 = no rescaling (TRELLIS.1 default)
                             0.7 = typical TRELLIS.2 value for sparse structure
                             0.5 = typical TRELLIS.2 value for shape generation
        """
        pred = super()._inference_model(model, x_t, t, cond, **kwargs)
        neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)

        # Standard CFG: pred_cfg = (1 + s) * pred_cond - s * pred_uncond
        pred_cfg = (1 + cfg_strength) * pred - cfg_strength * neg_pred

        # TRELLIS.2 guidance rescaling to prevent oversaturation
        if guidance_rescale > 0:
            # Rescale CFG output to match the standard deviation of the conditional prediction
            # This prevents color saturation and artifacts at high CFG values
            std_cond = pred.std()
            std_cfg = pred_cfg.std()
            if std_cfg > 1e-8:  # Avoid division by zero
                rescale_factor = std_cond / std_cfg
                # Interpolate between original and rescaled based on guidance_rescale
                pred_cfg = pred_cfg * (1 - guidance_rescale) + pred_cfg * rescale_factor * guidance_rescale

        return pred_cfg
