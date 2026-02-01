import torch
import torch.nn as nn


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store original dtype and convert to float32 for precision
        orig_dtype = x.dtype
        x_float = x.float()

        # Handle mixed precision: if weights are not float32, convert them temporarily
        if self.weight is not None and self.weight.dtype != torch.float32:
            weight = self.weight.float()
            bias = self.bias.float() if self.bias is not None else None
            out = nn.functional.layer_norm(x_float, self.normalized_shape, weight, bias, self.eps)
        else:
            out = super().forward(x_float)

        return out.to(orig_dtype)
    

class GroupNorm32(nn.GroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)
    
    
class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM-1, *range(1, DIM-1)).contiguous()
        return x
    