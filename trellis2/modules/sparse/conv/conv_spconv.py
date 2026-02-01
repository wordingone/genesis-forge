"""
spconv backend for TRELLIS.2 with checkpoint-compatible weight storage.

The official TRELLIS.2-4B checkpoint was trained with flex_gemm which stores weights
directly on self.weight/self.bias (not wrapped in self.conv).

Both flex_gemm and spconv use the same weight format: (Co, Kd, Kh, Kw, Ci)
This means checkpoints can be loaded directly without weight permutation.

This backend mimics flex_gemm's weight storage pattern for checkpoint compatibility
while using spconv for actual sparse convolution execution.
"""

import math
import torch
import torch.nn as nn
from .. import SparseTensor
from . import config
import spconv.pytorch as spconv


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    """
    Initialize SparseConv3d with flex_gemm-compatible weight storage.
    Weights are stored directly on self.weight (not self.conv.weight) for checkpoint compatibility.
    """
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3
    self.padding = padding
    self._use_bias = bias
    self._indice_key = indice_key

    # Store algorithm config
    algo = None
    if config.SPCONV_ALGO == 'native':
        algo = spconv.ConvAlgo.Native
    elif config.SPCONV_ALGO == 'implicit_gemm':
        algo = spconv.ConvAlgo.MaskImplicitGemm
    self._algo = algo

    # Determine if this is submanifold or regular sparse conv
    self._is_submanifold = (stride == 1 or self.stride == (1, 1, 1)) and (padding is None)

    # flex_gemm stores weight in permuted format: (Co, Kd, Kh, Kw, Ci)
    # This matches the checkpoint format
    self.weight = nn.Parameter(torch.empty((out_channels, *self.kernel_size, in_channels)))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    # Initialize parameters using kaiming uniform
    # Both flex_gemm and spconv use format: (Co, Kd, Kh, Kw, Ci)
    # For kaiming init, we need to compute fan_in correctly
    fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
    std = 1.0 / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    torch.nn.init.uniform_(self.weight, -bound, bound)

    if self.bias is not None:
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    # Cached spconv layer (built on first forward)
    self._spconv_layer = None
    self._last_weight_data_ptr = None


def _build_spconv_layer(self, device, dtype=None):
    """Build spconv layer with current weights."""
    # Both flex_gemm and spconv use the same weight format: (Co, Kd, Kh, Kw, Ci)
    # No permutation needed - weights can be copied directly

    if self._is_submanifold:
        layer = spconv.SubMConv3d(
            self.in_channels, self.out_channels, self.kernel_size,
            dilation=self.dilation, bias=False,  # bias handled separately
            indice_key=self._indice_key, algo=self._algo
        )
    else:
        layer = spconv.SparseConv3d(
            self.in_channels, self.out_channels, self.kernel_size,
            stride=self.stride, dilation=self.dilation, padding=self.padding,
            bias=False, indice_key=self._indice_key, algo=self._algo
        )

    layer = layer.to(device)

    # Convert layer to target dtype first, then copy weights
    if dtype is not None:
        if dtype == torch.float16:
            layer = layer.half()
        elif dtype == torch.bfloat16:
            layer = layer.bfloat16()
        # Copy weights (already in correct dtype after .half()/.bfloat16())
        layer.weight.data.copy_(self.weight.data.to(dtype))
    else:
        layer.weight.data.copy_(self.weight.data)

    return layer


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    """Forward pass using spconv with flex_gemm-compatible weights."""
    # Get input dtype to ensure weight dtype matches
    input_dtype = x.feats.dtype

    # Rebuild spconv layer if weights changed or dtype mismatches
    needs_rebuild = (
        self._spconv_layer is None or
        self._last_weight_data_ptr != self.weight.data_ptr() or
        self._spconv_layer.weight.dtype != input_dtype
    )
    if needs_rebuild:
        self._spconv_layer = _build_spconv_layer(self, self.weight.device, dtype=input_dtype)
        self._last_weight_data_ptr = self.weight.data_ptr()

    spatial_changed = any(s != 1 for s in self.stride) or (self.padding is not None)

    # Run spconv
    new_data = self._spconv_layer(x.data)

    # Add bias manually (since spconv layer has bias=False)
    if self.bias is not None:
        bias = self.bias.to(input_dtype) if self.bias.dtype != input_dtype else self.bias
        new_data = new_data.replace_feature(new_data.features + bias)

    new_shape = [x.shape[0], self.out_channels]
    new_layout = None if spatial_changed else x.layout

    if spatial_changed and (x.shape[0] != 1):
        # spconv with non-1 stride breaks contiguous output, sort by coords
        fwd = new_data.indices[:, 0].argsort()
        bwd = torch.zeros_like(fwd).scatter_(0, fwd, torch.arange(fwd.shape[0], device=fwd.device))
        sorted_feats = new_data.features[fwd]
        sorted_coords = new_data.indices[fwd]
        unsorted_data = new_data
        new_data = spconv.SparseConvTensor(sorted_feats, sorted_coords, unsorted_data.spatial_shape, unsorted_data.batch_size)

    out = SparseTensor(
        new_data, shape=torch.Size(new_shape), layout=new_layout,
        scale=tuple([s * stride for s, stride in zip(x._scale, self.stride)]),
        spatial_cache=x._spatial_cache,
    )

    if spatial_changed and (x.shape[0] != 1):
        out.register_spatial_cache(f'conv_{self.stride}_unsorted_data', unsorted_data)
        out.register_spatial_cache(f'conv_{self.stride}_sort_bwd', bwd)

    return out


def sparse_inverse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
    """Initialize SparseInverseConv3d with flex_gemm-compatible weight storage."""
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3
    self._use_bias = bias
    self._indice_key = indice_key

    # Store weight in flex_gemm format for checkpoint compatibility
    self.weight = nn.Parameter(torch.empty((out_channels, *self.kernel_size, in_channels)))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    # Initialize using kaiming uniform
    # Format: (Co, Kd, Kh, Kw, Ci) - same as spconv
    fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
    std = 1.0 / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    torch.nn.init.uniform_(self.weight, -bound, bound)

    if self.bias is not None:
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    self._spconv_layer = None
    self._last_weight_data_ptr = None


def _build_inverse_spconv_layer(self, device, dtype=None):
    """Build spconv inverse layer with current weights."""
    # Both flex_gemm and spconv use same format: (Co, Kd, Kh, Kw, Ci)
    layer = spconv.SparseInverseConv3d(
        self.in_channels, self.out_channels, self.kernel_size,
        bias=False, indice_key=self._indice_key
    )
    layer = layer.to(device)

    # Convert layer to target dtype first, then copy weights
    if dtype is not None:
        if dtype == torch.float16:
            layer = layer.half()
        elif dtype == torch.bfloat16:
            layer = layer.bfloat16()
        layer.weight.data.copy_(self.weight.data.to(dtype))
    else:
        layer.weight.data.copy_(self.weight.data)

    return layer


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    """Forward pass for inverse conv."""
    # Get input dtype to ensure weight dtype matches
    input_dtype = x.feats.dtype

    # Rebuild spconv layer if weights changed or dtype mismatches
    needs_rebuild = (
        self._spconv_layer is None or
        self._last_weight_data_ptr != self.weight.data_ptr() or
        self._spconv_layer.weight.dtype != input_dtype
    )
    if needs_rebuild:
        self._spconv_layer = _build_inverse_spconv_layer(self, self.weight.device, dtype=input_dtype)
        self._last_weight_data_ptr = self.weight.data_ptr()

    spatial_changed = any(s != 1 for s in self.stride)

    if spatial_changed:
        # Recover original spconv order
        data = x.get_spatial_cache(f'conv_{self.stride}_unsorted_data')
        bwd = x.get_spatial_cache(f'conv_{self.stride}_sort_bwd')
        data = data.replace_feature(x.feats[bwd])
    else:
        data = x.data

    new_data = self._spconv_layer(data)

    # Add bias manually
    if self.bias is not None:
        bias = self.bias.to(input_dtype) if self.bias.dtype != input_dtype else self.bias
        new_data = new_data.replace_feature(new_data.features + bias)

    new_shape = [x.shape[0], self.out_channels]
    new_layout = None if spatial_changed else x.layout

    out = SparseTensor(
        new_data, shape=torch.Size(new_shape), layout=new_layout,
        scale=tuple([s // stride for s, stride in zip(x._scale, self.stride)]),
        spatial_cache=x._spatial_cache,
    )

    return out
