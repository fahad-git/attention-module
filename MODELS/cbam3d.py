import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv3D(nn.Module):
    """
    A fundamental building block for 3D convolutional neural networks.

    This module combines a 3D convolutional layer, an optional batch normalization layer,
    and an optional ReLU activation function. It provides a consistent way to apply these
    common operations.
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        """
        Initializes the BasicConv3D layer.

        Args:
            in_planes (int): Number of input channels.
            out_planes (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel.
            stride (int or tuple): Stride of the convolution. Default: 1.
            padding (int or tuple): Padding added to the input. Default: 0.
            dilation (int or tuple): Spacing between kernel elements. Default: 1.
            groups (int): Number of blocked connections from input channels to output channels. Default: 1.
            relu (bool): If True, applies ReLU activation after convolution and batch normalization. Default: True.
            bn (bool): If True, applies batch normalization after convolution. Default: True.
            bias (bool): If True, adds a learnable bias to the convolution. Default: False.
        """
        super(BasicConv3D, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        """
        Defines the forward pass of the BasicConv3D layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor after convolution, batch normalization (if enabled), and ReLU (if enabled).
        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    """
    Flattens a multi-dimensional tensor into a 2D tensor.

    This module reshapes the input tensor to have the shape (batch_size, -1),
    where -1 indicates that the remaining dimensions are flattened into a single dimension.
    """
    def forward(self, x):
        """
        Defines the forward pass of the Flatten layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Flattened tensor of shape (batch_size, -1).
        """
        return x.view(x.size(0), -1)

def logsumexp_3d(tensor):
    """
    Computes the log-sum-exp trick over the spatial dimensions of a 5D tensor.

    This function is numerically stable for computing the logarithm of the sum of exponentials.
    It flattens the spatial dimensions (depth, height, width) and then applies the log-sum-exp
    operation.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

    Returns:
        torch.Tensor: Tensor with log-sum-exp values over the spatial dimensions,
                      shape (batch_size, channels, 1, 1, 1).
    """
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs.view(tensor.size(0), tensor.size(1), 1, 1, 1) # Reshape back to original spatial dims as 1

class ChannelGate3D(nn.Module):
    """
    A 3D Channel Gate module.

    This module applies attention to the channel dimension of an input tensor. It uses
    multiple pooling operations (average, max, and optionally others like LSE) followed
    by a multi-layer perceptron (MLP) to learn channel-wise attention weights.
    """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        """
        Initializes the ChannelGate3D module.

        Args:
            gate_channels (int): Number of input channels.
            reduction_ratio (int): Reduction factor in the MLP. Default: 16.
            pool_types (list): List of pooling types to use ('avg', 'max', 'lp', 'lse'). Default: ['avg', 'max'].
        """
        super(ChannelGate3D, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        """
        Defines the forward pass of the ChannelGate3D module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Input tensor scaled by the learned channel attention weights.
        """
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pool = F.avg_pool3d(x, kernel_size=x.size()[2:])
            elif pool_type == 'max':
                pool = F.max_pool3d(x, kernel_size=x.size()[2:])
            elif pool_type == 'lp':
                pool = F.lp_pool3d(x, norm_type=2, kernel_size=x.size()[2:])
            elif pool_type == 'lse':
                pool = logsumexp_3d(x)
            channel_att_raw = self.mlp(pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw

        scale = torch.sigmoid(channel_att_sum).view(x.size(0), x.size(1), 1, 1, 1)
        return x * scale.expand_as(x)

class ChannelPool3D(nn.Module):
    """
    Performs channel-wise max pooling and average pooling.

    This module takes a 5D tensor as input and performs max pooling and average pooling
    across the channel dimension. The results are then concatenated along the channel dimension.
    """
    def forward(self, x):
        """
        Defines the forward pass of the ChannelPool3D module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Tensor with concatenated max and average pooled features
                          along the channel dimension, shape (batch_size, 2, depth, height, width).
        """
        max_pool = torch.max(x, 1)[0].unsqueeze(1)
        mean_pool = torch.mean(x, 1).unsqueeze(1)
        return torch.cat((max_pool, mean_pool), dim=1)

class SpatialGate3D(nn.Module):
    """
    A 3D Spatial Gate module.

    This module applies attention to the spatial dimensions (depth, height, width) of an
    input tensor. It uses channel pooling (max and average) to aggregate channel information
    and then applies a 3D convolution to learn spatial attention weights.
    """
    def __init__(self):
        """
        Initializes the SpatialGate3D module.
        """
        super(SpatialGate3D, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool3D()
        self.spatial = BasicConv3D(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        """
        Defines the forward pass of the SpatialGate3D module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Input tensor scaled by the learned spatial attention weights.
        """
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale

class CBAM3D(nn.Module):
    """
    The 3D Convolutional Block Attention Module (CBAM).

    This module sequentially applies channel and spatial attention to an input tensor
    for 3D data. It enhances feature representation by adaptively recalibrating channel
    and spatial features.
    """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        """
        Initializes the CBAM3D module.

        Args:
            gate_channels (int): Number of input channels.
            reduction_ratio (int): Reduction factor in the channel attention MLP. Default: 16.
            pool_types (list): List of pooling types to use in channel attention. Default: ['avg', 'max'].
            no_spatial (bool): If True, disables the spatial attention module. Default: False.
        """
        super(CBAM3D, self).__init__()
        self.ChannelGate = ChannelGate3D(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate3D()

    def forward(self, x):
        """
        Defines the forward pass of the CBAM3D module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor after applying channel and (optionally) spatial attention.
        """
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out