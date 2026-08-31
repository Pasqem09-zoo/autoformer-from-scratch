import torch
import torch.nn as nn


class MyLayerNorm(nn.Module):
    """
    Special LayerNorm for the seasonal part.

    It applies standard LayerNorm and then removes the temporal mean.
    """

    def __init__(self, channels):
        super().__init__()

        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        # x has shape [batch_size, seq_len, channels]

        x_hat = self.layernorm(x)

        bias = torch.mean(x_hat, dim=1)
        bias = bias.unsqueeze(1).repeat(1, x.shape[1], 1)   # repeat the bias along the temporal dimension
        x_hat = x_hat - bias

        return x_hat