"""
This module implements the series decomposition block used in the Autoformer model. It decomposes a time series into its seasonal and trend components using a moving average.
The trend component is calculated using a moving average with a specified kernel size, while the seasonal component is obtained by subtracting the trend component from the original series.

This is the SeriesDecomp block, which takes a time series as input and returns two components: the seasonal component and the trend component.
"""



import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """
    Moving average block to extract the trend component of a time series.

    It uses a 1D average pooling operation to compute the moving average over the input sequence.
    The input sequence is padded on both sides to maintain the same length after pooling.
    The moving average is computed using a specified kernel size and stride.
    """

    def __init__(self, kernel_size, stride):
        super().__init__()

        self.kernel_size = kernel_size

        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )



    def forward(self, x):
        # x has shape [batch_size, seq_len, channels]

        padding_size = (self.kernel_size - 1) // 2      # number of elements to pad on each side of the sequence to maintain the same length after pooling

        # Pad the input sequence on both sides with the first and last values to maintain the same length after pooling
        front = x[:, 0:1, :].repeat(1, padding_size, 1)
        end = x[:, -1:, :].repeat(1, padding_size, 1)
        x = torch.cat([front, x, end], dim=1)           # concatenate the padded values along the sequence length dimension

        # AvgPool1d expects [batch_size, channels, seq_len]
        x = x.permute(0, 2, 1)

        x = self.avg_pool(x)

        # Back to [batch_size, seq_len, channels]
        x = x.permute(0, 2, 1)

        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block.

    It decomposes a time series into:
    - seasonal component
    - trend component
    """

    def __init__(self, kernel_size):
        super().__init__()

        self.moving_avg = MovingAvg(
            kernel_size=kernel_size,
            stride=1
        )



    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean      # original series and moving average have the same shape, so we can subtract them element-wise

        return residual, moving_mean