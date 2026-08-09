"""
questo blocco implementa la decomposizione di una serie temporale in due componenti: stagionale e di trend. 
La componente di trend viene calcolata utilizzando una media mobile con un kernel di dimensione specificata, 
mentre la componente stagionale è ottenuta sottraendo la componente di trend dalla serie originale.

stiamo implementando il blocco SeriesDecomp, che prende in input una serie temporale e restituisce due componenti: la componente stagionale e la componente di trend.
"""



import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """
    è la componente di trend della decomposizione, calcola la media mobile del trend della serie temporale, 
    utilizzando un kernel di dimensione kernel_size. il kernel è applicato con stride 1 e padding 0, quindi la 
    lunghezza della sequenza in output sarà inferiore a quella in input di kernel_size - 1.
    
    La media mobile viene calcolata con padding sui bordi della sequenza per mantenere la stessa lunghezza dell'input.
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

        # Padding on both ends of the time series
        padding_size = (self.kernel_size - 1) // 2

        # padding
        front = x[:, 0:1, :].repeat(1, padding_size, 1)
        end = x[:, -1:, :].repeat(1, padding_size, 1)
        x = torch.cat([front, x, end], dim=1)

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

    #separazione molto semplice: prende la serie originale, ne estrae una versione smussata tramite media mobile, e considera il resto come componente stagionale/residuale
    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean

        return residual, moving_mean