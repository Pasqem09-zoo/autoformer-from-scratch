import torch
import torch.nn as nn

from layers.decomposition import SeriesDecomp


class EncoderLayer(nn.Module):
    """
    Single Autoformer encoder layer.

    Structure:
    - Auto-Correlation
    - residual connection
    - series decomposition
    - feed-forward network
    - residual connection
    - series decomposition

    Input shape:
        x: [batch_size, seq_len, d_model]

    Output shape:
        x: [batch_size, seq_len, d_model]
    """

    def __init__(self, autocorrelation_layer, d_model, d_ff, moving_avg, dropout=0.1, activation="gelu"):
        super().__init__()

        self.autocorrelation_layer = autocorrelation_layer

        self.decomp1 = SeriesDecomp(kernel_size=moving_avg)
        self.decomp2 = SeriesDecomp(kernel_size=moving_avg)

        self.dropout = nn.Dropout(dropout)

        # Feed-forward network with Conv1d kernel size 1
        self.conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_ff,
            kernel_size=1,
            bias=False
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_ff,
            out_channels=d_model,
            kernel_size=1,
            bias=False
        )
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()



    def forward(self, x):
        # x has shape [batch_size, seq_len, d_model]

        # Self Auto-Correlation: Q, K and V all come from x. This is the case of encoder self-attention, where the queries, keys and values are all derived from the same input sequence
        autocorr_output = self.autocorrelation_layer(
            queries=x,
            keys=x,
            values=x
        )

        x = x + self.dropout(autocorr_output)       # residual connection
        x, _ = self.decomp1(x)                      # only seasonal part is passed to the next layer, trend part is discarded

        # Feed-forward network
        y = x.transpose(1, 2)
        y = self.conv1(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)

        x = x + y
        x, _ = self.decomp2(x)

        return x




class Encoder(nn.Module):
    """
    Autoformer encoder.

    It stacks multiple EncoderLayer blocks.

    Input shape:
        x: [batch_size, seq_len, d_model]

    Output shape:
        x: [batch_size, seq_len, d_model]
    """

    def __init__(self, encoder_layers, norm_layer=None):
        super().__init__()

        self.encoder_layers = nn.ModuleList(encoder_layers)     # ModulesList is a list of encoder layers, each of which is an instance of EncoderLayer. This allows us to stack multiple encoder layers to form the complete encoder
        self.norm_layer = norm_layer

    def forward(self, x):

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        return x