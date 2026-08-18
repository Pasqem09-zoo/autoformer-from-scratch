"""
Decoder blocks for Autoformer.
"""

import torch
import torch.nn as nn

from layers.decomposition import SeriesDecomp


class DecoderLayer(nn.Module):
    """
    Single Autoformer decoder layer.

    Structure:
    - self Auto-Correlation
    - residual connection
    - series decomposition
    - cross Auto-Correlation
    - residual connection
    - series decomposition
    - feed-forward network
    - residual connection
    - series decomposition
    - trend projection

    Input shapes:
        x:     [batch_size, dec_len, d_model]
        cross: [batch_size, enc_len, d_model]

    Output shapes:
        x:              [batch_size, dec_len, d_model]
        residual_trend: [batch_size, dec_len, c_out]
    """

    def __init__(
        self,
        self_attention_layer,
        cross_attention_layer,
        d_model,
        c_out,
        d_ff,
        moving_avg,
        dropout=0.1
    ):
        super().__init__()

        self.self_attention_layer = self_attention_layer
        self.cross_attention_layer = cross_attention_layer

        self.decomp1 = SeriesDecomp(kernel_size=moving_avg)
        self.decomp2 = SeriesDecomp(kernel_size=moving_avg)
        self.decomp3 = SeriesDecomp(kernel_size=moving_avg)

        self.dropout = nn.Dropout(dropout)

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

        self.activation = nn.ReLU()

        self.trend_projection = nn.Conv1d(
            in_channels=d_model,
            out_channels=c_out,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=False
        )

    def forward(self, x, cross):
        # x has shape [batch_size, dec_len, d_model]
        # cross has shape [batch_size, enc_len, d_model]

        # Self Auto-Correlation: Q, K and V all come from decoder input x
        self_output = self.self_attention_layer(
            queries=x,
            keys=x,
            values=x
        )

        x = x + self.dropout(self_output) #### residual connection con dropout al fine di evitare overfitting
        x, trend1 = self.decomp1(x) ### prima decomposizione: a differenza dell'encoder qui non buttiamo via il trend

        # Cross Auto-Correlation:
        # Q comes from decoder x
        # K and V come from encoder output cross
        cross_output = self.cross_attention_layer(
            queries=x,
            keys=cross,
            values=cross
        )

        x = x + self.dropout(cross_output)
        x, trend2 = self.decomp2(x)

        # Feed-forward network with Conv1d kernel size 1
        y = x.transpose(1, 2)

        y = self.conv1(y)
        y = self.activation(y)
        y = self.dropout(y)

        y = self.conv2(y)
        y = self.dropout(y)

        y = y.transpose(1, 2)

        x, trend3 = self.decomp3(x + y) ### x sarebbe la parte stagionale

        # Sum the trend components extracted in this decoder layer
        residual_trend = trend1 + trend2 + trend3 ### shape [B, dec_len, d_model]

        # Project trend from d_model to c_out (variable dimension, e.g., 1 for univariate forecasting)
        # [B, dec_len, d_model] -> [B, d_model, dec_len]
        residual_trend = residual_trend.permute(0, 2, 1) ### si fa perche conv1d vuole in input [B, C, L] e noi abbiamo [B, L, C]

        # [B, d_model, dec_len] -> [B, c_out, dec_len]
        residual_trend = self.trend_projection(residual_trend) ### è una conv1d con kernel size 3 e padding circolare, quindi non cambia la lunghezza della sequenza dec_len ma cambia il numero di canali da d_model a c_out

        # [B, c_out, dec_len] -> [B, dec_len, c_out]
        residual_trend = residual_trend.transpose(1, 2) ### si torna a [B, dec_len, c_out] perche' il decoder output deve avere la stessa forma di input

        return x, residual_trend ### stagionale aggiornata e trend prodotto da questo layer e proiettato in c_out




class Decoder(nn.Module):
    """
    Autoformer decoder.

    It stacks multiple DecoderLayer blocks and progressively updates the trend.

    Input shapes:
        x:     [batch_size, dec_len, d_model]
        cross: [batch_size, enc_len, d_model]
        trend: [batch_size, dec_len, c_out] (è la parte trend già nello spazio originario, con c_out variabili)

    Output shapes:
        x:     [batch_size, dec_len, d_model]
        trend: [batch_size, dec_len, c_out]
    """

    def __init__(self, decoder_layers, norm_layer=None, projection=None):
        super().__init__()

        self.decoder_layers = nn.ModuleList(decoder_layers) ### ModuleList è come una lista Python, ma compatibile con 
                                                            ### PyTorch: così PyTorch sa che quei layer hanno parametri allenabili
        self.norm_layer = norm_layer
        self.projection = projection

    def forward(self, x, cross, trend):
        # x has shape [batch_size, dec_len, d_model]
        # cross has shape [batch_size, enc_len, d_model]
        # trend has shape [batch_size, dec_len, c_out]

        for layer in self.decoder_layers:
            x, residual_trend = layer(x, cross)
            trend = trend + residual_trend

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, trend