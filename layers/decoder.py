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
        dropout=0.1,
        activation="gelu"
    ):
        super().__init__()

        self.self_attention_layer = self_attention_layer        # Q,V and K all come from decoder input x
        self.cross_attention_layer = cross_attention_layer      # Q comes from decoder input x, K and V come from encoder output cross

        self.decomp1 = SeriesDecomp(kernel_size=moving_avg)
        self.decomp2 = SeriesDecomp(kernel_size=moving_avg)
        self.decomp3 = SeriesDecomp(kernel_size=moving_avg)

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
        
        # Trend projection: project the trend from d_model to c_out (variable dimension)
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

        x = x + self.dropout(self_output)       # residual connection
        x, trend1 = self.decomp1(x)             # keep also the trend

        # Cross Auto-Correlation: Q comes from decoder x, K and V come from encoder output
        cross_output = self.cross_attention_layer(
            queries=x,
            keys=cross,
            values=cross
        )

        x = x + self.dropout(cross_output)
        x, trend2 = self.decomp2(x)

        # Feed-forward network
        y = x.transpose(1, 2)
        y = self.conv1(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)

        x, trend3 = self.decomp3(x + y)           # x is the updated seasonal part, trend3 is the trend extracted from this layer

        residual_trend = trend1 + trend2 + trend3                   # shape [B, dec_len, d_model]

        residual_trend = residual_trend.permute(0, 2, 1)            # [B, dec_len, d_model] -> [B, d_model, dec_len]

        residual_trend = self.trend_projection(residual_trend)      # [B, d_model, dec_len] -> [B, c_out, dec_len]

        residual_trend = residual_trend.transpose(1, 2)             # [B, c_out, dec_len] -> [B, dec_len, c_out]

        return x, residual_trend                                    # return the updated seasonal part and the projected trend part




class Decoder(nn.Module):
    """
    Autoformer decoder.

    It stacks multiple DecoderLayer blocks and progressively updates the trend.

    Input shapes:
        x:     [batch_size, dec_len, d_model]
        cross: [batch_size, enc_len, d_model]
        trend: [batch_size, dec_len, c_out]

    Output shapes:
        x:     [batch_size, dec_len, d_model]
        trend: [batch_size, dec_len, c_out]
    """

    def __init__(self, decoder_layers, norm_layer=None, projection=None):
        super().__init__()

        self.decoder_layers = nn.ModuleList(decoder_layers)
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