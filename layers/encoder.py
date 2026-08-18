"""
Encoder blocks for Autoformer.
"""

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

    def __init__(self, autocorrelation_layer, d_model, d_ff, moving_avg, dropout=0.1):
        super().__init__()

        self.autocorrelation_layer = autocorrelation_layer

        self.decomp1 = SeriesDecomp(kernel_size=moving_avg)
        self.decomp2 = SeriesDecomp(kernel_size=moving_avg)

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

    def forward(self, x):
        # x has shape [batch_size, seq_len, d_model]

        # Self Auto-Correlation: Q, K and V all come from x
        ### prima cosa che fa: siamo nel caso self quindi Q K e V vengono tutti da x, non cross
        autocorr_output = self.autocorrelation_layer(
            queries=x,
            keys=x,
            values=x
        )

        # Residual connection + first decomposition
        x = x + self.dropout(autocorr_output)
        x, _ = self.decomp1(x) ### teniamo solo la parte stagionale, la parte trend la buttiamo via

        # Feed-forward network with Conv1d kernel size 1
        # Conv1d expects [batch_size, channels, seq_len]
        y = x.transpose(1, 2)

        y = self.conv1(y)
        y = self.activation(y)
        y = self.dropout(y)

        y = self.conv2(y)
        y = self.dropout(y)

        # Back to [batch_size, seq_len, d_model]
        y = y.transpose(1, 2)

        # Residual connection + second decomposition
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
        ### ModulesList è una lista di moduli pytorch che permette di registrare i moduli in modo che vengano 
        ### considerati come parte del modello e quindi i loro parametri vengano aggiornati durante l'addestramento
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.norm_layer = norm_layer

    def forward(self, x):
        # x has shape [batch_size, seq_len, d_model]

        for encoder_layer in self.encoder_layers: ### per ogni encoder layer nella lista di encoder layers, applichiamo il layer all'input x
            x = encoder_layer(x)

        if self.norm_layer is not None: ### normalizzazione finale opzionale, se è stata passata una normalizzazione la applichiamo, altrimenti no
            x = self.norm_layer(x)

        return x