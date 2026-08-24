"""
Embedding layers for Autoformer.

Autoformer does not use positional embedding.
It uses:
- value embedding: embeds the observed time series values
- time feature embedding: embeds timestamp information
"""

import torch
import torch.nn as nn #torch.nn contiene i blocchi delle reti neurali, tipo: nn.Module, nn.Conv1d, nn.Linear
# Tutte le classi che stiamo scrivendo ereditano da nn.Module, che è la classe base per tutti i moduli di PyTorch


class TokenEmbedding(nn.Module):
    """
    Value embedding. Il nome TokenEmbedding viene dagli autori, ma nelle time series non abbiamo token come nelle frasi

    It maps the raw input time series from:
        [batch_size, seq_len, c_in]
    to:
        [batch_size, seq_len, d_model]
    where:
    - c_in is the number of input variables
    - d_model is the internal model dimension
    """

    def __init__(self, c_in, d_model):
        super().__init__()

        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular", # È una scelta pratica: ai bordi, invece di mettere zeri, PyTorch considera la serie come se girasse ad anello
            bias=False
        )

        ### Gli autori inizializzano esplicitamente la Conv1d del value embedding
        ### con kaiming_normal_. Questo rende il nostro TokenEmbedding più fedele al codice ufficiale
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="leaky_relu"
                )

    # usiamo una convoluzione 1D. Non è una CNN complicata: serve solo a trasformare ogni punto temporale 
    # guardando anche un pochino i vicini
    def forward(self, x):
        # x has shape [batch_size, seq_len, c_in]

        # Conv1d expects [batch_size, channels, seq_len] beacuse Conv1d require input like this
        x = x.permute(0, 2, 1)

        # Apply convolution
        x = self.token_conv(x)

        # Go back to [batch_size, seq_len, d_model] 
        x = x.transpose(1, 2)

        return x


class TimeFeatureEmbedding(nn.Module):
    """
    Time feature embedding.

    It maps timestamp features from:

        [batch_size, seq_len, d_inp]

    to:

        [batch_size, seq_len, d_model]

    For ETTm2 with freq="t", d_inp = 5.
    Usually these 5 features are related to:
    month, day, weekday, hour, minute.
    """

    def __init__(self, d_model, freq="h"):
        super().__init__()

        freq_map = {
            "h": 4,  # hourly data
            "t": 5,  # minutely data
            "s": 6,
            "m": 1,
            "a": 1,
            "w": 2,
            "d": 3,
            "b": 3,
        } # in base alla frequenza dei dati, il numero di features temporali cambia. Per esempio, per dati orari (h) 
        # abbiamo 4 features: month, day, weekday, hour. Per dati minutely (t) abbiamo 5 features: month, day, weekday, hour, minute

        d_inp = freq_map[freq]

        self.embed = nn.Linear( # trasformazione lineare del tipo y = Wx + b
            in_features=d_inp,
            out_features=d_model,
            bias=False
        )

    def forward(self, x_mark):
        # x_mark has shape [batch_size, seq_len, d_inp]

        x_mark = self.embed(x_mark)

        return x_mark


class DataEmbeddingWithoutPos(nn.Module):
    """
    Data embedding without positional embedding.

    This follows Autoformer:
    value embedding + timestamp embedding, without positional embedding.
    """

    def __init__(self, c_in, d_model, freq="h", dropout=0.1):
        super().__init__()

        self.value_embedding = TokenEmbedding(
            c_in=c_in,
            d_model=d_model
        ) # crea il blocco che trasforma i valori della serie

        self.time_embedding = TimeFeatureEmbedding(
            d_model=d_model,
            freq=freq
        ) # crea il blocco che trasforma le time features

        self.dropout = nn.Dropout(p=dropout)  #Il dropout è una piccola regolarizzazione. Durante il training 
                                              #spegne casualmente alcune componenti della rappresentazione, 
                                              # così il modello non si abitua troppo a usare sempre gli stessi 
                                              # segnali

    def forward(self, x, x_mark):
        # x has shape [batch_size, seq_len, c_in]
        # x_mark has shape [batch_size, seq_len, d_inp]

        x = self.value_embedding(x) + self.time_embedding(x_mark)

        x = self.dropout(x)

        return x