"""
Embedding layers for Autoformer.

Autoformer does not use positional embedding.
It uses:
- value embedding: embeds the observed time series values
- time feature embedding: embeds timestamp information (calendar information, such as month, day, weekday, hour, minute)
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Value embedding.

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

        self.token_conv = nn.Conv1d(    # Conv1d in this case, it is used to embed the input time series values into a higher-dimensional space
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",    # circular padding means that the input is treated as if it were circular, so the last element is followed by the first element
            bias=False
        )

        # initialization of the convolutional layer using Kaiming normal initialization
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="leaky_relu"
                )



    # [batch_size, seq_len, c_in] -> [batch_size, seq_len, d_model]
    def forward(self, x):

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
            "h": 4,         # month, day, weekday, hour of day
            "t": 5,
            "min": 5,
            "15min": 5,     # month, day, weekday, hour of day, minute of hour
            "10min": 5,     
            "s": 6,         # month, day, weekday, hour of day, minute of hour, second of minute
            "m": 1,
            "a": 1,
            "w": 2,
            "d": 3,
            "b": 3,
        }

        d_inp = freq_map[freq]

        self.embed = nn.Linear(     # linear layer that maps the input time features to the model dimension
            in_features=d_inp,
            out_features=d_model,
            bias=False
        )

    def forward(self, x_mark):

        x_mark = self.embed(x_mark)

        return x_mark


class DataEmbeddingWithoutPos(nn.Module):
    """
    Data embedding without positional embedding:

    value embedding + timestamp embedding, without positional embedding.
    """

    def __init__(self, c_in, d_model, freq="h", dropout=0.1):
        super().__init__()

        self.value_embedding = TokenEmbedding(          # transforms the input time series values into a higher-dimensional space
            c_in=c_in,
            d_model=d_model
        )

        self.time_embedding = TimeFeatureEmbedding(     # transforms the timestamp features into a higher-dimensional space
            d_model=d_model,
            freq=freq
        )

        self.dropout = nn.Dropout(p=dropout)



    def forward(self, x, x_mark):
        # x has shape [batch_size, seq_len, c_in]
        # x_mark has shape [batch_size, seq_len, d_inp]
        # x = x + x_mark has shape [batch_size, seq_len, d_model] after adding the two embeddings

        x = self.value_embedding(x) + self.time_embedding(x_mark)

        x = self.dropout(x)

        return x