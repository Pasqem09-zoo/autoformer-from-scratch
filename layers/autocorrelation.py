"""
Auto-Correlation mechanism used by Autoformer.

This module contains:
- AutoCorrelation: core mechanism based on FFT and time-delay aggregation
- AutoCorrelationLayer: multi-head wrapper that projects input into Q, K, V
"""

import torch
import torch.nn as nn
import math


class AutoCorrelation(nn.Module):
    """
    Core Auto-Correlation mechanism.

    Input shape:
        queries: [batch_size, seq_len, n_heads, d_head]
        keys:    [batch_size, seq_len, n_heads, d_head]
        values:  [batch_size, seq_len, n_heads, d_head]

    Output shape:
        output:  [batch_size, seq_len, n_heads, d_head]
    """

    def __init__(self, c=1):
        super().__init__()

        self.c = c



    def _resize_keys_values(self, queries, keys, values):
        """
        Resize keys and values to have the same temporal length as queries.

        This is needed for cross Auto-Correlation, where query length and
        key/value length may be different.
        """

        query_length = queries.shape[1]     # [batch_size, seq_len, n_heads, d_head]
        key_length = keys.shape[1]

        if key_length > query_length:
            keys = keys[:, :query_length, :, :]
            values = values[:, :query_length, :, :]

        elif key_length < query_length:
            padding_length = query_length - key_length
            keys_padding = torch.zeros(
                keys.shape[0],
                padding_length,
                keys.shape[2],
                keys.shape[3],
                device=keys.device,
                dtype=keys.dtype
            )
            values_padding = torch.zeros(
                values.shape[0],
                padding_length,
                values.shape[2],
                values.shape[3],
                device=values.device,
                dtype=values.dtype
            )
            keys = torch.cat([keys, keys_padding], dim=1)       # concatenate along the sequence dimension (dim=1)
            values = torch.cat([values, values_padding], dim=1)

        return keys, values




    def _time_delay_aggregation(self, values, corr):
        """
        Aggregate shifted versions of values using the most important delays.

        This follows the training speedup version of Autoformer:
        - average correlations over heads and channels
        - select global top-k delays using the batch average
        - use batch-specific weights for those delays
        - roll values according to the selected delays
        - combine shifted values with softmax weights
        """

        batch_size = values.shape[0]
        n_heads = values.shape[1]
        d_head = values.shape[2]
        seq_len = values.shape[3]

        top_k = int(self.c * math.log(seq_len))                     # Number of delays to keep
        top_k = max(1, top_k)

        mean_corr = torch.mean(torch.mean(corr, dim=1), dim=1)      # mean_corr has shape [B, L] and represents the mean correlation for each series in the batch over all heads and channels

        global_corr = torch.mean(mean_corr, dim=0)                  # global_corr has shape [L] and represents the global mean correlation for each lag over all series in the batch

        topk_indices = torch.topk(global_corr, top_k, dim=-1)[1]    # torch.topk: best-k lag indices with shape [top_k] over the whole batch

        weights = torch.stack(
            [mean_corr[:, topk_indices[i]] for i in range(top_k)],  # weights has shape [B, top_k] and represents the mean correlation for each series in the batch at the selected top-k lags
            dim=-1
        )
        weights = torch.softmax(weights, dim=-1)                    # normalize weights to sum to 1 for each series in the batch

        output = torch.zeros_like(values)                           # [B, H, E, L] output tensor initialized to zeros

        for i in range(top_k):
            delay = int(topk_indices[i].item())
            shifted_values = torch.roll(values, shifts=-delay, dims=-1)     # shift values by the selected "-delay" on the last dimension (L, temporal dimension)
            weight = weights[:, i].view(batch_size, 1, 1, 1)                # [B] -> [B, 1, 1, 1] to broadcast over [B, H, E, L]
            output = output + weight * shifted_values                       # new representation is a weighted sum of the shifted values according to the selected delays

        return output



    def _time_delay_aggregation_inference(self, values, corr):
        """
        Time-delay aggregation used during validation/test.

        Difference from training version:
        - training selects global delays shared by the whole batch;
        - inference selects delays separately for each element of the batch.
        """

        batch_size = values.shape[0]
        n_heads = values.shape[1]
        d_head = values.shape[2]
        seq_len = values.shape[3]

        # Create the base temporal indices [0, 1, ..., L-1] and expand them to shape [B, H, E, L]
        init_index = torch.arange(seq_len).to(values.device)
        init_index = init_index.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        init_index = init_index.repeat(batch_size, n_heads, d_head, 1)

        top_k = int(self.c * math.log(seq_len))
        top_k = max(1, top_k)

        mean_corr = torch.mean(torch.mean(corr, dim=1), dim=1)

        weights, delays = torch.topk(mean_corr, top_k, dim=-1)      # two tensors: weights and delays, both of shape [B, top_k]
        weights = torch.softmax(weights, dim=-1)

        tmp_values = values.repeat(1, 1, 1, 2)                      # values dimension: [B, H, E, 2*L]

        output = torch.zeros_like(values).float()

        for i in range(top_k):

            delay = delays[:, i]
            delay = delay.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            delay = delay.repeat(1, n_heads, d_head, seq_len)

            gather_index = init_index + delay

            pattern = torch.gather(                                 # torch.gather: gather values from tmp_values according to the indices in gather_index along the last dimension (L)
                tmp_values,
                dim=-1,
                index=gather_index
            )

            weight = weights[:, i]
            weight = weight.unsqueeze(1).unsqueeze(1).unsqueeze(1)

            output = output + pattern * weight

        return output
    


    def forward(self, queries, keys, values):
        """
        Forward pass of the AutoCorrelationLayer.

        It computes the auto-correlation between queries and keys, and then aggregates the values based on the most important time delays.
        """
        
        keys, values = self._resize_keys_values(        # resize keys and values to match the temporal length of queries
            queries=queries,
            keys=keys,
            values=values
        )

        seq_len = queries.shape[1]

        # [B, L, H, E] -> [B, H, E, L]
        queries = queries.permute(0, 2, 3, 1).contiguous()
        keys = keys.permute(0, 2, 3, 1).contiguous()
        values = values.permute(0, 2, 3, 1).contiguous()

        # FFT along the temporal dimension
        q_fft = torch.fft.rfft(queries, dim=-1)
        k_fft = torch.fft.rfft(keys, dim=-1)

        # Correlation in frequency domain
        res = q_fft * torch.conj(k_fft)

        # Back to time domain. corr represents the auto-correlation between queries and keys
        corr = torch.fft.irfft(res, n=seq_len, dim=-1)

        # Time-delay aggregation
        if self.training:
            output = self._time_delay_aggregation(values, corr)                 # speed-up training version
        else:
            output = self._time_delay_aggregation_inference(values, corr)       # inference version

        # [B, H, E, L] -> [B, L, H, E]
        output = output.permute(0, 3, 1, 2).contiguous()

        return output       # return the weighted sum of the shifted values


class AutoCorrelationLayer(nn.Module):
    """
    Multi-head Auto-Correlation layer.

    Input shape:
        queries: [batch_size, query_len, d_model]
        keys:    [batch_size, key_len, d_model]
        values:  [batch_size, key_len, d_model]

    Output shape:
        output:  [batch_size, query_len, d_model]
    """

    def __init__(self, autocorrelation, d_model, n_heads):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.autocorrelation = autocorrelation
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)



    def forward(self, queries, keys, values):

        batch_size = queries.shape[0]
        query_len = queries.shape[1]
        key_len = keys.shape[1]

        # Project input tensors into Q, K, V
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        # Reshape Q, K, V to have shape from [B, L, d_model] to [B, L, H, d_head]
        queries = queries.view(
            batch_size,
            query_len,
            self.n_heads,
            self.d_head
        )

        keys = keys.view(
            batch_size,
            key_len,
            self.n_heads,
            self.d_head
        )

        values = values.view(
            batch_size,
            key_len,
            self.n_heads,
            self.d_head
        )

        # Apply core Auto-Correlation
        output = self.autocorrelation(
            queries=queries,
            keys=keys,
            values=values
        )

        # Merge heads back: d_model = n_heads * d_head
        # [B, L, H, d_head] -> [B, L, d_model]
        output = output.reshape(
            batch_size,
            query_len,
            self.d_model
        )

        output = self.out_projection(output)

        return output