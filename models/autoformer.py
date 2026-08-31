"""
Complete Autoformer model.

This module will assemble embedding, encoder, decoder and final prediction.
"""


import torch
import torch.nn as nn

from layers.decomposition import SeriesDecomp
from layers.embedding import DataEmbeddingWithoutPos
from layers.autocorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.encoder import EncoderLayer, Encoder
from layers.decoder import DecoderLayer, Decoder
from layers.layer_norm import MyLayerNorm


class Autoformer(nn.Module):
    """
    Autoformer model for long-term time series forecasting.

    Input shapes:
        x_enc:      [batch_size, seq_len, enc_in]
        x_mark_enc: [batch_size, seq_len, time_features] (e.g., time of day, day of week, etc.)
        x_dec:      [batch_size, label_len + pred_len, dec_in]
        x_mark_dec: [batch_size, label_len + pred_len, time_features]

    Output shape:
        output:     [batch_size, pred_len, c_out]
    """

    def __init__(self, config):
        super().__init__()

        # Sequence lengths
        self.seq_len = config["seq_len"]
        self.label_len = config["label_len"]
        self.pred_len = config["pred_len"]

        # Input/output dimensions
        self.enc_in = config["enc_in"]
        self.dec_in = config["dec_in"]
        self.c_out = config["c_out"]

        # Model parameters
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.d_ff = config["d_ff"]
        self.enc_layers = config["enc_layers"]
        self.dec_layers = config["dec_layers"]
        self.moving_avg = config["moving_avg"]
        self.c = config["c"]
        self.dropout = config["dropout"]
        self.freq = config["freq"]

        # Activation function
        self.activation = config.get("activation", "gelu")

        # Initial decomposition used to build decoder inputs
        self.decomp = SeriesDecomp(kernel_size=self.moving_avg)

        # Embeddings for encoder and decoder inputs
        self.enc_embedding = DataEmbeddingWithoutPos(
            c_in=self.enc_in,
            d_model=self.d_model,
            freq=self.freq,
            dropout=self.dropout
        )
        self.dec_embedding = DataEmbeddingWithoutPos(
            c_in=self.dec_in,
            d_model=self.d_model,
            freq=self.freq,
            dropout=self.dropout
        )


        # Encoder: series of encoder layers, each containing an AutoCorrelation layer
        encoder_layers = []
        for _ in range(self.enc_layers):
            autocorrelation = AutoCorrelation(c=self.c)
            autocorrelation_layer = AutoCorrelationLayer(
                autocorrelation=autocorrelation,
                d_model=self.d_model,
                n_heads=self.n_heads
            )

            encoder_layer = EncoderLayer(
                autocorrelation_layer=autocorrelation_layer,
                d_model=self.d_model,
                d_ff=self.d_ff,
                moving_avg=self.moving_avg,
                dropout=self.dropout,
                activation=self.activation
            )
            encoder_layers.append(encoder_layer)

        self.encoder = Encoder(
            encoder_layers=encoder_layers,
            norm_layer=MyLayerNorm(self.d_model)
        )


        # Decoder: series of decoder layers, each containing a self-attention layer and a cross-attention layer
        decoder_layers = []
        for _ in range(self.dec_layers):
            self_autocorrelation = AutoCorrelation(c=self.c)
            self_attention_layer = AutoCorrelationLayer(
                autocorrelation=self_autocorrelation,
                d_model=self.d_model,
                n_heads=self.n_heads
            )

            cross_autocorrelation = AutoCorrelation(c=self.c)    # cross autocorrelation between encoder and decoder
            cross_attention_layer = AutoCorrelationLayer(
                autocorrelation=cross_autocorrelation,
                d_model=self.d_model,
                n_heads=self.n_heads
            )

            decoder_layer = DecoderLayer(
                self_attention_layer=self_attention_layer,
                cross_attention_layer=cross_attention_layer,
                d_model=self.d_model,
                c_out=self.c_out,
                d_ff=self.d_ff,
                moving_avg=self.moving_avg,
                dropout=self.dropout,
                activation=self.activation
            )

            decoder_layers.append(decoder_layer)

        self.decoder = Decoder(
            decoder_layers=decoder_layers,
            norm_layer=MyLayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)    # final linear projection to output dimension
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        batch_size = x_enc.shape[0]

        # for inizialization, we use the mean of the encoder input to give a reasonable starting point for the trend
        mean = torch.mean(x_enc, dim=1).unsqueeze(1)    # [B, pred_len, enc_in] -> mean on dimension 1 -> [B, 1, enc_in]
        mean = mean.repeat(1, self.pred_len, 1)         # [B, 1, enc_in] -> [B, pred_len, enc_in]

        # Zeros used to initialize the future seasonal part: zeros in [batch_size, pred_len, dec_in]
        zeros = torch.zeros(
            x_dec.shape[0],
            self.pred_len,      
            x_dec.shape[2],     # number of features in the decoder input
            device=x_enc.device,
            dtype=x_enc.dtype
        )

        # Build decomposition of encoder input: self.decomp takes the encoder input and decomposes it into seasonal and trend components
        seasonal_init, trend_init = self.decomp(x_enc)

        # Build decoder seasonal input: last label_len seasonal values + future zeros
        seasonal_init = torch.cat(
            [
                seasonal_init[:, -self.label_len:, :],
                zeros
            ],
            dim=1
        )

        # Build decoder trend input: last label_len trend values + future mean
        trend_init = torch.cat(
            [
                trend_init[:, -self.label_len:, :],
                mean
            ],
            dim=1
        )

        # Encoder: embedding + encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out)

        # Decoder: embedding + decoder
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            x=dec_out,
            cross=enc_out,
            trend=trend_init
        )

        # Final output: seasonal + trend
        output = seasonal_part + trend_part

        # Return only the prediction horizon
        output = output[:, -self.pred_len:, :]
        
        return output