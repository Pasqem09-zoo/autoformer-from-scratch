"""
Auto-Correlation mechanism used by Autoformer.

This module contains:
- AutoCorrelation: core mechanism based on FFT and time-delay aggregation
- AutoCorrelationLayer: multi-head wrapper that projects input into Q, K, V

Diversamente dallo pseudo codice della speedup versione dell'articolo, qui separiamo le cose che si fanno a single-head
con quelle multi-head:
AutoCorrelationLayer farà:
- proiezioni lineari Q, K, V
- reshape in heads

AutoCorrelation fa:
- resize K, V
- FFT
- correlazione
- mean
- top-k
- roll
- somma
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

    def __init__(self, c=1): ### c è l iperparametro che serve per decidere quanti ritardi temporali tenere nella A-C
        super().__init__()

        self.c = c

    def _resize_keys_values(self, queries, keys, values):
        """
        Resize keys and values to have the same temporal length as queries.

        This is needed for cross Auto-Correlation, where query length and
        key/value length may be different.
        """

        query_length = queries.shape[1] ### restituisce la dimensione della sequenza: [batch_size, seq_len, n_heads, d_head]
        key_length = keys.shape[1]

        if key_length > query_length:
            keys = keys[:, :query_length, :, :] ### gli dai la dimensione della sequenza di queries, quindi se keys è più lungo lo taglia
            values = values[:, :query_length, :, :]

        elif key_length < query_length: ### se è piu piccola, la paddi con degli zeri: padding a destra, quindi aggiunge zeri alla fine della sequenza
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
            keys = torch.cat([keys, keys_padding], dim=1) ### cat concatena keys e padding lungo la dimensione della sequenza (dim=1)
            values = torch.cat([values, values_padding], dim=1) ### cat concatena values e padding lungo la dimensione della sequenza (dim=1)

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

        batch_size, seq_len, n_heads, d_head = values.shape ### è pensata per prendere in input 4 dim cioè quelle dopo aver diviso in heads. per questo serve AutoCorrelationLayer

        top_k = int(self.c * math.log(seq_len))     # Number of delays to keep
        top_k = max(1, top_k)                       ### ci deve essere almeno 1 lag

        ### corr ha shape [B, L, H, E]
        ### Facciamo la media su heads e canali, ma NON sul batch
        ### Risultato: [B, L]
        mean_corr = torch.mean(corr, dim=(2, 3))  ### non tocco [L] perche nella speedup version vogliamo scegliere lag globali non lag diversi per ogni batch/head/canale
                                                  ### Non facciamo ancora la media sul batch perché vogliamo mantenere pesi diversi per ogni serie
                                                  ### torch.mean(corr, dim=(2, 3)) restituisce un tensore di dimensione [B, L] che rappresenta la media della correlazione su 
                                                  ### heads e canali per ogni batch e ogni lag

        ### Ora scegliamo i lag globali
        ### Per scegliere i lag facciamo la media anche sul batch
        ### Risultato: [L]
        global_corr = torch.mean(mean_corr, dim=0) ### torch.mean(mean_corr, dim=0) restituisce un tensore di dimensione [L] che rappresenta la media della correlazione su
                                                   ### tutti i batch per ogni lag. quindi global_corr è un tensore di dimensione [L] che rappresenta la correlazione media globale per ogni lag
        
        ### Prendiamo gli indici dei top_k lag più importanti
        ### topk_indices ha shape [top_k]
        topk_indices = torch.topk(global_corr, top_k, dim=-1)[1]

        ### Per ogni lag scelto, prendiamo il valore di correlazione
        ### separatamente per ogni serie del batch
        ### Risultato: [B, top_k]
        weights = torch.stack(
            [mean_corr[:, topk_indices[i]] for i in range(top_k)],
            dim=-1
        )

        ### Trasformiamo le correlazioni in pesi
        ### Ogni serie del batch ha i suoi pesi sui lag scelti
        weights = torch.softmax(weights, dim=-1)

        ### Tensore finale, stessa shape di values: [B, L, H, E]
        output = torch.zeros_like(values)

        for i in range(top_k):
            delay = int(topk_indices[i].item()) ### seleziona il lag i-esimo

            shifted_values = torch.roll(values, shifts=-delay, dims=1) ### shifta i valori di values di delay posizioni lungo la dimensione della sequenza 1 cioè [L] il tempo

            ### weights[:, i] ha shape [B]
            ### Lo trasformiamo in [B, 1, 1, 1] per moltiplicarlo con [B, L, H, E]
            weight = weights[:, i].view(batch_size, 1, 1, 1)

            output = output + weight * shifted_values ### è tipo il context vector dei transformers
        ### es. delay=24: se il modello ha scoperto che la serie si ripete ogni 24 passi, allora prende V 
        ### spostato di 24 e lo usa per costruire la nuova rappresentazione

        return output

    def forward(self, queries, keys, values): ### percorso completo di questa classe: prende queries, keys, values, 
        ### calcola le correlazioni con FFT, trova implicitamente i lag importanti tramite corr, poi chiama la time-delay aggregation
        
        # queries, keys, values have shape [B, L, H, E]
        keys, values = self._resize_keys_values(
            queries=queries,
            keys=keys,
            values=values
        )

        seq_len = queries.shape[1]

        # FFT along the temporal dimension
        q_fft = torch.fft.rfft(queries, dim=1)
        k_fft = torch.fft.rfft(keys, dim=1)

        # Correlation in frequency domain
        res = q_fft * torch.conj(k_fft)

        # Back to time domain
        corr = torch.fft.irfft(res, n=seq_len, dim=1) ### parte FFT che trova le correlazioni sui lag 

        # Time-delay aggregation
        output = self._time_delay_aggregation(values, corr) ### usa i lag piu importanti per fare una media pesata dei valori shiftati: spostare e aggregare V

        return output


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
        self.d_head = d_model // n_heads ### numero dimensioni per head, quindi se d_model=512 e n_heads=8 allora d_head=64

        ### insieme a keys e values creiamo Q K e V come nella attention classica
        self.query_projection = nn.Linear(d_model, d_model) ### d_model è la dimensione dell'input, quindi 512, e l'output è sempre 512, ma poi lo split in heads farà in modo che ogni head abbia dimensione d_head=64
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values): ### prende in input queries, keys e values di dimensione [B, L, d_model] e dopo proiezioni, split in d_heads, autocorrelation, reshape restituisce output di dimensione [B, L, d_model]

        ### salviamo le dim principali. separatamente per queries e keys perche possono avere lunghezze diverse nella fase di cross attention (K e V dall encoder e Q dal decoder)
        batch_size = queries.shape[0]
        query_len = queries.shape[1]
        key_len = keys.shape[1]

        # Project input tensors into Q, K, V
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        # Split d_model into multiple heads
        # [B, L, d_model] -> [B, L, H, d_head]
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
        ### qui entra in gioco la classe creata sopra che prende in input queries, keys e values di dimensione [B, L, H, d_head]
        output = self.autocorrelation(
            queries=queries,
            keys=keys,
            values=values
        )

        # Merge heads back
        # [B, L, H, d_head] -> [B, L, d_model]
        output = output.reshape( ### reshape richiede che d_model = n_heads * d_head. quindi mergia le ultime due dim
            batch_size,
            query_len,
            self.d_model
        )

        output = self.out_projection(output) ### proiezione finale

        return output