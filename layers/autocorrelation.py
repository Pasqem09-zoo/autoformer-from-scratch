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

        ### è pensata per prendere in input 4 dim cioè quelle dopo aver diviso in heads. per questo serve AutoCorrelationLayer
        batch_size = values.shape[0]
        n_heads = values.shape[1]
        d_head = values.shape[2]
        seq_len = values.shape[3]

        top_k = int(self.c * math.log(seq_len))     # Number of delays to keep
        top_k = max(1, top_k)                       ### ci deve essere almeno 1 lag

        ### corr ha shape [B, L, H, E]
        ### Facciamo la media su heads e canali, ma NON sul batch
        ### Risultato: [B, L]
        mean_corr = torch.mean(torch.mean(corr, dim=1), dim=1)  ### corr ora ha shape [B, H, E, L], come nel codice ufficiale
                                                                ### facciamo la media prima sugli heads H e poi sui canali E
                                                                ### NON facciamo ancora la media sul batch, perché vogliamo mantenere pesi diversi per ogni serie
                                                                ### il risultato ha shape [B, L]
                                                                ### quindi per ogni elemento del batch otteniamo una correlazione media per ogni possibile lag temporale

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

            shifted_values = torch.roll(values, shifts=-delay, dims=-1)  ### values ora ha shape [B, H, E, L]
                                                                        ### quindi la dimensione temporale L è l'ultima, cioè dim=-1
                                                                        ### facciamo lo stesso roll degli autori: spostiamo i values di -delay lungo il tempo
                                                                        ### in questo modo allineiamo la sequenza rispetto al lag periodico selezionato

            ### weights[:, i] ha shape [B]
            ### Lo trasformiamo in [B, 1, 1, 1] per moltiplicarlo con [B, L, H, E]
            weight = weights[:, i].view(batch_size, 1, 1, 1)

            output = output + weight * shifted_values ### è tipo il context vector dei transformers
        ### es. delay=24: se il modello ha scoperto che la serie si ripete ogni 24 passi, allora prende V 
        ### spostato di 24 e lo usa per costruire la nuova rappresentazione

        return output


    def _time_delay_aggregation_inference(self, values, corr):
        """
        Time-delay aggregation used during validation/test.

        Difference from training version:
        - training selects global delays shared by the whole batch;
        - inference selects delays separately for each element of the batch.

        values has shape [B, H, E, L]
        corr has shape [B, H, E, L]
        """

        batch_size = values.shape[0]
        n_heads = values.shape[1]
        d_head = values.shape[2]
        seq_len = values.shape[3]

        ### Creo gli indici temporali base [0, 1, ..., L-1]
        ### e li espando alla shape [B, H, E, L]
        init_index = torch.arange(seq_len).to(values.device)
        init_index = init_index.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        init_index = init_index.repeat(batch_size, n_heads, d_head, 1)

        top_k = int(self.c * math.log(seq_len))
        top_k = max(1, top_k)

        ### corr ha shape [B, H, E, L]
        ### facciamo la media su heads e canali
        ### risultato: [B, L]
        mean_corr = torch.mean(torch.mean(corr, dim=1), dim=1)

        ### In inference scegliamo i top_k lag migliori per ogni elemento del batch
        ### weights ha shape [B, top_k]
        ### delays ha shape [B, top_k]
        weights, delays = torch.topk(mean_corr, top_k, dim=-1)

        ### trasformiamo le correlazioni in pesi
        weights = torch.softmax(weights, dim=-1)

        ### raddoppiamo values lungo il tempo per evitare problemi di indice
        ### quando usiamo gather con i delay
        tmp_values = values.repeat(1, 1, 1, 2)

        output = torch.zeros_like(values).float()

        for i in range(top_k):
            ### delay del lag i-esimo per ogni elemento del batch
            ### shape: [B]
            delay = delays[:, i]

            ### espandiamo delay a [B, H, E, L]
            delay = delay.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            delay = delay.repeat(1, n_heads, d_head, seq_len)

            ### indici temporali spostati secondo il delay
            gather_index = init_index + delay

            ### prendiamo i values shiftati usando gather
            pattern = torch.gather(
                tmp_values,
                dim=-1,
                index=gather_index
            )

            ### peso del lag i-esimo per ogni elemento del batch
            weight = weights[:, i]
            weight = weight.unsqueeze(1).unsqueeze(1).unsqueeze(1)

            output = output + pattern * weight

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

        ### Portiamo la dimensione temporale alla fine, come nel codice ufficiale:
        ### [B, L, H, E] -> [B, H, E, L]
        queries = queries.permute(0, 2, 3, 1).contiguous()
        keys = keys.permute(0, 2, 3, 1).contiguous()
        values = values.permute(0, 2, 3, 1).contiguous()

        # FFT along the temporal dimension
        q_fft = torch.fft.rfft(queries, dim=-1)
        k_fft = torch.fft.rfft(keys, dim=-1)

        # Correlation in frequency domain
        res = q_fft * torch.conj(k_fft)

        # Back to time domain
        corr = torch.fft.irfft(res, n=seq_len, dim=-1) ### parte FFT che trova le correlazioni sui lag 

        # Time-delay aggregation
        if self.training:
            output = self._time_delay_aggregation(values, corr) ### training: usa lag globali condivisi dal batch, come nella speedup version degli autori
        else:
            output = self._time_delay_aggregation_inference(values, corr) ### validation/test: usa lag specifici per ogni serie del batch, come nel codice ufficiale
            
        ### Torniamo alla shape usata dal resto del nostro codice:
        ### [B, H, E, L] -> [B, L, H, E]
        output = output.permute(0, 3, 1, 2).contiguous()

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