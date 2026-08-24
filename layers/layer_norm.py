"""
Custom LayerNorm used in Autoformer.

The official Autoformer implementation uses a special LayerNorm
designed for the seasonal part.

---

Normalizzazione speciale usata dagli autori di Autoformer per la parte seasonal.
Questa classe parte da una normale LayerNorm di PyTorch, quindi normalizza ogni
vettore temporale rispetto alla dimensione delle feature. Se l'input ha shape:

    [batch_size, seq_len, channels]

la LayerNorm standard lavora sull'ultima dimensione, cioè sui channels, e produce
un tensore normalizzato con la stessa shape.

Gli autori però aggiungono un passaggio in più: dopo la LayerNorm calcolano la
media lungo la dimensione temporale, cioè lungo seq_len, e la sottraggono da tutti
i punti della sequenza.

In formule, prima si calcola:

    x_hat = LayerNorm(x)

poi si calcola il bias temporale medio:

    bias = mean(x_hat, dim=tempo)

e infine si restituisce:

    x_hat - bias

Quindi, rispetto a una LayerNorm classica, questa normalizzazione non si limita a
normalizzare le feature localmente in ogni istante temporale, ma rimuove anche la
media residua lungo tutta la sequenza temporale.

L'idea è coerente con la decomposizione di Autoformer: il modello separa la serie
in una componente trend e una componente seasonal. La parte seasonal dovrebbe
rappresentare oscillazioni e variazioni periodiche, non contenere uno spostamento
medio persistente nel tempo. Sottrarre la media temporale aiuta quindi a mantenere
la rappresentazione seasonal più centrata, lasciando che il trend venga gestito
dalla componente trend del modello.

In breve:
- LayerNorm classica: normalizza rispetto alle feature;
- MyLayerNorm degli autori: normalizza rispetto alle feature e poi rimuove la
  media temporale residua.

È una piccola modifica, ma è importante perché rispetta meglio l'idea centrale di
Autoformer: separare ciò che oscilla, cioè seasonal, da ciò che si muove lentamente,
cioè trend.
"""

import torch
import torch.nn as nn


class MyLayerNorm(nn.Module):
    """
    Special LayerNorm for the seasonal part.

    It applies standard LayerNorm and then removes the temporal mean.
    """

    def __init__(self, channels):
        super().__init__()

        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        # x has shape [batch_size, seq_len, channels]

        x_hat = self.layernorm(x)

        bias = torch.mean(x_hat, dim=1)
        bias = bias.unsqueeze(1).repeat(1, x.shape[1], 1)

        x_hat = x_hat - bias

        return x_hat