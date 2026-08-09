# Worklog - Autoformer from scratch

## Obiettivo del progetto

Reimplementare from scratch l'architettura Autoformer per long-term time series forecasting, seguendo il paper:

**Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting**

L'obiettivo è costruire una implementazione pulita, comprensibile e il più possibile fedele al paper, con risultati sperimentali confrontabili su alcuni dataset selezionati.

---

## Decisioni iniziali del progetto

- Modello principale: Autoformer.
- Task: long-term time series forecasting.
- Tipo di dati: multivariate time series.
- Dataset iniziale per debug: ETTm2.
- Primo setting sperimentale:
  - `seq_len = 96`
  - `label_len = 48`
  - `pred_len = 96`
- Loss: MSE.
- Metriche finali: MSE e MAE.
- Ambiente virtuale usato: `DLvenv`.
- Repository: `autoformer-from-scratch`.

---

## Struttura della repository

```text
autoformer-from-scratch/
│
├── main.py
├── README.md
├── WORKLOG.md
├── EXPERIMENTS.md
├── requirements.txt
│
├── configs/
├── data/
├── notebooks/
├── results/
├── scripts/
└── src/
    ├── train.py
    ├── evaluate.py
    ├── data_loader.py
    ├── metrics.py
    ├── utils.py
    └── models/
        ├── decomposition.py
        ├── embedding.py
        ├── autocorrelation.py
        ├── encoder.py
        ├── decoder.py
        └── autoformer.py
```
---

## Ruolo dei file principali

- `main.py`: punto di ingresso principale del progetto. Servirà per lanciare un esperimento completo da terminale.
- `src/train.py`: contiene il ciclo di training.
- `src/evaluate.py`: contiene la fase di valutazione.
- `src/data_loader.py`: costruisce le finestre temporali input/target.
- `src/metrics.py`: contiene MSE e MAE.
- `src/utils.py`: funzioni di supporto, come seed, device, salvataggi, early stopping.
- `src/models/decomposition.py`: Moving Average e Series Decomposition.
- `src/models/embedding.py`: embedding dei dati temporali.
- `src/models/autocorrelation.py`: meccanismo di Auto-Correlation.
- `src/models/encoder.py`: EncoderLayer e Encoder.
- `src/models/decoder.py`: DecoderLayer e Decoder.
- `src/models/autoformer.py`: modello Autoformer completo.
- `configs/`: configurazioni degli esperimenti.
- `notebooks/`: notebook per debugging e sanity checks.
- `results/`: checkpoint, plot e tabelle finali.
- `scripts/`: script `.sh` per lanciare esperimenti.
- `EXPERIMENTS.md`: tabella ordinata degli esperimenti lanciati.

---

## Log giornaliero

### 2026-08-08

**Obiettivo della sessione**

Preparare la struttura iniziale della repository e sistemare l'ambiente di lavoro.

**Fatto**

- Creata o aperta la repository `autoformer-from-scratch`.
- Sistemato l'ambiente virtuale `DLvenv`.
- Aggiornato `.gitignore`
- Creata struttura iniziale del progetto
- Deciso di aggiungere `main.py` come punto di ingresso principale del progetto
- wandb loggato
- provi l'accesso alla gpu con il codice di martina - hai un esempio nel notebook
- riffata la struttura del progetto fedelmente agli autori di atuformer

**Da fare**
- scrivere dentro ogni file dentro " " cosa fa quel modulo
- src/models/decomposition.py

---

### 2026-08-09

**Obiettivo della sessione: circa 3 ore**
Iniziare l'implementazione del primo blocco reale di Autoformer: la decomposizione della serie temporale in trend e componente residual/seasonal; embedding.

**Fatto**
- Riorganizzata la repository in stile simile a quella ufficiale degli autori:
  - `layers/`
  - `models/`
  - `data_provider/`
  - `exp/`
  - `utils/`
- Deciso di usare `layers/` per i blocchi dell'architettura e `models/` per il modello completo
- `layers/decomposition.py`:
  - Confrontato il blocco di decomposizione con il codice ufficiale degli autori in `Autoformer_EncDec.py`
  - `moving_mean` rappresenta il trend
  - `residual = x - moving_mean` rappresenta la componente seasonal/residual
  - testato `layers/decomposition.py` nel notebook e funziona tutto
- Verificato che il backend `mps` del Mac funziona. Testato `AvgPool1d` su CPU e MPS
- Iniziata la costruzione del primo file di configurazione:
  - `configs/ettm2_96.yaml`
- Deciso di usare un config leggibile, diviso in sezioni commentate. Inseriti per ora solo i parametri già discussi:
  - `dataset`
  - `seq_len`
  - `label_len`
  - `pred_len`
  - `moving_avg`
  - `use_mps`
  - `wandb_enabled`
- Scrivere all'inizio di ogni file una breve descrizione del modulo
- `layers/embedding.py`:
  - Autoformer non usa il positional embedding classico del Transformer.
  - Dal paper: Autoformer mantiene `value embedding` e `time stamp embedding`.
  - `TokenEmbedding` = value embedding;
  - `TimeFeatureEmbedding` = embedding delle informazioni temporali;
  - `DataEmbeddingWithoutPos` = somma di value embedding e time feature embedding, senza positional embedding.
  - `TokenEmbedding` prende i valori grezzi della serie, ad esempio `[B, L, 7]`, e li porta in `[B, L, d_model]`.
  - `TimeFeatureEmbedding` prende le feature temporali, ad esempio mese, giorno, weekday, ora, minuto, e le porta anch’esse in `[B, L, d_model]`
  - Verificato nel notebook che l'embedding trasforma
    x: [B, L, 7], x_mark: [B, L, 5]

    in
  
    out: [B, L, 512]

**note da ricordare su decomposition.py**
- `MovingAvg` calcola una media mobile della serie temporale. La media mobile viene usata come stima della componente lenta della serie, cioè il trend. ricorda che è fatta con avg(pooling(x))
- Il parametro importante è `moving_avg = 25`. va nel config
- Il padding non è un iperparametro indipendente: serve solo a mantenere invariata la lunghezza della sequenza.
- Gli autori mettono `moving_avg` e `series_decomp` dentro `Autoformer_EncDec.py`; noi li separiamo in `layers/decomposition.py` per chiarezza.
- `CUDA available: False` è normale sul Mac: CUDA riguarda GPU NVIDIA. Sul Mac useremo `mps`
- Il parametro `moving_avg: 25` controlla la finestra della media mobile usata in `SeriesDecomp`
- Il padding non viene messo nel config, perché viene calcolato automaticamente a partire da `moving_avg`



---

### 2026-08-10

**Obiettivo della sessione: circa 3 ore**
- Prossima sessione: iniziare `layers/autocorrelation.py`.
- ripassare velocemente la formula dell'Auto-Correlation;
- guardare il codice ufficiale degli autori;
- decidere se implementare subito la speedup version;
- creare `layers/autocorrelation.py`;
- implementare una prima versione testabile;
- testarla nel notebook con tensori random;
- verificare solo le shape, non ancora le performance.

**Fatto**

**Da fare**

**note da ricordare su...**


