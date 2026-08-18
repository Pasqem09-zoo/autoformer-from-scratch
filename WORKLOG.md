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

**note da ricordare su embedding.py**
- Autoformer prende ispirazione dall'embedding di Informer, perché mantiene l'idea di combinare i valori osservati della serie con le informazioni temporali associate ai timestamp.
- Tuttavia, Autoformer non usa il positional embedding classico: secondo gli autori, il meccanismo di Auto-Correlation fornisce già abbastanza informazione sequenziale attraverso le dipendenze periodiche tra sottoserie.
- Per questo nel nostro codice usiamo `DataEmbeddingWithoutPos`, cioè:
  `value_embedding(x) + time_embedding(x_mark)`.


---

### 2026-08-10

**Obiettivo della sessione: circa 3 ore**

Iniziare il modulo più importante e delicato di Autoformer: `layers/autocorrelation.py`.  
L'obiettivo non è necessariamente completarlo tutto oggi, ma capire bene la struttura del meccanismo Auto-Correlation e iniziare una prima implementazione testabile nel notebook.

**Fatto**
- `layers/autocorrelation.py`

**note da ricordare su AC e ACLayer**
- Questo modulo sostituisce la self-attention classica del Transformer.
- Invece di confrontare tutti i punti uno a uno, Autoformer cerca ritardi temporali importanti, cioè pattern periodici.
- Il test minimo dovrà verificare che il modulo riceva una sequenza e restituisca un tensore con shape compatibile.
- il flusso si capisce nel forward di A-C:

    Q, K, V

    ↓

    resize K,V se necessario

    ↓

    FFT(Q), FFT(K)

    ↓

    correlazione Q-K nel dominio delle frequenze

    ↓

    IFFT per ottenere correlazioni sui lag

    ↓

    time-delay aggregation su V

    ↓

    output

    la relazione tra le due funzioni è:

    AutoCorrelationLayer lavora fuori:[B, L, d_model]

    AutoCorrelation lavora dentro:[B, L, H, d_head]

    Poi AutoCorrelationLayer richiude:[B, L, H, d_head] → [B, L, d_model]

- in time_delay_aggregation() abbiamo cambiato una cosa dopo aver visto il codice degli autori.
Prima: stessi lag + stessi pesi per tutti gli elementi del batch (facevamo una sola media sugli istanti temporali).  
Versione più fedele agli autori: stessi lag + pesi batch-specific. ora diamo non solo un peso diverso ad ogni lag ma anche diverso tra le serie dentro il batch! i lag scelti sono gli stessi per tutto il batch ma i pesi cambiano tra le `B` serie. durante il training vogliono una versione veloce: scelgono pochi lag globali condivisi, così non esplode il costo computazionale. Però lasciano comunque un po’ di flessibilità: ogni serie del batch può dire “ok, i lag sono questi, ma per me il lag 24 conta più del lag 48”.


----
**Note teoriche su Auto-Correlation: Q, K, V e multi-head**

Nel meccanismo di Auto-Correlation, l’idea di base è simile all’attention classica:

- `Q` e `K` servono a trovare le dipendenze importanti;
- `V` contiene l’informazione che verrà spostata e aggregata;
- il cuore del meccanismo è `AutoCorrelation(Q, K, V)`.

Nel caso Autoformer, però, `Q` e `K` non servono a costruire una matrice di attention punto-per-punto, ma a individuare i ritardi temporali più importanti, cioè i lag/periodi più rilevanti della serie.

Prima del multi-head immaginiamo una sequenza embedded:

`x.shape = [B, L, d_model]`

Per esempio:

`x.shape = [4, 96, 512]`

Nel wrapper multi-head, questa sequenza viene proiettata in `Q`, `K` e `V` tramite layer lineari:

`Q = linear_q(x)`

`K = linear_k(x)`

`V = linear_v(x)`

All’inizio `Q`, `K` e `V` hanno ancora la stessa forma:

`Q.shape = [B, L, d_model]`

`K.shape = [B, L, d_model]`

`V.shape = [B, L, d_model]`

Poi `d_model` viene diviso in più teste. Per esempio, se:

`d_model = 512`

`n_heads = 8`

allora ogni testa lavora con:

`d_head = 512 / 8 = 64`

Dopo il reshape, i tensori diventano:

`[B, L, n_heads, d_head]`

Questa è la forma su cui lavora il meccanismo interno di Auto-Correlation.

In generale, dentro `AutoCorrelation` useremo:

`queries.shape = [B, L, H, E]`

`keys.shape = [B, S, H, E]`

`values.shape = [B, S, H, E]`

dove:

- `B` = batch size;
- `L` = lunghezza delle query;
- `S` = lunghezza di keys/values;
- `H` = numero di heads;
- `E` = dimensione di ogni head.

Nel caso di self Auto-Correlation, `Q`, `K` e `V` vengono dalla stessa sequenza, quindi normalmente:

`L = S`

Nel caso di cross Auto-Correlation nel decoder, invece:

- `Q` viene dal decoder;
- `K` e `V` vengono dall’encoder.

Quindi le lunghezze possono essere diverse. Per questo il codice ufficiale gestisce il caso in cui `K` e `V` devono essere adattati alla lunghezza di `Q`.

La distinzione importante è questa:

`AutoCorrelationLayer`

- input: `[B, L, d_model]`
- output: `[B, L, d_model]`

`AutoCorrelationLayer` è il wrapper esterno: crea `Q`, `K`, `V`, divide in heads, chiama `AutoCorrelation`, poi rimette insieme le teste.

`AutoCorrelation`

- input: `[B, L, H, d_head]`
- output: `[B, L, H, d_head]`

`AutoCorrelation` è il motore interno: lavora già sulle heads separate e implementa FFT, scelta dei top-k ritardi e time-delay aggregation.

-------
**Note teoriche su Auto-Correlation: FFT e scelta dei lag**

Autoformer non costruisce una matrice di attention punto-per-punto di dimensione `L × L`.

Invece, calcola una correlazione tra `Q` e `K` lungo la dimensione temporale, per capire quali ritardi temporali sono più importanti.

L’idea è:

- `Q` e `K` vengono trasformati con FFT;
- nel dominio delle frequenze si calcola la correlazione;
- con la trasformata inversa si torna nel dominio del tempo;
- il risultato indica quanto sono importanti i diversi lag temporali.

Concettualmente:

`q_fft = FFT(Q)`

`k_fft = FFT(K)`

`corr = inverse_FFT(q_fft * conjugate(k_fft))`

La dimensione temporale di `corr` rappresenta i possibili ritardi della serie.

Dopo aver calcolato `corr`, Autoformer seleziona solo i ritardi più importanti usando `topk`.

Il numero di lag scelti è circa:

`top_k = factor * log(L)`

dove:

- `L` è la lunghezza della sequenza;
- `factor` è un iperparametro;
- `top_k` è il numero di ritardi selezionati.

Quindi Autoformer non usa tutti i lag possibili, ma solo quelli più rilevanti.

La logica è:

1. calcolare la correlazione tra `Q` e `K`;
2. individuare i lag più importanti;
3. usare questi lag per spostare `V`;
4. aggregare le versioni spostate di `V`.

In sintesi, la FFT serve a calcolare in modo efficiente le correlazioni per tutti i ritardi, mentre `topk` serve a tenere solo i ritardi più informativi.

Dopo FFT e topK abbiamo una lista di lag importanti e a quel punto autoformer prende V e lo sposta secondo quei ritardi:  
`torch.roll(values, shifts=-delay, dims=1)`  
perche il senso è "se il modello ha scoperto che la serie si ripete ogni 24 passi allora prende i valori spostati di 24 posizioni e li usa per costruire la nuova rappresentazione". poi non li somma tutti allo stesso modo ma usa dei pesi dati dalla softmax calcolata sulle autocorrelazioni: questa è la **time-delay aggregation**.


----

## 2026-08-11

### Obiettivo della sessione

In questa sessione vogliamo iniziare a costruire il blocco `encoder.py`, cioè la prima parte vera dell’architettura Autoformer che mette insieme i moduli già implementati.

L’obiettivo principale è passare da moduli isolati a un primo blocco composito, usando:

- `AutoCorrelationLayer`
- `SeriesDecomp`
- feed-forward network
- connessioni residuali

In particolare, vogliamo capire e poi implementare la struttura dell’`EncoderLayer`, che nel paper corrisponde alla sequenza:

1. Auto-Correlation sul segnale in input
2. residual connection
3. series decomposition
4. feed-forward network
5. residual connection
6. seconda series decomposition

L’obiettivo minimo della sessione è avere una prima versione di `EncoderLayer` testabile con tensori finti.


### da fare
- come feedforward avevo messo un linear ma gli autori usano conv1d: Il senso è questo: Conv1d(kernel_size=1) non guarda finestre temporali, quindi non mescola istanti diversi. Lavora istante per istante sulle feature, come una Linear, però nella forma usata dagli autori.

- TODO: my_Layernorm, è una normalizzazione speciale che gli autori applicano alla parte seasonal. Non è la normale LayerNorm, perché dopo la normalizzazione toglie anche una media temporale. È una finezza del codice ufficiale, non necessaria adesso per testare il nostro encoder. La possiamo aggiungere più avanti se vogliamo essere più fedeli.

### fatto


### Note sull'encoder

In Autoformer l’encoder è diverso dal Transformer classico perché dopo i blocchi principali fa sempre decomposizione seasonal/trend. In particolare il trend viene soppresso ma fa parte dell'output:

S1, _ = SeriesDecomp(AutoCorrelation(x) + x)

S2, _ = SeriesDecomp(FeedForward(S1) + S1)

output = S2

Perché? Perché il trend viene gestito soprattutto dal decoder. L’encoder deve produrre una rappresentazione più pulita della parte stagionale/periodica, come se dicesse: “tolgo la nebbia lenta del trend e tengo il disegno ricorrente”. l'architettura è:

x

↓

AutoCorrelationLayer: trova dipendenze periodiche

↓

residual connection: x + autocorr(x)

↓

SeriesDecomp: separa seasonal e trend

↓

FeedForward: continua solo con seasonal

↓

residual connection

↓

SeriesDecomp

↓

output encoder layer: trend viene scartato

Le dimensioni sono sempre le stesse: [B, L, d_model] → [B, L, d_model]. la cosa bella è che `EncoderLayer` non deve sapere come funziona internamente la FFT. usa `ACLayer` come blocco già pronto. Una cosa da notare: nel paper/feed-forward ufficiale spesso usano Conv1d con kernel 1, che è equivalente a fare una trasformazione punto-per-punto sulla dimensione feature. Noi per ora usiamo Linear, che è più leggibile e per una prima implementazione va benissimo. È come usare una chiave inglese normale invece di una chiave super-professionale: fa lo stesso lavoro e capiamo meglio cosa succede.


### Note sul decoder

la novità è sicuramente la cross-attention: hai due input x=decoder seasonal input; cross=encoder output. infatti cross-att usa Q = x dal decoder; K = cross  dall’encoder; V = cross  dall’encoder.


x = x + SelfAutoCorrelation(x):                   self Auto-Correlation sul decoder input

↓

x, trend1 = SeriesDecomp(x):                      decomposition → produce trend1

x = x + CrossAutoCorrelation(x, encoder_output):  cross Auto-Correlation con encoder output

↓

x, trend2 = SeriesDecomp(x):                      decomposition → produce trend2

y = FeedForward(x)

↓

x, trend3 = SeriesDecomp(x + y):                  decomposition → produce trend3

residual_trend = trend iniziale + trend1 + trend2 + trend3


----

## Sessione 2026-08-12

### Obiettivo della sessione

In questa sessione vogliamo continuare il lavoro sul decoder di Autoformer.

Nella sessione precedente abbiamo iniziato `layers/decoder.py`, introducendo il `DecoderLayer`, cioè il blocco che combina:

- self Auto-Correlation sul decoder input;
- cross Auto-Correlation con l’output dell’encoder;
- tre decomposizioni progressive;
- accumulo della componente trend;
- proiezione del trend da `d_model` a `c_out`.

L’obiettivo principale di oggi è rendere il decoder testabile e iniziare a costruire il decoder completo come stack di più `DecoderLayer`.


### Cose da fare
- TODO: GLI AUTORI USANO argomenti x_mask e cross_mask. Noi per ora non usiamo maschere, coerente con una prima versione pulita.
- TODO: GLI AUTORI usano relu o gelu.
- TODO: gli autori usano `my_Layernorm` mentre io `nn.LayerNorm`. in futuro per essere più fedeli al codice ufficiale, più avanti possiamo implementare MyLayerNorm in encoder.py o in un file utility. Non è urgente per il forward test, però per riproduzione risultati potrebbe contare.


### Fatto
- Riprendere velocemente la struttura teorica del `DecoderLayer`.
- Controllare che il codice del `DecoderLayer` sia coerente con il codice ufficiale degli autori.
- Testare `DecoderLayer` nel notebook con tensori finti.
- Verificare le shape attese:
  - decoder input seasonal: `[B, dec_len, d_model]`
  - encoder output: `[B, enc_len, d_model]`
  - decoder output seasonal: `[B, dec_len, d_model]`
  - residual trend: `[B, dec_len, c_out]`
- Implementare la classe `Decoder`, cioè lo stack di più `DecoderLayer`.
- Testare il `Decoder` completo con tensori finti.
- `Autoformer` e test


### Note sul Decoder
L'idea del decoder come M decoderLayers è la seguente:

trend iniziale

↓

DecoderLayer 1 produce residual_trend_1   
trend = trend + residual_trend_1

↓

DecoderLayer 2 produce residual_trend_2  
trend = trend + residual_trend_2

↓

...

↓

output seasonal finale + trend finale


### Note su Autoformer e shape

Prima di implementare `models/autoformer.py`, abbiamo chiarito come Autoformer costruisce gli input del decoder.

Il decoder non riceve direttamente un input già pronto, ma costruisce due inizializzazioni a partire da `x_enc`, cioè l’input grezzo dell’encoder:

- `seasonal_init`, ottenuta dalla decomposizione di `x_enc`;
- `trend_init`, ottenuta dalla stessa decomposizione.

Attenzione: `seasonal_init` non è l’output dell’ultimo layer dell’encoder.  
L’output finale dell’encoder si chiamerà `enc_out` e verrà usato nel decoder durante la cross Auto-Correlation. In quella fase `enc_out` resta fisso e fornisce al decoder le informazioni estratte dal passato.

NOTAZIONE:  
`I = seq_len`  
`O = pred_len`  
`I/2 = label_len`  
`I/2 + O = label_len + pred_len = dec_len`

i 3 paramentri di una serie multivariata sono: `seq_len = quanto passato do all’encoder`, `label_len = quanto passato noto do anche al decoder`, `pred_len  = quanto futuro voglio prevedere`. Nel primo esperimento abbiamo fatto `seq_len = 96`, `label_len = 48` e `pred_len = 96`. quindi x_enc.shape = [B, 96, 7] con B batchsize mentre il decoder non parte da 0 ma gli diamo gli ultimi 48 istanti e poi gli lasciamo lo spazio per generare i 96 istanti futuri, e quindi la lunghezza del decoder è:

`dec_len = label_len + pred_len = 144`

La parte seasonal del decoder viene costruita prendendo gli ultimi `label_len` passi di `seasonal_init` e concatenando zeri per i `pred_len` passi futuri:  
seasonal decoder input = ultimi 48 punti seasonal + 96 zeri futuri => [B, 48, 7] + [B, 96, 7] = [B, 144, 7]

La parte trend viene costruita prendendo gli ultimi `label_len` passi di `trend_init` e concatenando la media della serie ripetuta per i `pred_len` passi futuri:  
trend decoder input = ultimi 48 punti trend + media ripetuta 96 volte => [B, 48, 7] + [B, 96, 7] = [B, 144, 7]  
Perché la media? Perché il trend è la parte lenta della serie. Se devo inizializzare il futuro, una stima semplice e stabile è dire: “parto dal livello medio recente”.

Quindi:

`seasonal future → inizializzato a zeri`

`trend future → inizializzato con la media`

Questo serve a dare al decoder due binari separati:

- una parte seasonal, che verrà poi portata nello spazio `d_model` tramite embedding e diventerà [B, 144, 512];
- una parte trend, che rimane nello spazio originario `c_out`, quindi [B, 144, 7], e viene aggiornata progressivamente dal decoder.

**Il decoder riceve un pezzo di passato noto per orientarsi, e uno spazio futuro inizializzato in modo semplice: zeri per la seasonal, media per il trend.**

**se vuoi capire meglio guarda le dimensioni I/2+O del paper!!!**


### Note riassuntive su Autoformer
Encoder input:
x_enc = passato osservato
shape: [B, seq_len, enc_in]

Encoder:
embedding → encoder → enc_out
shape: [B, seq_len, d_model]

Decoder initialization:
seasonal_init = ultimi label_len punti seasonal + zeri futuri
trend_init    = ultimi label_len punti trend + media futura

Decoder:
seasonal_init embedded → decoder seasonal path
trend_init resta in c_out → decoder trend path

Output:
seasonal_part + trend_part
poi prendiamo solo gli ultimi pred_len istanti

**Autoformer non predice direttamente da x_enc.
Prima separa seasonal/trend, poi fa lavorare il decoder su due binari:
uno per la stagionalità e uno per il trend.**

----

## Sessione 2026-08-14

### Obiettivo della sessione

In questa sessione vogliamo iniziare il blocco di **data loading**, cioè passare dai tensori finti usati nei test ai dati reali del dataset ETTm2.

Finora il modello Autoformer completo è stato implementato e testato con una forward end-to-end usando tensori casuali. Il prossimo obiettivo è costruire correttamente gli input reali richiesti dal modello:

- `x_enc`
- `x_mark_enc`
- `x_dec`
- `x_mark_dec`
- `y`

L’obiettivo minimo della sessione è iniziare `data_provider/data_loader.py` e capire bene come funzionano le sliding windows per il forecasting.



### Cose fatte
- Abbiamo chiarito meglio la differenza tra:
  - **finestra temporale**, cioè un singolo esempio del dataset;
  - **batch**, cioè un gruppo di finestre passato insieme al modello;
  - **Dataset**, che costruisce una finestra alla volta;
  - **DataLoader**, che raggruppa più finestre in batch.
- Abbiamo lavorato su `data_provider/data_loader.py`, costruendo la struttura principale della classe `ETTDataset`. La classe ora contiene:
  - `__init__`, per salvare i parametri principali;
  - `_read_data()`, per leggere il CSV, selezionare le colonne, fare split temporale, normalizzazione e time features;
  - `__len__`, per calcolare quante sliding windows sono disponibili;
  - `__getitem__`, per costruire una singola finestra temporale;
  - `get_data_loader()`, per creare Dataset e DataLoader PyTorch.
- Abbiamo chiarito anche che lo `StandardScaler` deve fare `fit` solo sul training set, ma poi la trasformazione va applicata a tutto il dataset usando le statistiche del train.
- Infine abbiamo aggiunto la funzione `get_data_loader()`, che crea un `ETTDataset` e lo passa al `DataLoader` PyTorch, impostando `batch_size`, `shuffle` e `drop_last`.




### Note su Dataset, DataLoader, batch, epoche e backpropagation

Nel training di un modello di deep learning non passiamo tutto il dataset al modello in un colpo solo. Il dataset viene diviso in piccoli gruppi chiamati **batch**. Un batch è un insieme di esempi presi dal dataset. Nel nostro caso, un esempio sarà una finestra temporale costruita dalla serie, mentre un batch sarà un gruppo di finestre.

Per esempio, con:

`batch_size = 32`

il modello riceve 32 finestre temporali alla volta.

Nel nostro caso un batch avrà forme tipo:

`x_enc: [32, 96, 7]`

`x_mark_enc: [32, 96, 5]`

`x_dec: [32, 144, 7]`

`x_mark_dec: [32, 144, 5]`

`y: [32, 96, 7]`

Dove:

- `32` è il numero di esempi nel batch;
- `96` è la lunghezza del passato o del futuro da predire;
- `144` è la lunghezza del decoder, cioè `label_len + pred_len`;
- `7` è il numero di variabili della serie;
- `5` è il numero di feature temporali.

Il **Dataset** definisce come costruire un singolo esempio. Nel nostro caso dovrà prendere una porzione della serie temporale e costruire:

- input encoder;
- input decoder;
- target vero;
- feature temporali associate.

Il **DataLoader** prende il Dataset e si occupa di creare automaticamente i batch. Quindi non siamo noi a prendere manualmente 32 esempi alla volta: lo fa PyTorch.

Un’**epoca** corrisponde a un giro completo su tutto il dataset di training. Se il dataset ha 3200 esempi e `batch_size = 32`, allora in una epoca avremo circa:

`3200 / 32 = 100 batch`

Quindi durante una epoca il modello vede tutti gli esempi di training, ma divisi in batch.

Per ogni batch succede questo ciclo:

1. il DataLoader fornisce un batch;
2. il modello fa la forward;
3. si confronta la previsione con il target vero;
4. si calcola la loss;
5. si fa la backpropagation;
6. l’optimizer aggiorna i pesi del modello.

Nel nostro caso la forward sarà:

`output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)`

e l’output avrà shape:

`[batch_size, pred_len, c_out]`

Il target `y` dovrà avere la stessa shape:

`[batch_size, pred_len, c_out]`


La loss misura quanto la previsione è lontana dal valore vero. Per Autoformer useremo probabilmente la MSE loss:

`loss = MSE(output, y)`

La **backpropagation** calcola come ogni peso del modello ha contribuito all’errore. In pratica PyTorch calcola i gradienti della loss rispetto ai parametri del modello. Il ciclo tipico è:

```python
optimizer.zero_grad()
output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
loss = criterion(output, y)
loss.backward()
optimizer.step()
```

Il significato dei passaggi è:

- `optimizer.zero_grad()` azzera i gradienti del batch precedente;
- `output = model(...)` esegue la forward pass, cioè produce la previsione;
- `loss = criterion(output, y)` confronta la previsione con il target vero;
- `loss.backward()` calcola i gradienti con la backpropagation;
- `optimizer.step()` aggiorna i pesi del modello usando quei gradienti.

È importante azzerare i gradienti prima di ogni batch perché PyTorch, di default, li accumula. Se non facessimo `zero_grad()`, i gradienti del batch attuale si sommerebbero a quelli del batch precedente.

Durante una epoca, questo ciclo viene ripetuto per tutti i batch del training set. Quindi il modello non aggiorna i pesi una volta sola per epoca, ma li aggiorna dopo ogni batch.

Per esempio, se abbiamo 3200 finestre di training e `batch_size = 32`, una epoca contiene circa 100 batch. Questo significa che in una sola epoca il modello farà circa 100 aggiornamenti dei pesi.

Dopo ogni epoca possiamo valutare il modello sul validation set. In questa fase non facciamo backpropagation, perché non vogliamo aggiornare i pesi: vogliamo solo misurare quanto il modello generalizza su dati non usati direttamente per il training.

La logica quindi è:

`training batch → forward → loss → backward → update pesi`

mentre in validazione:

`validation batch → forward → loss → nessun update`

Dopo molte epoche, se tutto funziona, la loss di training dovrebbe diminuire. La loss di validazione invece ci dice se il modello sta imparando in modo utile oppure se sta solo memorizzando il training set.

In sintesi:

- il **Dataset** costruisce i singoli esempi;
- il **DataLoader** raggruppa gli esempi in batch;
- il **batch** è ciò che il modello vede a ogni aggiornamento;
- una **epoca** è un giro completo su tutto il training set;
- la **loss** misura l’errore della previsione;
- la **backpropagation** calcola i gradienti;
- l’**optimizer** aggiorna i pesi.

Il training è quindi un ciclo ripetuto: il modello guarda un batch, fa una previsione, misura quanto ha sbagliato, corregge un po’ i pesi, e passa al batch successivo.


### Schema sintetico: batch, loss e aggiornamento dei pesi

Il dataset di training è formato da tanti esempi:

$$
\mathcal{D}_{train} = \{(x^{(i)}, y^{(i)})\}_{i=1}^{N}
$$

Nel nostro caso ogni esempio è una finestra temporale: il modello riceve un pezzo di passato e deve prevedere un pezzo di futuro.

Il `DataLoader` non passa tutto il dataset al modello in una volta sola, ma costruisce dei batch. Un batch è un sottoinsieme di esempi:

$$
\mathcal{B} = \{(x^{(i)}, y^{(i)})\}_{i=1}^{B}
$$

dove $$B$$ è il `batch_size`.

Per ogni batch, Autoformer fa una previsione:

$$
\hat{y} = f_{\theta}(x)
$$

dove:

- $f_{\theta}$ è il modello Autoformer;
- $\theta$ rappresenta tutti i pesi del modello;
- $x$ rappresenta gli input del modello;
- $\hat{y}$ è la previsione;
- $y$ è il valore vero.

Poi confrontiamo la previsione con il target vero usando una loss. Nel nostro caso useremo probabilmente la MSE:

$$
\mathcal{L}(\theta)
=
\frac{1}{M}
\sum_{m=1}^{M}
(\hat{y}_m - y_m)^2
$$

dove $$M$$ è il numero totale di valori confrontati nel batch.

La loss misura quanto il modello sta sbagliando su quel batch.

A questo punto entra la backpropagation, che calcola il gradiente della loss rispetto ai pesi:

$$
\nabla_{\theta}\mathcal{L}(\theta)
$$

Questo gradiente dice in che direzione modificare i pesi per ridurre l’errore.

L’optimizer aggiorna i pesi. In forma semplificata:

$$
\theta_{new}
=
\theta_{old}
-
\eta \nabla_{\theta}\mathcal{L}(\theta)
$$

dove $$\eta$$ è il learning rate.

Il ciclo di training su un batch è quindi:

```python
optimizer.zero_grad()

output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

loss = criterion(output, y)

loss.backward()

optimizer.step()
```

Il significato è:

- `zero_grad()` azzera i gradienti vecchi;
- `model(...)` produce la previsione;
- `criterion(...)` calcola la loss;
- `backward()` calcola i gradienti;
- `step()` aggiorna i pesi.

Una **epoca** è un giro completo su tutto il training set. Se il dataset ha $$N$$ esempi e il batch size è $$B$$, allora il numero di batch per epoca è circa:

$$
\frac{N}{B}
$$

Durante il training, questo ciclo viene ripetuto batch dopo batch ed epoca dopo epoca.

In sintesi:

$$
\text{batch}
\rightarrow
\text{forward}
\rightarrow
\text{loss}
\rightarrow
\text{backpropagation}
\rightarrow
\text{aggiornamento pesi}
$$

Durante la validation, invece, facciamo solo:

$$
\text{forward}
\rightarrow
\text{loss}
$$

senza aggiornare i pesi. La validation loss serve a capire se il modello sta generalizzando oppure se sta solo imparando troppo bene il training set.


---

## Sessione 2026-08-17

### Obiettivo della sessione

In questa sessione vogliamo completare e testare il blocco di **data loading** per Autoformer usando dati reali.

Finora abbiamo scritto la struttura principale di `data_provider/data_loader.py`, ma il Dataset e il DataLoader non sono ancora stati testati su un CSV vero. L’obiettivo della sessione è quindi scaricare o inserire il dataset ETTm2, verificare che venga letto correttamente e controllare che le finestre temporali e i batch abbiano le shape attese.


### Cose da fare

- Scaricare il dataset ETTm2.

- Inserire il file nella cartella:

```text
data/raw/ETTm2.csv
```

- Controllare che il CSV venga letto correttamente con `pandas`.

- Verificare le colonne del dataset, in particolare:
  - `date`;
  - variabili numeriche;
  - target `OT`.

- Testare la classe `ETTDataset` nel notebook.

- Creare un dataset di training con:

```python
flag="train"
```

- Controllare la lunghezza del dataset con:

```python
len(train_dataset)
```

- Estrarre un singolo esempio con:

```python
sample = train_dataset[0]
```

- Controllare le shape di:
  - `x_enc`;
  - `x_mark_enc`;
  - `x_dec`;
  - `x_mark_dec`;
  - `y`.

- Creare il primo `DataLoader` usando `get_data_loader()`.

- Estrarre un batch dal DataLoader.

- Controllare che il batch abbia le shape attese.

- Verificare che il batch possa essere passato al modello Autoformer.

- Eseguire una prima forward del modello usando dati reali invece di tensori casuali.

- Se la forward funziona, preparare il passaggio successivo verso il training loop.



### Cose fatte

- scaricato dataset ETTm2 da qui 
  https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small

- messo dataset nel gitignore e non comprarirà nemmeno la cartella dei dataset

- TODO: IN FUTURO NEL README METTICI I COMANDI PER SCARICARE I DATASET


### Note eventuali
