# On Long-Term Time Series Forecasting with Autoformer

This repository contains a PyTorch reimplementation of **Autoformer**, a Transformer-based architecture for long-term time series forecasting. 

The project is based on the paper:

> **Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting**  
> Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long  
> NeurIPS 2021  
> [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/bcc0d400288793e8bdcd7c19a8ac0c2b-Paper.pdf)

The main goal of this work is to study and reproduce the core ideas of Autoformer with a clean and readable implementation. In particular, the project focuses on:

- the decomposition of time series into seasonal and trend components;
- the Auto-Correlation mechanism used instead of standard self-attention;
- the encoder-decoder architecture for long-term forecasting;
- the experimental evaluation on standard multivariate time series datasets.

For the completed experiments, see the W&B workspace:  
  [Autoformer from Scratch - W&B](https://wandb.ai/pasqem-university-of-florence/autoformer-from-scratch/workspace?nw=nwuserpasqem).  
For each run, Weights & Biases tracks the main training and evaluation quantities, including training loss, validation loss and final test metrics such as MSE and MAE.


## Project overview

Long-term time series forecasting consists of predicting future values from a past observation window.

Autoformer is based on the idea that a time series can be progressively decomposed into two main components: a trend component and a seasonal/residual component. Given a sequence $X$, the decomposition block computes

$$
X_{\text{trend}} = \text{MovingAvg}(X),
$$

and

$$
X_{\text{seasonal}} = X - X_{\text{trend}}.
$$

Therefore,

$$
X = X_{\text{seasonal}} + X_{\text{trend}}.
$$

The trend component captures the smoother long-term direction of the series, while the seasonal component contains local variations, oscillations and residual patterns.

Unlike the standard Transformer, Autoformer does not rely on classical positional embeddings. Instead, temporal information is provided through time features, while long-range temporal dependencies are modeled through the Auto-Correlation mechanism.

The Auto-Correlation mechanism replaces standard self-attention. Instead of computing pairwise attention scores between all time steps, it searches for relevant time delays and aggregates information from the most correlated shifted versions of the sequence. In simplified form, it can be seen as

$$
\text{AutoCorrelation}(Q,K,V)
\approx
\sum_{\tau \in \mathcal{T}}
w_{\tau} \cdot \text{Roll}(V,\tau),
$$

where $\mathcal{T}$ is the set of selected time delays, $w_{\tau}$ is the importance assigned to delay $\tau$, and $\text{Roll}(V,\tau)$ shifts the value sequence by $\tau$ time steps.

The decoder maintains two paths: a seasonal path and a trend path. During decoding, the model progressively updates both components. The final prediction is obtained by summing them:

$$
\hat{Y}
=
\hat{Y}_{\text{seasonal}}
+
\hat{Y}_{\text{trend}}.
$$

In this project, the model is implemented from scratch in PyTorch, following this decomposition-based encoder-decoder structure.



<details>
<summary><b>How to run the project?</b></summary>

### 1. Create and activate a virtual environment

```bash
python -m venv DLvenv
source DLvenv/bin/activate
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the datasets

The benchmark datasets can be downloaded from the official Autoformer repository:
[Autoformer](https://github.com/thuml/Autoformer)

The original authors provide the preprocessed benchmark datasets through Google Drive. After downloading them, place the CSV files inside the `data/raw/` directory.


### 4. Run an experiment

Each experiment is defined by a YAML configuration file inside `configs/`.  
For example, to run Autoformer on ETTm2 with prediction length 96:

```bash
python main.py --config configs/ettm2_96.yaml
```

### 5. Outputs

- **Checkpoints**: saved in the directory specified by `checkpoint_dir`. They contain the trained model weights.
- **Results**: saved in the directory specified by `results_dir`. They include final metrics such as MSE and MAE, together with a short experiment summary.
- **Weights & Biases**: if `wandb_enabled: true`, training and validation metrics are logged online, together with the final test metrics.
</details>



<details>
<summary><b>Project structure</b></summary>

```text
.
├── configs/                         # YAML configuration files for the experiments
│   ├── ettm2_96.yaml                 # ETTm2 experiment with prediction length 96
|
│
├── data_provider/                    # Dataset and DataLoader logic
│   └── data_loader.py
│
├── exp/                              # Experiment pipeline: training, validation and testing loop
│   └── exp_main.py                   
│
├── layers/                           # Autoformer architectural blocks
│   ├── autocorrelation.py            # Auto-Correlation mechanism and multi-head wrapper
│   ├── decomposition.py              # Moving average and series decomposition blocks
│   ├── decoder.py                    # Autoformer decoder layers and decoder stack
│   ├── embedding.py                  # Value embedding and time feature embedding
│   ├── encoder.py                    # Autoformer encoder layers and encoder stack
│   └── layer_norm.py                 # Custom layer normalization used by Autoformer
│
├── models/                           # Full Autoformer model assembly
│   └── autoformer.py                
│
├── results/                          # Final lightweight experiment outputs
|
│
├── utils/                            # Utility functions
│   ├── early_stopping.py             
│   ├── experiment_summary.py     
│   ├── metrics.py                    
│   └── timefeatures.py               # Time feature generation from timestamps
│
└── main.py                           # Main entry point for running experiments
```
</details>





## Results

The table below reports the final test performance of the executed experiments.  
For each dataset and prediction length, we report Mean Squared Error (MSE) and Mean Absolute Error (MAE). Lower values are better for both metrics.

| Dataset | Prediction length | MSE | MAE |
| ------- | ----------------- | --- | --- |
| ETTm2 | 96 | 0.226320 | 0.311904 |
| ETTm2 | 192 | 0.307347 | 0.352652 |
| ETTm2 | 336 | 0.336734 | 0.370989 |
| ETTm2 | 720 | 0.425686 | 0.417435 |
| Electricity | 96 | 0.201191 | 0.315621 |
| Electricity | 192 | 0.213100 | 0.321610 |
| Electricity | 336 | 0.222914 | 0.332650 |
| Electricity | 720 | 0.270358 | 0.370536 |
| Exchange | 96 | 0.146914 | 0.278760 |
| Exchange | 192 | 0.268973 | 0.378014 |
| Exchange | 336 | 0.442639 | 0.494495 |
| Exchange | 720 | 1.436560 | 0.914484 |
| Weather | 96 | 0.244225 | 0.316946 |
| Weather | 192 | 0.298179 | 0.355041 |
| Weather | 336 | 0.364687 | 0.405623 |
| Weather | 720 | 0.410205 | 0.426098 |
| Traffic | 96 | 0.637362 | 0.398426 |
| Traffic | 192 | 0.648346 | 0.408926 |
| Traffic | 336 | 0.632986 | 0.392751 |
| Traffic | 720 | 0.692797 | 0.428425 |
| ILI | 24 | 3.739083 | 1.327708 |
| ILI | 36 | 2.608348 | 1.047174 |
| ILI | 48 | 2.968705 | 1.135123 |
| ILI | 60 | 3.057795 | 1.172061 |
