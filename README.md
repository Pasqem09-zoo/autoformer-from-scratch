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

The main numerical results are also included in the `results/` directory.

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
[Autoformer](https://github.com/thuml/Autoformer).

The original authors provide the preprocessed benchmark datasets through Google Drive. Since the raw datasets are not included in this repository, they must be downloaded separately and placed inside the `data/raw/` directory before running the experiments.


### 4. Run an experiment

Each experiment is defined by a YAML configuration file inside `configs/`.  
For example, to run Autoformer on ETTm2 with prediction length 96:

```bash
python main.py --config configs/ettm2_96.yaml
```


### 5. Outputs

- **Checkpoints**: during training, model checkpoints are saved in the directory specified by `checkpoint_dir`. These files contain the trained model weights, but they are not included in the repository.

- **Results**: final lightweight outputs are saved in the directory specified by `results_dir`. For each experiment, the repository includes a short summary file and the training/validation loss curve.

- **Weights & Biases**: if `wandb_enabled: true`, training and validation metrics are logged online, together with the final test metrics.
</details>



<details>
<summary><b>Project structure</b></summary>

```text
.
├── configs/                         # YAML configuration files for the experiments
│   ├── ettm2_96.yaml                 
|   ├── ...
|   └── traffic_720.yaml
│
├── data_provider/                    # Dataset and DataLoader logic
│   └── data_loader.py
│
├── exp/                              # Experiment pipeline: training, validation and testing loop
│   └── exp_main.py                   
│
├── layers/                           # Autoformer architectural blocks
│   ├── autocorrelation.py            
│   ├── decomposition.py              
│   ├── decoder.py                    
│   ├── embedding.py                  
│   ├── encoder.py                    
│   └── layer_norm.py                 
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
│   └── timefeatures.py               
│
└── main.py                           # Main entry point
```
</details>





## Results

The table below reports the final test performance of the executed experiments.  
For each dataset and prediction length, we report Mean Squared Error (MSE) and Mean Absolute Error (MAE).

| Dataset | Prediction length | MSE | MAE |
| ------- | ----------------- | --- | --- |
| ETTm2 | 96 | 0.226 | 0.312 |
| ETTm2 | 192 | 0.307 | 0.353 |
| ETTm2 | 336 | 0.337 | 0.371 |
| ETTm2 | 720 | 0.426 | 0.417 |
| Electricity | 96 | 0.201 | 0.316 |
| Electricity | 192 | 0.213 | 0.322 |
| Electricity | 336 | 0.223 | 0.333 |
| Electricity | 720 | 0.270 | 0.371 |
| Exchange | 96 | 0.147 | 0.279 |
| Exchange | 192 | 0.269 | 0.378 |
| Exchange | 336 | 0.443 | 0.494 |
| Exchange | 720 | 1.437 | 0.914 |
| Weather | 96 | 0.244 | 0.317 |
| Weather | 192 | 0.298 | 0.355 |
| Weather | 336 | 0.365 | 0.406 |
| Weather | 720 | 0.410 | 0.426 |
| Traffic | 96 | 0.637 | 0.398 |
| Traffic | 192 | 0.648 | 0.409 |
| Traffic | 336 | 0.633 | 0.393 |
| Traffic | 720 | 0.693 | 0.428 |
| ILI | 24 | 3.739 | 1.328 |
| ILI | 36 | 2.608 | 1.047 |
| ILI | 48 | 2.969 | 1.135 |
| ILI | 60 | 3.058 | 1.172 |