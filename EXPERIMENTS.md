# Experiments

## ETTm2 - pred_len 96

| ID | Date | Experiment | LR | Epochs | Grad clip | Best val loss | Test MSE | Test MAE | Notes |
|---|---|---|---:|---:|---|---:|---:|---:|---|
| 1 | 2026-08-24 | `ettm2_96_debug` | `1e-6` | 3 | yes, `1.0` | `0.565717` | `0.655773` | `0.569181` | First stable end-to-end baseline. |
| 2 | 2026-08-24 | `ettm2_96_archfix` | `1e-6` | 3 | yes, `1.0` | `0.353043` | `0.419823` | `0.474948` | Architecture closer to official implementation. |
| 3 | 2026-08-24 | `ettm2_96_official_lr_debug` | `1e-4` | 3 | no | TBD | TBD | TBD | Run in progress. |

---

## Experiment 1 - `ettm2_96_debug`

**Config**

    learning_rate: 0.000001
    epochs: 3
    patience: 2
    use_grad_clip: true
    grad_clip: 1.0
    checkpoint_dir: checkpoints/ettm2_96
    results_dir: results/ettm2_96

**Training**

    Epoch 1/3 | train loss: 3.519540 | val loss: 0.703592
    Epoch 2/3 | train loss: 0.687172 | val loss: 0.565717
    Epoch 3/3 | train loss: 0.661568 | val loss: 0.573692

**Test**

    Test MSE: 0.655773
    Test MAE: 0.569181

---

## Experiment 2 - `ettm2_96_archfix`

**Config**

    learning_rate: 0.000001
    epochs: 3
    patience: 2
    use_grad_clip: true
    grad_clip: 1.0
    checkpoint_dir: checkpoints/ettm2_96_archfix
    results_dir: results/ettm2_96_archfix

**Main changes**

    MyLayerNorm
    AutoCorrelation closer to official implementation
    train/eval time-delay aggregation
    Kaiming init in TokenEmbedding
    activation from config

**Training**

    Epoch 1/3 | train loss: 9.845787 | val loss: 0.751138
    Epoch 2/3 | train loss: 0.628675 | val loss: 0.380335
    Epoch 3/3 | train loss: 0.529024 | val loss: 0.353043

**Test**

    Test MSE: 0.419823
    Test MAE: 0.474948

---

## Experiment 3 - `ettm2_96_official_lr_debug`

**Config**

    learning_rate: 0.0001
    epochs: 3
    patience: 2
    use_grad_clip: false
    checkpoint_dir: checkpoints/ettm2_96_official_lr_debug
    results_dir: results/ettm2_96_official_lr_debug

**Training**

    Epoch 1/3 | train loss: 5893945.384311 | val loss: 1443947.192752
    Epoch 2/3 | train loss: 5041317.302607 | val loss: 401363.655506
    Epoch 3/3 | train loss: 2959292.856378 | val loss: 223702.235447

**Test**

    Test MSE: 222113.343750
    Test MAE: 347.936035

**Notes**

    Unstable run.
    The official-like learning rate 1e-4 without gradient clipping causes exploding losses.
    This configuration is not usable for the current implementation.

---

## Current best result

    Experiment: ettm2_96_archfix
    Test MSE: 0.419823
    Test MAE: 0.474948


---

## Experiment 4 - `ettm2_96_archfix_lr1e6_no_clip`

**Config**

    learning_rate: 0.000001
    epochs: 3
    patience: 2
    use_grad_clip: false
    checkpoint_dir: checkpoints/ettm2_96_officiallr_lr1e6_no_clip
    results_dir: results/ettm2_96_officiallr_lr1e6_no_clip

**Training**

    Epoch 1/3 | train loss: 11.495857 | val loss: 2.364762
    Epoch 2/3 | train loss: 1.390411 | val loss: 0.813910
    Epoch 3/3 | train loss: 0.865983 | val loss: 0.654547

**Test**

    Test MSE: 0.770887
    Test MAE: 0.652291

**Notes**

    Stable run without gradient clipping.
    Performance is worse than the run with gradient clipping.
    This suggests that clipping is not required for stability at lr=1e-6, but improves final metrics.

---

## Current best result

    Experiment: ettm2_96_archfix
    Test MSE: 0.419823
    Test MAE: 0.474948


---

## Experiment 5 - `ettm2_96_lr1e6_no_clip_10epochs`

**Config**

    learning_rate: 0.000001
    epochs: 10
    patience: 3
    lradj: type1
    use_grad_clip: false
    checkpoint_dir: checkpoints/ettm2_96_lr1e6_no_clip_10epochs
    results_dir: results/ettm2_96_lr1e6_no_clip_10epochs

**Final training epochs**

    Epoch 8/10  | train loss: 0.668977 | val loss: 0.521131
    Epoch 9/10  | train loss: 0.665321 | val loss: 0.518145
    Epoch 10/10 | train loss: 0.663587 | val loss: 0.516634

**Best validation loss**

    Best val loss: 0.516634
    Best epoch: 10

**Test**

    Test MSE: 0.599740
    Test MAE: 0.579474

**Saved files**

    checkpoint: checkpoints/ettm2_96_lr1e6_no_clip_10epochs/checkpoint.pth
    metrics: results/ettm2_96_lr1e6_no_clip_10epochs/metrics.npy
    predictions: results/ettm2_96_lr1e6_no_clip_10epochs/pred.npy
    targets: results/ettm2_96_lr1e6_no_clip_10epochs/true.npy

**Notes**

    First serious 10-epoch run without gradient clipping.
    The training is stable and the validation loss keeps improving until epoch 10.
    The final result is better than the short 3-epoch run without clipping, but still worse than the best run with gradient clipping.

    Comparison with previous no-clipping run:

        3 epochs, no clipping:
            Test MSE: 0.770887
            Test MAE: 0.652291

        10 epochs, no clipping:
            Test MSE: 0.599740
            Test MAE: 0.579474

    This suggests that the model benefits from longer training when gradient clipping is disabled, although the clipped run remains better.

---

## Current best result

    Experiment: ettm2_96_archfix
    Test MSE: 0.419823
    Test MAE: 0.474948


---

## Experiment 6 - `ettm2_96_timefeatures_fix_lr1e6_clip_3epochs`

**Config**

    learning_rate: 0.000001
    epochs: 3
    patience: 2
    lradj: type1
    use_grad_clip: true
    grad_clip: 1.0
    checkpoint_dir: checkpoints/ettm2_96_timefeatures_fix_lr1e6_clip_3epochs
    results_dir: results/ettm2_96_timefeatures_fix_lr1e6_clip_3epochs

**Main change**

    Fixed the time feature encoding used with embed="timeF".
    The previous implementation used raw calendar features such as month, day, hour and minute.
    The new implementation follows the official Autoformer time_features logic and uses normalized temporal features in [-0.5, 0.5].

**Training**

    Epoch 1/3 | train loss: 0.387542 | val loss: 0.174240
    Epoch 2/3 | train loss: 0.318897 | val loss: 0.175363
    Epoch 3/3 | train loss: 0.306376 | val loss: 0.173270

**Best validation loss**

    Best val loss: 0.173270
    Best epoch: 3

**Test**

    Test MSE: 0.246956
    Test MAE: 0.327919

**Saved files**

    checkpoint: checkpoints/ettm2_96_timefeatures_fix_lr1e6_clip_3epochs/checkpoint.pth
    metrics: results/ettm2_96_timefeatures_fix_lr1e6_clip_3epochs/metrics.npy
    predictions: results/ettm2_96_timefeatures_fix_lr1e6_clip_3epochs/pred.npy
    targets: results/ettm2_96_timefeatures_fix_lr1e6_clip_3epochs/true.npy

**Notes**

    This is the best result obtained so far.
    The result is close to, and slightly better than, the reference MSE around 0.255 for Autoformer on ETTm2 with pred_len=96.

    Comparison with the previous best run:

        Previous best, before time feature fix:
            Test MSE: 0.419823
            Test MAE: 0.474948

        After time feature fix:
            Test MSE: 0.246956
            Test MAE: 0.327919

    This confirms that the time feature encoding was a major discrepancy with the official implementation.

---

## Current best result

    Experiment: ettm2_96_timefeatures_fix_lr1e6_clip_3epochs
    Test MSE: 0.246956
    Test MAE: 0.327919