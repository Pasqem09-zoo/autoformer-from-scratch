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



---

## Experiment 7 - `ettm2_96_timefeatures_fix_lr1e6_NOclip_3epochs`

**Config**

    learning_rate: 0.000001
    epochs: 3
    patience: 2
    lradj: type1
    use_grad_clip: false
    checkpoint_dir: checkpoints/ettm2_96_timefeatures_fix_lr1e6_NOclip_3epochs
    results_dir: results/ettm2_96_timefeatures_fix_lr1e6_NOclip_3epochs

**Main change**

    Same configuration as the previous time feature fix experiment, but without gradient clipping.

**Training**

    Epoch 1/3 | train loss: 0.390391 | val loss: 0.176535
    Epoch 2/3 | train loss: 0.322860 | val loss: 0.179653
    Epoch 3/3 | train loss: 0.309482 | val loss: 0.178263

**Best validation loss**

    Best val loss: 0.176535
    Best epoch: 1

**Test**

    Test MSE: 0.249890
    Test MAE: 0.339192

**Saved files**

    checkpoint: checkpoints/ettm2_96_timefeatures_fix_lr1e6_NOclip_3epochs/checkpoint.pth
    metrics: results/ettm2_96_timefeatures_fix_lr1e6_NOclip_3epochs/metrics.npy
    predictions: results/ettm2_96_timefeatures_fix_lr1e6_NOclip_3epochs/pred.npy
    targets: results/ettm2_96_timefeatures_fix_lr1e6_NOclip_3epochs/true.npy

**Notes**

    Stable run without gradient clipping after fixing the time feature encoding.
    The result is very close to the clipped version, showing that after the time feature fix the model no longer depends strongly on gradient clipping.

    Comparison with clipped run:

        Time feature fix, with clipping:
            Test MSE: 0.246956
            Test MAE: 0.327919

        Time feature fix, without clipping:
            Test MSE: 0.249890
            Test MAE: 0.339192

    This suggests that the main issue was the time feature encoding, not the absence of gradient clipping.

---

## Current best result

    Experiment: ettm2_96_timefeatures_fix_lr1e6_clip_3epochs
    Test MSE: 0.246956
    Test MAE: 0.327919



---

## Experiment 8 - `ettm2_96_timefeatures_fix_lr1e4_no_clip_3epochs`

**Config**

    learning_rate: 0.0001
    epochs: 3
    patience: 2
    lradj: type1
    use_grad_clip: false
    checkpoint_dir: checkpoints/ettm2_96_timefeatures_fix_lr1e4_no_clip_3epochs
    results_dir: results/ettm2_96_timefeatures_fix_lr1e4_no_clip_3epochs

**Main change**

    Same corrected time feature encoding as the previous experiments, but using the official learning rate 1e-4 without gradient clipping.

**Training**

    Epoch 1/3 | train loss: 16303178.926811 | val loss: 175882228.974790
    Epoch 2/3 | train loss: 2032609703549.020508 | val loss: 1179970889814.050537

**Interrupted epoch**

    Epoch 3 was manually interrupted because the loss was clearly exploding.

    Example batch losses during epoch 3:

        iters: 100 | loss: 8884006682624.0000000
        iters: 200 | loss: 10001548574720.0000000
        iters: 300 | loss: 12196333486080.0000000
        iters: 400 | loss: 14495031230464.0000000
        iters: 500 | loss: 19682358722560.0000000

**Test**

    No final test was executed because the run was interrupted.

**Notes**

    Unstable run.
    Even after fixing the time feature encoding, the official learning rate 1e-4 without gradient clipping causes the training loss to explode.

    The first batches of epoch 1 were initially reasonable:

        iters: 100 | loss: 0.2599852
        iters: 200 | loss: 0.2624408
        iters: 300 | loss: 0.1747247

    However, the loss started exploding within the same epoch:

        iters: 500 | loss: 36865.0351562
        iters: 700 | loss: 800670.3750000
        iters: 900 | loss: 349901248.0000000

    This confirms that the time feature fix solved the main data encoding issue, but learning rate 1e-4 remains numerically unstable in this reimplementation when gradient clipping is disabled.

    Comparison with stable no-clipping run:

        lr = 1e-6, no clipping:
            Test MSE: 0.249890
            Test MAE: 0.339192

        lr = 1e-4, no clipping:
            Training diverged.
            No valid test metrics.

---

## Current best result

    Experiment: ettm2_96_timefeatures_fix_lr1e6_clip_3epochs
    Test MSE: 0.246956
    Test MAE: 0.327919



---

## Experiment 10 - `ettm2_96_serverunifi_lr1e4_noclipping_5epochs`

**Config**

    learning_rate: 0.0001
    epochs: 5
    patience: 2
    lradj: type1
    use_grad_clip: false
    device: cuda
    checkpoint_dir: checkpoints/ettm2_96_serverunifi_lr1e4_noclipping_5epochs
    results_dir: results/ettm2_96_serverunifi_lr1e4_noclipping_5epochs

**Main change**

    Diagnostic run on the university server using CUDA and the official Autoformer learning rate.

    This run tests whether the instability previously observed with learning_rate = 0.0001 was mainly related to the MPS backend or to the implementation itself.

**Training**

    Epoch 1/5 | train loss: 0.271248 | val loss: 0.158426
    Epoch 2/5 | train loss: 0.239331 | val loss: 0.162216
    Epoch 3/5 | train loss: 0.209984 | val loss: 0.170994

**Best validation loss**

    Best val loss: 0.158426
    Best epoch: 1
    Early stopped: true

**Test**

    Test MSE: 0.235195
    Test MAE: 0.319404

**Saved files**

    checkpoint: checkpoints/ettm2_96_serverunifi_lr1e4_noclipping_5epochs/checkpoint.pth
    metrics: results/ettm2_96_serverunifi_lr1e4_noclipping_5epochs/metrics.npy
    predictions: results/ettm2_96_serverunifi_lr1e4_noclipping_5epochs/pred.npy
    targets: results/ettm2_96_serverunifi_lr1e4_noclipping_5epochs/true.npy
    summary: results/ettm2_96_serverunifi_lr1e4_noclipping_5epochs/summary.txt
    loss plot: results/ettm2_96_serverunifi_lr1e4_noclipping_5epochs/loss_curve.png

**Notes**

    This is an important result because the official learning rate no longer explodes on CUDA.

    On MPS, the same learning rate produced unstable training and extremely large losses.
    On CUDA, instead, the training remains stable and reaches good validation and test performance.

    The best checkpoint is obtained at epoch 1.
    After that, the training loss continues to decrease, but the validation loss gets worse:

        epoch 1:
            train loss: 0.271248
            val loss: 0.158426

        epoch 2:
            train loss: 0.239331
            val loss: 0.162216

        epoch 3:
            train loss: 0.209984
            val loss: 0.170994

    Early stopping is triggered at epoch 3 because the validation loss does not improve for 2 consecutive epochs.

    Comparison with previous best runs:

        Mac MPS, learning_rate = 0.000001, 10 epochs:
            Test MSE: 0.243797
            Test MAE: 0.324120

        Server CUDA, learning_rate = 0.000001, 3 epochs:
            Test MSE: 0.219496
            Test MAE: 0.302725

        Server CUDA, learning_rate = 0.0001, 5 epochs with early stopping:
            Test MSE: 0.235195
            Test MAE: 0.319404

    This confirms that CUDA gives a much more reliable backend for the official-like configuration.
    However, for this specific run, learning_rate = 0.0001 is stable but not better than the previous CUDA run with learning_rate = 0.000001.

    The current best result remains:

        Experiment: ettm2_96_serverunifi_txt_lr1e6_noclipping_3epochs
        Test MSE: 0.219496
        Test MAE: 0.302725

---

## Experiment 11 - `ettm2_96_serverunifi_lr1e4_noclipping_10epochs`

**Config**

    learning_rate: 0.0001
    epochs: 10
    patience: 3
    lradj: type1
    use_grad_clip: false
    device: cuda
    checkpoint_dir: checkpoints/ettm2_96_serverunifi_lr1e4_noclipping_10epochs
    results_dir: results/ettm2_96_serverunifi_lr1e4_noclipping_10epochs

**Main change**

    Official-like run on the university server using CUDA.

    This run uses the learning rate adopted by the Autoformer authors, together with batch size 32, early stopping, type1 learning rate adjustment and no gradient clipping.

**Training**

    Epoch 1/10 | train loss: 0.271577 | val loss: 0.156904
    Epoch 2/10 | train loss: 0.240857 | val loss: 0.159059
    Epoch 3/10 | train loss: 0.212853 | val loss: 0.164363
    Epoch 4/10 | train loss: 0.197730 | val loss: 0.157898

**Best validation loss**

    Best val loss: 0.156904
    Best epoch: 1
    Early stopped: true

**Test**

    Test MSE: 0.230977
    Test MAE: 0.316881

**Saved files**

    checkpoint: checkpoints/ettm2_96_serverunifi_lr1e4_noclipping_10epochs/checkpoint.pth
    metrics: results/ettm2_96_serverunifi_lr1e4_noclipping_10epochs/metrics.npy
    predictions: results/ettm2_96_serverunifi_lr1e4_noclipping_10epochs/pred.npy
    targets: results/ettm2_96_serverunifi_lr1e4_noclipping_10epochs/true.npy
    summary: results/ettm2_96_serverunifi_lr1e4_noclipping_10epochs/summary.txt
    loss plot: results/ettm2_96_serverunifi_lr1e4_noclipping_10epochs/loss_curve.png

**Notes**

    This is the current official-like reference run for ETTm2 with pred_len = 96.

    The run confirms that learning_rate = 0.0001 is stable on CUDA.
    This is important because the same learning rate was unstable on MPS.

    The best checkpoint is obtained at epoch 1.
    After epoch 1, the training loss continues to decrease, but the validation loss does not improve:

        epoch 1:
            train loss: 0.271577
            val loss: 0.156904

        epoch 2:
            train loss: 0.240857
            val loss: 0.159059

        epoch 3:
            train loss: 0.212853
            val loss: 0.164363

        epoch 4:
            train loss: 0.197730
            val loss: 0.157898

    Early stopping is triggered at epoch 4 because the validation loss does not improve for 3 consecutive epochs.

    Comparison with previous official-like CUDA run:

        Server CUDA, learning_rate = 0.0001, 5 epochs with patience 2:
            Test MSE: 0.235195
            Test MAE: 0.319404

        Server CUDA, learning_rate = 0.0001, 10 epochs with patience 3:
            Test MSE: 0.230977
            Test MAE: 0.316881

    The 10-epoch configuration with patience 3 gives a slightly better test result.
    This run should be used as the official-like result for pred_len = 96.

---

## Current official-like result for ETTm2 96

    Experiment: ettm2_96_serverunifi_lr1e4_noclipping_10epochs
    Test MSE: 0.230977
    Test MAE: 0.316881



---

## Experiment 12 - `ettm2_96_serverunifi_txt_lr1e6_noclipping_3epochs`

**Config**

    learning_rate: 0.000001
    epochs: 3
    patience: 2
    lradj: type1
    use_grad_clip: false
    device: cuda
    checkpoint_dir: checkpoints/ettm2_96_serverunifi_txt_lr1e6_noclipping_3epochs
    results_dir: results/ettm2_96_serverunifi_txt_lr1e6_noclipping_3epochs

**Main change**

    First stable CUDA run on the university server with corrected time features and a reduced learning rate.

    This run was mainly used to verify that the model, the dataset pipeline, CUDA execution, local result saving and W&B logging were working correctly on the server.

**Training**

    Epoch 1/3 | train loss: 0.384187 | val loss: 0.162175
    Epoch 2/3 | train loss: 0.312848 | val loss: 0.153215
    Epoch 3/3 | train loss: 0.298199 | val loss: 0.151331

**Best validation loss**

    Best val loss: 0.151331
    Best epoch: 3
    Early stopped: false

**Test**

    Test MSE: 0.219496
    Test MAE: 0.302725

**Saved files**

    checkpoint: checkpoints/ettm2_96_serverunifi_txt_lr1e6_noclipping_3epochs/checkpoint.pth
    metrics: results/ettm2_96_serverunifi_txt_lr1e6_noclipping_3epochs/metrics.npy
    predictions: results/ettm2_96_serverunifi_txt_lr1e6_noclipping_3epochs/pred.npy
    targets: results/ettm2_96_serverunifi_txt_lr1e6_noclipping_3epochs/true.npy
    summary: results/ettm2_96_serverunifi_txt_lr1e6_noclipping_3epochs/summary.txt
    loss plot: results/ettm2_96_serverunifi_txt_lr1e6_noclipping_3epochs/loss_curve.png

**Notes**

    This is an important diagnostic run because it confirms that the full training pipeline works correctly on CUDA.

    Compared to the previous Mac/MPS run with the same reduced learning rate, CUDA gives both faster training and better metrics.

    Comparison with the best previous Mac/MPS run:

        Mac MPS, learning_rate = 0.000001, 10 epochs:
            Test MSE: 0.243797
            Test MAE: 0.324120

        Server CUDA, learning_rate = 0.000001, 3 epochs:
            Test MSE: 0.219496
            Test MAE: 0.302725

    The validation loss improves at every epoch:

        epoch 1:
            val loss: 0.162175

        epoch 2:
            val loss: 0.153215

        epoch 3:
            val loss: 0.151331

    Early stopping is not triggered.
    The best checkpoint is obtained at the final epoch.

    Although this run is not official-like because it uses a lower learning rate than the Autoformer authors, it currently gives the best test performance obtained so far for pred_len = 96.

---

## Current best empirical result for ETTm2 96

    Experiment: ettm2_96_serverunifi_txt_lr1e6_noclipping_3epochs
    Test MSE: 0.219496
    Test MAE: 0.302725