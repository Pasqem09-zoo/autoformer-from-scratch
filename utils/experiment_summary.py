"""
Utilities to save experiment summaries and loss plots.
"""

import os
import matplotlib.pyplot as plt


def save_loss_plot(config, epoch_history):
    """
    Save a plot with training and validation loss curves.
    """

    results_dir = config.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    plot_path = os.path.join(results_dir, "loss_curve.png")

    if len(epoch_history) == 0:
        print("No epoch history found. Loss plot was not saved.")
        return

    epochs = [item["epoch"] for item in epoch_history]
    train_losses = [item["train_loss"] for item in epoch_history]
    val_losses = [item["val_loss"] for item in epoch_history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train loss")
    plt.plot(epochs, val_losses, marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(config.get("experiment_name", "Experiment loss curve"))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved loss curve to: {plot_path}")


def save_experiment_summary(
    config,
    epoch_history,
    mae_score,
    mse_score,
    device,
    train_size,
    val_size,
    test_size,
    train_steps,
    early_stopped
):
    """
    Save a human-readable summary of an experiment.

    The summary contains:
    - experiment information
    - device information
    - dataset sizes
    - model configuration
    - training configuration
    - epoch history
    - early stopping information
    - final test metrics
    """

    results_dir = config.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "summary.txt")

    if len(epoch_history) > 0:
        best_item = min(epoch_history, key=lambda item: item["val_loss"])
        best_epoch = best_item["epoch"]
        best_val_loss = best_item["val_loss"]
    else:
        best_epoch = "unknown"
        best_val_loss = "unknown"

    with open(summary_path, "w") as file:
        file.write("Experiment summary\n")
        file.write("==================\n\n")

        file.write("General information\n")
        file.write("-------------------\n")
        file.write(f"experiment_name: {config.get('experiment_name', 'unknown')}\n")
        file.write(f"dataset: {config.get('dataset', 'unknown')}\n")
        file.write(f"data_path: {config.get('data_path', 'unknown')}\n")
        file.write(f"features: {config.get('features', 'unknown')}\n")
        file.write(f"target: {config.get('target', 'unknown')}\n")
        file.write(f"freq: {config.get('freq', 'unknown')}\n")
        file.write(f"device: {device}\n\n")

        file.write("Dataset sizes\n")
        file.write("-------------\n")
        file.write(f"train_size: {train_size}\n")
        file.write(f"val_size: {val_size}\n")
        file.write(f"test_size: {test_size}\n")
        file.write(f"train_steps_per_epoch: {train_steps}\n\n")

        file.write("Forecasting setup\n")
        file.write("-----------------\n")
        file.write(f"seq_len: {config.get('seq_len')}\n")
        file.write(f"label_len: {config.get('label_len')}\n")
        file.write(f"pred_len: {config.get('pred_len')}\n\n")

        file.write("Model configuration\n")
        file.write("-------------------\n")
        file.write(f"model: {config.get('model', 'Autoformer')}\n")
        file.write(f"d_model: {config.get('d_model')}\n")
        file.write(f"n_heads: {config.get('n_heads')}\n")
        file.write(f"enc_layers: {config.get('enc_layers')}\n")
        file.write(f"dec_layers: {config.get('dec_layers')}\n")
        file.write(f"d_ff: {config.get('d_ff')}\n")
        file.write(f"moving_avg: {config.get('moving_avg')}\n")
        file.write(f"factor_c: {config.get('c')}\n")
        file.write(f"dropout: {config.get('dropout')}\n")
        file.write(f"embed: {config.get('embed')}\n")
        file.write(f"activation: {config.get('activation')}\n\n")

        file.write("Training configuration\n")
        file.write("----------------------\n")
        file.write(f"learning_rate: {config.get('learning_rate')}\n")
        file.write(f"epochs: {config.get('epochs')}\n")
        file.write(f"patience: {config.get('patience')}\n")
        file.write(f"batch_size: {config.get('batch_size')}\n")
        file.write(f"lradj: {config.get('lradj')}\n")

        file.write("Output paths\n")
        file.write("------------\n")
        file.write(f"checkpoint_dir: {config.get('checkpoint_dir')}\n")
        file.write(f"results_dir: {config.get('results_dir')}\n")
        file.write(f"checkpoint_file: {config.get('checkpoint_dir')}/checkpoint.pth\n")
        file.write(f"predictions_file: {results_dir}/pred.npy\n")
        file.write(f"targets_file: {results_dir}/true.npy\n")
        file.write(f"metrics_file: {results_dir}/metrics.npy\n")
        file.write(f"loss_plot_file: {results_dir}/loss_curve.png\n\n")

        file.write("Epoch history\n")
        file.write("-------------\n")
        file.write("epoch, train_loss, val_loss, epoch_time_seconds, learning_rate\n")

        for item in epoch_history:
            file.write(
                f"{item['epoch']}, "
                f"{item['train_loss']:.6f}, "
                f"{item['val_loss']:.6f}, "
                f"{item['epoch_time']:.2f}, "
                f"{item['learning_rate']}\n"
            )

        file.write("\nBest validation checkpoint\n")
        file.write("--------------------------\n")
        file.write(f"best_epoch: {best_epoch}\n")

        if isinstance(best_val_loss, float):
            file.write(f"best_val_loss: {best_val_loss:.6f}\n")
        else:
            file.write(f"best_val_loss: {best_val_loss}\n")

        file.write(f"early_stopped: {early_stopped}\n\n")

        file.write("Final test metrics\n")
        file.write("------------------\n")
        file.write(f"test_mse: {mse_score:.6f}\n")
        file.write(f"test_mae: {mae_score:.6f}\n")

    print(f"Saved experiment summary to: {summary_path}")

    save_loss_plot(config, epoch_history)