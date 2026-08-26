"""
Entry point for Autoformer experiments.

This script:
- loads the YAML config
- sets the random seed
- creates the experiment
- runs training and testing

la catena è:
terminale
   ↓
python main.py --config configs/ettm2_96.yaml
   ↓
main.py legge il config
   ↓
set_seed(...)
   ↓
ExpMain(config)
   ↓
train()
   ↓
test()
"""

import argparse
import yaml
import wandb

from exp.exp_main import ExpMain
from utils.tools import set_seed


def load_config(config_path):
    """
    Load YAML configuration file.
    """

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def main():
    """
    Main function.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/ettm2_96.yaml",
        help="Path to the YAML config file"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if config.get("wandb_enabled", False):
        wandb.init(
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            config=config
        )

        wandb.define_metric("epoch", hidden=True)
        wandb.define_metric("epoch_progress", hidden=True)

        wandb.define_metric("train_loss", step_metric="epoch")
        wandb.define_metric("val_loss", step_metric="epoch")
        wandb.define_metric("epoch_time", step_metric="epoch")
        wandb.define_metric("learning_rate", step_metric="epoch")
        wandb.define_metric("epoch_value", step_metric="epoch")

        wandb.define_metric("batch_loss_every_100", step_metric="epoch_progress")

        wandb.define_metric("test_mse")
        wandb.define_metric("test_mae")

    set_seed(config["seed"])

    exp = ExpMain(config)

    exp.train()
    exp.test()

    if config.get("wandb_enabled", False):
        wandb.finish()


if __name__ == "__main__":
    main()