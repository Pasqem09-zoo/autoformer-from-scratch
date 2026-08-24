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

    set_seed(config["seed"])

    exp = ExpMain(config)

    exp.train()
    exp.test()


if __name__ == "__main__":
    main()