import argparse

from algorithms.utils.runner import main_base_wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run POLO training or evaluation.")
    parser.add_argument(
        "--config",
        default="yamls/base_actower.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--project",
        default="polo",
        help="Weights & Biases project name when --wandb is enabled.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    args = parser.parse_args()

    main_base_wandb(
        config_path=args.config,
        use_wandb=args.wandb,
        project_name=args.project,
    )
