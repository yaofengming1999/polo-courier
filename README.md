# POLO

POLO is a route-aware multi-agent reinforcement learning project for courier-order matching and dispatch simulation. The repository includes the simulator, feature extraction pipeline, and a PyTorch-based POLO controller.

## Structure

- `algorithms/`: controllers, policy/value networks, and training logic
- `env/`: simulator, courier/order models, and dispatch environment wrapper
- `yamls/base_actower.yaml`: main experiment configuration
- `run_tower.py`: command-line entry point for running experiments

## Requirements

- Python 3.10
- Conda is recommended
- Local CSV data under `env/meituan_order_instance/`

The simulator expects files such as `orders_combined_small.csv` and `couriers_small.csv` in `env/meituan_order_instance/`. That data directory is ignored by Git and is not included in this repository.

## Setup

```bash
conda env create -f environment.yml
conda activate courier
```

## Run

Run with local logging only:

```bash
python run_tower.py
```

Run with Weights & Biases enabled:

```bash
python run_tower.py --wandb --project polo
```

Use a different config file if needed:

```bash
python run_tower.py --config yamls/base_actower.yaml
```

## Notes

- Generated artifacts such as model checkpoints, notebook checkpoints, and `__pycache__` files are ignored.
- The default configuration is stored in `yamls/base_actower.yaml`.
- `run_tower.py` now keeps WandB optional so the project can run without account-specific settings.
