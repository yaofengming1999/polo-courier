# POLO

POLO is a route-aware multi-agent reinforcement learning project for courier-order matching and dispatch simulation. The repository includes the simulator, feature extraction pipeline, and a PyTorch-based POLO controller.

## Structure

- `algorithms/`: controllers, policy/value networks, and training logic
- `env/`: simulator, courier/order models, and dispatch environment wrapper
- `yamls/base_polo_train.yaml`: training configuration
- `yamls/base_polo_test.yaml`: evaluation configuration
- `run_polo.py`: command-line entry point for running experiments
- `saved/`: saved checkpoints such as `latest_model.pt` and `best_model.pt`

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

Run evaluation with local logging only:

```bash
python run_polo.py
```

This uses `yamls/base_polo_test.yaml` by default.

Run training:

```bash
python run_polo.py --config yamls/base_polo_train.yaml
```

Run evaluation with an explicit test config:

```bash
python run_polo.py --config yamls/base_polo_test.yaml
```

Run with Weights & Biases enabled:

```bash
python run_polo.py --config yamls/base_polo_train.yaml --wandb --project polo
```

## Checkpoints

The test config loads a checkpoint through `model_load_path`, for example:

```bash
saved/large/P=2/<run_name>/latest_model.pt
```

Update `model_load_path` in `yamls/base_polo_test.yaml` to point at the checkpoint you want to evaluate.

## Notes

- Generated artifacts such as model checkpoints, notebook checkpoints, and `__pycache__` files are ignored.
- Use `yamls/base_polo_train.yaml` for training and `yamls/base_polo_test.yaml` for evaluation.
- `run_polo.py` keeps WandB optional so the project can run without account-specific settings.
