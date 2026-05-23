import wandb
from algorithms.utils.runner import make_sweep_fn, main_base_wandb

if __name__ == "__main__":
    # wandb.agent(sweep_id='4zrx9o8t',
    #             function=make_sweep_fn('./yamls/base_actower.yaml'),
    #             project='polo-cikm-2026')
    
    main_base_wandb(config_path='yamls/base_actower.yaml', project_name='polo-cikm-2026')

