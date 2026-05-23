import os
import time
import wandb
import numpy as np
import tracemalloc
import torch
from typing import List, Dict
from env.matching_wrapper import Wrapper
from algorithms.controller import Controller
from env.utils.order_generator import InstanceGenerator
import yaml
from datetime import datetime


class Runner:
    def __init__(self, config):
        self.config = config

        self.is_learning = self.config.get('is_learning', False)
        self.is_deep = self.config.get('is_deep', False)
        self.model_load_path = self.config.get('model_load_path', None)
        self.model_save_path = self.config.get('model_save_path', None)

        self.instance_generator = self._create_instance_generator()
        self.wrapper = self._create_wrapper()
        self.controller: Controller = self._initialize_controller()
        self.best_model_path = None

        if self.model_load_path:
            self._load_checkpoint()

    def _create_instance_generator(self) -> InstanceGenerator:
        return InstanceGenerator(
            scale=self.config.get('instance_size', 'small'),
            platform_num=self.config.get('number_of_platforms', 1),
            courier_appearance_ratio=self.config.get('courier_appearance_ratio'),
            time_start=self.config.get('time_start', None),
            time_end=self.config.get('time_end', None),
            spatial_platform_diff=self.config.get('spatial_platform_diff', False),
            temporal_platform_diff=self.config.get('temporal_platform_diff', False)
        )

    def _create_wrapper(self) -> Wrapper:
        return Wrapper(config=self.config)

    def _initialize_controller(self) -> Controller:
        controller_type = self.config.get('controller_type', 'RandomController')

        if controller_type == 'RandomController':
            from algorithms.random_controller import RandomController
            controller = RandomController(self.config)
        elif controller_type == 'PRandomController':
            from algorithms.random_controller import RandomController
            controller = RandomController(self.config)
            controller.is_prune = True
        elif controller_type == 'polo':
            from algorithms.polo_controller import POLOController
            controller = POLOController(self.config)
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")

        if not self.is_learning and self.is_deep:
            assert self.model_load_path is not None, "model_load_path must be specified for inference mode"
            controller.load(self.model_load_path)

        return controller

    def _load_checkpoint(self):
        if not os.path.exists(self.model_load_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.model_load_path}")
        try:
            self.controller.load(self.model_load_path)
            print(f"✓ Checkpoint loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

    def run(self):
        num_episodes = self.controller.num_episodes
        all_episode_metrics = []

        if self.controller.is_learning:
            cumulative_time = 0.0
            assert self.model_save_path is not None, 'model_save_path must be set for training'
            save_dir = os.path.dirname(self.model_save_path)
            os.makedirs(save_dir, exist_ok=True)
            tracemalloc.start()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            latest_model_path = os.path.join(save_dir, 'latest_model.pt')
            self.best_model_path = os.path.join(save_dir, 'best_model.pt')
            best_global_revenue = float('-inf')

            for episode in range(num_episodes):
                orders, couriers = self.instance_generator.generate(seed=episode + 2000)
                self.wrapper.reset(orders=orders, couriers=couriers)

                t0 = time.perf_counter()
                episode_metrics = self.controller.train([self.wrapper])
                episode_time = time.perf_counter() - t0
                cumulative_time += episode_time
                all_episode_metrics.append(episode_metrics)

                # Memory snapshot
                cpu_current, cpu_peak = tracemalloc.get_traced_memory()
                mem_dict = {
                    'memory/cpu_current_mb': cpu_current / 1024 ** 2,
                    'memory/cpu_peak_mb':    cpu_peak    / 1024 ** 2,
                }
                if torch.cuda.is_available():
                    mem_dict['memory/gpu_allocated_mb'] = torch.cuda.memory_allocated()     / 1024 ** 2
                    mem_dict['memory/gpu_reserved_mb']  = torch.cuda.memory_reserved()      / 1024 ** 2
                    mem_dict['memory/gpu_peak_mb']      = torch.cuda.max_memory_allocated() / 1024 ** 2

                if self.wandb_run is not None:
                    self._log_episode(episode + 1, episode_metrics)
                    extra = {'time/episode_seconds': episode_time,
                             'time/cumulative_seconds': cumulative_time,
                             **mem_dict}
                    if hasattr(self.controller, 'epsilon'):
                        extra['train/epsilon'] = self.controller.epsilon
                    if hasattr(self.controller, 'replay_buffer') and hasattr(self.controller.replay_buffer, 'size'):
                        buf_cap = getattr(self.controller, 'buffer_size', None) or self.config.get('buffer_size', 1)
                        extra['train/buffer_fill'] = self.controller.replay_buffer.size() / buf_cap
                    wandb.log(extra, step=episode + 1)

                if (episode + 1) % 200 == 0:
                    try:
                        self.controller.save(latest_model_path)
                        print(f"[Episode {episode + 1}] Latest model saved to: {latest_model_path}")
                    except Exception as e:
                        print(f"[Episode {episode + 1}] Warning: failed to save latest model: {e}")

                global_revenue = None
                for metrics in episode_metrics:
                    rev = metrics.get('platform_revenue', {})
                    if 'global' in rev:
                        global_revenue = rev['global']
                        break
                if global_revenue is not None and global_revenue > best_global_revenue:
                    best_global_revenue = global_revenue
                    try:
                        self.controller.save(self.best_model_path)
                        print(f"[Episode {episode + 1}] Best model saved (global revenue={global_revenue:.2f})")
                    except Exception as e:
                        print(f"[Episode {episode + 1}] Warning: failed to save best model: {e}")

            _, cpu_final_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            if self.wandb_run is not None:
                self.wandb_run.summary['memory/cpu_peak_mb'] = cpu_final_peak / 1024 ** 2
                if torch.cuda.is_available():
                    self.wandb_run.summary['memory/gpu_peak_mb']      = torch.cuda.max_memory_allocated() / 1024 ** 2
                    self.wandb_run.summary['memory/gpu_reserved_mb']  = torch.cuda.memory_reserved()      / 1024 ** 2

            try:
                self.controller.save(latest_model_path)
                print(f"\nFinal model saved to: {latest_model_path}")
                print(f"Best model (revenue={best_global_revenue:.2f}) saved to: {self.best_model_path}")
            except Exception as e:
                print(f"Warning: failed to save final model: {e}")

            if self.wandb_run is not None:
                self.wandb_run.summary['time/train_total_seconds'] = cumulative_time
                self.wandb_run.summary['time/avg_episode_seconds'] = cumulative_time / num_episodes if num_episodes else 0

            print(f"\n{'='*60}")
            print(f"Training done. Running post-training evaluation...")
            print(f"{'='*60}")
            self.controller.is_learning = False
            test_metrics, test_time = self._run_test_phase(step_offset=num_episodes)
            self.controller.is_learning = True

            return {'episode_metrics': all_episode_metrics, 'test_metrics': test_metrics,
                    'train_time': cumulative_time, 'total_time': cumulative_time + test_time}
        else:
            test_metrics, test_time = self._run_test_phase()
            return {'instance_metrics': test_metrics, 'total_time': test_time}

    def _run_test_phase(self, step_offset: int = 0):
        num_test_instances = self.config.get('num_test_instances')
        test_metrics = []
        cumulative_time = 0.0
        # if self.best_model_path !=None:
        #     self.controller.load(self.best_model_path)

        for i in range(num_test_instances):
            print(f"\n=== Testing on Instance #{i + 1}/{num_test_instances} ===")
            orders, couriers = self.instance_generator.generate(seed=i + 3950)
            self.wrapper.reset(orders=orders, couriers=couriers)

            t0 = time.perf_counter()
            instance_metric = self.controller.test([self.wrapper])
            instance_time = time.perf_counter() - t0
            cumulative_time += instance_time

            step = step_offset + i + 1
            if self.wandb_run is not None:
                self._log_episode(step, instance_metric)
                wandb.log({'time/test_instance_seconds': instance_time,
                           'time/test_cumulative_seconds': cumulative_time}, step=step)

            test_metrics.append(instance_metric)

        if self.wandb_run is not None:
            self._log_summary(test_metrics, num_summary_episodes=num_test_instances)
            self.wandb_run.summary['time/test_total_seconds'] = cumulative_time
            self.wandb_run.summary['time/avg_test_instance_seconds'] = cumulative_time / num_test_instances if num_test_instances else 0

        return test_metrics, cumulative_time

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _build_log_dict(self, episode: int, episode_metrics: list) -> dict:
        """Build the per-episode metric dict shared by _log_episode and _log_summary."""
        log_dict = {'Step/episode': episode}
        for metrics in episode_metrics:
            log_dict['Step/num_steps'] = metrics['wrapper_num_steps']
            log_dict['courier/(1)avg_courier_distance'] = metrics['average_courier_distance_travelled']
            log_dict['courier/(2)avg_courier_income']   = metrics['average_courier_income']
            income_stats = metrics.get('courier_income_stats', {})
            log_dict['courier/(3)income_std']  = income_stats.get('std', 0.0)
            log_dict['courier/(4)income_gini'] = income_stats.get('gini', 0.0)
            log_dict['service/(1)avg_response_time_s'] = metrics.get('average_response_time', 0.0)
            log_dict['service/(2)avg_delivery_time_s'] = metrics.get('average_delivery_time', 0.0)
            refusal = metrics.get('courier_refusal_stats', {})
            log_dict['service/(3)courier_refusal_rate']  = refusal.get('rate', 0.0)
            log_dict['service/(4)courier_refused_count'] = refusal.get('refused', 0)
            if 'loss' in metrics and metrics['loss'] is not None:
                log_dict['actor_loss']   = metrics['loss']['actor_loss']
                log_dict['critic_loss']  = metrics['loss']['critic_loss']
                log_dict['entropy_loss'] = metrics['loss']['entropy_loss']
            for platform_id in metrics['platform_order_response_rate'].keys():
                pfx = f'platform_{platform_id}'
                log_dict[f'{pfx}/(1)order_numbers']    = metrics['platform_order_numbers'][platform_id]
                log_dict[f'{pfx}/(2)courier_numbers']  = metrics['platform_courier_numbers'][platform_id]
                log_dict[f'{pfx}/(3)potential_revenue']= metrics['platform_potential_revenue'][platform_id]
                log_dict[f'{pfx}/(4)revenue']          = metrics['platform_revenue'][platform_id]
                potential = metrics['platform_potential_revenue'][platform_id]
                log_dict[f'{pfx}/(5)revenue_rate']  = metrics['platform_revenue'][platform_id] / potential if potential > 0 else 0.0
                log_dict[f'{pfx}/(6)response_rate'] = metrics['platform_order_response_rate'][platform_id]
                log_dict[f'{pfx}/(7)overdue_rate']  = metrics['platform_order_overdue_rate'][platform_id]
        return log_dict

    def _log_episode(self, episode: int, episode_metrics: list):
        wandb.log(self._build_log_dict(episode, episode_metrics), step=episode)

    def _log_summary(self, all_episode_metrics: List[List[Dict]], num_summary_episodes: int = 50):
        if self.wandb_run is None:
            return
        summary_metrics = all_episode_metrics[-num_summary_episodes:]
        if not summary_metrics:
            return
        metric_values: Dict[str, list] = {}
        for episode_metrics in summary_metrics:
            for metrics in episode_metrics:
                d = self._build_log_dict(0, [metrics])
                for k, v in d.items():
                    if k == 'Step/episode':
                        continue
                    self._collect_metric(metric_values, k, v)
        for metric_name, values in metric_values.items():
            if values:
                self.wandb_run.summary[f'summary/{metric_name}_mean'] = float(np.mean(values))
                self.wandb_run.summary[f'summary/{metric_name}_std']  = float(np.std(values))

    def _collect_metric(self, metric_values: Dict[str, List], key: str, value):
        if value is not None:
            metric_values.setdefault(key, []).append(value)


def make_sweep_fn(config_path: str):
    """Return a wandb sweep callback for the given base config path.

    Usage:
        wandb.agent(sweep_id='...', function=make_sweep_fn('yamls/base_xxx.yaml'), project='...')
    """
    def _sweep():
        with open(config_path) as f:
            base_cfg = yaml.safe_load(f)
        with wandb.init(config=base_cfg) as run:
            config = dict(run.config)
            scale = config.get('instance_size')
            n_platforms = config.get('number_of_platforms')
            controller_type = config.get('controller_type')
            run.name = f"{controller_type}_{scale}_{n_platforms}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run.tags = [controller_type, scale, f"{n_platforms}platforms"]
            runner = Runner(config)
            runner.wandb_run = run
            runner.model_save_path = f'./saved/{scale}/P={n_platforms}/{run.name}/{run.name}.pt'
            runner.run()
    return _sweep


def main_base_wandb(config_path, use_wandb: bool = True, project_name: str = 'minicourier',
                    overrides: dict = None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if overrides:
        config.update(overrides)

    scale = config.get('instance_size')
    number_of_platforms = config.get('number_of_platforms')
    controller_type = config.get('controller_type')
    run_name = f"{controller_type}_{scale}_{number_of_platforms}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path = f'./{scale}/P={number_of_platforms}/{run_name}'

    if use_wandb:
        run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            tags=[controller_type, scale, f"{number_of_platforms}platforms"],
        )
        print(f"\n{'='*60}")
        print(f"Wandb Initialized")
        print(f"{'='*60}")
        print(f"Project: {project_name}")
        print(f"Run URL: {run.get_url()}")
        print(f"{'='*60}\n")
    else:
        run = None
        print(f"\n{'='*60}")
        print(f"Running WITHOUT Wandb")
        print(f"{'='*60}\n")

    print(f"{'='*60}")
    print(f"Config")
    print(f"{'='*60}")
    for k, v in sorted(config.items()):
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    runner = Runner(config)
    runner.wandb_run = run
    runner.model_save_path = f'./saved/{save_path}/{run_name}.pt'

    results = runner.run()

    if use_wandb:
        wandb.finish()
        print(f"\n{'='*60}")
        print(f"Wandb run finished: {run.get_url()}")
        print(f"{'='*60}\n")

    return results
