"""
Base Controller Class for MiniCourier

This module provides a base controller class that can be extended for both:
1. Heuristic controllers (greedy, rule-based) - no learning
2. Deep RL controllers (DQN, MADDPG, etc.) - with learning and replay buffer
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import yaml
import json
import os
from pathlib import Path
from datetime import datetime
from env.matching_wrapper import Wrapper


class Controller(ABC):
    """
    Base class for all controllers (both heuristic and RL-based).

    Controllers take environment state and generate actions for couriers.

    Two types of controllers:
    1. Heuristic: Rule-based, no learning (e.g., GreedyController)
    2. RL-based: Learning from experience, using replay buffer (e.g., MADDPGController)
    """

    def __init__(self, config: dict):
        """
        Initialize controller from config file.

        Args:
            config_path: Path to controller configuration YAML file
        """
        self.config = config

        # Extract common config parameters
        self.controller_type = self.config.get('controller_type', 'BaseController')
        self.is_learning = self.config.get('is_learning')
        self.is_deep = self.config.get('is_deep')
        self.decision_times_per_step = self.config.get('decision_times_per_step')


        # Learning-related parameters (only used if is_learning=True)
        self.num_episodes = self.config.get('num_episodes')
        self.batch_size = self.config.get('batch_size')
        self.buffer_size = self.config.get('buffer_size')

        # Initialize replay buffer for RL controllers
        self.replay_buffer = None
        if self.is_learning:
            self.replay_buffer = []  # Simple list-based buffer (can be improved)

        # Cache for decision information to avoid recomputation in store_experience
        self._decision_cache = None

        # Episode tracking
        self.current_episode = 0

        # Step logging configuration
        # self.save_step_json = self.config.get('save_step_json', False)
        # self.step_json_dir = self.config.get('step_json_dir', 'step_logs')
        # self.step_json_interval = self.config.get('step_json_interval', 1)  # Save every N steps

        # Create step log directory if saving is enabled
        # if self.save_step_json:
        #     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        #     self.current_log_dir = Path(self.step_json_dir) / f"{self.controller_type}_{timestamp}"
        #     self.current_log_dir.mkdir(parents=True, exist_ok=True)
        #     print(f"  - Step logging enabled: {self.current_log_dir}")

        print(f"Initialized {self.controller_type}")
        print(f"  - Learning: {self.is_learning}")
        if self.is_learning:
            print(f"  - Episodes: {self.num_episodes}")
            print(f"  - Batch size: {self.batch_size}")
            print(f"  - Buffer size: {self.buffer_size}")

    @abstractmethod
    def make_decision(self, wrapper_state: Dict, dispatch_round: int) -> Dict[str, List[Tuple]]:
        """
        Generate actions for all platforms based on current state.

        This is the core decision-making method that must be implemented by all controllers.

        Args:
            wrapper_state: Dict mapping platform_id to platform state
                Format: {platform_id: state_dict}

        Returns:
            Dict mapping platform_id to list of actions
            Format: {platform_id: [(courier_id, order_id), ...]}
        """
        pass

    def run_episode(self, wrapper: Wrapper, is_training: bool = True) -> Dict[str, Any]:
        """
        Run a single episode on one wrapper.

        Args:
            wrapper: Environment wrapper instance
            is_training: Whether this is training (True) or testing (False)

        Returns:
            Dict containing episode metrics:
            {
                'total_revenue': dict,  # {platform_id: revenue}
                'order_response_rate': dict,  # {platform_id: rate}
                'order_overdue_rate': dict,  # {platform_id: rate}
                'num_steps': int
            }
        """
        # Reset controller state (for stateful controllers like GateController)
        self.order_info = {}
        for order in wrapper.simulator.order_pool:
            self.order_info[order.order_id] = {
                'create_time': order.create_time,
                'deadline': order.delivery_patience,
                'value': order.value
            }
        if hasattr(self, 'reset') and callable(self.reset):
            self.reset()

        # Reset environment
        wrapper_state, wrapper_done = wrapper.reset()
        num_steps = 0

        # Run episode until done
        while not wrapper_done:

            # Dispatch rounds (micro steps without advancing real time)
            # print('  - Episode {}, Step {}'.format(self.current_episode, num_steps))
            # Transfer step into real time seconds = 120 seconds / 1 step 
            # clock_hour = wrapper.simulator.time // 3600
            # clock_minute = (wrapper.simulator.time % 3600) // 60
            # clock_second = wrapper.simulator.time % 60
            # print('    - Current time: {:02d}:{:02d}:{:02d}'.format(clock_hour, clock_minute, clock_second))

            # 
            advance_real_time = False
            for dispatch_round in range(self.decision_times_per_step):
                # if all(wrapper_done.values()):
                #     break
                # If no unassigned orders, break early
                # print('    - Dispatch round {}'.format(dispatch_round))
                if all(len(state.get_unassigned_orders()) == 0 for state in wrapper_state.values()):
                    break
                # Save state before action
                s = wrapper_state
                
                a = self.make_decision(s, dispatch_round)
                # Micro step: don't advance real time
                is_last_dispatch = (dispatch_round == self.decision_times_per_step - 1)
                s_next, done_next = wrapper.step(realgo=is_last_dispatch, wrapper_actions=a)
                # Store experience (use s, not wrapper_state after update)
                if self.is_learning and is_training:
                    self.store_experience(
                        wrapper_state=s,
                        wrapper_action=a,
                        next_wrapper_state=s_next,
                        next_wrapper_done=done_next
                    )
                # Update state for next dispatch round
                wrapper_state, wrapper_done = s_next, done_next
                if is_last_dispatch:
                    advance_real_time = True
            # Advance real time once (with empty action)
            if (not advance_real_time) and (not wrapper_done):
                # s = wrapper_state
                empty_action = {platform_id: [] for platform_id in wrapper_state.keys()}
                s_next, done_next = wrapper.step(realgo=True, wrapper_actions=empty_action)
                wrapper_state, wrapper_done = s_next, done_next
                advance_real_time = True
                
            if advance_real_time:
                num_steps += 1  # Only increment on real time steps

            # Update parameters if RL controller
        if (self.current_episode+50) % 50 == 0:
            print('one episode ends at step {}'.format(num_steps))
        if self.is_learning and is_training:
            # print('controller updating parameters at step {}'.format(num_steps))
            loss = self.update_parameters()

        # Increment episode counter
        self.current_episode += 1

        # Collect metrics
        metrics = {
            'platform_order_numbers': wrapper.get_total_orders_appeared(),
            'platform_courier_numbers': wrapper.get_total_couriers(),
            'platform_potential_revenue': wrapper.get_potential_platform_revenue(),
            'platform_revenue': wrapper.get_platform_revenue(),
            'platform_order_response_rate': wrapper.get_order_response_rate(),
            'platform_order_overdue_rate': wrapper.get_order_overdue_rate(),
            'average_courier_distance_travelled': wrapper.get_average_courier_distance_travelled(),
            'average_courier_income': wrapper.get_average_courier_income(),
            'courier_income_stats': wrapper.get_courier_income_stats(),
            'average_response_time': wrapper.get_average_response_time(),
            'average_delivery_time': wrapper.get_average_delivery_time(),
            'courier_refusal_stats': wrapper.get_courier_refusal_stats(),
            'wrapper_num_steps': num_steps,
        }
        if self.is_learning and is_training:
            metrics['loss'] = loss
        
        # adjust epsilon if applicable
        if self.is_learning and hasattr(self, 'epsilon'):
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return metrics

    def train(self, train_wrappers: List[Wrapper]) -> List[Dict[str, Any]]:
        """
        Train controller on multiple wrappers for one episode each.

        For heuristic controllers: Just runs episodes without learning
        For RL controllers: Runs episodes and updates parameters

        Args:
            train_wrappers: List of training environment wrappers

        Returns:
            List of episode metrics for each wrapper
            Format: [wrapper_0_metrics, wrapper_1_metrics, ...]
        """
        all_wrappers_metrics = []

        for wrapper_idx, wrapper in enumerate(train_wrappers):

            episode_metrics = self.run_episode(wrapper, is_training=True)
            all_wrappers_metrics.append(episode_metrics)

        return all_wrappers_metrics

    def test(self, test_wrappers: List) -> List[Dict[str, Any]]:
        """
        Test controller on multiple wrappers (no learning).

        Args:
            test_wrappers: List of test environment wrappers

        Returns:
            List of episode metrics for each wrapper
        """
        all_wrappers_metrics = []

        for wrapper_idx, wrapper in enumerate(test_wrappers):
            print(f"\n  Testing on wrapper {wrapper_idx + 1}/{len(test_wrappers)}")

            episode_metrics = self.run_episode(wrapper, is_training=False)
            all_wrappers_metrics.append(episode_metrics)

        return all_wrappers_metrics

    def store_experience(self, wrapper_state, wrapper_action, wrapper_reward, next_wrapper_state, done):
        """
        Store experience in replay buffer (only used by RL controllers).

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Done flag
        """
        if not self.is_learning:
            return  # Heuristic controllers don't store experience

        experience = {
            'state': wrapper_state,
            'action': wrapper_action,
            'reward': wrapper_reward,
            'next_state': next_wrapper_state,
            'done': done
        }

        self.replay_buffer.append(experience)

        # Keep buffer size limited
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

    def update_parameters(self):
        """
        Update controller parameters from replay buffer (only for RL controllers).

        This method should be overridden by RL-based controllers.
        Heuristic controllers can leave this as no-op.
        """
        if not self.is_learning:
            return  # Heuristic controllers don't learn

        # Default: no-op (to be overridden by RL controllers)
        pass

    def save(self, path: str):
        """
        Save controller state (for RL controllers with learnable parameters).

        Args:
            path: Path to save controller
        """
        if not self.is_learning:
            print(f"Heuristic controller {self.controller_type} has no parameters to save")
            return

        # To be implemented by RL controllers
        # Example implementation should save:
        # - Model parameters (network weights, etc.)
        # - Optimizer state
        # - Training statistics
        # - Config used for training
        print(f"Saving controller to {path}")
        raise NotImplementedError(f"{self.controller_type} must implement save() method")

    def load(self, path: str):
        """
        Load controller state (for RL controllers with learnable parameters).

        Args:
            path: Path to load controller from
        """
        # Note: This can be called even when is_learning=False
        # because we want to load trained params for inference

        # To be implemented by RL controllers
        # Example implementation should load:
        # - Model parameters (network weights, etc.)
        # - Optionally: optimizer state (if continuing training)
        # - Training statistics for logging
        print(f"Loading controller from {path}")
        raise NotImplementedError(f"{self.controller_type} must implement load() method")
    
    def _save_step_json(self, wrapper_state: Dict, wrapper_action: Dict, step_num: int):
        """
        Save the current step's state and action into a JSON file for analysis.

        Args:
            wrapper_state: Current state of the wrapper (Dict[platform_id -> SimulatorState])
            wrapper_action: Action taken at the current step (Dict[platform_id -> List[Tuple]])
            step_num: Current step number
        """
        # Skip if saving is disabled
        if not self.save_step_json:
            return

        # Skip if not at save interval
        if step_num % self.step_json_interval != 0:
            return

        try:
            # Prepare data structure using SimulatorState.to_json()
            step_data = {
                'state': {},
                'action': {}
            }
            # Serialize wrapper_state using SimulatorState.to_json()
            for platform_id, simulator_state in wrapper_state.items():
                # Use built-in to_json() method
                step_data['state'][platform_id] = json.loads(simulator_state.to_json())

            # Serialize wrapper_action
            for platform_id, actions in wrapper_action.items():
                step_data['action'][platform_id] = [
                    {
                        'courier_id': action[0] if len(action) > 0 else None,
                        'order_id': action[1] if len(action) > 1 else None
                    }
                    for action in actions
                ]
            # Save to file
            episode_num = self.current_episode
            filename = f"episode_{episode_num:04d}_step_{step_num:05d}.json"
            filepath = self.current_log_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(step_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Warning: Failed to save step JSON at step {step_num}: {e}")
        