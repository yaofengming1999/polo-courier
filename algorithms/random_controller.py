"""
Random Controller for MiniCourier

A random heuristic controller that randomly assigns orders to available couriers.
Uses a fixed random seed for reproducibility.
No learning involved - pure random decision making.
"""

import random
from algorithms.controller import Controller
from algorithms.utils.grid_utils import prune_couriers_per_platform
from typing import Dict, List, Tuple
import numpy as np
from src.env.simulator.models.state_action import SimulatorState


class RandomController(Controller):
    """
    Random heuristic controller.

    Strategy:
    1. For each platform, iterate through unassigned orders
    2. For each order, find the available courier (optionally pruned to k nearest)
    3. Assign the order to a randomly selected available courier with a fixed incentive
    """

    def __init__(self, config: dict):
        """
        Initialize random controller.

        Args:
            config: Controller configuration dict
        """
        super().__init__(config)

        # Random-specific parameters
        self.random_seed = self.config.get('random_seed')  # Default seed

        # Pruning parameters
        # self.is_prune = self.config.get('is_prune')  # Whether to prune couriers
        self.pruning_k = self.config.get('pruning_k')    # Number of nearest couriers per order
        self.rng = np.random.default_rng()
        self.is_prune = None


        # Set random seed for reproducibility
        random.seed(self.random_seed)

        # Ensure is_learning is False for RandomController
        self.is_learning = False


    def make_decision(self, wrapper_state: Dict[str, SimulatorState], dispatch_round: int) -> Dict[str, List[Tuple[str, str]]]:
        """
        Generate random actions for all platforms.

        For each platform:
        1. Get unassigned orders from SimulatorState
        2. Filter couriers to only include available ones (not signed out)
        3. For each order select an available courier (from pruned candidates if is_prune=True)

        Args:
            wrapper_state: Dict[platform_id -> SimulatorState]

        Returns:
            Platform actions dict: {platform_id: [(courier_id, order_id), ...]}
        """
        assert dispatch_round == 0, "RandomController only supports single-round dispatching."
        platform_actions = {}

        # wrapper_state = {k: v for k, v in wrapper_state.items() if k != 'global'}
        for platform_id, state in wrapper_state.items():
            if platform_id == 'global':
                continue
            orders = state.get_unassigned_orders()
            couriers = state.courier_pool
            # randomly forgive some orders
            # orders = [o for o in orders if rng.random() < 0.9]

            if not orders or not couriers:
                platform_actions[platform_id] = []
                continue

            if self.is_prune:
                # Use pruned candidates: for each order, randomly select from k nearest couriers
                order_courier_map = prune_couriers_per_platform(orders, couriers, self.pruning_k)
                actions = []
                for order in orders:
                    candidate_indices = order_courier_map.get(order.order_id, [])
                    if candidate_indices:
                        selected_idx = self.rng.choice(candidate_indices)
                        actions.append((couriers[selected_idx].courier_id, order.order_id))
                platform_actions[platform_id] = actions
            else:
                # Original behavior: randomly select from all couriers
                courier_ids = [c.courier_id for c in couriers]
                n = len(courier_ids)
                idx = self.rng.integers(0, n, size=len(orders))
                actions = [(courier_ids[i], o.order_id) for i, o in zip(idx, orders)]

                platform_actions[platform_id] = actions

        return platform_actions
