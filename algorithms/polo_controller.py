import numpy as np
import torch
from typing import Dict, List, Tuple
from algorithms.controller import Controller
from algorithms.POLO.AC import TowerActorCritic, ReplayBuffer
from env.models.state_action import SimulatorState
from algorithms.utils.agent_registry import GridManager
from algorithms.utils.feature_extractor import FeatureExtractor, feature_builder
from algorithms.POLO.reward_simulator import RewardSimulator
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

class POLOController(Controller):
    """Multi-Agent Tower Controller.

    Each grid cell acts as an independent agent, making decisions for its first order.
    All grids make decisions simultaneously in each dispatch round.
    """

    def __init__(self, config: dict):

        super().__init__(config)

        # Hyperparameters
        self.actor_lr = config.get('actor_lr', 0.0001)
        self.critic_lr = config.get('critic_lr', 0.0001)
        self.gamma = config.get('gamma', 0.9)
        self.epsilon = config.get('epsilon', 0.9)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.01)

        # Grid manager parameters
        self.city_scale = config.get('instance_size', 'small')
        self.hex_size = config.get('hex_size', 0.5)
        self.max_agents = config.get('max_agents', 500)
        self.number_of_platforms = config.get('number_of_platforms')
        self.platforms = [f'Platform{chr(65+i)}' for i in range(self.number_of_platforms)]
        self.pruning_k = config.get('pruning_k', 20)

        # Attention mechanism configuration
        self.use_attention = config.get('use_attention', True)
        self.attention_heads = config.get('attention_heads', 4)
        self.use_route = config.get('use_route', False)
        self.reshape_reward = config.get('reshape_reward', True)
        self.use_hold_action = config.get('use_hold_action', False)
        self.global_reward = config.get('global_reward', False)

        # Dimensions
        self.courier_feature_dim = config.get('courier_feature_dim', 5)
        self.order_feature_dim = config.get('order_feature_dim', 6)
        self.pair_feature_dim = config.get('pair_feature_dim', 2)

        self.action_dim = self.order_feature_dim + self.pair_feature_dim

        self.batch_size = config.get('batch_size', 512)  
        self.buffer_size = config.get('buffer_size', 100000)
        self.use_direction = config.get('use_direction', False)

        # Trajectory / route sequence lengths
        self.back_length    = config.get('back_length', 10)   # past trajectory steps
        self.forward_length = config.get('forward_length', 10) # future route steps

        self.update_steps = config.get('update_steps', 10)

        # Route random masking (data augmentation / robustness test)
        self.route_random_mask = config.get('route_random_mask', False)
        self.route_mask_ratio  = config.get('route_mask_ratio', 0.3)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.counterfactual_ratio = config.get('counterfactual_ratio', 0.1)

        self.distance_penalty_ratio = config.get('distance_penalty_ratio', 0.1)
        self.order_info = None

        # Initialize GridManager
        self.grid_manager = GridManager(
            city_scale=self.city_scale,
            hex_size=self.hex_size,
            max_agents=self.max_agents,
            use_direction=self.use_direction,
            grid_type='hex',
            platforms=self.platforms
        )

        # Initialize FeatureExtractor
        self.feature_extractor = FeatureExtractor(
            order_dim=self.order_feature_dim,
            courier_dim=self.courier_feature_dim,
            pruning_k=self.pruning_k,
            device=self.device,
            forward_length=self.forward_length,
            back_length=self.back_length,
        )

        self.supply_demand_dim = self.grid_manager.get_grid_dim()
        self.courier_state_dim = self.courier_feature_dim + self.supply_demand_dim 
        self.state_dim = self.courier_state_dim + self.action_dim

        # Initialize Actor-Critic agent (shared across all grid agents)
        self.agent = TowerActorCritic(
            stateDim=self.courier_state_dim,
            actionDim=self.action_dim,
            actorLr=self.actor_lr,
            criticLr=self.critic_lr,
            gamma=self.gamma,
            epsilon=self.epsilon,
            batchSize=self.batch_size,
            device=self.device,
            use_attention=self.use_attention,
            attention_heads=self.attention_heads,
            use_route=self.use_route,
            use_hold_action=self.use_hold_action
        )

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.buffer_size,
            batchSize=self.batch_size
        )

        self.reward_simulator = RewardSimulator()
        self.experience_cache = {}

        print(f"MACTowerController initialized:")
        print(f"  - Device: {self.device}")
        print(f"  - State dim: {self.courier_state_dim}, Action dim: {self.action_dim}")
        print(f"  - City scale: {self.city_scale}, Grid dim: {self.supply_demand_dim}")
        print(f"  - Batch size: {self.batch_size}, Buffer size: {self.buffer_size}")
        print(f"  - Use route GRU: {self.use_route}  (back_length={self.back_length}, forward_length={self.forward_length})")
        print(f"  - Attention enable: {self.use_attention}")
        print(f"  - Reshape reward: {self.reshape_reward}")
        print(f"  - diection enable: {self.use_direction}")
        print(f" - Hold action enable: {self.use_hold_action}")
        print(f" - Global reward enable: {self.global_reward}")
        print(f" - Route random mask: {self.route_random_mask} (ratio={self.route_mask_ratio})")

    def _apply_route_mask(self, routes: torch.Tensor) -> torch.Tensor:
        """Randomly zero out a fraction of waypoints in each courier's route.

        Args:
            routes: (num_couriers, seq_len, 2)
        Returns:
            routes with route_mask_ratio fraction of waypoints zeroed, same shape.
        """
        if not self.route_random_mask or not self.use_route:
            return routes
        routes = routes.clone()
        num_couriers, seq_len, _ = routes.shape
        num_masked = max(1, int(seq_len * self.route_mask_ratio))
        for i in range(num_couriers):
            mask_idx = torch.randperm(seq_len)[:num_masked]
            routes[i, mask_idx, :] = 0.0
        return routes

    def make_decision(self, wrapper_state: Dict[str, SimulatorState], dispatch_round: int) -> Dict[str, List[Tuple[str, str]]]:
        """Make decisions for all grids simultaneously."""

        self.experience_cache = {}
        platform_actions = {}
        courier_assigned_count = {}

        # Get features for all grids
        grid_states, grid_dispatch_map, grid_routes  = feature_builder(
            wrapper_state, self.feature_extractor, self.grid_manager,
            self.state_dim, self.order_feature_dim, self.controller_type, use_grid_decision=True
        )

        if len(grid_states) == 0:
            return platform_actions

        for grid_key, state in grid_states.items():
            platform_id, cell, direction = grid_key

            if state.shape[0] == 1 and torch.sum(state) == 0:
                continue

            order_id, courier_list, pair_features = grid_dispatch_map[grid_key]
            routes = self._apply_route_mask(grid_routes[grid_key])
            assert len(courier_list) == state.shape[0], "State and courier list size mismatch"
            assert len(courier_list) == routes.shape[0], "Routes and courier list size mismatch"

            # Take action
            action = self.agent.take_action(state, routes)

            # Get selected courier
            if action >= len(courier_list):
                assert action == len(courier_list), "Invalid action index, should be hold action"
                # Hold action, no assignment
                if self.is_learning:
                    # Cache experience
                    self.experience_cache[grid_key] = {
                        'state': state,
                        'action': action,
                        'routes': routes,
                        'pair_features': pair_features,
                        'order_id': order_id,
                        'courier_id': None,
                        'platform_id': platform_id
                    }
                continue
            index_in_all = courier_list[action]
            courier_id = wrapper_state[platform_id].courier_pool[int(index_in_all)].courier_id
            if courier_id not in courier_assigned_count:
                courier_assigned_count[courier_id] = 0
            courier_assigned_count[courier_id] += 1
            # Store action
            if platform_id not in platform_actions:
                platform_actions[platform_id] = []
            platform_actions[platform_id].append((courier_id, order_id))

            # Cache experience
            if self.is_learning:
                self.experience_cache[grid_key] = {
                    'state': state,
                    'action': action,
                    'routes': routes,
                    'pair_features': pair_features,
                    'order_id': order_id,
                    'courier_id': courier_id,
                    'platform_id': platform_id
                }

        return platform_actions

    def store_experience(self, wrapper_state, wrapper_action, next_wrapper_state, next_wrapper_done):
        """Store experiences for all grid agents."""

        # Get next states for all grids
        next_grid_states, _, next_grid_routes = feature_builder(
            next_wrapper_state, self.feature_extractor, self.grid_manager,
            self.state_dim, self.order_feature_dim, self.controller_type, use_grid_decision=True
        )
        # self.order_info = self.reward_simulator.build_order_info(next_wrapper_state['global'])

        for grid_key, exp in self.experience_cache.items():
            platform_id = exp['platform_id']
            courier_id = exp['courier_id']
            state = exp['state']
            action = exp['action']
            routes = exp['routes']
            order_id = exp['order_id']
            pair_features = exp['pair_features']

            # Calculate rewar
            if self.global_reward:
                old_courier = wrapper_state['global'].get_courier_by_id(courier_id)
                new_courier = next_wrapper_state['global'].get_courier_by_id(courier_id)
            else:
                old_courier = wrapper_state[platform_id].get_courier_by_id(courier_id)
                new_courier = next_wrapper_state[platform_id].get_courier_by_id(courier_id)

             # Get next state for this grid
            # assert grid_key in next_grid_states, "Next state for grid not found"

            if grid_key in next_grid_states:
                next_state = next_grid_states[grid_key]
                next_routes = next_grid_routes.get(grid_key)
            else:
                # Find any state from same platform, or use zeros
                next_state = None
                next_routes = None
                for k, v in next_grid_states.items():
                    if k[0] == platform_id:
                        next_state = v
                        next_routes = next_grid_routes.get(k)
                        break
                if next_state is None:
                    next_courier_number = state.shape[0]
                    next_state = torch.zeros((next_courier_number, self.state_dim))
                    next_routes = torch.zeros((next_courier_number, self.back_length + self.forward_length, 2)) if self.use_route else None

            if next_routes is not None:
                next_routes = self._apply_route_mask(next_routes)

            if old_courier is None or new_courier is None:

                reward = 0.0
            else:
                old_reward = self.reward_simulator.calculate_route_reward(
                    old_courier.route, old_courier, self.order_info, wrapper_state[platform_id].time
                )

                no_this_order_route = self.reward_simulator._remove_order_from_route(new_courier.route, order_id)
                new_no_this_order_reward = self.reward_simulator.calculate_route_reward(no_this_order_route, new_courier, self.order_info, next_wrapper_state[platform_id].time)

                real_new_reward = self.reward_simulator.calculate_route_reward(
                    new_courier.route, new_courier, self.order_info, next_wrapper_state[platform_id].time
                )
                
                real_reward = real_new_reward - old_reward
                if self.reshape_reward is True:
                    no_this_order_reward = new_no_this_order_reward - old_reward

                    add_distance = pair_features[action][1].item()
                    # 1. Extract the order value and match the reward scale.
                    # calculate_route_reward is scaled by 10, so align here too.
                    target_val = self.order_info[order_id]['value'] / 10.0

                    # 2. Compute the actual marginal contribution.
                    # This is the incremental score gained by adding the order.
                    # With perfect delivery and no interference, it should match target_val.
                    actual_contribution = real_reward - no_this_order_reward

                    # 3. Measure the disturbance to the existing route.
                    # diff < 0 means the added order hurt overall performance.
                    diff = actual_contribution - target_val

                    # 4. Build the final reward.
                    # Only penalize negative side effects; positive spillover is left untouched.
                    penalty = self.counterfactual_ratio * (diff ** 2) if diff < 0 else 0

                    distance_penalty = add_distance * self.distance_penalty_ratio  # Small penalty for longer distances

                    reward = real_reward - penalty - distance_penalty
                else:
                    reward = real_reward

           

            assert state.shape[0] == next_state.shape[0], "State and next state dimension mismatch"
            # assert state.shape[0] == self.pruning_k, "State dimension mismatch"
            # assert next_state.shape[0]  == self.pruning_k, "Next state dimension mismatch"
            if self.use_route:
                assert self.back_length + self.forward_length == next_routes.shape[1], "Routes and next routes dimension mismatch"

            if self.use_route:
                self.replay_buffer.add(state, action, reward, next_state, routes, next_routes)
            else:
                self.replay_buffer.add(state, action, reward, next_state)

    def update_parameters(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        # Clear CUDA cache before update to prevent memory fragmentation
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        for _ in range(self.update_steps):
            batch = self.replay_buffer.sample()
            self.agent.update(*batch)

    def save(self, path):
        torch.save(self.agent.actor.state_dict(), path)

    def load(self, path):
        self.agent.actor.load_state_dict(torch.load(path, map_location=self.device))
