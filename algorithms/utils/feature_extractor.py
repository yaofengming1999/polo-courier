"""Feature Extractor - Extract features for orders, couriers, routes.

Uses shared grid utilities from algorithms.grid and local utils for Tower-specific features.
"""
import math
import torch
import numpy as np

from algorithms.utils.grid_utils import prune_couriers_per_platform
from typing import Dict, Tuple, List
from env.models.state_action import SimulatorState
from env.route_decision_engine import greedy_route_insertion_by_order

class FeatureExtractor:
    """Extract features for dispatch decisions."""
    CITY_BOUNDS = {
        'small': (0.0, 3.29, 0.0, 3.74),
        'middle': (0.0, 4.86, 0.0, 4.49),
        'large': (0.0, 13.47, 0.0, 13.02),
    }

    def __init__(self, order_dim, courier_dim, pruning_k, device, scale='small', forward_length=10, back_length=10):
        """Initialize feature extractor.

        Args:
            order_dim: Order feature dimension
            courier_dim: Courier feature dimension
            pruning_k: Number of top couriers to keep per order
            device: PyTorch device
            scale: City scale key ('small', 'middle', 'large') for coordinate normalization
        """
        self.order_dim = order_dim
        self.courier_dim = courier_dim
        self.pruning_k = pruning_k
        self.device = device
        self.scale = scale
        self.forward_length = forward_length
        self.back_length = back_length

        # Get city bounds for coordinate scaling
        if scale not in self.CITY_BOUNDS:
            raise ValueError(f"Unknown scale: {scale}. Must be one of {list(self.CITY_BOUNDS.keys())}")
        self.min_lng, self.max_lng, self.min_lat, self.max_lat = self.CITY_BOUNDS[scale]
        self.lng_range = self.max_lng - self.min_lng
        self.lat_range = self.max_lat - self.min_lat
        self._courier_state_cache = {}
        # Cache insertion costs: (order_id, courier_id) → (courier_version, distance_to_pickup, add_distance)
        # courier.version bumps on both movement and route changes, so this is always exact.
        # Key insight: between realgo=False dispatch rounds, unassigned couriers don't move and
        # their version is unchanged → ~90%+ cache hit rate across a time step's dispatch rounds.
        self._pair_insertion_cache: dict = {}

    def _scale_coordinates(self, lng: float, lat: float) -> Tuple[float, float]:
        """Scale coordinates to [0, 1] range based on city bounds.

        Args:
            lng: Longitude value
            lat: Latitude value

        Returns:
            Tuple of (scaled_lng, scaled_lat)
        """
        scaled_lng = (lng - self.min_lng) / self.lng_range if self.lng_range > 0 else 0.0
        scaled_lat = (lat - self.min_lat) / self.lat_range if self.lat_range > 0 else 0.0
        return scaled_lng, scaled_lat

    def extract(self, wrapper_state, controller_type):

        platform_states = {k: v for k, v in wrapper_state.items() if k != 'global'}

        order_state_dict, order_lookup_dict = self._extract_order_features(platform_states, controller_type) # {platform_id: order features tensor}
        order_courier_map = self._prune_couriers_per_order(platform_states, self.pruning_k,controller_type)

        courier_feature_dict, courier_trajectory_dict, courier_route_dict, courier_id_lookup = self._extract_courier_features(platform_states,order_courier_map)
        order_courier_pair_features_dict= self._extract_order_courier_pair_features(platform_states, order_courier_map,controller_type)
        
        order_part1 = order_state_dict
        order_part2 = order_lookup_dict
        courier_part1 = {platform_id: courier_feature_dict[platform_id][...,:self.courier_dim] for platform_id in courier_feature_dict.keys()}
        courier_part2 = courier_trajectory_dict
        courier_part3 = courier_route_dict
        courier_part4 = courier_id_lookup
        pair_part1 = order_courier_map
        pair_part2 = order_courier_pair_features_dict
        # pair_part3 = courier_order_map
        # pair_part4 = courier_order_pair_features_dict

        return order_part1, order_part2, courier_part1, courier_part2, courier_part3, courier_part4, pair_part1, pair_part2 #, pair_part3, pair_part4
    
    def _extract_order_courier_pair_features(self, platform_state, order_courier_map,controller_type) -> Dict[str, Dict[str, torch.Tensor]]:
        all_orders_courier_pair_feature_dict = {}
        order_courier_feature_dict = {}

        # all_couriers_order_pair_feature_dict = {}
        # courier_order_feature_dict = {}

        for platform_id in platform_state.keys():
            orders = platform_state[platform_id].get_unassigned_orders()
            couriers = platform_state[platform_id].courier_pool
            if len(orders) == 0:
                continue
            if controller_type=='RouteACBGMController' or controller_type=='FairRouteACBGMController':
                orders = [orders[0]]
            order_map = order_courier_map[platform_id]
            for order in orders:
                order_id = order.order_id
                if order_id in order_map:
                    selected_indices = order_map[order_id]
                    K = len(selected_indices)
                    # Pre-allocate output array — avoids list.append + torch.tensor overhead
                    pair_arr = np.empty((K, 3), dtype=np.float32)
                    val_scaled = order.value * 0.1
                    plng = order.pickup_location[2]
                    plat = order.pickup_location[3]
                    for i, idx in enumerate(selected_indices):
                        courier = couriers[idx]
                        cache_key = (order_id, courier.courier_id)
                        cached = self._pair_insertion_cache.get(cache_key)
                        if cached is not None and cached[0] == courier.version:
                            distance_to_pickup, add_distance = cached[1], cached[2]
                        else:
                            dx = courier.location[0] - plng
                            dy = courier.location[1] - plat
                            distance_to_pickup = math.sqrt(dx * dx + dy * dy)
                            _, add_distance, _ = greedy_route_insertion_by_order(
                                courier.location, courier.route, courier.arrive_time,
                                order, courier.speed, skip_arrive_time=True
                            )
                            self._pair_insertion_cache[cache_key] = (courier.version, distance_to_pickup, add_distance)
                        pair_arr[i, 0] = distance_to_pickup * 0.01
                        pair_arr[i, 1] = add_distance * 0.01
                        pair_arr[i, 2] = val_scaled
                    order_courier_feature_dict[order_id] = torch.from_numpy(pair_arr)
                else:
                    order_courier_feature_dict[order_id] = torch.zeros((0, 3), dtype=torch.float32)

            all_orders_courier_pair_feature_dict[platform_id] = order_courier_feature_dict
        
        return all_orders_courier_pair_feature_dict #, all_couriers_order_pair_feature_dict
    
    def _extract_courier_features(self, platform_state: Dict[str, SimulatorState], order_courier_map) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Dict[int, str]]]:
        """Extract features for all available couriers.

        Args:
            platform_state: Dict mapping platform_id to SimulatorState

        Returns:
            Tuple of (courier_features_dict, trajectory_dict, route_dict)
        """
        courier_feature_dict = {}
        courier_route_dict = {}
        courier_trajectory_dict = {}
        courier_id_lookup = {}
        platform_averagre_income = {platform_id: np.mean([courier.total_income[platform_id] for courier in platform_state[platform_id].courier_pool]) for platform_id in platform_state.keys()}



        for platform_id in platform_state.keys():
            couriers = list(platform_state[platform_id].courier_pool)
            courier_id_lookup[platform_id] = {courier_idx: courier.courier_id for courier_idx, courier in enumerate(couriers)}
            current_time = platform_state[platform_id].time
            features = []
            trajectories = []
            routes = []
            for courier in couriers:
                # Determine if new courier state needs to be computed
                cache = self._courier_state_cache.get(platform_id, {}).get(courier.courier_id)

                if cache is not None and cache[0] == courier.version:
                    # Use cached state
                    features.append(cache[1])
                    trajectories.append(cache[2])
                    routes.append(cache[3])
                # Compute new courier state
                else:
                    if platform_id not in self._courier_state_cache:
                        self._courier_state_cache[platform_id] = {}
                    # Update cache after computing new state
                    route_to_use = courier.route

                    # Current load
                    current_load = len(route_to_use)
                    is_idle = (current_load == 0)
                    total_task_time = courier.arrive_time[-1] if len(courier.arrive_time) > 0 else 0.0

                    # Scale current location
                    scaled_lng, scaled_lat = self._scale_coordinates(courier.location[0], courier.location[1])
                    assert len(courier.total_income.keys()) == 1, "platform should only see one income record"

                    feat = [
                        scaled_lng,  # Scaled lng
                        scaled_lat,  # Scaled lat
                        is_idle * 1.0,
                        current_time / 86400.0,  # Normalize time of day
                        total_task_time / 3600.0, # time interval
                        platform_averagre_income[platform_id] / 30.0,  # Normalize average income
                        courier.total_income[platform_id] / 30.0,  # Normalize courier income
                        current_load / 10.0,
                        courier.direction[0],
                        courier.direction[1],
                    ]
                    features.append(feat)

                    # Scale trajectory
                    trajectory = np.zeros((self.back_length, 2))
                    for index, point in enumerate(courier.trajectory):
                        if index <= self.back_length - 1:
                            traj_lng_scaled, traj_lat_scaled = self._scale_coordinates(point[0], point[1])
                            trajectory[index][0] = traj_lng_scaled  # Scaled lng
                            trajectory[index][1] = traj_lat_scaled  # Scaled lat

                    trajectories.append(trajectory)

                    # Scale route (use the updated route)
                    route = np.zeros((self.forward_length, 2))
                    for index, point in enumerate(route_to_use):
                        if index <= self.forward_length - 1:
                            # Extract lng and lat from route point tuple (order_id, location_type, lng, lat)
                            point_lng = point[2] if len(point) > 2 else point[0]
                            point_lat = point[3] if len(point) > 3 else point[1]
                            route_lng_scaled, route_lat_scaled = self._scale_coordinates(point_lng, point_lat)
                            route[index][0] = route_lng_scaled  # Scaled lng
                            route[index][1] = route_lat_scaled  # Scaled lat

                    routes.append(route)
                        # Update cache
                    self._courier_state_cache[platform_id][courier.courier_id] = (courier.version, feat, trajectory, route)


            courier_feature_dict[platform_id] = torch.from_numpy(np.asarray(features, dtype=np.float32))
            courier_trajectory_dict[platform_id] = torch.from_numpy(np.asarray(trajectories, dtype=np.float32))
            courier_route_dict[platform_id] = torch.from_numpy(np.asarray(routes, dtype=np.float32))

        return courier_feature_dict, courier_trajectory_dict, courier_route_dict, courier_id_lookup

    def _extract_order_features(self, platform_state: Dict[str, SimulatorState], controller_type) -> Dict[str, torch.Tensor]:
  
        order_feature_dict = {}
        order_lookup_dict = {}
        for platform_id in platform_state.keys():

            features = []
            order_lookup = {}
            orders = platform_state[platform_id].get_unassigned_orders()

            if len(orders) == 0:
                order_feature_dict[platform_id] = []
                order_lookup_dict[platform_id] = {}
                continue

            if controller_type=='RouteACBGMController'or controller_type=='FairRouteACBGMController':
                orders = [orders[0]]

            # Build lookup once — was O(N²) when placed inside the loop below
            order_lookup = {order.order_id: idx for idx, order in enumerate(orders)}
            current_time = platform_state[platform_id].time

            for idx, order in enumerate(orders):
                # Scale pickup and delivery locations
                pickup_lng_scaled, pickup_lat_scaled = self._scale_coordinates(
                    order.pickup_location[2], order.pickup_location[3]
                )
                delivery_lng_scaled, delivery_lat_scaled = self._scale_coordinates(
                    order.delivery_location[2], order.delivery_location[3]
                )

                remaining_time = order.delivery_patience - (current_time - order.create_time)

                feat = [
                    pickup_lng_scaled,
                    pickup_lat_scaled,
                    delivery_lng_scaled,
                    delivery_lat_scaled,
                    remaining_time / 1800.0,  # Normalize patience
                    # order.value / 10.0,  # Normalize value
                ]
                features.append(feat)
            order_feature_dict[platform_id] = torch.tensor(features, dtype=torch.float32)
            order_lookup_dict[platform_id] = order_lookup

        return order_feature_dict, order_lookup_dict
    
    def _prune_couriers_per_order(self, platform_state, k_per_order,controller_type):

        order_courier_map = {}
        # courier_order_map = {}
        for platform_id in platform_state.keys():
            orders = platform_state[platform_id].get_unassigned_orders()
            couriers = list(platform_state[platform_id].courier_pool)
            if len(orders) == 0:
                order_courier_map[platform_id] = {}
                continue
            if controller_type=='RouteACBGMController' or controller_type=='FairRouteACBGMController':
                orders = [orders[0]]

            # Use shared pruning utility
            order_courier_map_platform = prune_couriers_per_platform(orders, couriers, self.pruning_k)
            order_courier_map[platform_id] = order_courier_map_platform
            # courier_order_map[platform_id] = courier_order_map_platform

        return order_courier_map #, courier_order_map


def build_single_courier_state_from_parts(order_part1, order_part2, courier_part1, courier_part3,
                                          pair_part1, pair_part2, supply_demand_features,
                                          state_dim, order_feature_dim,
                                          platform_id: str, courier_index_in_all: int):
    courier_feature = courier_part1[platform_id][courier_index_in_all]
    courier_route_feature = courier_part3[platform_id][courier_index_in_all]
    selected_order_id = None
    pair_map = pair_part1[platform_id]
    least_distance = float('inf')
    for order_id, candi_courier_index in pair_map.items():
        if courier_index_in_all in candi_courier_index:
            inside_index = candi_courier_index.index(courier_index_in_all)
            pair_features = pair_part2[platform_id][order_id][inside_index]
            distance = pair_features[1]
            if distance < least_distance:
                least_distance = distance
                selected_order_id = order_id
                selected_pair_features = pair_features
    if selected_order_id is None:
        selected_order_id = list(pair_map.keys())[0] if len(pair_map) > 0 else None
        selected_pair_features = pair_part2[platform_id][selected_order_id][0] if selected_order_id is not None else torch.zeros(3)

    order_feature = order_part1[platform_id][order_part2[platform_id][selected_order_id]] if selected_order_id is not None else torch.zeros(order_feature_dim)
    pair_features = selected_pair_features
    current_supply_demand_features = supply_demand_features[platform_id]
    state = np.concatenate((
        courier_feature.numpy(),
        current_supply_demand_features.numpy(),
        order_feature.numpy(),
        pair_features.numpy(),
    ))
    state = np.array(state)
    assert state.shape[-1] == state_dim, "State dimension mismatch"
    return state, selected_order_id, courier_route_feature


def feature_builder(wrapper_state: Dict[str, SimulatorState], feature_extractor, grid_manager, state_dim, order_feature_dim, controller_type, platform_id: str = None, courier_index_in_all: str = None, use_grid_decision: bool = False) -> Dict[str, torch.Tensor]:
   

    features = feature_extractor.extract(wrapper_state, controller_type)

    order_part1, order_part2, courier_part1, courier_part2, courier_part3, courier_part4, pair_part1, pair_part2 = features
    
    grid_manager.clear_all_queues()
    for pid in wrapper_state.keys():
        grid_manager.add_orders_to_queues(wrapper_state[pid].get_unassigned_orders(), wrapper_state[pid].time)
    
    supply_demand_features = grid_manager.get_demand_supply_vector(wrapper_state)

    if courier_index_in_all is None:
        # Grid-based decision making: process first order in each grid cell
        if use_grid_decision and grid_manager is not None:
            grid_state_built = {}
            grid_dispatch_map = {}
            grid_routes = {}

            # Get all agents (grid cells) with non-empty queues
            all_agents_data = grid_manager.get_all_agents_data(sort_by='create_time', descending=False)

            if len(all_agents_data) == 0:
                # No orders in any grid
                return grid_state_built, grid_dispatch_map, grid_routes

            for agent_data in all_agents_data:
                agent_platform_id = agent_data['platform_id']
                cell = agent_data['cell']
                direction = agent_data['direction']
                sorted_orders = agent_data['sorted_orders']

                # Key for this grid agent
                grid_key = (agent_platform_id, cell, direction)

                if len(sorted_orders) == 0:
                    continue

                # Get the first order in this grid (sorted by create_time ascending)
                first_order_data = sorted_orders[0]
                first_order_id = first_order_data['order_id']

                # Check if this order exists in the feature extractor's data
                if agent_platform_id not in order_part2 or first_order_id not in order_part2[agent_platform_id]:
                    continue
                if first_order_id not in pair_part1[agent_platform_id]:
                    continue

                # Build features for this order
                order_idx = order_part2[agent_platform_id][first_order_id]
                order_features = order_part1[agent_platform_id][order_idx].unsqueeze(0)

                candidate_courier_indices = pair_part1[agent_platform_id][first_order_id]
                # assert len(candidate_courier_indices) == feature_extractor.pruning_k, "Pruned courier count mismatch"
                # if len(candidate_courier_indices) == 0:
                #     grid_state_built[grid_key] = torch.zeros((feature_extractor.pruning_k, state_dim))
                #     continue

                pair_features = pair_part2[agent_platform_id][first_order_id]
                courier_features = torch.stack([courier_part1[agent_platform_id][idx] for idx in candidate_courier_indices])
                courier_route_features = torch.stack([courier_part3[agent_platform_id][idx] for idx in candidate_courier_indices])
                courier_trajectory_features = torch.stack([courier_part2[agent_platform_id][idx] for idx in candidate_courier_indices])

                # concate the traject to route
                if controller_type =='polo':
                    courier_route_features = torch.cat((courier_trajectory_features, courier_route_features), dim=1)

                order_features = order_features.repeat(len(candidate_courier_indices), 1)
                current_supply_demand_features = supply_demand_features[agent_platform_id].unsqueeze(0).repeat(len(candidate_courier_indices), 1)

                state = torch.cat((courier_features, current_supply_demand_features, order_features, pair_features), dim=1)
                assert state.shape[-1] == state_dim, "State dimension mismatch"
                assert state.shape[0] == len(candidate_courier_indices), "State dimension mismatch"
                # print(courier_route_features.shape[1])
                # print(feature_extractor.back_length + feature_extractor.forward_length)
                assert courier_route_features.shape[1] == feature_extractor.back_length + feature_extractor.forward_length, "Route length mismatch"

                grid_state_built[grid_key] = state
                # Store order_id instead of order object since we only have order_id from queue
                grid_dispatch_map[grid_key] = (first_order_id, candidate_courier_indices, pair_features)
                grid_routes[grid_key] = courier_route_features

            return grid_state_built, grid_dispatch_map, grid_routes

        # Original behavior: process only the first global order per platform
        platform_state_built = {}
        order_dispatch_map = {}
        platform_routes = {}  # Store route features separately

        for platform_id in wrapper_state.keys():
            if platform_id == 'global':
                continue


            platform_unassigned_order = wrapper_state[platform_id].get_unassigned_orders()
            if len(platform_unassigned_order) == 0:
                # print(f"No unassigned orders for platform {platform_id}")
                platform_state_built[platform_id] = torch.zeros((1, state_dim))
                order_dispatch_map[platform_id] = (None, [], torch.zeros((1, order_feature_dim)))
                platform_routes[platform_id] = None
                continue


            order = platform_unassigned_order[0] # 取第一个订单

            order_features = order_part1[platform_id][order_part2[platform_id][order.order_id]].unsqueeze(0) # 1*订单特征维度
            candidate_courier_indices = pair_part1[platform_id][order.order_id] # 该订单对应的骑手索引列表
            pair_features = pair_part2[platform_id][order.order_id] # 订单-骑手特征列表
            courier_features = torch.stack([courier_part1[platform_id][idx] for idx in candidate_courier_indices])# 骑手特征列表
            courier_route_features = torch.stack([courier_part3[platform_id][idx] for idx in candidate_courier_indices]) # 骑手路线特征列表 (num_couriers, route_length, 2)
            courier_trajectory_features = torch.stack([courier_part2[platform_id][idx] for idx in candidate_courier_indices])
            # concate the traject to route
            if controller_type =='MACTowerController':
                courier_route_features = torch.cat((courier_trajectory_features, courier_route_features), dim=1)
            order_features = order_features.repeat(len(candidate_courier_indices), 1)

            current_supply_demand_features = supply_demand_features[platform_id].unsqueeze(0).repeat(len(candidate_courier_indices), 1)

            state = torch.cat((courier_features, current_supply_demand_features, order_features, pair_features), dim=1)
            assert state.shape[-1] == state_dim, "State dimension mismatch"
            platform_state_built[platform_id] = state
            order_dispatch_map[platform_id] = (order, candidate_courier_indices, pair_features)
            platform_routes[platform_id] = courier_route_features

        return platform_state_built, order_dispatch_map, platform_routes

    else:
        courier_feature = courier_part1[platform_id][courier_index_in_all]# 1*骑手特征维度
        courier_route_feature = courier_part3[platform_id][courier_index_in_all]  # 骑手路线特征
        selected_order_id = None
        pair_map = pair_part1[platform_id]
        least_distance = float('inf')
        for order_id, candi_courier_index in pair_map.items():
            if courier_index_in_all in candi_courier_index:
                inside_index = candi_courier_index.index(courier_index_in_all)
                pair_features = pair_part2[platform_id][order_id][inside_index]
                distance = pair_features[1]
                if distance < least_distance:
                    least_distance = distance
                    selected_order_id = order_id
                    selected_pair_features = pair_features
        if selected_order_id is None:
            # raise ValueError("No suitable order found for the given courier.")
            selected_order_id = list(pair_map.keys())[0] if len(pair_map) > 0 else None
            selected_pair_features = pair_part2[platform_id][selected_order_id][0] if selected_order_id is not None else torch.zeros(3)

        order_feature = order_part1[platform_id][order_part2[platform_id][selected_order_id]] if selected_order_id is not None else torch.zeros(order_feature_dim)
        pair_features = selected_pair_features
        current_supply_demand_features = supply_demand_features[platform_id]
        state = np.concatenate((
            courier_feature.numpy(),
            current_supply_demand_features.numpy(),
            order_feature.numpy(),
            pair_features.numpy(),
        ))
        state = np.array(state)
        # print(state.shape[-1], state_dim)
        
        assert state.shape[-1] == state_dim, "State dimension mismatch"
        return state, selected_order_id, courier_route_feature