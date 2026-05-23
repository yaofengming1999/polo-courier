
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
                    continue
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

                    # Scale trajectory (reversed, padded at the beginning with initial location)
                    whole_traj = list(courier.trajectory)
                    traj_len = min(len(whole_traj), self.back_length)
                    trajectory = np.zeros((self.back_length, 2))
                    # Place reversed trajectory at the end of the array
                    offset = self.back_length - traj_len
                    for index in range(traj_len):
                        point = whole_traj[index]
                        traj_lng_scaled, traj_lat_scaled = self._scale_coordinates(point[0], point[1])
                        trajectory[offset + index][0] = traj_lng_scaled
                        trajectory[offset + index][1] = traj_lat_scaled
                    # Pad the beginning with the first point of the reversed trajectory
                    if traj_len > 0:
                        pad_lng, pad_lat = self._scale_coordinates(whole_traj[0][0], whole_traj[0][1])
                        for index in range(offset):
                            trajectory[index][0] = pad_lng
                            trajectory[index][1] = pad_lat

                    trajectories.append(trajectory)

                    # Scale route (padded at the end with final location)
                    route_len = min(len(route_to_use), self.forward_length)
                    route = np.zeros((self.forward_length, 2))
                    for index in range(route_len):
                        point = route_to_use[index]
                        point_lng = point[2] if len(point) > 2 else point[0]
                        point_lat = point[3] if len(point) > 3 else point[1]
                        route_lng_scaled, route_lat_scaled = self._scale_coordinates(point_lng, point_lat)
                        route[index][0] = route_lng_scaled
                        route[index][1] = route_lat_scaled
                    # Pad the end with the final location of the route
                    if route_len > 0:
                        last_point = route_to_use[route_len - 1]
                        last_lng = last_point[2] if len(last_point) > 2 else last_point[0]
                        last_lat = last_point[3] if len(last_point) > 3 else last_point[1]
                        pad_lng, pad_lat = self._scale_coordinates(last_lng, last_lat)
                        for index in range(route_len, self.forward_length):
                            route[index][0] = pad_lng
                            route[index][1] = pad_lat

                    routes.append(route)
                    # Update cache
                    self._courier_state_cache[platform_id][courier.courier_id] = (courier.version, feat, trajectory, route)


            courier_feature_dict[platform_id] = torch.from_numpy(np.asarray(features, dtype=np.float32))
            courier_trajectory_dict[platform_id] = torch.from_numpy(np.asarray(trajectories, dtype=np.float32))
            courier_route_dict[platform_id] = torch.from_numpy(np.asarray(routes, dtype=np.float32))

        return courier_feature_dict, courier_trajectory_dict, courier_route_dict, courier_id_lookup