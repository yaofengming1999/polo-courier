from typing import List, Union, Tuple

from .location import Location, OrderLocation
from .order import Order

# Maximum trajectory entries kept in memory.
# Must be >= back_length used by any feature extractor (default back_length=10).
# Increase this constant if you raise back_length beyond 20.
_TRAJ_CAP: int = 20



class Courier:

    def __init__(self,
                 courier_id: str,
                 current_location: Union[Location, Tuple[float, float]],
                 platforms: List[str],
                 speed: float = 0.003): #5.56
        if not isinstance(courier_id, str):
            raise TypeError("courier_id must be a string")
        if not isinstance(platforms, list) or not platforms:
            raise ValueError("platforms must be a non-empty list")
        if not all(isinstance(p, str) for p in platforms):
            raise TypeError("All platform IDs must be strings")
        if not isinstance(speed, (int, float)) or speed <= 0:
            raise ValueError("speed must be a positive number")

        self.courier_id = courier_id

        # Handle location input - convert tuple to Location if needed
        if isinstance(current_location, tuple):
            self.current_location = Location(current_location[0], current_location[1])
        else:
            self.current_location = current_location

        self.platforms = platforms.copy()  # Create a copy to avoid external modifications
        self.speed = float(speed)

        # Route management - ordered list of pickup and delivery locations
        self.route: List[OrderLocation] = []

        self.arrive_time: List[float] = []  # Expected arrival times at each location in the route
        

        # Movement tracking — capped at _TRAJ_CAP entries (feature extractors use at most back_length=10)
        self.trajectory: List[Location] = [self.current_location]
        self._total_distance: float = 0.0

        # Direction vector: [dx, dy] where [0, 0] means stationary
        self.direction: List[float] = [0.0, 0.0]

        # Income tracking
        self.total_income = {platform_id: 0.0 for platform_id in platforms}

        # Profit threshold for order acceptance
        self.profit_threshold: float = 1

        # Route metrics at decision point (before moving)
        # self.start_revenue: float = 0.0
        # self.start_distance: float = 0.0

        # Version tracking for caching optimization
        self._version: int = 0

    @property
    def version(self) -> int:
        return self._version

    def bump_version(self) -> None:
        self._version += 1

    # def calculate_route_metrics(self, current_time: int, order_pool: List['Order']) -> None:
    #     """
    #     Calculate start_revenue and start_distance for the current route.

    #     This should be called at decision points (before courier moves) to record
    #     the expected revenue and remaining distance of the route.

    #     Args:
    #         current_time: Current simulation time
    #         order_pool: List of all orders to look up order details
    #     """
    #     if not self.route:
    #         self.start_revenue = 0.0
    #         self.start_distance = 0.0
    #         return

    #     # Build order map
    #     order_map = {order.order_id: order for order in order_pool}

    #     # Calculate distance
    #     current_pos = self.current_location
    #     total_distance = 0.0

    #     for i, waypoint in enumerate(self.route):
    #         order_id, location_type, lng, lat = waypoint
    #         # Distance to this waypoint
    #         dist = ((lng - current_pos.longtitude) ** 2 + (lat - current_pos.latitude) ** 2) ** 0.5
    #         total_distance += dist
    #         current_pos = Location(lng, lat)

    #     self.start_distance = total_distance

    #     # Calculate expected revenue by simulating traversal
    #     current_pos = self.current_location
    #     estimated_time = current_time
    #     seen_deliveries = set()
    #     revenue = 0.0

    #     for waypoint in self.route:
    #         order_id, location_type, lng, lat = waypoint

    #         # Travel time to this waypoint
    #         distance = ((lng - current_pos.longtitude) ** 2 + (lat - current_pos.latitude) ** 2) ** 0.5
    #         travel_time = distance / self.speed if self.speed > 0 else 0
    #         estimated_time += travel_time

    #         # Check if delivery point
    #         if location_type == 1:  # Delivery
    #             if order_id in seen_deliveries:
    #                 continue
    #             seen_deliveries.add(order_id)

    #             order = order_map.get(order_id)
    #             if order is None:
    #                 continue

    #             # Check if on-time
    #             delivery_time = estimated_time - order.create_time
    #             if delivery_time <= order.delivery_patience:
    #                 revenue += order.value

    #         current_pos = Location(lng, lat)

    #     self.start_revenue = revenue

    def update(self, time_interval: int) -> List[OrderLocation]:
        if time_interval < 0:
            raise ValueError("time_interval must be non-negative")

        # Don't move if has no route
        if not self.route:
            return []

        # Convert route tuples to OrderLocation objects for movement calculation
        # from .state_action import RouteState
        # route_state = RouteState(sequence=self.route)
        # route_locations = route_state.get_location_objects()
        # route_locations = [OrderLocation(lon, lat, order_id, location_type) for order_id, location_type, lon, lat in self.route]

        # Calculate movement and update position
        new_location, remain_route, visited_order_locations = self.courier_move_forward(
            self.current_location,
            self.route,
            self.speed,
            time_interval
        )

        # Accumulate distance BEFORE overwriting current_location so `prev` starts
        # at the old position (using new_location after the update was the bug).
        prev = self.current_location
        for loc in visited_order_locations:
            stop = Location(loc.longtitude, loc.latitude)
            self._total_distance += prev.distance_to(stop)
            prev = stop
        self._total_distance += prev.distance_to(new_location)

        # Update courier state
        self.current_location = new_location

        # Convert remaining OrderLocation objects back to tuples
        remaining_tuples = [loc for loc in remain_route]
        self.route.clear()
        self.route.extend(remaining_tuples)

        self.arrive_time = self.arrive_time[len(visited_order_locations):]  # 更新预计到达时间列表
        self.arrive_time = [t - time_interval for t in self.arrive_time]  # 减去时间间隔

        # Track movement in trajectory — cap at _TRAJ_CAP entries
        if visited_order_locations:
            self.trajectory.extend(Location(loc.longtitude, loc.latitude) for loc in visited_order_locations)
        self.trajectory.append(new_location)
        if len(self.trajectory) > _TRAJ_CAP:
            del self.trajectory[:-_TRAJ_CAP]

        # State changed due to movement
        self.bump_version()

        return visited_order_locations

    def get_total_distance(self) -> float:
        return self._total_distance


    def add_income(self, platform, amount: float) -> None:
        # if amount < 0:
        #     raise ValueError("Income amount cannot be negative")
        self.total_income[platform] += amount

    def get_total_income(self) -> float:
        return sum(self.total_income.values())


    def courier_move_forward(self, current_location: Location, route: List[OrderLocation], speed: float, time_interval: int) -> Tuple[Location, List[OrderLocation], List[OrderLocation]]:

        if not route:
            # No route, courier is stationary
            self.direction = [0.0, 0.0]
            return current_location, [], []  # 如果没有路由，直接返回当前位置和空的路由列表

        # Calculate total distance courier can travel
        # speed is in km/s, time_interval is in seconds, so speed * time_interval gives kilometers
        # Convert to meters for the movement calculation
        remaining_distance_meters = speed * time_interval * 1000  # Convert km to meters
        new_location = Location(current_location.longtitude, current_location.latitude)
        visited_locations: List[OrderLocation] = []
        route_idx = 0

        # Track initial location to calculate direction vector
        initial_location = Location(current_location.longtitude, current_location.latitude)

        while route_idx < len(route) and remaining_distance_meters > 0:
            next_stop = route[route_idx]
            distance_to_next_meters = new_location.distance_to_meters(next_stop)

            if distance_to_next_meters <= remaining_distance_meters:
                # Can reach the next stop
                visited_locations.append(next_stop)
                new_location = Location(next_stop.longtitude, next_stop.latitude)
                route_idx += 1
                remaining_distance_meters -= distance_to_next_meters
            else:
                # Cannot reach next stop, move partially
                ratio = remaining_distance_meters / distance_to_next_meters
                delta_lon = (next_stop.longtitude - new_location.longtitude) * ratio
                delta_lat = (next_stop.latitude - new_location.latitude) * ratio
                new_location = Location(new_location.longtitude + delta_lon, new_location.latitude + delta_lat)
                remaining_distance_meters = 0

        remain_route = route[route_idx:]  # single slice instead of L pop(0) calls

        # Calculate direction vector based on movement
        if new_location.longtitude != initial_location.longtitude or new_location.latitude != initial_location.latitude:
            self.direction = [
                new_location.longtitude - initial_location.longtitude,
                new_location.latitude - initial_location.latitude
            ]
        else:
            # No movement occurred
            self.direction = [0.0, 0.0]

        return new_location, remain_route, visited_locations

