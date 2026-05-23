import pandas as pd
import os
from typing import List, Optional, Dict, Any, Tuple

from traitlets import Int

from env.models.courier import Courier
from env.models.order import Order
from env.models.location import Location, OrderLocation
from env.models.state_action import SimulatorState, SimulatorAction


class Simulator:

    def __init__(self,
                 time_interval: int = 60,
                 orders: Optional[List[dict]] = None,
                 couriers: Optional[List[dict]] = None):
        if time_interval <= 0:
            raise ValueError("time_interval must be positive")

        # Simulation parameters
        self.time_interval = time_interval

        # Store raw data for reset
        self._orders_data = orders
        self._couriers_data = couriers

        # Initialize simulation state
        self.courier_pool: List[Courier] = []
        self.order_pool: List[Order] = []
        
        self.order_pointer = 0
        self.time = 0
        self.done = False
        self.order_by_id: Dict[str, Order] = {}
        self.courier_by_id: Dict[str, Courier] = {}

        # Cache for OrderState objects: {order_id: (version, OrderState)}
        self._order_state_cache: Dict[str, Tuple[int, Any]] = {}
        # Cache for CourierState objects: {courier_id: (version, CourierState)}
        self._courier_state_cache: Dict[str, Tuple[int, Any]] = {}
        self._courier_state_by_platform_cache: Dict[str, Dict[str, Tuple[int, Any]]] = {}

        # Maintain a set for unassigned orders for quick access
        self.unassigned_order_ids = set()
        self.order_platform_map: Dict[str, str] = {}

        # Load data if provided
        if orders:
            self._load_orders_from_list(orders)
        if couriers:
            self._load_couriers_from_list(couriers)

    def _load_couriers_from_list(self, couriers: List[dict]) -> None:
        self.courier_pool = []
        for c in couriers:
            # Handle platform - can be list or string
            platforms = c['platform']
            if isinstance(platforms, str):
                platforms = [platforms]

            courier = Courier(
                courier_id=str(c['courier_id']),
                current_location=Location(
                    longtitude=float(c['current_x']),
                    latitude=float(c['current_y'])
                ),
                platforms=platforms
            )
            self.courier_pool.append(courier)
        # Create lookup dict
        self.courier_by_id = {c.courier_id: c for c in self.courier_pool}

    def _load_orders_from_list(self, orders: List[dict]) -> None:
        self.order_pool = []
        for o in orders:
            order = Order(
                order_id=str(o['order_id']),
                pickup_location=(float(o['pickup_x']), float(o['pickup_y'])),
                delivery_location=(float(o['delivery_x']), float(o['delivery_y'])),
                create_time=int(o['create_time']),
                delivery_patience=int(o['delivery_patience']),
                platform=str(o['platform_id']),
                value=float(o['value'])
            )
            self.order_pool.append(order)

        # Sort orders by create_time for proper release
        self.order_pool.sort(key=lambda x: x.create_time)
        self.order_pointer = 0
        # Create lookup dict
        self.order_by_id = {o.order_id: o for o in self.order_pool}
        self.unassigned_order_ids = {o.order_id for o in self.order_pool if o.status == Order.UNASSIGNED}
        self.order_platform_map = {o.order_id: o.platform for o in self.order_pool}

    def reset(self, orders: Optional[List[dict]] = None,
              couriers: Optional[List[dict]] = None) -> SimulatorState:
        # Update stored data if new data provided
        if orders is not None:
            self._orders_data = orders
        if couriers is not None:
            self._couriers_data = couriers

        # Reset simulation state
        self.courier_pool = []
        self.order_pool = []
        self.order_pointer = 0
        self.time = 0
        self.done = False

        # Clear state caches
        self._order_state_cache.clear()
        self._courier_state_cache.clear()
        self._courier_state_by_platform_cache.clear()


        # Reload data from stored lists
        if self._orders_data:
            self._load_orders_from_list(self._orders_data)
        if self._couriers_data:
            self._load_couriers_from_list(self._couriers_data)

        return self.get_state()

    def step(self, real_go, action: Optional[SimulatorAction] = None) -> SimulatorState:
        # Apply algorithm action
        if action:
            self.apply_action(action)

        # Move simulation forward if dispatch has finished
        if real_go:
            self._move_forward()

        return self.get_state()

    def _move_forward(self,) -> None:

        self.time += self.time_interval

        # Move all couriers and collect visited locations
        all_visited_locations = []
        for courier in self.courier_pool:
            visited_locations = courier.update(self.time_interval)
            all_visited_locations.extend(visited_locations)

        # Update order statuses based on visits
        self._update_order_statuses(all_visited_locations)

        # Release new orders based on time
        self._release_orders()

        # Cancelled orders based on current time and order.delivery_patience
        self._cancel_orders()

    def _update_order_statuses(self, visited_order_locations: List[OrderLocation]) -> None:
        for order_location in visited_order_locations:
            # Find the corresponding order
            order = self._find_order_by_id(order_location.order_id)
            if not order:
                continue

            try:
                if order_location.is_pickup():
                    # Order picked up
                    if order.status == Order.ASSIGNED:
                        order.mark_picked_up(self.time)

                elif order_location.is_delivery():
                    # Order delivered
                    if order.status == Order.DELIVERING:
                        order.mark_delivered(self.time)

            except ValueError as e:
                pass  # Skip invalid order status update

    def _release_orders(self) -> None:
        released_count = 0

        while self.order_pointer < len(self.order_pool):
            order = self.order_pool[self.order_pointer]

            if order.create_time <= self.time:
                order.release_for_assignment()
                self.order_pointer += 1
                released_count += 1
                self.unassigned_order_ids.add(order.order_id)
            else:
                break  # Orders are sorted by create_time
        if self.order_pointer >= len(self.order_pool)-1 and all(order.is_terminal() for order in self.order_pool):
            self.done = True
    def _cancel_orders(self) -> None:
        canceled_count = 0

        for order in self.order_pool:
            if order.status != Order.UNASSIGNED:
                continue

            # Only cancel if promised delivery time has passed
            if self.time >= order.create_time + order.delivery_patience:
                try:
                    order.cancel_order(self.time)
                    canceled_count += 1
                    self.unassigned_order_ids.discard(order.order_id)
                except ValueError:
                    pass  # Skip failed cancellation

    def _find_order_by_id(self, order_id: str) -> Optional[Order]:
        return self.order_by_id.get(order_id)


    def apply_action(self, action: SimulatorAction) -> None:
        try:
            action.apply_to_simulator(self, self.time)
        except Exception as e:
            raise ValueError(f"Invalid action: {e}")

    def get_state(self) -> SimulatorState:
        return SimulatorState.from_simulator(self)


    def get_all_courier_ids(self) -> List[str]:
        return [courier.courier_id for courier in self.courier_pool]

    def get_platforms(self) -> List[str]:
        platforms = set()
        for order in self.order_pool:
            platforms.add(order.platform)
        return sorted(list(platforms))

    def get_orders_by_platform(self, platform: str) -> List[Order]:
        return [order for order in self.order_pool if order.platform == platform]

    def get_orders_num_by_platform(self, platform: str) -> int:
        return sum(1 for order in self.order_pool if order.platform == platform)

    def get_num_delivered_orders_by_platform(self, platform: str) -> Int:
        return sum(1 for order in self.order_pool if order.platform == platform and order.is_completed())

    def get_num_completed_orders(self) -> Int:
        return sum(1 for order in self.order_pool if order.is_completed())

    def get_num_couriers_by_platform(self, platform: str) -> Int:
        return sum(1 for courier in self.courier_pool if platform in courier.platforms)

    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        return self.order_by_id.get(order_id)
    def get_order_platform_map(self) -> Dict[str, str]:
        return self.order_platform_map
    
    def get_couriers_state_by_platform(self, platform: str):
        return self._courier_state_by_platform_cache.setdefault(platform, {})
