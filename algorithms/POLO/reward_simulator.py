"""Reward Simulator - Counterfactual reward calculation.

Computes marginal reward as the difference in courier route revenue
before and after an order assignment.
"""
from env.models.location import Location


class RewardSimulator:
    """Calculate counterfactual rewards based on route revenue change."""

    def __init__(self, rejection_penalty=0.05, overdue_penalty_factor=1.0):
        """Initialize reward simulator.

        Args:
            rejection_penalty: Penalty for rejected/invalid assignments
            overdue_penalty_factor: Multiplier for overdue penalties
        """
        # self.rejection_penalty = rejection_penalty
        self.overdue_penalty_factor = overdue_penalty_factor

    def compute_counterfactual_rewards(self, dispatch_records, next_state):
        """Compute counterfactual reward for each agent's dispatch.

        For each agent:
        reward = courier_revenue(with all actions) - courier_revenue(without this agent's action)

        Args:
            dispatch_records: List of dispatch records, each with:
                - order_id, courier_id
            next_state: State after all dispatches applied
                (order_pool contains order info, courier_pool contains routes)

        Returns:
            List of rewards (float) for each dispatch record
        """
        if next_state is None or not dispatch_records:
            return [0.0] * len(dispatch_records)

        # Build order info from state's order_pool
        
        order_info = self.build_order_info(next_state)
        current_time = next_state.time if hasattr(next_state, 'time') else 0

        # Get all couriers from next_state (with all actions applied)
        courier_dict = {c.courier_id: c for c in next_state.courier_pool}

        rewards = []
        for record in dispatch_records:
            courier_id = record['courier_id']
            order_id = record['order_id']

            courier = courier_dict.get(courier_id)
            if courier is None:
                rewards.append(0.0)
                continue

            # Courier's revenue with all actions (current route)
            revenue_with_all = self.calculate_route_reward(
                courier.route, courier, order_info, current_time
            )

            # Courier's revenue without this agent's action (remove this order)
            route_without = self._remove_order_from_route(courier.route, order_id)
            revenue_without_this = self.calculate_route_reward(
                route_without, courier, order_info, current_time
            )

            reward = revenue_with_all - revenue_without_this
            rewards.append(reward)

        return rewards

    def _remove_order_from_route(self, route, order_id):
        """Create a new route without the specified order."""
        if route is None:
            return None
        new_sequence = [loc for loc in route if loc[0] != order_id]
        return new_sequence

    def build_order_info(self, state):
        """Build order info lookup from state."""
        order_info = {}
        for order in state.order_pool:
            order_info[order.order_id] = {
                'create_time': order.create_time,
                'deadline': order.delivery_patience,
                'value': order.value
            }
        return order_info

    def calculate_route_reward(self, route, courier, order_info, current_time):
        """Calculate total expected reward for a courier's route.

        Simulates route traversal and calculates reward based on
        whether deliveries meet their deadlines.

        Args:
            route: RouteState object
            courier: Courier object
            order_info: Dict mapping order_id to {create_time, deadline, value}
            current_time: Current simulation time

        Returns:
            Total route reward (float)
        """
        if route is None or len(route) == 0:
            return 0.0

        courier_speed = courier.speed if courier.speed > 0 else 0.001
        current_loc = Location(courier.location[0], courier.location[1])
        accumulated_time = current_time
        order_delivery_times = {}

        # Traverse the route
        for location_tuple in route:
            order_id, location_type, lng, lat = location_tuple
            next_loc = Location(lng, lat)

            # Calculate travel distance and time
            distance_km = current_loc.distance_to(next_loc)
            travel_time = distance_km / courier_speed
            accumulated_time += travel_time

            # If delivery location (type=1), record delivery time
            if location_type == 1:
                order_delivery_times[order_id] = accumulated_time

            current_loc = next_loc

        # Calculate total reward
        total_reward = 0.0
        for oid, delivery_time in order_delivery_times.items():
            if oid not in order_info:
                continue
            info = order_info[oid]
            deadline_time = info['create_time'] + info['deadline']

            if delivery_time <= deadline_time:
                total_reward += info['value']
            else:
                total_reward -= info['value'] * self.overdue_penalty_factor

        return total_reward / 10  # Scale down
