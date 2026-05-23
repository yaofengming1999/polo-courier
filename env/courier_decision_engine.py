import random
from typing import List, Tuple, Dict, Optional
from env.models.state_action import  OrderState, CourierState, SimulatorState
from env.models.location import Location
from env.route_decision_engine import greedy_route_insertion_by_order


# # Hardcoded refusal probability tables (loaded from CSV files)
ETA_REFUSAL_TABLE_1 = {
    2.5: 10.85106,
    7.5: 12.12766,
    12.5: 11.48936,
    17.5: 10.42553,
    22.5: 10.85106,
    27.5: 12.76596,
    32.5: 15.10638,
    37.5: 16.17021,
    42.5: 18.7234,
    47.5: 13.19149,
    52.5: 21.06383,
    57.5: 6.38298
}

ON_HAND_ORDER_REFUSAL_TABLE_1 = {
    0.0: 29.74526,
    1.0: 13.82838,
    2.0: 11.07474,
    3.0: 12.20765,
    4.0: 9.45641,
    5.0: 12.31596,
    6.0: 20.35544,
    7.0: 14.15012,
    8.0: 37.29771
}

PICK_DISTANCE_REFUSAL_TABLE_1 = {
    0.1: 23.61992,
    0.3: 24.07056,
    0.5: 26.41302,
    0.7: 23.07046,
    0.9: 21.41398,
    1.1: 18.49341,
    1.3: 15.36233,
    1.5: 9.70594,
    1.7: 13.52015,
    1.9: 10.39192,
    2.1: 5.78814,
    2.3: 8.76026,
    2.5: 5.84161,
    2.7: 2.92104,
    2.9: 3.78891,
    3.1: 7.8165,
    3.3: 12.89765,
    3.5: 1.97537,
    3.7: 14.21424,
    3.9: 2.03074
}


# testing with zero refusal
ETA_REFUSAL_TABLE_0 = {
    2.5: 0,
}

ON_HAND_ORDER_REFUSAL_TABLE_0 = {
    0.0: 0,

}

PICK_DISTANCE_REFUSAL_TABLE_0 = {
    0.1: 0,
}

class CourierDecisionEngine:

    def __init__(self, is_courier_refuse):
        if is_courier_refuse:
            self.eta_refusal = ETA_REFUSAL_TABLE_1
            self.on_hand_refusal = ON_HAND_ORDER_REFUSAL_TABLE_1
            self.pick_distance_refusal = PICK_DISTANCE_REFUSAL_TABLE_1
        elif is_courier_refuse is False:
            self.eta_refusal = ETA_REFUSAL_TABLE_0
            self.on_hand_refusal = ON_HAND_ORDER_REFUSAL_TABLE_0
            self.pick_distance_refusal = PICK_DISTANCE_REFUSAL_TABLE_0

    def _lookup_refusal_probability(self, value: float, table: Dict[float, float]) -> float:
        if not table:
            return 0.0

        # Find the closest key (nearest neighbor)
        nearest_key = min(table.keys(), key=lambda k: abs(k - value))
        return table[nearest_key]

    def calculate_refusal_probability(
        self,
        eta_minutes: float,
        on_hand_orders: int,
        pick_distance_km: float,
        aggregation_method: str = 'average'
    ) -> float:
        # Get individual refusal probabilities (all use nearest neighbor lookup)
        eta_prob = self._lookup_refusal_probability(eta_minutes, self.eta_refusal)
        distance_prob = self._lookup_refusal_probability(pick_distance_km, self.pick_distance_refusal)
        on_hand_prob = self._lookup_refusal_probability(float(on_hand_orders), self.on_hand_refusal)


        # Probabilistic product (assuming independence)
        # Convert percentages to probabilities, combine, convert back
        p_eta = eta_prob / 100.0
        p_on_hand = on_hand_prob / 100.0
        p_distance = distance_prob / 100.0

        # Combined probability: 1 - (1-p1)*(1-p2)*(1-p3)
        if on_hand_orders == 0:
            # If no on-hand orders, ignore that factor
            p_accept_all = (1 - p_on_hand) * (1 - p_distance)
        else:

            p_accept_all = (1 - p_eta) * (1 - p_on_hand) * (1 - p_distance)

        overall_prob = (1 - p_accept_all) * 100.0


        return overall_prob

    def should_refuse_order(
        self,
        eta_minutes: float,
        on_hand_orders: int,
        pick_distance_km: float,
        aggregation_method: str = 'average'
    ) -> bool:
        refusal_prob = self.calculate_refusal_probability(
            eta_minutes, on_hand_orders, pick_distance_km, aggregation_method
        )

        # Convert percentage to probability (0-1)
        refusal_prob_normalized = refusal_prob / 100.0

        # Random decision based on probability
        return random.random() < refusal_prob_normalized

    def process_courier_order_decisions(
        self,
        courier_state: CourierState,
        order_candidates: List[Tuple[str, float, str]],
        simulator_state,
        get_order_func,
        aggregation_method: str = 'average'
    ) -> Tuple[List[Tuple[str, int, float, float]], List[Tuple[str, float, str, bool]]]:
        current_route = courier_state.route
        arrive_time = courier_state.arrive_time
        decision_list = []

        # Compute once; increment by 1 each time an order is accepted into the route
        on_hand_orders = len(set(loc[0] for loc in current_route)) if current_route else 0

        # Process orders sorted by incentive (highest first)
        for order_id, incentive, platform_id in order_candidates:
            is_accepted, updated_route, updated_arrive_time = self._courier_accept_order(
                courier_state=courier_state,
                current_route=current_route,
                current_arrive_time=arrive_time,
                order_id=order_id,
                incentive=incentive,
                on_hand_orders=on_hand_orders,
                simulator_state=simulator_state,
                get_order_func=get_order_func,
                aggregation_method=aggregation_method
            )

            decision_list.append((order_id, incentive, platform_id, is_accepted))

            if is_accepted:
                current_route = updated_route
                arrive_time = updated_arrive_time
                on_hand_orders += 1

        return [current_route, arrive_time], decision_list

    def _courier_accept_order(
        self,
        courier_state: CourierState,
        current_route: List[Tuple[str, int, float, float]],
        current_arrive_time: List[float],
        order_id: str,
        incentive: float,
        on_hand_orders: int,
        simulator_state,
        get_order_func,
        aggregation_method: str = 'average'
    ) -> Tuple[bool, List[Tuple[str, int, float, float]]]:
        order_state = get_order_func(order_id)
        if order_state is None:
            return False, current_route, current_arrive_time

        # Extract three features
        eta_minutes = self._get_most_urgent_eta(current_route, simulator_state)
        pick_distance_km = courier_state.get_current_location().distance_to(
            order_state.get_pickup_location()
        )

        # Make decision based on refusal probability
        should_refuse = self.should_refuse_order(
            eta_minutes, on_hand_orders, pick_distance_km, aggregation_method
        )

        if should_refuse:
            return False, current_route, current_arrive_time  

        # Accept - insert order into route using greedy insertion
        updated_route, _ , new_arrive_time = greedy_route_insertion_by_order(courier_state.location, current_route, current_arrive_time, order_state, courier_state.speed)
        return True, updated_route, new_arrive_time

    def _get_most_urgent_eta(self, route: List[Tuple[str, int, float, float]], simulator_state:SimulatorState) -> float:
        if not route:
            return 0.0

        # Find the earliest promised delivery time
        min_deadline = None
        for order_id, location_type, _, _ in route:
            if location_type == 1:  # Delivery location
                order = simulator_state.get_order_by_id(order_id)
                if order and hasattr(order, 'delivery_patience'):
                    if min_deadline is None or order.delivery_patience + order.create_time < min_deadline:
                        min_deadline = order.delivery_patience + order.create_time

        if min_deadline is None:
            return 0.0

        # Calculate minutes from now
        current_time = simulator_state.time
        eta_minutes = max(0, (min_deadline - current_time) / 60.0)
        return eta_minutes
