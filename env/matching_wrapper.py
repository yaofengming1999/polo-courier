import numpy as np
import logging
import yaml
import os
import importlib.util
from typing import Dict, List, Optional, Tuple, Union
from env.models.order import Order
from collections import defaultdict, deque


from env.simulator import Simulator
from env.models.state_action import SimulatorState, SimulatorAction
from env.models.order import Order
from env.models.courier import Courier
from env.courier_decision_engine import CourierDecisionEngine


class Wrapper:

    # =============================================================================
    # PUBLIC INTERFACE (for RL algorithms)
    # =============================================================================

    def __init__(self, config: dict):
        self.config = config

        # Initialize simulator (empty, will be populated on reset)
        self.simulator = Simulator(time_interval=self.config.get('time_interval'))

        self.simulator_state_all: None
        self.platforms = []
        self.late_delivery_compensation_ratio = self.config.get('late_delivery_compensation_ratio')
        if self.config.get('fixed_incentive_ratio'):
            self.incentive_ratio = self.config.get('incentive_ratio')

        # Initialize tracking (will be populated on reset)
        self.revenues = {}
        self.accepted_orders_tracking = {}
        self.total_orders_appeared = {}
        self.total_orders_accepted = {}
        self.total_orders_delivered = {}
        self.total_orders_delivered_late = {}
        self._total_courier_refused = 0
        self.done = False

        # Initialize courier decision engine
        self.courier_decision_engine = CourierDecisionEngine(is_courier_refuse=self.config.get('is_courier_refuse'))

    def _apply_courier_config_to_all_couriers(self):
        courier_speed = self.config.get('courier_speed')
        courier_profit_threshold = self.config.get('courier_profit_threshold')

        for courier in self.simulator.courier_pool:
            courier.speed = courier_speed
            courier.profit_threshold = courier_profit_threshold

    def reset(self, orders: List[dict] = None, couriers: List[dict] = None) -> Tuple[Dict[str, Dict], Dict[str, bool]]:
        # Reset simulator with data
        self.simulator_state_all = self.simulator.reset(orders=orders, couriers=couriers)


        # Update platforms after reset
        self.platforms = self.simulator.get_platforms()

        # Reset tracking
        self.revenues = {platform: 0.0 for platform in self.platforms}
        self.accepted_orders_tracking = {}
        self.total_orders_appeared = {platform: 0 for platform in self.platforms}
        self.total_orders_accepted = {platform: 0 for platform in self.platforms}
        self.total_orders_delivered = {platform: 0 for platform in self.platforms}
        self.total_orders_delivered_late = {platform: 0 for platform in self.platforms}
        self._seen_order_ids = set()
        self._total_courier_refused = 0
        self.done = False

        # Apply courier configurations after reset
        self._apply_courier_config_to_all_couriers()

        # Separate the State into different platforms
        wrapper_state, wrapper_done = self.simulator_state_all

        return wrapper_state, wrapper_done

    def get_total_orders_appeared(self) -> Dict[str, int]:
        assert self.done == True, "Total orders appeared can only be retrieved after the environment is done."
        # after done, the total orders appeared should equal to the number of all orders in simulator
        for platform in self.platforms:

            assert self.total_orders_appeared[platform] == self.simulator.get_orders_num_by_platform(platform), "Total orders appeared mismatch."
        # return the global total orders appeared as well

        self.total_orders_appeared['global'] = sum(self.total_orders_appeared.values())
        return self.total_orders_appeared

    def get_total_couriers(self) -> Dict[str, int]:
        assert self.done == True, "Total couriers can only be retrieved after the environment is done."
        total_couriers = {}
        for platform in self.platforms:
            platform_couriers_amount = self.simulator.get_num_couriers_by_platform(platform)
            total_couriers[platform] = platform_couriers_amount

        # return the global total couriers as well
        # assert 每个平台的快递员数目相同, 因为快递员是共享的
        assert len(set(total_couriers.values())) == 1, "Total couriers across platforms should be the same."
        total_couriers['global'] = set(total_couriers.values()).pop()
        return total_couriers

    def get_platform_revenue(self) -> Dict[str, float]:
        assert self.done == True, "Platform revenue can only be retrieved after the environment is done."
            # return the global revenue as well
        self.revenues['global'] = sum(self.revenues.values())
        return self.revenues

    def get_potential_platform_revenue(self) -> Dict[str, float]:
        potential_revenues = {platform: 0.0 for platform in self.platforms}

        for order in self.simulator.order_pool:
            platform_id = order.platform
            order_value = order.get_value()
            # Assume incentive is calculated using fixed ratio if applicable
            if self.config.get('fixed_incentive_ratio', True):
                incentive = order_value * self.incentive_ratio
            else:
                # If not fixed ratio, assume zero incentive for potential calculation
                incentive = 0.0

            potential_revenues[platform_id] += (order_value - incentive)

        # return the global potential revenue as well
        potential_revenues['global'] = sum(potential_revenues.values())
        return potential_revenues


    def get_order_response_rate(self) -> Dict[str, float]:

        response_rates = {}
        total_orders_appeared = 0
        for platform in self.platforms:
            # assert 计算wrapper的订单响应率应该在done之后进行
            # assert 被response的订单数 = wrapper结束状态时所有被送达的订单数
            assert self.done == True
            assert self.simulator.get_num_delivered_orders_by_platform(platform) == self.total_orders_accepted[platform]
            if self.total_orders_appeared[platform] == 0:
                response_rates[platform] = 0.0
            else:
                response_rates[platform] = (
                    self.total_orders_accepted[platform] / self.total_orders_appeared[platform]
                )
                total_orders_appeared += self.total_orders_appeared[platform]
            # return the global response rate as well
            response_rates['global'] = (
                sum(self.total_orders_accepted.values()) / total_orders_appeared
            ) if sum(self.total_orders_appeared.values()) > 0 else 0.0
        return response_rates

    def get_order_overdue_rate(self) -> Dict[str, float]:
        overdue_rates = {}
        assert self.done == True, "Order overdue rate can only be retrieved after the environment is done."
        assert sum(self.total_orders_delivered.values()) == self.simulator.get_num_completed_orders() , "Total delivered orders mismatch."
        for platform in self.platforms:
            if self.total_orders_delivered[platform] == 0:
                overdue_rates[platform] = 0.0
            else:
                overdue_rates[platform] = (
                    self.total_orders_delivered_late[platform] / self.total_orders_delivered[platform]
                )
            # return the global overdue rate as well
            overdue_rates['global'] = (
                sum(self.total_orders_delivered_late.values()) / sum(self.total_orders_delivered.values())
            ) if sum(self.total_orders_delivered.values()) > 0 else 0.0
        return overdue_rates

    def get_average_courier_income(self) -> float:
        total_income = 0.0
        total_couriers = 0

        for courier in self.simulator.courier_pool:
            total_income += courier.get_total_income()
            total_couriers += 1

        if total_couriers == 0:
            return 0.0

        average_income = total_income / total_couriers
        return average_income

    def get_average_courier_distance_travelled(self) -> float:
        total_distance = 0.0
        total_couriers = 0

        for courier in self.simulator.courier_pool:
            total_distance += courier.get_total_distance()
            total_couriers += 1

        if total_couriers == 0:
            return 0.0

        average_distance = total_distance / total_couriers
        return average_distance

    def get_courier_income_stats(self) -> dict:
        """Return mean, std, and Gini coefficient of total courier income."""
        incomes = [c.get_total_income() for c in self.simulator.courier_pool]
        if not incomes:
            return {'mean': 0.0, 'std': 0.0, 'gini': 0.0}
        arr = np.array(incomes, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        # Gini coefficient: 0 = perfect equality, 1 = maximum inequality
        if mean == 0.0:
            gini = 0.0
        else:
            n = len(arr)
            sorted_arr = np.sort(arr)
            gini = float((2.0 * np.sum((np.arange(1, n + 1)) * sorted_arr) / (n * np.sum(sorted_arr))) - (n + 1) / n)
            gini = max(0.0, gini)  # clamp numerical noise
        return {'mean': mean, 'std': std, 'gini': gini}

    def get_average_response_time(self) -> float:
        """Average time (seconds) from order creation to assignment, over accepted orders."""
        durations = []
        for order in self.simulator.order_pool:
            if order.response_time is not None:
                durations.append(order.response_time - order.create_time)
        return float(np.mean(durations)) if durations else 0.0

    def get_average_delivery_time(self) -> float:
        """Average end-to-end time (seconds) from order creation to delivery, over delivered orders."""
        durations = []
        for order in self.simulator.order_pool:
            if order.is_completed() and order.delivered_time is not None:
                durations.append(order.delivered_time - order.create_time)
        return float(np.mean(durations)) if durations else 0.0

    def get_courier_refusal_stats(self) -> dict:
        """Return total dispatched attempts, courier-refused count, and refusal rate."""
        total_dispatched = sum(self.total_orders_accepted.values())
        total_refused = getattr(self, '_total_courier_refused', 0)
        total_attempts = total_dispatched + total_refused
        rate = total_refused / total_attempts if total_attempts > 0 else 0.0
        return {'refused': total_refused, 'attempts': total_attempts, 'rate': rate}

    def step(self, realgo, wrapper_actions: Dict[str, List[Tuple]]) -> Tuple[Dict[str, Dict], Dict[str, bool], Dict[str, List[float]]]:
        # Track new orders that appeared in this step (before action execution)
        self._track_new_orders()

        # Transfer the action into the format for simulator
        action = self._wrapper_action_to_simulator(wrapper_actions)

        # Execute step in the simulator
        self.simulator_state_all = self.simulator.step(realgo, action)

        # Update revenues based on completed orders
        self._update_revenue()

        # Separate state by platform
        wrapper_state, wrapper_done = self.simulator_state_all

        self.done = wrapper_done

        return wrapper_state, wrapper_done

    # =============================================================================
    # PRIVATE HELPER METHODS
    # =============================================================================

    def _track_new_orders(self):
        # Use a set to track which orders we've already seen
        if not hasattr(self, '_seen_order_ids'):
            self._seen_order_ids = set()

        for platform in self.platforms:
            platform_orders = self.simulator.get_orders_by_platform(platform)

            for order in platform_orders:
                order_id = order.order_id
                if order_id not in self._seen_order_ids:
                    # This is a new order
                    self._seen_order_ids.add(order_id)
                    self.total_orders_appeared[platform] += 1

    # def _check_platform_done(self, platform: str) -> bool:
    #     platform_orders = self.simulator.get_orders_by_platform(platform)

    #     if not platform_orders:
    #         return True

    #     # Platform is done when all orders are either delivered or canceled
    #     for order in platform_orders:
    #         if not (order.is_completed() or order.is_canceled()):
    #             return False

    #     return True

    # def _separate_state_by_platform(self, state: SimulatorState):
    #     wrapper_state = {}
    #     for platform in self.platforms:
    #         # Get platform-specific couriers and orders
    #         platform_couriers = state.get_couriers_by_platform(platform)
    #         platform_orders = state.get_orders_by_platform(platform)
    #         platform_unassigned_order_ids = state.get_unassigned_order_ids_by_platform(platform)
    #         order_by_id = {order.order_id: order for order in platform_orders}
    #         Courier_by_id = {courier.courier_id: courier for courier in platform_couriers}

    #         # Create platform-specific SimulatorState
    #         platform_state = SimulatorState(
    #             courier_pool=platform_couriers,
    #             order_pool=platform_orders,
    #             time=state.time,
    #             unassigned_order_ids=platform_unassigned_order_ids,
    #             order_by_id=order_by_id,
    #             courier_by_id=Courier_by_id
    #         )
    #         wrapper_state[platform] = platform_state

    #     wrapper_state['global'] = state  # include the global state as well

    #     wrapper_done = {}
    #     for platform in self.platforms:
    #         wrapper_done[platform] = self._check_platform_done(platform)

    #     if all(wrapper_done.values()):
    #         self.done = True

    #     return wrapper_state, wrapper_done

    def _wrapper_action_to_simulator(self, platform_actions) -> Tuple[SimulatorAction, Dict]:
        # Step 1: Convert platform actions to courier-centered format sorted by incentive
        order_candidates_by_courier = self._platform_action_to_courier_center(platform_actions)

        # Step 2: Process each courier's decisions using courier decision engine
        accepted_orders_by_platform = defaultdict(list)
        rejected_orders = []  # Track rejected orders
        updated_routes_by_courier = {}

        for courier_id, order_candidates in order_candidates_by_courier.items():
            courier_state = self.simulator_state_all[0]['global'].get_courier_by_id(courier_id)

            # Use courier decision engine to process all order candidates
            new_route, order_decisions = self.courier_decision_engine.process_courier_order_decisions(
                courier_state=courier_state,
                order_candidates=order_candidates,
                simulator_state=self.simulator_state_all[0]['global'],
                get_order_func=self.simulator_state_all[0]['global'].get_order_by_id
            )

            # Process decisions for tracking and metrics
            for order_id, incentive, platform_id, is_accepted in order_decisions:
                if is_accepted:
                    # Track accepted orders for revenue calculation after delivery
                    order = self.simulator.get_order_by_id(order_id)
                    self.accepted_orders_tracking[order_id] = (platform_id, order.value, incentive)
                    accepted_orders_by_platform[platform_id].append((order_id, courier_id, incentive))
                    self.total_orders_accepted[platform_id] += 1
                else:
                    # Track rejected orders for rejection count
                    rejected_orders.append(order_id)
                    self._total_courier_refused += 1

            updated_routes_by_courier[courier_id] = new_route

        # Step 3: Generate SimulatorAction from accepted orders, rejected orders, and updated routes
        simulator_action = self._generate_simulator_action(
            accepted_orders_by_platform,
            updated_routes_by_courier,
            rejected_orders
        )

        return simulator_action

    def _platform_action_to_courier_center(self, platform_actions) -> Dict[str, List[Tuple]]:

        order_candidate_by_courier = defaultdict(list)

        # Convert platform actions to courier-centered format
        for platform_id, assignments in platform_actions.items():
            if not assignments:
                continue
            else:
                if self.config.get('fixed_incentive_ratio', True):
                    # assert assignments中每个元素的长度为2
                    assert all(len(a) == 2 for a in assignments), "When fixed_incentive_ratio is True, each assignment must be (courier_id, order_id)."
                    # Calculate incentive based on order value
                    for courier_id, order_id in assignments:
                        order = self.simulator.get_order_by_id(order_id)
                        incentive = order.value * self.incentive_ratio
                        order_candidate_by_courier[courier_id].append((order_id, incentive, platform_id))
                else:
                    # assert assignments中每个元素的长度为3
                    assert all(len(a) == 3 for a in assignments), "When fixed_incentive_ratio is False, each assignment must be (courier_id, order_id, incentive)."
                    for courier_id, order_id, incentive in assignments:
                        order_candidate_by_courier[courier_id].append((order_id, incentive, platform_id))

        # Sort by incentive (higher incentive has higher priority)
        for courier_id in order_candidate_by_courier:
            order_candidate_by_courier[courier_id].sort(key=lambda x: x[1], reverse=True)

        return dict(order_candidate_by_courier)

    def _generate_simulator_action(self, accepted_orders_by_platform: Dict[str, List[Tuple]],
                                    updated_routes_by_courier: Dict[str, List[Tuple[str, int, float, float]]], rejected_orders: List[str]) -> SimulatorAction:
        assign_orders = {}
        courier_actions = {}

        # Create order assignments from accepted orders
        for platform_id, accepted_list in accepted_orders_by_platform.items():
            for order_id, courier_id, incentive in accepted_list:
                assign_orders[order_id] = (courier_id, incentive)

        # Create courier route actions
        for courier_id, route_state in updated_routes_by_courier.items():
            # route_action = RouteAction(route=route_state.sequence)
            courier_actions[courier_id] = route_state

        return SimulatorAction(assign_orders=assign_orders, courier_actions=courier_actions, rejected_orders=rejected_orders)

    def _update_revenue(self) -> Dict[str, float]:
        # Check all tracked orders for completion
        completed_orders = []
        for order_id, (platform_id, order_value, incentive_paid) in self.accepted_orders_tracking.items():
            order = self.simulator.get_order_by_id(order_id)

            # Check if order was just delivered in this step
            if order.is_completed():
                # Update metrics: increment delivered count
                self.total_orders_delivered[platform_id] += 1

                # Check if delivery was on time
                is_on_time = order.is_delivered_on_time()

                if is_on_time:
                    # On-time: full revenue
                    revenue = order_value - incentive_paid
                else:
                    # Late: deduct compensation
                    compensation = self.late_delivery_compensation_ratio * order_value
                    revenue = order_value - incentive_paid - compensation

                    # Update metrics: increment late delivery count
                    self.total_orders_delivered_late[platform_id] += 1

                completed_orders.append(order_id)

                # Update platform revenue tracking
                self.revenues[platform_id] += revenue

        # Remove completed orders from tracking
        for order_id in completed_orders:
            del self.accepted_orders_tracking[order_id]



    # def _get_order_value_from_simulator(self, order_id: str) -> float:
    #     try:
    #         # Get current simulator state
    #         current_state = self.simulator.get_state()

    #         # Look for the order in the simulator's order pool
    #         for order in current_state.order_pool:
    #             if order.order_id == order_id:
    #                 return order.get_value() if hasattr(order, 'get_value') else getattr(order, 'value', 0.0)

    #         return 0.0
    #     except Exception:
    #         return 0.0
