from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
import json
from .location import OrderLocation, Location
if TYPE_CHECKING:
    from ..simulator import Simulator
    from .courier import Courier
    from .order import Order


@dataclass(frozen=True)
class CourierState:

    courier_id: str
    location: Tuple[float, float]  # (longtitude, latitude)
    platforms: List[str]
    route: List[Tuple[str, int, float, float]]  # Only one unified route
    arrive_time: List[float] 
    trajectory: List[Tuple[float, float]]
    total_income: Dict[str, float]

    direction: Tuple[float, float] = (0.0, 0.0)
    speed: float = 0.0
    version: int = 0

    @staticmethod
    def from_courier(courier: 'Courier') -> 'CourierState':
        return CourierState(
            courier_id=courier.courier_id,
            location=courier.current_location.to_tuple(),
            platforms=courier.platforms,
            route=[loc.to_tuple() for loc in courier.route],
            arrive_time=[t for t in courier.arrive_time],
            trajectory=[loc.to_tuple() for loc in courier.trajectory],
            direction=tuple(courier.direction),
            speed=courier.speed,
            version=courier.version,
            total_income={platform: income for platform, income in courier.total_income.items()}
        )

    def get_current_location(self) -> Location:
        return Location(self.location[0], self.location[1])

@dataclass(frozen=True)
class OrderState:
    order_id: str
    pickup_location: Tuple[str, int, float, float]
    delivery_location: Tuple[str, int, float, float]
    platform: str
    create_time: int
    delivery_patience: int
    status: int
    assigned_courier_id: Optional[str]  # Str or None
    value: float
    num_rejections: int = 0  # Number of times rejected by couriers
    delivered_time: Optional[int] = None
    response_time: Optional[int] = None  # Time taken to assign the order
    version: int = 0  # Version for caching optimization

    @staticmethod
    def from_order(order: 'Order') -> 'OrderState':
        return OrderState(
            order_id=order.order_id,
            pickup_location=order.pickup_location.to_tuple(),
            delivery_location=order.delivery_location.to_tuple(),
            platform=order.platform,
            create_time=order.create_time,
            delivery_patience=order.delivery_patience,
            status=order.status,
            assigned_courier_id=order.assigned_courier_id,
            value=order.value,
            num_rejections=order.num_rejections,
            delivered_time=order.delivered_time,
            response_time=order.response_time,
            version=order.version
        )

    def get_pickup_location(self) -> OrderLocation:
        return OrderLocation(
            self.pickup_location[2], self.pickup_location[3],  # lon, lat
            self.pickup_location[0], self.pickup_location[1]   # order_id, location_type
        )

    def get_delivery_location(self) -> OrderLocation:
        return OrderLocation(
            self.delivery_location[2], self.delivery_location[3],  # lon, lat
            self.delivery_location[0], self.delivery_location[1]   # order_id, location_type
        )

    def is_unassigned(self) -> bool:
        return self.status == 1



@dataclass(frozen=True)
class SimulatorState:

    courier_pool: List[CourierState]
    order_pool: List[OrderState]
    time: int
    unassigned_order_ids: List[str]
    order_by_id: Dict[str, OrderState] = field(default_factory=dict, repr=False)
    courier_by_id: Dict[str, CourierState] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_simulator(sim: 'Simulator') -> 'SimulatorState':
        # Build order_pool with caching
        order_pool = []
        order_by_id = {}
        order_by_platform = {}

        for order_id in sim.unassigned_order_ids:
            order = sim.order_by_id[order_id]
            cached = sim._order_state_cache.get(order_id)
            if cached is not None and cached[0] == order.version:
                order_state = cached[1]
            else:
                order_state = OrderState.from_order(order)
                sim._order_state_cache[order_id] = (order.version, order_state)

            order_pool.append(order_state)
            if order_state.platform not in order_by_platform:
                order_by_platform[order_state.platform] = []
            
            order_by_platform[order_state.platform].append(order_state)
            order_by_id[order_id] = order_state

        courier_pool = []
        courier_by_id = {}
        for courier in sim.courier_pool:
            courier_id = courier.courier_id
            cached = sim._courier_state_cache.get(courier_id)

            if cached is not None and cached[0] == courier.version:
                courier_state = cached[1]
            else:
                courier_state = CourierState.from_courier(courier)
                sim._courier_state_cache[courier_id] = (courier.version, courier_state)

            courier_pool.append(courier_state)
            courier_by_id[courier_id] = courier_state

        courier_pool_by_platform = {}
        for platform in sim.get_platforms():
            if platform not in sim._courier_state_by_platform_cache:
                sim._courier_state_by_platform_cache[platform] = {}
            for courier in sim.courier_pool:
                courier_id = courier.courier_id
                cached = sim._courier_state_by_platform_cache[platform].get(courier_id)

                if cached is not None and cached[0] == courier.version:
                    courier_state = cached[1]
                else:
                    # Filter courier state for this platform
                    filter_route = [loc.to_tuple() for loc in courier.route
                                    if sim.order_platform_map.get(loc.order_id) == platform]
                    filter_arrive_time = [courier.arrive_time[i] for i, loc in enumerate(courier.route)
                                            if sim.order_platform_map.get(loc.order_id) == platform]
                    courier_state = CourierState(
                        courier_id=courier.courier_id,
                        location=courier.current_location.to_tuple(),
                        platforms=courier.platforms,
                        route=filter_route,
                        arrive_time=filter_arrive_time,
                        trajectory=[loc.to_tuple() for loc in courier.trajectory],
                        direction=tuple(courier.direction),
                        speed=courier.speed,
                        version=courier.version,
                        total_income={platform: courier.total_income.get(platform, 0.0)}
                    )
                    sim._courier_state_by_platform_cache[platform][courier_id] = (courier.version, courier_state)
            courier_pool_by_platform[platform] = list(sim._courier_state_by_platform_cache[platform].values())
                

        global_simulator_state = SimulatorState(
            courier_pool=courier_pool,
            order_pool=order_pool,
            time=sim.time,
            order_by_id=order_by_id,
            courier_by_id=courier_by_id,
            unassigned_order_ids=list(sim.unassigned_order_ids)
        )

        all_state = {}
        for platform in sim.get_platforms():
            platform_couriers = [entry[1] for entry in courier_pool_by_platform[platform]]
            platform_orders = order_by_platform.get(platform, [])
            platform_state = SimulatorState(
                courier_pool=platform_couriers,
                order_pool=platform_orders,
                time=sim.time,
                order_by_id={order.order_id: order for order in platform_orders},
                courier_by_id={courier.courier_id: courier for courier in platform_couriers},
                unassigned_order_ids=[order_id for order_id in sim.unassigned_order_ids
                                    if order_by_id[order_id].platform == platform]
            )
            all_state[platform] = platform_state
        all_state['global'] = global_simulator_state

        return all_state, sim.done
 

    def get_unassigned_orders(self) -> List[OrderState]:
        return [self.order_by_id[order_id] for order_id in self.unassigned_order_ids]

    def get_orders_by_platform(self, platform: str) -> List[OrderState]:
        return [order for order in self.order_pool if order.platform == platform]

    def get_unassigned_order_ids_by_platform(self, platform: str) -> List[str]:
        return [order_id for order_id in self.unassigned_order_ids
                if self.order_by_id[order_id].platform == platform]


    def get_courier_by_id(self, courier_id: str) -> CourierState:
        return self.courier_by_id.get(courier_id)

    def get_order_by_id(self, order_id: str) -> Optional[OrderState]:
        return self.order_by_id.get(order_id)


@dataclass(frozen=True)
class SimulatorAction:

    assign_orders: Dict[str, Tuple[str, float]]  # order_id -> (courier_id, pricing)
    courier_actions: Dict[str, Tuple[List[Tuple[str, int, float, float]], List[float]]]  # courier_id -> RouteAction (simplified from CourierAction)
    rejected_orders: List[str] = None  # order_ids that were rejected

    def apply_to_simulator(self, sim: 'Simulator', current_time: int) -> None:
        # 1. 分配订单, 更改order_stream 中的订单状态
        for order_id, (courier_id, pricing) in self.assign_orders.items():
            order = sim.order_by_id.get(order_id)
            courier = sim.courier_by_id.get(courier_id)
            if order and courier:
                order.status = 2  # assigned
                order.assigned_courier_id = courier_id
                order.response_time = current_time - order.create_time  # 记录响应时间
                order.bump_version()  # Increment version after state change
                sim.unassigned_order_ids.discard(order_id)  # Remove from unassigned set
                courier.add_income(order.platform, pricing)

        # 1.5. 处理被拒绝的订单 - 增加拒绝计数
        if self.rejected_orders:
            for order_id in self.rejected_orders:
                order = sim.order_by_id.get(order_id)
                if order and order.status == 1:  # Only count rejections for unassigned orders
                    order.increment_rejections()

        # 2. 更新每个快递员的 route 和状态
        for courier_id, route_action in self.courier_actions.items():
            courier = sim.courier_by_id.get(courier_id)
            if not courier:
                continue
            if len(route_action) == 0:
                continue

            # 更新 route directly
            courier.route = [OrderLocation(lon, lat, order_id, location_type)
                            for order_id, location_type, lon, lat in route_action[0]]
            
            courier.arrive_time = [arrive_t for arrive_t in route_action[1]]  # 更新预计到达时间列表
            # Increment version after route update
            courier.bump_version()  

