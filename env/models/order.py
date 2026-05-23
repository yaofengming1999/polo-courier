from typing import Optional, Tuple, Union
from .location import Location, OrderLocation


class Order:

    # Status constants for better code readability
    UNRELEASED = 0
    UNASSIGNED = 1
    ASSIGNED = 2
    DELIVERING = 3
    DELIVERED = 4
    CANCELED = 5

    STATUS_NAMES = {
        UNRELEASED: "unreleased",
        UNASSIGNED: "unassigned",
        ASSIGNED: "assigned",
        DELIVERING: "delivering",
        DELIVERED: "delivered",
        CANCELED: "canceled"
    }

    def __init__(self,
                 order_id: str,
                 pickup_location: Union[Tuple[float, float], OrderLocation],
                 delivery_location: Union[Tuple[float, float], OrderLocation],
                 create_time: int,
                 delivery_patience: int,
                 platform: str,
                 num_rejections: int = 0,
                 value: float = 0.0):
        if not isinstance(order_id, str):
            raise TypeError("order_id must be a string")
        if not isinstance(create_time, int):
            raise TypeError("create_time must be an integer")
        if create_time < 0:
            raise ValueError("create_time must be non-negative")
        if not isinstance(platform, str):
            raise TypeError("platform must be a string")
        if not isinstance(value, (int, float)):
            raise TypeError("value must be a number")
        if value < 0:
            raise ValueError("value must be non-negative")

        self.order_id = order_id

        # Handle location inputs - convert tuples to OrderLocation if needed
        if isinstance(pickup_location, tuple):
            self.pickup_location = OrderLocation(
                order_id=order_id,
                location_type=OrderLocation.PICKUP,
                longtitude=pickup_location[0],
                latitude=pickup_location[1]
            )
        else:
            self.pickup_location = pickup_location

        if isinstance(delivery_location, tuple):
            self.delivery_location = OrderLocation(
                order_id=order_id,
                location_type=OrderLocation.DELIVERY,
                longtitude=delivery_location[0],
                latitude=delivery_location[1]
            )
        else:
            self.delivery_location = delivery_location

        self.create_time = create_time
        self.delivery_patience = delivery_patience
        self.platform = platform
        self.value = float(value)
        self.num_rejections = num_rejections  # Track number of times rejected by couriers

        # Initialize status tracking
        self.status = self.UNRELEASED
        self.assigned_courier_id: Optional[str] = None
        self.picked_time: Optional[int] = None
        self.delivered_time: Optional[int] = None
        self.response_time: Optional[int] = None  # Time when successfully assigned

        # Version tracking for caching optimization
        self._version: int = 0

    @property
    def version(self) -> int:
        return self._version

    def bump_version(self) -> None:
        self._version += 1

    def get_status_name(self) -> str:
        return self.STATUS_NAMES.get(self.status, f"unknown_status_{self.status}")

    def is_assigned(self) -> bool:
        return self.status >= self.ASSIGNED

    def is_completed(self) -> bool:
        return self.status == self.DELIVERED

    def mark_picked_up(self, pickup_time: int) -> None:
        if self.status != self.ASSIGNED:
            raise ValueError(f"Order {self.order_id} cannot be picked up (status: {self.get_status_name()})")

        self.status = self.DELIVERING
        self.picked_time = pickup_time
        self.bump_version()

    def mark_delivered(self, delivery_time: int) -> None:
        if self.status != self.DELIVERING:
            raise ValueError(f"Order {self.order_id} cannot be delivered (status: {self.get_status_name()})")

        self.status = self.DELIVERED
        self.delivered_time = delivery_time
        self.bump_version()

    def release_for_assignment(self) -> None:
        if self.status == self.UNRELEASED:
            self.status = self.UNASSIGNED
            self.bump_version()

    def cancel_order(self, cancel_time: int) -> None:
        if self.status != self.UNASSIGNED:
            raise ValueError(f"Order {self.order_id} cannot be canceled (status: {self.get_status_name()})")

        self.status = self.CANCELED
        self.bump_version()

    def is_canceled(self) -> bool:
        return self.status == self.CANCELED

    def increment_rejections(self) -> None:
        self.num_rejections += 1
        self.bump_version()

    def is_delivered_on_time(self) -> bool:
        if self.status != self.DELIVERED or self.delivered_time is None:
            return False

        # Delivery deadline = create_time + delivery_patience
        delivery_deadline = self.create_time + self.delivery_patience
        return self.delivered_time <= delivery_deadline

    def get_total_distance(self) -> float:
        return self.pickup_location.distance_to(self.delivery_location)

    def get_delivery_duration(self) -> Optional[int]:
        if self.picked_time is not None and self.delivered_time is not None:
            return self.delivered_time - self.picked_time
        return None
    def is_terminal(self) -> bool:
        return self.status in {self.DELIVERED, self.CANCELED}
    def get_value(self) -> float:
        return self.value

    def get_value_per_distance(self) -> float:
        distance = self.get_total_distance()
        if distance > 0:
            return self.value / distance
        return 0.0

    def __str__(self) -> str:
        return f"Order({self.order_id}, {self.get_status_name()}, {self.platform})"

    def __repr__(self) -> str:
        return (f"Order(order_id='{self.order_id}', status={self.status}, "
                f"platform='{self.platform}', create_time={self.create_time})")

