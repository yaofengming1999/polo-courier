import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Location:

    longtitude: float
    latitude: float

    def __post_init__(self):
        if not (-180 <= self.longtitude <= 180):
            raise ValueError(f"longtitude must be between -180 and 180, got {self.longtitude}")
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"Latitude must be between -90 and 90, got {self.latitude}")

    def to_tuple(self) -> Tuple[float, float]:
        return (self.longtitude, self.latitude)

    def distance_to(self, other: 'Location') -> float:
        if not isinstance(other, Location):
            raise TypeError("Distance calculation requires another Location object")
        # # Calculate distance in kilometers
        distance_km = math.sqrt((self.longtitude - other.longtitude)**2 + (self.latitude - other.latitude)**2)
        return distance_km

    def distance_to_meters(self, other: 'Location') -> float:
        return self.distance_to(other) * 1000  # Convert km to meters

    def __str__(self) -> str:
        return f"Location({self.longtitude:.4f}, {self.latitude:.4f})"

    def is_equal_within_tolerance(self, other: 'Location', tolerance: float = 1e-6) -> bool:
        if not isinstance(other, Location):
            return False
        return (abs(self.longtitude - other.longtitude) < tolerance and
                abs(self.latitude - other.latitude) < tolerance)



@dataclass(frozen=True)
class OrderLocation(Location):

    order_id: str
    location_type: int


    def __post_init__(self):
        # First validate parent Location coordinates
        super().__post_init__()

        if not isinstance(self.order_id, str):
            raise TypeError("order_id must be a string")
        if self.location_type not in (0, 1):
            raise ValueError(f"location_type must be 0 (pickup) or 1 (delivery), got {self.location_type}")

    def is_pickup(self) -> bool:
        return self.location_type == 0

    def is_delivery(self) -> bool:
        return self.location_type == 1

    # def get_location_type_name(self) -> str:
    #     return "pickup" if self.is_pickup() else "delivery"

    def is_equal_within_tolerance(self, other: 'OrderLocation', tolerance: float = 1e-6) -> bool:
        if not isinstance(other, OrderLocation):
            return False
        return (self.order_id == other.order_id and
                self.location_type == other.location_type and
                super().is_equal_within_tolerance(other, tolerance))

    def to_tuple(self) -> Tuple[str, int, float, float]:
        return (self.order_id, self.location_type, self.longtitude, self.latitude)

    def __str__(self) -> str:
        # type_name = self.get_location_type_name()
        return f"OrderLocation({self.order_id}, {self.location_type}, {self.longtitude:.4f}, {self.latitude:.4f})"


# Location type constants for better code readability
OrderLocation.PICKUP = 0
OrderLocation.DELIVERY = 1
