import math
from typing import List, Tuple
from env.models.state_action import OrderState


def greedy_route_insertion_by_order(
    current_location,
    current_route: List[Tuple[str, int, float, float]],
    current_arrive_time: List[float],
    order: OrderState,
    speed: float,
    *,
    skip_arrive_time: bool = False,
) -> Tuple[List[Tuple[str, int, float, float]], float, List[float]]:
    """Insert an order (pickup then delivery) into the route using greedy insertion.

    When skip_arrive_time=True the returned arrive_time list is None — use this
    when only the insertion cost (add_distance) is needed, e.g. for pair features.
    """
    pickup_pos, pickup_delta, new_route, new_arrive_time = greedy_route_insertion_by_location(
        current_location, current_route, current_arrive_time,
        order.pickup_location, None, speed, skip_arrive_time=skip_arrive_time
    )
    _, delivery_delta, new_route, new_arrive_time = greedy_route_insertion_by_location(
        current_location, new_route, new_arrive_time,
        order.delivery_location, pickup_pos, speed, skip_arrive_time=skip_arrive_time
    )
    return new_route, pickup_delta + delivery_delta, new_arrive_time


def greedy_route_insertion_by_location(
    current_location,
    current_route: List[Tuple[str, int, float, float]],
    current_arrive_time: List[float],
    new_location,
    pickup_pos,
    speed: float,
    *,
    skip_arrive_time: bool = False,
) -> Tuple[int, float, List[Tuple[str, int, float, float]], List[float]]:
    """Insert a single location into the cheapest position in the route."""
    seq = current_route
    new_order_id, new_type, new_lng, new_lat = new_location

    if not seq:
        dx = new_lng - current_location[0]
        dy = new_lat - current_location[1]
        d = math.sqrt(dx * dx + dy * dy)
        return 0, d * d, [new_location], [d / speed]

    L = len(seq)

    # Find existing delivery stop (only relevant when inserting pickup, new_type==0)
    delivery_position = None
    if new_type == 0:
        for idx, loc in enumerate(seq):
            if loc[0] == new_order_id and loc[1] == 1:
                delivery_position = idx
                break

    # Pre-compute valid insertion range — avoids per-iteration branch checks
    lo = (pickup_pos + 1) if (new_type == 1 and pickup_pos is not None) else 0
    hi = delivery_position if (new_type == 0 and delivery_position is not None) else L

    best_pos = lo
    best_delta = float("inf")

    # Iterate only valid positions; access seq tuples directly (no coords listcomp)
    for i in range(lo, hi + 1):
        if i == 0:
            s = seq[0]
            delta = (new_lng - s[2]) ** 2 + (new_lat - s[3]) ** 2
        elif i == L:
            s = seq[L - 1]
            delta = (s[2] - new_lng) ** 2 + (s[3] - new_lat) ** 2
        else:
            p = seq[i - 1]; n = seq[i]
            px = p[2]; py = p[3]; nx = n[2]; ny = n[3]
            dpx = new_lng - px; dpy = new_lat - py
            dnx = nx - new_lng; dny = ny - new_lat
            delta = (dpx*dpx + dpy*dpy
                     + dnx*dnx + dny*dny
                     - (nx - px)**2 - (ny - py)**2)

        if delta < best_delta:
            best_delta = delta
            best_pos = i

    # Slice-concat avoids list copy + O(L) element shift from list.insert
    new_seq = seq[:best_pos] + [new_location] + seq[best_pos:]

    if skip_arrive_time:
        return best_pos, best_delta, new_seq, None

    new_arrive_time = _partial_arrive_times(
        current_location, new_seq, current_arrive_time, best_pos, speed
    )

    return best_pos, best_delta, new_seq, new_arrive_time


def _partial_arrive_times(
    current_location,
    route: List[Tuple[str, int, float, float]],
    old_arrive_time: List[float],
    insert_pos: int,
    speed: float
) -> List[float]:
    """
    Recompute arrival times only from insert_pos onward; reuse earlier values unchanged.
    old_arrive_time has len = len(route) - 1. Returns list of len = len(route).
    """
    L = len(route)
    new_times = list(old_arrive_time[:insert_pos])

    # Carry forward previous position and cumulative time to avoid back-indexing
    if insert_pos == 0:
        px, py = current_location[0], current_location[1]
        t = 0.0
    else:
        prev = route[insert_pos - 1]
        px, py = prev[2], prev[3]
        t = old_arrive_time[insert_pos - 1]

    inv_speed = 1.0 / speed

    for i in range(insert_pos, L):
        loc = route[i]
        cx, cy = loc[2], loc[3]
        dx = cx - px; dy = cy - py
        # Inline sqrt — eliminates _dist function-call overhead (was 0.627s / 4.4M calls)
        t += math.sqrt(dx * dx + dy * dy) * inv_speed
        new_times.append(t)
        px, py = cx, cy

    return new_times


def calculate_route_distance(route: List[Tuple[str, int, float, float]]) -> float:
    if len(route) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(route)):
        p = route[i - 1]; c = route[i]
        dx = c[2] - p[2]; dy = c[3] - p[3]
        total += math.sqrt(dx * dx + dy * dy)
    return total
