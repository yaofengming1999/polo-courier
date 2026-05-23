"""Unified grid utilities for grid-based controllers.

Supports two modes:
1. Region-only mode (use_direction=False): Orders grouped by grid cell only
2. Region+Direction mode (use_direction=True): Orders grouped by (cell, direction)

Used by: TowerController, CoopController, ACBGMController, FairController
"""
import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Any

# Direction constants (0=CENTER, 1-6=E, NE, NW, W, SW, SE)
NUM_DIRECTIONS = 7
DIRECTION_NAMES = ['CENTER', 'E', 'NE', 'NW', 'W', 'SW', 'SE']


# ============================================================================
# Grid Creation
# ============================================================================

def create_hex_grid(min_lat: float, max_lat: float, min_lng: float, max_lng: float,
                    hex_size: float) -> Dict:
    """Create hexagonal grid covering geographic region.

    Args:
        min_lat, max_lat: Latitude bounds
        min_lng, max_lng: Longitude bounds
        hex_size: Size of hexagonal cells (in coordinate units)

    Returns:
        Grid dictionary with cells, bounds, and params
    """
    hex_width = math.sqrt(3) * hex_size
    hex_height = 2 * hex_size

    # Convert bounds to axial coordinates
    q1, r1 = _geo_to_axial(min_lng, min_lat, min_lat, min_lng, hex_width, hex_height)
    q2, r2 = _geo_to_axial(max_lng, max_lat, min_lat, min_lng, hex_width, hex_height)

    # Ensure min/max are in correct order (handles negative coordinates)
    q_min, q_max = min(q1, q2), max(q1, q2)
    r_min, r_max = min(r1, r2), max(r1, r2)

    # Generate all cells
    cells = [(q, r) for q in range(q_min, q_max + 1) for r in range(r_min, r_max + 1)]

    return {
        'cells': cells,
        'bounds': {'q_min': q_min, 'q_max': q_max, 'r_min': r_min, 'r_max': r_max},
        'params': {
            'min_lat': min_lat, 'max_lat': max_lat,
            'min_lng': min_lng, 'max_lng': max_lng,
            'hex_size': hex_size, 'hex_width': hex_width, 'hex_height': hex_height
        }
    }


def create_rect_grid(min_lat: float, max_lat: float, min_lng: float, max_lng: float,
                     grid_size: float) -> Dict:
    """Create rectangular grid covering geographic region.

    Args:
        min_lat, max_lat: Latitude bounds
        min_lng, max_lng: Longitude bounds
        grid_size: Size of each grid cell (in coordinate units)

    Returns:
        Grid dictionary with cells, bounds, and params
    """
    # Calculate grid dimensions
    grid_cols = max(1, int((max_lng - min_lng) / grid_size) + 1)
    grid_rows = max(1, int((max_lat - min_lat) / grid_size) + 1)

    # Generate all cells (row, col)
    cells = [(row, col) for row in range(grid_rows) for col in range(grid_cols)]

    return {
        'cells': cells,
        'bounds': {'row_min': 0, 'row_max': grid_rows - 1, 'col_min': 0, 'col_max': grid_cols - 1},
        'params': {
            'min_lat': min_lat, 'max_lat': max_lat,
            'min_lng': min_lng, 'max_lng': max_lng,
            'grid_size': grid_size, 'grid_rows': grid_rows, 'grid_cols': grid_cols
        },
        'grid_type': 'rect'
    }


# ============================================================================
# Cell Lookup
# ============================================================================

def get_cell(grid: Dict, lng: float, lat: float) -> Optional[Tuple[int, int]]:
    """Get cell containing the point.

    Args:
        grid: Grid dictionary from create_hex_grid or create_rect_grid
        lng: Longitude
        lat: Latitude

    Returns:
        (q, r) for hex grid or (row, col) for rect grid, or None if out of bounds
    """
    if grid.get('grid_type') == 'rect':
        return _get_rect_cell(grid, lng, lat)
    else:
        return _get_hex_cell(grid, lng, lat)


def _get_hex_cell(grid: Dict, lng: float, lat: float) -> Optional[Tuple[int, int]]:
    """Get hexagonal cell (q, r) containing the point."""
    params = grid['params']
    bounds = grid['bounds']

    q, r = _geo_to_axial(lng, lat, params['min_lat'], params['min_lng'],
                         params['hex_width'], params['hex_height'])

    # Clamp to bounds
    q = max(bounds['q_min'], min(bounds['q_max'], q))
    r = max(bounds['r_min'], min(bounds['r_max'], r))

    return (q, r) if (q, r) in grid['cells'] else None


def _get_rect_cell(grid: Dict, lng: float, lat: float) -> Optional[Tuple[int, int]]:
    """Get rectangular cell (row, col) containing the point."""
    params = grid['params']
    bounds = grid['bounds']

    col = int((lng - params['min_lng']) / params['grid_size'])
    row = int((lat - params['min_lat']) / params['grid_size'])

    # Clamp to bounds
    col = max(bounds['col_min'], min(bounds['col_max'], col))
    row = max(bounds['row_min'], min(bounds['row_max'], row))

    return (row, col)


def get_cell_index(grid: Dict, cell: Tuple[int, int]) -> int:
    """Convert cell coordinates to linear index.

    Args:
        grid: Grid dictionary
        cell: Cell coordinates

    Returns:
        Linear index (for use in neural network inputs)
    """
    if grid.get('grid_type') == 'rect':
        params = grid['params']
        row, col = cell
        return row * params['grid_cols'] + col
    else:
        bounds = grid['bounds']
        q, r = cell
        q_offset = q - bounds['q_min']
        r_offset = r - bounds['r_min']
        num_r = bounds['r_max'] - bounds['r_min'] + 1
        return q_offset * num_r + r_offset


def get_num_cells(grid: Dict) -> int:
    """Get total number of cells in grid."""
    return len(grid['cells'])


# ============================================================================
# Direction Calculation
# ============================================================================

def get_direction(lng1: float, lat1: float, lng2: float, lat2: float) -> int:
    """Determine direction from point 1 to point 2.

    Args:
        lng1, lat1: Origin point
        lng2, lat2: Destination point

    Returns:
        Direction index: 0=CENTER, 1=E, 2=NE, 3=NW, 4=W, 5=SW, 6=SE
    """
    dx = lng2 - lng1
    dy = lat2 - lat1

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0  # CENTER

    # Calculate angle (0° = East, counterclockwise)
    angle_deg = math.degrees(math.atan2(dy, dx))
    if angle_deg < 0:
        angle_deg += 360

    # Map to 6 directions (60° sectors)
    if angle_deg < 30 or angle_deg >= 330:
        return 1  # E
    elif angle_deg < 90:
        return 2  # NE
    elif angle_deg < 150:
        return 3  # NW
    elif angle_deg < 210:
        return 4  # W
    elif angle_deg < 270:
        return 5  # SW
    else:
        return 6  # SE


# ============================================================================
# Queue Management
# ============================================================================

def create_queue_manager(grid: Dict, platform_ids: List, use_direction: bool = True) -> Dict:
    """Create queue manager for tracking orders in each cell (optionally by direction).

    Args:
        grid: Grid dictionary
        platform_ids: List of platform IDs
        use_direction: If True, create queues for (cell, direction, platform).
                       If False, create queues for (cell, platform) only (direction=0).

    Returns:
        Queue manager dictionary
    """
    queues = {}
    directions = range(NUM_DIRECTIONS) if use_direction else [0]

    for cell in grid['cells']:
        for direction in directions:
            for platform in platform_ids:
                key = (cell[0], cell[1], direction, platform)
                queues[key] = {
                    'orders': [],
                    'cell': cell,
                    'direction': direction,
                    'platform': platform
                }

    return {
        'queues': queues,
        'grid': grid,
        'platforms': platform_ids,
        'use_direction': use_direction
    }


def add_order_to_queue(queue_mgr: Dict, order_dict: Dict) -> bool:
    """Add order to appropriate queue based on location, direction, and platform.

    Args:
        queue_mgr: Queue manager from create_queue_manager
        order_dict: Order dictionary with pickup_lng, pickup_lat, delivery_lng,
                    delivery_lat, platform, and other order fields

    Returns:
        True if order was added, False otherwise
    """
    grid = queue_mgr['grid']
    use_direction = queue_mgr['use_direction']

    # Get pickup cell
    cell = get_cell(grid, order_dict['pickup_lng'], order_dict['pickup_lat'])
    if cell is None:
        return False

    # Get direction (or 0 if not using direction)
    if use_direction:
        direction = get_direction(
            order_dict['pickup_lng'], order_dict['pickup_lat'],
            order_dict['delivery_lng'], order_dict['delivery_lat']
        )
    else:
        direction = 0

    # Get platform
    platform = order_dict['platform']

    # Add to queue
    key = (cell[0], cell[1], direction, platform)
    if key in queue_mgr['queues']:
        queue_mgr['queues'][key]['orders'].append(order_dict)
        return True
    return False


def get_queue_orders(queue_mgr: Dict, cell: Tuple[int, int], direction: int,
                     platform: Any) -> List[Dict]:
    """Get orders from specific queue.

    Args:
        queue_mgr: Queue manager
        cell: Cell coordinates
        direction: Direction index (0 if not using direction)
        platform: Platform ID

    Returns:
        List of order dictionaries
    """
    key = (cell[0], cell[1], direction, platform)
    return queue_mgr['queues'][key]['orders'] if key in queue_mgr['queues'] else []


def get_cell_orders(queue_mgr: Dict, cell: Tuple[int, int], platform: Any) -> List[Dict]:
    """Get all orders in a cell (across all directions) for a platform.

    Args:
        queue_mgr: Queue manager
        cell: Cell coordinates
        platform: Platform ID

    Returns:
        List of all order dictionaries in the cell
    """
    orders = []
    if queue_mgr['use_direction']:
        for direction in range(NUM_DIRECTIONS):
            orders.extend(get_queue_orders(queue_mgr, cell, direction, platform))
    else:
        orders.extend(get_queue_orders(queue_mgr, cell, 0, platform))
    return orders


def clear_all_queues(queue_mgr: Dict) -> None:
    """Clear all orders from all queues."""
    for queue_data in queue_mgr['queues'].values():
        queue_data['orders'] = []


def remove_orders_from_queues(queue_mgr: Dict, order_ids: List) -> None:
    """Remove specific orders from all queues by order_id."""
    order_id_set = set(order_ids)
    for queue_data in queue_mgr['queues'].values():
        queue_data['orders'] = [
            order for order in queue_data['orders']
            if order['order_id'] not in order_id_set
        ]


# ============================================================================
# Grid State Features (for neural network inputs)
# ============================================================================

def compute_demand_supply_grids(grid: Dict, orders: List, couriers: List,
                                platform_id: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute demand (order) and supply (courier) grids.

    Args:
        grid: Grid dictionary
        orders: List of order objects with pickup_location attribute
        couriers: List of courier objects with location attribute
        platform_id: If provided, only count orders for this platform

    Returns:
        (demand_grid, supply_grid) numpy arrays
    """
    num_cells = get_num_cells(grid)

    # Create index mapping for cells
    cell_to_idx = {cell: i for i, cell in enumerate(grid['cells'])}

    demand = np.zeros(num_cells, dtype=np.float32)
    supply = np.zeros(num_cells, dtype=np.float32)

    # Count orders (demand) by pickup location
    for order in orders:
        if platform_id is not None:
            order_platform = getattr(order, 'platform', None)
            if order_platform != platform_id:
                continue

        # Handle both object and dict formats
        if hasattr(order, 'pickup_location'):
            lng, lat = order.pickup_location[2], order.pickup_location[3]
        else:
            lng, lat = order['pickup_lng'], order['pickup_lat']

        cell = get_cell(grid, lng, lat)
        if cell and cell in cell_to_idx:
            demand[cell_to_idx[cell]] += 1

    # Count couriers (supply) by current location
    for courier in couriers:
        if hasattr(courier, 'location'):
            lng, lat = courier.location[0], courier.location[1]
        else:
            lng, lat = courier['lng'], courier['lat']

        cell = get_cell(grid, lng, lat)
        if cell and cell in cell_to_idx:
            supply[cell_to_idx[cell]] += 1

    return demand, supply


def get_local_demand_supply(grid: Dict, demand_grid: np.ndarray, supply_grid: np.ndarray,
                            lng: float, lat: float, radius: int = 1) -> Tuple[float, float]:
    """Get local demand and supply around a location.

    Args:
        grid: Grid dictionary
        demand_grid: Demand array from compute_demand_supply_grids
        supply_grid: Supply array from compute_demand_supply_grids
        lng, lat: Center location
        radius: Number of cells to consider in each direction

    Returns:
        (total_demand, total_supply) in local neighborhood
    """
    center_cell = get_cell(grid, lng, lat)
    if center_cell is None:
        return 0.0, 0.0

    cell_to_idx = {cell: i for i, cell in enumerate(grid['cells'])}

    total_demand = 0.0
    total_supply = 0.0

    # Get neighboring cells
    for cell in grid['cells']:
        if _cell_distance(center_cell, cell) <= radius:
            idx = cell_to_idx.get(cell)
            if idx is not None:
                total_demand += demand_grid[idx]
                total_supply += supply_grid[idx]

    return total_demand, total_supply


def _cell_distance(cell1: Tuple[int, int], cell2: Tuple[int, int]) -> int:
    """Calculate distance between two cells (Chebyshev distance for rect, axial for hex)."""
    return max(abs(cell1[0] - cell2[0]), abs(cell1[1] - cell2[1]))


# ============================================================================
# Hexagonal Distance (for Fair-style distance calculation)
# ============================================================================

def cal_hex_distance(ori_index: int, dest_index: int, grid_size: int = 10) -> int:
    """Calculate hexagon grid distance between two grid indices.

    Adapted from Fair algorithm for configurable grid size.

    Args:
        ori_index: Origin grid index (0 to grid_size^2 - 1)
        dest_index: Destination grid index
        grid_size: Size of the grid (default 10)

    Returns:
        Integer distance in grid units
    """
    ori_y = int(ori_index / grid_size)
    ori_x = ori_index % grid_size
    dest_y = int(dest_index / grid_size)
    dest_x = dest_index % grid_size

    dist_y = abs(ori_y - dest_y)
    dist_x = abs(ori_x - dest_x)

    if dist_y % 2 == 0:
        distance = dist_y if dist_y >= 2 * dist_x else dist_x + (1 / 2) * dist_y
    else:
        if dist_y >= (2 * dist_x + 1):
            distance = dist_y
        else:
            actual_dist_x = dest_x - ori_x
            if actual_dist_x > 0:  # Moving right
                if dest_y % 2 == 0:  # From odd to even row
                    distance = (1 / 2) * (dist_y - 1) + actual_dist_x
                else:
                    distance = (1 / 2) * (dist_y - 1) + actual_dist_x + 1
            else:  # Moving left
                if dest_y % 2 == 0:  # From odd to even row
                    distance = (1 / 2) * (dist_y - 1) + (-actual_dist_x) + 1
                else:
                    distance = (1 / 2) * (dist_y - 1) + (-actual_dist_x)

    return int(distance)


# ============================================================================
# Courier Pruning (for action space reduction)
# ============================================================================

def prune_couriers_per_platform(orders: List, couriers: List, k: int) -> Dict[Any, List[int]]:
    if len(couriers) == 0:
        return {}

    # If fewer couriers than k, return all courier indices for each order
    if len(couriers) <= k:
        all_indices = list(range(len(couriers)))
        return {order.order_id: all_indices for order in orders}

    # Build (C, 2) and (O, 2) location arrays once
    courier_locs = np.array([[c.location[0], c.location[1]] for c in couriers], dtype=np.float32)   # (C, 2)
    order_locs   = np.array([[o.pickup_location[2], o.pickup_location[3]] for o in orders], dtype=np.float32)  # (O, 2)

    # Broadcast: (O, C, 2) → squared distance matrix (O, C)
    diff = courier_locs[np.newaxis, :, :] - order_locs[:, np.newaxis, :]  # (O, C, 2)
    dist_matrix = diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2                # (O, C)

    # argpartition selects k nearest in O(C) per row (vs O(C log C) for argsort)
    # and is fully vectorised across all O rows — replaces the Python for-loop
    k_idx = np.argpartition(dist_matrix, k, axis=1)[:, :k]  # (O, k) unordered nearest

    return {order.order_id: k_idx[i].tolist() for i, order in enumerate(orders)}


def get_unique_courier_indices(order_courier_map: Dict[Any, List[int]]) -> List[int]:
    """Get unique courier indices across all orders.

    Useful when you need the set of all couriers that are candidates for any order.

    Args:
        order_courier_map: Dict from prune_couriers_per_order

    Returns:
        Sorted list of unique courier indices
    """
    all_indices = set()
    for indices in order_courier_map.values():
        all_indices.update(indices)
    return sorted(all_indices)


# ============================================================================
# Private Helper Functions
# ============================================================================

def _geo_to_axial(lng: float, lat: float, min_lat: float, min_lng: float,
                  hex_width: float, hex_height: float) -> Tuple[int, int]:
    """Convert geographic to axial hex coordinates."""
    x = (lng - min_lng) / hex_width
    y = (lat - min_lat) / hex_height
    q = int(round(x))
    r = int(round(y - x / 2))
    return q, r
