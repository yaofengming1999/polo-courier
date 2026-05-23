"""Agent Registry - Unified spatial grid, queue, and agent ID management.

Supports two modes:
1. Region-only mode (use_direction=False): One agent per (cell, platform)
2. Region+Direction mode (use_direction=True): One agent per (cell, direction, platform)

Used by: TowerController, CoopController, ACBGMController, FairController
"""
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from algorithms.utils import grid_utils as grid
import torch


class GridManager:
    """Registry for managing spatial grid, queues, and agent IDs.

    Responsibilities:
    1. Grid initialization and management (hex or rect)
    2. Order queue management (by cell or cell+direction)
    3. Agent ID mapping: (cell, direction, platform) → unique ID
    4. Query agents with non-empty queues
    5. Demand/supply tracking for grid cells
    """

    # City scale to bounds mapping (in km)
    # Format: (min_x, max_x, min_y, max_y)
    CITY_BOUNDS = {
        'small': (0.0, 3.29, 0.0, 3.74),
        'middle': (0.0, 4.86, 0.0, 4.49),
        'large': (0.0, 13.47, 0.0, 13.02),
    }

    def __init__(self, city_scale: str, hex_size: float = 1.0, grid_size: float = 0.001,
                 max_agents: int = 500, use_direction: bool = True,
                 grid_type: str = 'hex', platforms: Optional[List] = None):
        """Initialize agent registry with grid based on city scale.

        Args:
            city_scale: City size ('small', 'middle', 'large')
            hex_size: Size of hexagonal grid cells (for hex grid)
            grid_size: Size of rectangular grid cells (for rect grid)
            max_agents: Maximum number of agents to track
            use_direction: If True, agents are (cell, direction, platform).
                          If False, agents are (cell, platform) only.
            grid_type: 'hex' for hexagonal grid, 'rect' for rectangular grid
            platforms: List of platform IDs (default: empty list)
        """
        self.city_scale = city_scale
        self.hex_size = hex_size
        self.grid_size = grid_size
        self.max_agents = max_agents
        self.use_direction = use_direction
        self.grid_type = grid_type
        self.platforms = platforms if platforms is not None else []

        # Agent ID mapping
        self.agent_id_map: Dict[Tuple, int] = {}  # (cell_x, cell_y, direction, platform) -> int
        self.next_agent_id = 0

        # Demand/supply tracking (combined across all platforms)
        self.demand_grid: Optional[np.ndarray] = None
        self.supply_grid: Optional[np.ndarray] = None

        # Per-platform demand/supply tracking
        self.platform_demand_grids: Dict[Any, np.ndarray] = {}
        self.platform_supply_grids: Dict[Any, np.ndarray] = {}

        # Initialize grid from city scale
        self._initialize_grid_from_city_scale()

    # ===========================
    # Grid Management
    # ===========================

    def _initialize_grid_from_city_scale(self) -> None:
        """Initialize grid based on city_scale bounds."""
        if self.city_scale not in self.CITY_BOUNDS:
            raise ValueError(f"Unknown city_scale: {self.city_scale}. "
                           f"Must be one of {list(self.CITY_BOUNDS.keys())}")

        min_x, max_x, min_y, max_y = self.CITY_BOUNDS[self.city_scale]

        # Create grid based on type
        if self.grid_type == 'rect':
            self.grid = grid.create_rect_grid(min_y, max_y, min_x, max_x, self.grid_size)
        else:
            self.grid = grid.create_hex_grid(min_y, max_y, min_x, max_x, self.hex_size)

        # Create queue manager
        self.queue_mgr = grid.create_queue_manager(self.grid, self.platforms,
                                                   use_direction=self.use_direction)

        # Initialize demand/supply grids
        num_cells = grid.get_num_cells(self.grid)
        self.demand_grid = np.zeros(num_cells, dtype=np.float32)
        self.supply_grid = np.zeros(num_cells, dtype=np.float32)

        # Initialize per-platform grids
        self.platform_demand_grids = {
            plat_id: np.zeros(num_cells, dtype=np.float32) for plat_id in self.platforms
        }
        self.platform_supply_grids = {
            plat_id: np.zeros(num_cells, dtype=np.float32) for plat_id in self.platforms
        }


    def add_orders_to_queues(self, orders: List, current_time: float) -> None:
        """Add orders to direction queues.

        Args:
            orders: List of order objects
            current_time: Current simulation time
        """
        if not self.queue_mgr:
            return

        for order in orders:
            deadline = order.delivery_patience - (current_time - order.create_time)
            if deadline <= 0:
                continue

            pickup_lng = order.pickup_location[2]
            pickup_lat = order.pickup_location[3]
            delivery_lng = order.delivery_location[2]
            delivery_lat = order.delivery_location[3]

            # Get cell and direction
            cell = grid.get_cell(self.grid, pickup_lng, pickup_lat)
            if cell is None:
                continue

            direction = grid.get_direction(pickup_lng, pickup_lat, delivery_lng, delivery_lat) if self.use_direction else 0

            # Build queue key and add order
            key = (cell[0], cell[1], direction, order.platform)
            if key not in self.queue_mgr['queues']:
                continue

            self.queue_mgr['queues'][key]['orders'].append({
                'order_id': order.order_id,
                'pickup_lng': pickup_lng,
                'pickup_lat': pickup_lat,
                'delivery_lng': delivery_lng,
                'delivery_lat': delivery_lat,
                'deadline': deadline,
                'reward': order.value,
                'platform': order.platform,
                'queued_time': current_time,
                'create_time': order.create_time
            })

    def get_all_agents_data(self, sort_by: str = 'reward', descending: bool = True) -> List[Dict]:
        """Get all agents with non-empty queues across all platforms.

        Args:
            sort_by: Key to sort orders by ('reward', 'deadline', 'create_time')
            descending: Sort in descending order if True

        Returns:
            List of agent dicts with platform_id, cell, direction, sorted_orders
        """
        if not self.grid or not self.queue_mgr:
            return []

        all_agents_data = []

        for platform_id in self.platforms:
            if self.use_direction:
                for cell in self.grid['cells']:
                    for direction in range(grid.NUM_DIRECTIONS):
                        orders = grid.get_queue_orders(self.queue_mgr, cell, direction, platform_id)
                        if orders:
                            sorted_orders = sorted(orders, key=lambda o: o[sort_by], reverse=descending)
                            all_agents_data.append({
                                'platform_id': platform_id,
                                'cell': cell,
                                'direction': direction,
                                'sorted_orders': sorted_orders,
                            })
            else:
                for cell in self.grid['cells']:
                    orders = grid.get_cell_orders(self.queue_mgr, cell, platform_id)
                    if orders:
                        sorted_orders = sorted(orders, key=lambda o: o[sort_by], reverse=descending)
                        all_agents_data.append({
                            'platform_id': platform_id,
                            'cell': cell,
                            'direction': 0,
                            'sorted_orders': sorted_orders,
                        })

        return all_agents_data

    def get_cell_agents(self, cell: Tuple[int, int], platform_id: Any) -> List[Dict]:
        """Get all agents in a specific cell for a platform.

        Args:
            cell: Cell coordinates
            platform_id: Platform ID

        Returns:
            List of agent dicts (one per direction if use_direction=True)
        """
        if not self.queue_mgr:
            return []

        agents = []
        directions = range(grid.NUM_DIRECTIONS) if self.use_direction else [0]

        for direction in directions:
            orders = grid.get_queue_orders(self.queue_mgr, cell, direction, platform_id)
            if orders:
                agents.append({
                    'cell': cell,
                    'direction': direction,
                    'platform': platform_id,
                    'orders': orders
                })

        return agents

    def clear_platform_queues(self, platform_id: Any) -> None:
        """Clear all queues for a specific platform."""
        if not self.queue_mgr or not self.grid:
            return

        directions = range(grid.NUM_DIRECTIONS) if self.use_direction else [0]

        for cell in self.grid['cells']:
            for direction in directions:
                key = (cell[0], cell[1], direction, platform_id)
                if key in self.queue_mgr['queues']:
                    self.queue_mgr['queues'][key]['orders'] = []

    def clear_all_queues(self) -> None:
        """Clear all queues across all platforms."""
        if self.queue_mgr:
            grid.clear_all_queues(self.queue_mgr)


    def get_agent_id(self, cell: Tuple[int, int], direction: int, platform: Any) -> int:
        """Get or create agent ID for (cell, direction, platform).

        Args:
            cell: (x, y) tuple
            direction: Direction index (0 if not using direction)
            platform: Platform ID

        Returns:
            agent_id: Unique integer ID

        Raises:
            ValueError: If max_agents exceeded
        """
        # Normalize direction to 0 if not using direction
        if not self.use_direction:
            direction = 0

        key = (cell[0], cell[1], direction, platform)

        if key not in self.agent_id_map:
            if self.next_agent_id >= self.max_agents:
                raise ValueError(f"Exceeded max_agents={self.max_agents}")
            self.agent_id_map[key] = self.next_agent_id
            self.next_agent_id += 1

        return self.agent_id_map[key]

    def get_num_agents(self) -> int:
        """Get current number of registered agents."""
        return self.next_agent_id

    # ===========================
    # Demand/Supply Tracking
    # ===========================

    def update_demand_supply(self, orders: List, couriers: List,
                             platform_id: Optional[Any] = None) -> None:
        """Update demand (order) and supply (courier) grids.

        Args:
            orders: List of order objects
            couriers: List of courier objects
            platform_id: If provided, only count orders for this platform
        """
        if not self.grid:
            return

        # Update combined grids
        self.demand_grid, self.supply_grid = grid.compute_demand_supply_grids(
            self.grid, orders, couriers, platform_id
        )

        # Update per-platform grids
        for plat_id in self.platforms:
            plat_demand, plat_supply = grid.compute_demand_supply_grids(
                self.grid, orders, couriers, plat_id
            )
            self.platform_demand_grids[plat_id] = plat_demand
            self.platform_supply_grids[plat_id] = plat_supply

    def get_demand_supply_at(self, lng: float, lat: float) -> Tuple[float, float]:
        """Get demand and supply at a location.

        Returns:
            (demand, supply) at the cell containing the location
        """
        if not self.grid or self.demand_grid is None:
            return 0.0, 0.0

        cell = grid.get_cell(self.grid, lng, lat)
        if cell is None:
            return 0.0, 0.0

        idx = grid.get_cell_index(self.grid, cell)
        return self.demand_grid[idx], self.supply_grid[idx]

    def get_local_demand_supply(self, lng: float, lat: float, radius: int = 1) -> Tuple[float, float]:
        """Get local demand and supply around a location.

        Args:
            lng, lat: Center location
            radius: Number of cells to consider

        Returns:
            (total_demand, total_supply) in neighborhood
        """
        if not self.grid or self.demand_grid is None:
            return 0.0, 0.0

        return grid.get_local_demand_supply(
            self.grid, self.demand_grid, self.supply_grid, lng, lat, radius
        )

    def get_demand_supply_score(self, lng: float, lat: float) -> float:
        """Calculate demand-supply score (ratio) at a location."""
        demand, supply = self.get_local_demand_supply(lng, lat)
        if supply == 0:
            return demand if demand > 0 else 0.0
        return demand / (supply + 1e-6)

    def get_demand_supply_vector(self, normalize: bool = False,
                                   platform_separate: bool = True) -> Union[torch.Tensor, Dict[Any, torch.Tensor]]:
        """Get flattened demand-supply state vector.

        Args:
            normalize: If True, normalize by max value
            platform_separate: If True (default), return dict with per-platform vectors

        Returns:
            If platform_separate=True: Dict mapping platform_id to [demand_grid, supply_grid] tensor
            If platform_separate=False: Concatenated [demand_grid, supply_grid] as 1D tensor
        """
        if platform_separate:
            result = {}
            for plat_id in self.platforms:
                plat_demand = self.platform_demand_grids.get(plat_id)
                plat_supply = self.platform_supply_grids.get(plat_id)

                if plat_demand is None or plat_supply is None:
                    num_cells = grid.get_num_cells(self.grid) if self.grid else 1
                    result[plat_id] = torch.zeros(num_cells * 2, dtype=torch.float32)
                    continue

                vector = np.concatenate([plat_demand, plat_supply])
                if normalize:
                    max_val = max(vector.max(), 1)
                    result[plat_id] = torch.tensor(vector / max_val, dtype=torch.float32)
                else:
                    result[plat_id] = torch.tensor(vector, dtype=torch.float32)
            return result

        # Original behavior: combined vector
        if self.demand_grid is None or self.supply_grid is None:
            return torch.zeros(2, dtype=torch.float32)
        vector = np.concatenate([self.demand_grid, self.supply_grid])
        if normalize:
            max_val = max(vector.max(), 1)
            return torch.tensor(vector / max_val, dtype=torch.float32)
        return torch.tensor(vector, dtype=torch.float32)

    def get_normalized_demand_supply(self, state) -> np.ndarray:
        """Update grids from state and return normalized demand-supply vector.

        Combines update_grids() and get_demand_supply_vector(normalize=True).

        Args:
            state: SimulatorState with order_pool and courier_pool

        Returns:
            Normalized demand-supply vector
        """
        self.update_grids(state)
        return self.get_demand_supply_vector(normalize=True)

    def get_grid_state_vector(self) -> 'torch.Tensor':
        """Get grid state as torch tensor (compatible with GridStateManager).

        Returns:
            Tensor of shape (grid_dim,) with demand/supply grids
        """
        import torch
        vector = self.get_demand_supply_vector()
        return torch.tensor(vector, dtype=torch.float32)

    def get_grid_dim(self) -> int:
        """Get dimension of grid state vector.

        Returns:
            Total dimension (num_cells * 2 for demand + supply)
        """
        if not self.grid:
            return 2
        return grid.get_num_cells(self.grid) * 2

    def update_grids(self, state) -> None:
        """Update demand/supply grids from simulator state (compatible with GridStateManager).

        Args:
            state: SimulatorState with order_pool and courier_pool
        """
        # Get unassigned orders
        orders = [o for o in state.order_pool if o.is_unassigned()]
        couriers = list(state.courier_pool)

        self.update_demand_supply(orders, couriers)

    # ===========================
    # Reset
    # ===========================

    def reset(self) -> None:
        """Reset registry to initial state (keeps grid structure, clears queues and agent IDs)."""
        self.agent_id_map = {}
        self.next_agent_id = 0

        # Clear queues
        if self.queue_mgr:
            grid.clear_all_queues(self.queue_mgr)

        # Reset demand/supply grids
        if self.grid:
            num_cells = grid.get_num_cells(self.grid)
            self.demand_grid = np.zeros(num_cells, dtype=np.float32)
            self.supply_grid = np.zeros(num_cells, dtype=np.float32)

            # Reset per-platform grids
            self.platform_demand_grids = {
                plat_id: np.zeros(num_cells, dtype=np.float32) for plat_id in self.platforms
            }
            self.platform_supply_grids = {
                plat_id: np.zeros(num_cells, dtype=np.float32) for plat_id in self.platforms
            }

    def soft_reset(self) -> None:
        """Clear queues and agent IDs without clearing grid structure (alias for reset)."""
        self.reset()
