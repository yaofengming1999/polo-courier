"""Unified grid utilities for grid-based controllers.

This module provides:
- grid_utils: Low-level grid creation and manipulation
- AgentRegistry: High-level agent and queue management

Usage Examples:

1. Region-only mode (for ACBGM, Fair style):
    ```python
    from algorithms.grid import AgentRegistry

    registry = AgentRegistry(
        grid_size=0.001,      # ~100m cells
        use_direction=False,  # No direction segmentation
        grid_type='rect'      # Rectangular grid
    )
    registry.initialize_grid(platform_state, platforms)
    agents = registry.get_platform_agents(platform_id)  # One agent per cell
    ```

2. Region+Direction mode (for Tower, Coop style):
    ```python
    from algorithms.grid import AgentRegistry

    registry = AgentRegistry(
        hex_size=0.01,       # Hex cell size
        use_direction=True,  # 7 directions per cell
        grid_type='hex'      # Hexagonal grid
    )
    registry.initialize_grid(platform_state, platforms)
    agents = registry.get_platform_agents(platform_id)  # 7 agents per cell
    ```
"""

from algorithms.utils.grid_utils import (
    # Grid creation
    create_hex_grid,
    create_rect_grid,
    # Cell operations
    get_cell,
    get_cell_index,
    get_num_cells,
    # Direction
    get_direction,
    NUM_DIRECTIONS,
    DIRECTION_NAMES,
    # Queue management
    create_queue_manager,
    add_order_to_queue,
    get_queue_orders,
    get_cell_orders,
    clear_all_queues,
    remove_orders_from_queues,
    # Features
    compute_demand_supply_grids,
    get_local_demand_supply,
    cal_hex_distance,
    # Courier pruning
    prune_couriers_per_platform,
    get_unique_courier_indices,
)

from algorithms.utils.agent_registry import GridManager
# from algorithms.grid.feature_extractor import FeatureExtractor, FeatureConfig

__all__ = [
    # Grid creation
    'create_hex_grid',
    'create_rect_grid',
    # Cell operations
    'get_cell',
    'get_cell_index',
    'get_num_cells',
    # Direction
    'get_direction',
    'NUM_DIRECTIONS',
    'DIRECTION_NAMES',
    # Queue management
    'create_queue_manager',
    'add_order_to_queue',
    'get_queue_orders',
    'get_cell_orders',
    'clear_all_queues',
    'remove_orders_from_queues',
    # Features
    'compute_demand_supply_grids',
    'get_local_demand_supply',
    'cal_hex_distance',
    # Courier pruning
    'prune_couriers_per_platform',
    'get_unique_courier_indices',
    # Registry
    'GridManager',
    # Feature extraction
    'FeatureExtractor',
    'FeatureConfig',
]
