"""
Instance Generator - samples from real order data.
Generates orders and couriers for simulator.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

# Reuse the city boundary constants defined in the feature extractor so spatial
# strips are consistent with coordinate normalisation used during learning.
_CITY_BOUNDS = {
    'small':  (0.0,  3.29, 0.0,  3.74),
    'middle': (0.0,  4.86, 0.0,  4.49),
    'large':  (0.0, 13.47, 0.0, 13.02),
}  # (min_lng, max_lng, min_lat, max_lat)


class InstanceGenerator:
    """
    Generates orders and couriers for simulator.

    Orders: Randomly sampled from combined 8-day real data (1/8 per instance).
    Couriers: Loaded from pre-processed data (km coords), platform added dynamically.

    Time window:
        time_start / time_end (decimal hours, e.g. 17.0 = 17:00, 18.5 = 18:30).
        When set, only orders whose create_time falls within the converted second
        range are eligible.  The constructor converts hours → seconds internally;
        callers never need to pass raw seconds.
        Example: time_start=12.0, time_end=14.0 → noon-to-2pm peak window.
    """

    DATA_DIR = 'env/meituan_order_instance'
    OUTPUT_DIR = 'env/meituan_order_instance/generated_instances'

    @staticmethod
    def _hours_to_seconds(h: float) -> int:
        """Convert decimal hours to seconds (e.g. 18.5 → 66600)."""
        return int(h * 3600)

    def __init__(self,
                 scale: str = 'small',
                 platform_num: int = 1,
                 courier_appearance_ratio: float = 1.0,
                 time_start: Optional[float] = None,
                 time_end: Optional[float] = None,
                 temporal_platform_diff: bool = False,
                 spatial_platform_diff: bool = False):
        """
        temporal_platform_diff:
            Odd-indexed platforms are 'budget': delivery_patience += Uniform[600,1800]s,
            value × Uniform[0.6, 0.8].  Even-indexed platforms keep original values.

        spatial_platform_diff:
            Divide the pickup_x range into platform_num equal strips.  Each platform
            owns one strip: orders whose pickup_x falls in that strip are assigned to
            that platform with 70 % probability (remaining 30 % split uniformly among
            the other platforms).
        """
        self.scale = scale
        self.platform_num = platform_num
        self.courier_appearance_ratio = courier_appearance_ratio
        self.temporal_platform_diff = temporal_platform_diff
        self.spatial_platform_diff = spatial_platform_diff
        # For large scale, default to 5:00pm-6:00pm peak window to keep instance
        # size manageable. Other scales use the full day by default.
        if scale == 'large' and time_start is None and time_end is None:
            time_start = 17.0  # 17:00
            time_end   = 18.0  # 18:00

        # Convert decimal hours → seconds for internal use.
        self.time_start = self._hours_to_seconds(time_start) if time_start is not None else None
        self.time_end   = self._hours_to_seconds(time_end)   if time_end   is not None else None

        # Platform names
        self.platforms = [f'Platform{chr(65+i)}' for i in range(platform_num)]

        # Spatial strips: divide the x (longitude) range from city bounds into
        # platform_num equal strips. Platform i owns strip i.
        if self.spatial_platform_diff and platform_num > 1:
            min_lng, max_lng, _, _ = _CITY_BOUNDS.get(scale, (0.0, 1.0, 0.0, 1.0))
            self._x_breaks = np.linspace(min_lng, max_lng, platform_num + 1)
        else:
            self._x_breaks = None

        # Load combined order data (8 days)
        all_orders = pd.read_csv(os.path.join(self.DATA_DIR, f'orders_combined_{scale}.csv'))

        # Apply time window filter at load time so _generate_orders works on the
        if self.time_start is not None or self.time_end is not None:
            lo = self.time_start if self.time_start is not None else 0
            hi = self.time_end   if self.time_end   is not None else int(all_orders['create_time'].max())
            mask = (all_orders['create_time'] >= lo) & (all_orders['create_time'] <= hi)
            all_orders = all_orders[mask].copy()
            if len(all_orders) == 0:
                raise ValueError(f"No orders found in time window [{lo}, {hi}].")

        self.all_orders = all_orders.reset_index(drop=True)

        # Load pre-processed courier data (already in km, no platform)
        self.base_couriers = pd.read_csv(os.path.join(self.DATA_DIR, f'couriers_{scale}.csv'))

    def _assign_platform(self, pickup_x: float) -> str:
        """Return a platform id for one order.

        Spatial preference (when enabled): the platform whose strip contains
        pickup_x gets a 70 % draw probability; the remaining 30 % is split
        uniformly among the other platforms.
        Uniform fallback when spatial_platform_diff is False or platform_num==1.
        """
        n = self.platform_num
        if n == 1:
            return self.platforms[0]

        if self._x_breaks is not None:
            # Clip to handle floating-point edge at max boundary
            strip = int(np.searchsorted(self._x_breaks[1:], pickup_x, side='right'))
            strip = min(strip, n - 1)
            weights = np.full(n, 0.3 / (n - 1))
            weights[strip] = 0.7
            return self.platforms[self.rng.choice(n, p=weights)]

        return self.platforms[self.rng.integers(0, n)]

    def _generate_orders(self) -> List[dict]:
        """Sample 1/8 of the eligible order pool (one day's worth)."""
        n_total = len(self.all_orders)
        n_sample = max(1, n_total // 8)

        sample_indices = self.rng.choice(n_total, size=n_sample, replace=False)
        sampled = self.all_orders.iloc[sample_indices].copy()
        sampled = sampled.sort_values('create_time').reset_index(drop=True)

        orders = []
        for i, row in sampled.iterrows():
            platform_id = self._assign_platform(row['pickup_x'])
            patience = int(row['delivery_patience'])
            value = round(row['value'], 2)

            # Temporal platform differentiation:
            # odd-indexed platforms are 'budget' — more patient but lower pay.
            if self.temporal_platform_diff and self.platform_num > 1:
                p_idx = self.platforms.index(platform_id)
                if p_idx % 2 == 1:
                    patience += int(self.rng.integers(600, 1801))   # +10..30 min
                    value = round(value * self.rng.uniform(0.6, 0.8), 2)

            orders.append({
                'order_id': i,
                'pickup_x': round(row['pickup_x'], 4),
                'pickup_y': round(row['pickup_y'], 4),
                'delivery_x': round(row['delivery_x'], 4),
                'delivery_y': round(row['delivery_y'], 4),
                'create_time': int(row['create_time']),
                'delivery_patience': patience,
                'platform_id': platform_id,
                'value': value,
            })

        return orders

    def _generate_couriers(self) -> List[dict]:
        """Generate couriers from pre-processed data with random masking."""
        n_couriers = len(self.base_couriers)
        n_keep = max(1, int(n_couriers * self.courier_appearance_ratio))
        keep_indices = self._courier_rng.choice(n_couriers, size=n_keep, replace=False)

        couriers = []
        for idx in sorted(keep_indices):
            row = self.base_couriers.iloc[idx]
            couriers.append({
                'courier_id': int(row['courier_id']),
                'current_x': row['current_x'],
                'current_y': row['current_y'],
                'platform': self.platforms
            })

        return couriers

    def generate(self, seed) -> Tuple[List[dict], List[dict]]:
        """Generate one instance (orders + couriers) for the given seed."""
        self.seed = seed
        # Independent RNG streams so changing platform_num or order count
        # never shifts the courier draw.
        self.rng = np.random.default_rng(seed)
        self._courier_rng = np.random.default_rng(seed ^ 0xDEADBEEF)

        orders = self._generate_orders()
        couriers = self._generate_couriers()

        return orders, couriers
