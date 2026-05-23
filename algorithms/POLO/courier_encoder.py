"""Courier Encoder for TowerController."""
import torch
import torch.nn as nn


class CourierEncoder(nn.Module):
    """Encode variable-length courier sets with trajectory and route GRUs."""

    def __init__(self, courier_dim, hidden_dim=64, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # GRU for trajectory (past locations)
        self.trajectory_gru = nn.GRU(
            input_size=2,  # (lng, lat)
            hidden_size=hidden_dim // 4,
            num_layers=1,
            batch_first=True
        )

        # GRU for route (future planned locations)
        self.route_gru = nn.GRU(
            input_size=2,  # (lng, lat)
            hidden_size=hidden_dim // 4,
            num_layers=1,
            batch_first=True
        )

        # Project static features + GRU outputs to hidden_dim
        # courier_dim + trajectory_hidden + route_hidden
        combined_dim = courier_dim + 3
        self.proj = nn.Linear(combined_dim, hidden_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=2
        )

    def forward(self, x, trajectory_x=None, route_x=None, order_courier_pair_x=None):
        """x: (batch, M, courier_dim) or (M, courier_dim)
        trajectory_x: (batch, M, back_length, 2) or (M, back_length, 2)
        route_x: (batch, M, forward_length, 2) or (M, forward_length, 2)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if order_courier_pair_x.dim() == 2:
            order_courier_pair_x = order_courier_pair_x.unsqueeze(0)

        batch_size, M, _ = x.shape

        # Process trajectory with GRU
        if trajectory_x is not None:
            if trajectory_x.dim() == 3:
                trajectory_x = trajectory_x.unsqueeze(0)
            # Reshape to (batch * M, back_length, 2)
            traj_flat = trajectory_x.reshape(-1, trajectory_x.shape[2], 2)
            _, traj_hidden = self.trajectory_gru(traj_flat)  # (1, batch*M, hidden//4)
            traj_hidden = traj_hidden.squeeze(0).reshape(batch_size, M, -1)  # (batch, M, hidden//4)
        else:
            traj_hidden = torch.zeros(batch_size, M, self.hidden_dim // 4, device=x.device)

        # Process route with GRU
        if route_x is not None:
            if route_x.dim() == 3:
                route_x = route_x.unsqueeze(0)
            # Reshape to (batch * M, forward_length, 2)
            route_flat = route_x.reshape(-1, route_x.shape[2], 2)
            _, route_hidden = self.route_gru(route_flat)  # (1, batch*M, hidden//4)
            route_hidden = route_hidden.squeeze(0).reshape(batch_size, M, -1)  # (batch, M, hidden//4)
        else:
            route_hidden = torch.zeros(batch_size, M, self.hidden_dim // 4, device=x.device)

        # Concatenate static features with GRU outputs
        # x_combined = torch.cat([x, traj_hidden, route_hidden], dim=-1)  # (batch, M, combined_dim)
        x_combined = torch.cat([x, order_courier_pair_x], dim=-1)  # (batch, M, combined_dim)

        # Project and encode
        x = self.proj(x_combined)
        x = self.encoder(x)
        return x  # (batch, M, hidden_dim)
