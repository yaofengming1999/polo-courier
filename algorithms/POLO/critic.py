"""Tower Critic for value estimation."""
import torch
import torch.nn as nn


class TowerCritic(nn.Module):
    """Centralized critic with global view for value estimation.

    Takes agent ID + local state + all other agents' states for centralized training.
    """

    def __init__(self, order_encoder, courier_encoder, hidden_dim=64, max_agents=100):
        super().__init__()
        # Separate encoders (same architecture as actor but different parameters)
        self.order_encoder = order_encoder
        self.courier_encoder = courier_encoder
        self.hidden_dim = hidden_dim
        self.max_agents = max_agents

        # Agent ID embedding (to distinguish agents)
        self.agent_id_embedding = nn.Embedding(max_agents, hidden_dim)

        # Additional features encoder (dispatch_count, time, etc.)
        self.context_net = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),  # [dispatch_count, normalized_time]
            nn.ReLU()
        )

        # Global state aggregator
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim + hidden_dim // 2, 256),  # local + agent_id + context
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, agent_id, order_features, courier_features, trajectory_features, route_features, order_courier_pair_x,
                dispatch_count, current_time, order_mask=None, courier_mask=None):
        """
        Process agent state with global view.

        Args:
            agent_id: (batch,) int - agent identifier
            order_features: (batch, N, order_dim) - current agent's orders
            courier_features: (batch, M, courier_dim) - all couriers
            trajectory_features: (batch, M, back_length, 2)
            route_features: (batch, M, forward_length, 2)
            dispatch_count: (batch,) int - number of dispatches made
            current_time: (batch,) float - current time
            order_mask: (batch, N) bool - optional
            courier_mask: (batch, M) bool - optional
        Returns:
            values: (batch,) - state values
        """
        batch_size = order_features.shape[0] if order_features.dim() == 3 else 1

        # Encode agent ID
        if isinstance(agent_id, int):
            agent_id = torch.tensor([agent_id], device=order_features.device)
        agent_id_emb = self.agent_id_embedding(agent_id)  # (batch, hidden)

        # Encode context (dispatch_count, time)
        context_features = torch.stack([
            dispatch_count.float() if isinstance(dispatch_count, torch.Tensor) else torch.tensor([dispatch_count], dtype=torch.float32, device=order_features.device),
            current_time.float() if isinstance(current_time, torch.Tensor) else torch.tensor([current_time / 3600.0], dtype=torch.float32, device=order_features.device)  # Normalize time
        ], dim=-1)  # (batch, 2)

        if context_features.dim() == 1:
            context_features = context_features.unsqueeze(0)

        context_emb = self.context_net(context_features)  # (batch, hidden//2)

        # Encode local state
        order_emb = self.order_encoder(order_features)  # (batch, N, hidden)
        courier_emb = self.courier_encoder(courier_features, trajectory_features, route_features, order_courier_pair_x)  # (batch, M, hidden)

        # Aggregate local state
        if order_mask is not None:
            order_sum = (order_emb * order_mask.unsqueeze(-1)).sum(dim=1)
            order_count = order_mask.sum(dim=1, keepdim=True).clamp(min=1)
            order_global = order_sum / order_count
        else:
            order_global = order_emb.mean(dim=1)

        if courier_mask is not None:
            courier_sum = (courier_emb * courier_mask.unsqueeze(-1)).sum(dim=1)
            courier_count = courier_mask.sum(dim=1, keepdim=True).clamp(min=1)
            courier_global = courier_sum / courier_count
        else:
            courier_global = courier_emb.mean(dim=1)

        # Concatenate: local_state + agent_id + context
        state_emb = torch.cat([order_global, courier_global, agent_id_emb, context_emb], dim=-1)
        values = self.value_head(state_emb)

        return values.squeeze(-1)  # (batch,)

    def estimate_value(self, agent_id, dispatch_count, current_time,
                       order_features, courier_features, trajectory_features, route_features, order_courier_pair_x):
        """Estimate value for one agent.

        Args:
            agent_id: Agent ID
            dispatch_count: Dispatch count
            current_time: Current time
            order_features: (1, order_dim) local features
            courier_features: (M, courier_dim) local features
            trajectory_features: Local trajectory features
            route_features: Local route features

        Returns:
            value: Scalar value estimate
        """
        device = order_features.device
        M = courier_features.shape[0] if courier_features.dim() > 1 else 0

        if M == 0:
            return 0.0

        # Add batch dimension
        order_features_batch = order_features.unsqueeze(0)  # (1, 1, order_dim)
        courier_features_batch = courier_features.unsqueeze(0)  # (1, M, courier_dim)
        trajectory_features_batch = trajectory_features.unsqueeze(0)
        route_features_batch = route_features.unsqueeze(0)

        value = self.forward(
            agent_id=torch.tensor([agent_id], device=device),
            order_features=order_features_batch,
            courier_features=courier_features_batch,
            trajectory_features=trajectory_features_batch,
            route_features=route_features_batch,
            order_courier_pair_x=order_courier_pair_x.unsqueeze(0),
            dispatch_count=torch.tensor([dispatch_count], device=device),
            current_time=torch.tensor([current_time], device=device)
        )

        value_scalar = value.item() if value.dim() == 0 else value.squeeze().item()
        return value_scalar
