"""Tower Actor for order-courier matching."""
import torch
import torch.nn as nn


class TowerActor(nn.Module):
    """Two-tower actor for order-courier matching."""

    def __init__(self, order_encoder, courier_encoder, hidden_dim=64):
        super().__init__()
        self.order_encoder = order_encoder
        self.courier_encoder = courier_encoder
        self.hidden_dim = hidden_dim
        # HOLD logit network (per-order)
        self.hold_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, order_features, courier_features, trajectory_features, route_features, order_courier_pair_features):
        """
        Args:
            order_features: (batch, N, order_dim)
            courier_features: (batch, M, courier_dim)
        Returns:
            action_logits: (batch, N, M+1)  [HOLD, C1, ..., CM]
        """
        order_emb = self.order_encoder(order_features)  # (batch, N, hidden)
        courier_emb = self.courier_encoder(courier_features, trajectory_features, route_features,order_courier_pair_features)

        # Similarity scores (dot product)
        scores = torch.matmul(order_emb, courier_emb.transpose(-2, -1))
        scores = scores / (self.hidden_dim ** 0.5)  # (batch, N, M)

        # HOLD logits
        hold_logits = self.hold_net(order_emb)  # (batch, N, 1)

        # Concatenate
        action_logits = torch.cat([hold_logits, scores], dim=-1)  # (batch, N, M+1)

        return action_logits

    def sample_action(self, order_features, courier_features, trajectory_features, route_features, order_courier_pair_features, deterministic=False):
        """Select action given features.

        Args:
            order_features: (1, order_dim) or (batch, N, order_dim) tensor
            courier_features: (M, courier_dim) or (batch, M, courier_dim) tensor
            trajectory_features: (M, back_length, 2) or (batch, M, back_length, 2)
            route_features: (M, forward_length, 2) or (batch, M, forward_length, 2)
            deterministic: If True, select argmax; otherwise sample

        Returns:
            action: Selected action index (int or tensor)
            log_prob: Log probability of action (float or tensor)
        """
        M = courier_features.shape[0] if courier_features.dim() == 2 else courier_features.shape[1]

        if M == 0:
            return 0, 0.0

        # Add batch dimension if needed
        if order_features.dim() == 2:
            order_features = order_features.unsqueeze(0)
            courier_features = courier_features.unsqueeze(0)
            trajectory_features = trajectory_features.unsqueeze(0)
            route_features = route_features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Forward pass
        action_logits = self.forward(order_features, courier_features, trajectory_features, route_features, order_courier_pair_features)  

        # Sample or argmax
        if deterministic:
            actions = torch.argmax(action_logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=action_logits)
            actions = dist.sample()

        # Compute log prob
        dist = torch.distributions.Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        joint_log_prob = log_probs.sum(dim=-1)

        if squeeze_output:
            return actions.squeeze().item(), joint_log_prob.item()
        return actions, joint_log_prob
