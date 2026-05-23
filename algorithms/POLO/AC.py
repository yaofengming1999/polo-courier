import torch
import numpy as np
import torch.nn.functional as F
import collections
import random
from dataclasses import dataclass
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CourierAttention(torch.nn.Module):

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(CourierAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head self-attention
        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        # Output projection
        self.output_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x, return_weights=False):

        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, num_couriers, embed_dim)
            squeeze_output = True

        x = x.contiguous()

        attn_output, attn_weights = self.multihead_attn(x, x, x)

        x = self.layer_norm(x + attn_output)

        courier_importance = attn_weights.mean(dim=1)  # (batch_size, num_couriers)
        courier_importance = F.softmax(courier_importance, dim=-1)  # Normalize

        aggregated = torch.bmm(courier_importance.unsqueeze(1), x).squeeze(1)  # (batch_size, embed_dim)
        aggregated = self.output_proj(aggregated)

        if squeeze_output:
            aggregated = aggregated.squeeze(0)
            courier_importance = courier_importance.squeeze(0)

        if return_weights:
            return aggregated, courier_importance
        return aggregated


class PolicyNet(torch.nn.Module):
    def __init__(self, stateDim, actionDim, use_attention=False, attention_heads=2,
                 use_route=False, use_hold_action = True, route_input_dim=2, route_hidden_dim=32):
        super(PolicyNet, self).__init__()
        self.stateDim = stateDim   # 骑手状态
        self.actionDim = actionDim  # 动作状态
        self.use_attention = use_attention
        self.use_route = use_route
        self.route_hidden_dim = route_hidden_dim
        self.use_hold_action = use_hold_action
        self.S = torch.nn.Linear(stateDim, 64).to(device)
        self.A = torch.nn.Linear(actionDim, 4).to(device)
        base_dim = 68  # courier features (64 + 4)
        if use_hold_action:
            self.hold_net = torch.nn.Sequential(
                torch.nn.Linear(base_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1)).to(device)

        # GRU for route sequence embedding (optional)
        if use_route:
            self.route_gru = torch.nn.GRU(
                input_size=route_input_dim,
                hidden_size=route_hidden_dim,
                batch_first=True
            ).to(device)
            base_dim = 68 + route_hidden_dim # courier features + route embedding

        # Attention module for context aggregation (optional)
        if use_attention:
            self.courier_attention = CourierAttention(
                embed_dim=base_dim,
                num_heads=attention_heads
            ).to(device)
            # Combine individual features with global context
            self.L1 = torch.nn.Linear(base_dim * 2, 32).to(device)
        else:
            self.L1 = torch.nn.Linear(base_dim, 32).to(device)

        self.L2 = torch.nn.Linear(32, 8).to(device)
        self.f = torch.nn.Linear(8, 1).to(device)

    def forward(self, X, routes=None):

        is_batched = X.dim() == 3
        if not is_batched:
            X = X.unsqueeze(0)  # (1, num_couriers, features)
            if routes is not None:
                routes = routes.unsqueeze(0)  # (1, num_couriers, seq_len, 2)

        batch_size, num_couriers, _ = X.shape
        s = X[:, :, :self.stateDim]
        s1 = F.relu(self.S(s))  # (batch, num_couriers, 64)
        a = X[:, :, -self.actionDim:]
        a1 = F.relu(self.A(a))  # (batch, num_couriers, 4)
        if self.use_hold_action:
            hold_values = self.hold_net(torch.cat((s1, a1), dim=2))  # (batch, num_couriers, 1)
            hold_values = torch.mean(hold_values, dim=1, keepdim=True)  # (batch, 1, 1)

        if self.use_route and routes is not None:
            # GRU route embedding
            routes_flat = routes.reshape(batch_size * num_couriers, routes.shape[2], routes.shape[3])
            _, route_hidden = self.route_gru(routes_flat)  # (1, batch*num_couriers, route_hidden_dim)
            route_embed = route_hidden.squeeze(0).reshape(batch_size, num_couriers, self.route_hidden_dim)
            individual_features = torch.cat((s1, route_embed, a1), dim=2)  # (batch, num_couriers, 64+route_hidden_dim+4)
        else:
            individual_features = torch.cat((s1, a1), dim=2)  # (batch, num_couriers, 68)

        if self.use_attention:
            # Get global context via attention for each sample in batch
            global_context = self.courier_attention(individual_features)  # (batch, 68)
            global_context_expanded = global_context.unsqueeze(1).expand(-1, num_couriers, -1)
            y1 = torch.cat((individual_features, global_context_expanded), dim=2)
        else:
            y1 = individual_features

        l1 = F.relu(self.L1(y1))
        l2 = F.relu(self.L2(l1))
        out = self.f(l2)  # (batch, num_couriers, 1)
        if self.use_hold_action:
            final_out = torch.cat((out, hold_values), dim=1)  # (batch, num_couriers + 1, 1)
        else:
            final_out = out  # (batch, num_couriers, 1)
        if not is_batched:
            final_out = final_out.squeeze(0)  # (num_couriers, 1)
        
        return final_out


# critic
class ValueNet(torch.nn.Module):
    def __init__(self, stateDim, actionDim, use_attention=False, attention_heads=2,
                 use_route=False, route_input_dim=2, route_hidden_dim=32):
        super(ValueNet, self).__init__()
        self.courierDim = stateDim  # 骑手状态
        # assert self.courierDim == 40
        self.orderDim = actionDim  # 动作状态
        # assert self.orderDim == 8
        self.use_attention = use_attention
        self.use_route = use_route
        self.route_hidden_dim = route_hidden_dim
        self.S = torch.nn.Linear(stateDim, 64).to(device)
        self.A = torch.nn.Linear(actionDim, 4).to(device)

        base_dim = 64 + 4

        # GRU for route sequence embedding (optional)
        if use_route:
            self.route_gru = torch.nn.GRU(
                input_size=route_input_dim,
                hidden_size=route_hidden_dim,
                batch_first=True
            ).to(device)
            base_dim = 64 + 4 + route_hidden_dim

        self.L1 = torch.nn.Linear(base_dim, 32).to(device)
        self.L2 = torch.nn.Linear(32, 8).to(device)

        # Attention module for aggregation (replaces mean pooling)
        if use_attention:
            self.courier_attention = CourierAttention(
                embed_dim=8,  # Applied after L2
                num_heads=min(attention_heads, 8)  # Ensure num_heads divides embed_dim
            ).to(device)

        self.f = torch.nn.Linear(8, 1).to(device)

    def forward(self, X, routes=None):
        """
        Args:
            X: (batch_size, num_couriers, state_dim) tensor
            routes: (batch_size, num_couriers, seq_len, 2) tensor
        Returns:
            value: (batch_size, 1) tensor
        """
        s = X[:, :, :self.courierDim]
        a = X[:, :, -self.orderDim:]
        s1 = F.relu(self.S(s))  # (batch_size, num_couriers, 64)
        a1 = F.relu(self.A(a))  # (batch_size, num_couriers, 4)

        if self.use_route and routes is not None:
            batch_size, num_couriers = s1.shape[0], s1.shape[1]
            routes_flat = routes.reshape(batch_size * num_couriers, routes.shape[2], routes.shape[3])
            _, route_hidden = self.route_gru(routes_flat)
            route_embed = route_hidden.squeeze(0).reshape(batch_size, num_couriers, self.route_hidden_dim)
            y1 = torch.cat((s1, a1, route_embed), dim=2)  # (batch_size, num_couriers, 68+route_hidden_dim)
        else:
            y1 = torch.cat((s1, a1), dim=2)  # (batch_size, num_couriers, 68)

        l1 = F.relu((self.L1(y1)))  # (batch_size, num_couriers, 32)
        l2 = F.relu((self.L2(l1)))  # (batch_size, num_couriers, 8)

        if self.use_attention:
            # Use attention to aggregate courier features
            aggregated = self.courier_attention(l2)  # (batch_size, 8)
        else:
            # Use mean pooling (original behavior)
            aggregated = torch.mean(l2, dim=1)  # (batch_size, 8)

        return self.f(aggregated)  # (batch_size, 1)


class TowerActorCritic:
    def __init__(self, stateDim, actionDim, actorLr, criticLr, gamma, epsilon, batchSize, device,
                 use_attention=False, attention_heads=4, use_route=False, use_hold_action=False):
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.use_route = use_route
        self.use_hold_action = use_hold_action
        self.epsilon = epsilon

        self.actor = PolicyNet(stateDim, actionDim, use_attention, attention_heads, use_route, use_hold_action).to(device)
        self.critic = ValueNet(stateDim, actionDim, use_attention, attention_heads, use_route).to(device)
        self.actorLr = actorLr
        self.criticLr = criticLr
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actorLr)
        self.criticOptimizer = torch.optim.Adam(self.critic.parameters(), lr=self.criticLr)
        self.gamma = gamma
        self.stateDim = stateDim
        self.batchSize = batchSize
        self.device = device

    # actor:采取动作
    def take_action(self, state, routes=None):  # 训练
        state = state.clone().detach().float().to(device)
        if routes is not None:
            routes = routes.clone().detach().float().to(device)

        vOutput = self.actor(state, routes)
        if self.use_hold_action:
            assert vOutput.shape[0] == state.shape[0] + 1  # num_couriers + 1
        vOutput = vOutput.reshape(-1)  # 将二维张量变为一维张量
        actionProb = torch.softmax(vOutput, dim=0)
        if random.random() > self.epsilon:
            action = torch.max(actionProb,0)[1].cpu()
        else:
            actionDist = torch.distributions.Categorical(actionProb)
            action = actionDist.sample().cpu()
            
        return action.cpu().item()  # 对softmax函数求导时的用法


    def update(self, state, action, reward, nextState, routes=None, nextRoutes=None):
        # Convert to tensors - all in one batch, no loop
        action = torch.tensor(action).view(-1, 1).to(device)  # (batch, 1)
        reward = torch.tensor(reward, dtype=torch.float).to(device)  # (batch,)
        nextState = torch.tensor(nextState, dtype=torch.float).to(device)  # (batch, num_couriers, features)
        stateBatch = torch.tensor(state, dtype=torch.float).to(device)  # (batch, num_couriers, features)

        if routes is not None:
            routes = torch.tensor(np.array(routes), dtype=torch.float).to(device)
        if nextRoutes is not None:
            nextRoutes = torch.tensor(np.array(nextRoutes), dtype=torch.float).to(device)

        # Critic update
        tdTarget = reward.unsqueeze(1) + self.gamma * self.critic(nextState, nextRoutes)  # (batch, 1)
        V = self.critic(stateBatch, routes)  # (batch, 1)
        criticLoss = F.mse_loss(V, tdTarget.detach())
        tdDelta = tdTarget - V  # (batch, 1)

        actorOut = self.actor(stateBatch, routes).squeeze(-1)  # (batch, num_couriers)
        probs = torch.softmax(actorOut, dim=1)  # softmax over couriers
        logProb = torch.log(probs.gather(1, action) + 1e-8)  # (batch, 1)
        actorLoss = torch.mean(-logProb * tdDelta.detach())

        self.criticOptimizer.zero_grad()
        self.actorOptimizer.zero_grad()
        criticLoss.backward()
        actorLoss.backward()
        self.criticOptimizer.step()
        self.actorOptimizer.step()

    def reset_learning_rate(self):
        self.criticLr = self.criticLr / 5
        self.criticOptimizer.param_groups[0]['lr'] = self.criticLr



class ReplayBuffer:
    def __init__(self, capacity=100000, batchSize=50):
        self.buffer = collections.deque(maxlen=capacity)
        self.batchSize = batchSize

    def add(self, state, action, reward, nextState, routes=None, nextRoutes=None):
        self.buffer.append((state, action, reward, nextState, routes, nextRoutes))

    def sample(self):
        transitions = random.sample(self.buffer, self.batchSize)
        state, action, reward, nextState, routes, nextRoutes = zip(*transitions)
        # Stack as numpy arrays for efficient tensor conversion
        state = np.array([x.numpy() if hasattr(x, 'numpy') else x for x in state])
        # Handle routes (may be None if use_route=False)
        if routes[0] is not None:
            routes = [x.numpy() if hasattr(x, 'numpy') else x for x in routes]
            nextRoutes = [x.numpy() if hasattr(x, 'numpy') else x for x in nextRoutes]
        else:
            routes = None
            nextRoutes = None
        return state, action, reward, np.array(nextState), routes, nextRoutes

    def size(self):
        return len(self.buffer)
