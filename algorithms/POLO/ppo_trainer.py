"""PPO Trainer - PPO parameter updates."""
import numpy as np
import torch
import torch.nn as nn


class PPOTrainer:
    """PPO training with separate actor-critic updates."""

    def __init__(self, actor, critic, actor_optimizer, critic_optimizer, device,
                 epsilon_clip=0.2, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=0.5, ppo_epochs=4, batch_size=64):
        """Initialize PPO trainer.

        Args:
            actor: TowerActor network
            critic: TowerCritic network
            actor_optimizer: Adam optimizer for actor
            critic_optimizer: Adam optimizer for critic
            device: PyTorch device
            epsilon_clip: PPO clipping parameter
            entropy_coef: Entropy loss coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Gradient clipping norm
            ppo_epochs: Number of PPO epochs per update
            batch_size: Batch size for PPO updates
        """
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.device = device

        self.epsilon_clip = epsilon_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
    
    def compute_gae(self, trajectory, gamma=0.99, gae_lambda=0.95):
        """
        Compute Generalized Advantage Estimation.

        GAE computed for dispatch-level temporal ordering:
        - Group by agent_id
        - Sort by (time, dispatch_count) for temporal order
        - Each dispatch = one timestep

        Args:
            trajectory: list of dicts with keys
                ['agent_id', 'dispatch_count', 'time', 'log_prob', 'value', 'reward', 'done']
            gamma: discount factor
            gae_lambda: GAE lambda

        Returns:
            advantages: tensor (same order as input trajectory)
            returns: tensor (same order as input trajectory)
        """
        # Group trajectory by agent identity
        agent_trajectories = {}
        for idx, step in enumerate(trajectory):
            agent_id = step['agent_id']
            if agent_id not in agent_trajectories:
                agent_trajectories[agent_id] = []
            agent_trajectories[agent_id].append((idx, step))

        # Initialize advantages array (same order as trajectory)
        advantages = np.zeros(len(trajectory))

        # Compute GAE for each agent independently
        for agent_id, agent_steps in agent_trajectories.items():
            # Sort by (time, dispatch_count) for temporal order
            agent_steps.sort(key=lambda x: (x[1]['time'], x[1]['dispatch_count']))

            # Extract temporal data for this agent
            indices = [idx for idx, _ in agent_steps]
            rewards = np.array([step['reward'] for _, step in agent_steps])
            values = np.array([step['value'] for _, step in agent_steps])
            dones = np.array([step['done'] for _, step in agent_steps])

            # Compute GAE for this agent's dispatch sequence
            agent_advantages = np.zeros_like(rewards)
            last_gae = 0

            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                    next_non_terminal = 1.0 - dones[t]
                else:
                    next_value = values[t + 1]  # Next dispatch of SAME agent
                    next_non_terminal = 1.0 - dones[t]

                delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
                agent_advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

            # Put advantages back into original trajectory order
            for i, orig_idx in enumerate(indices):
                advantages[orig_idx] = agent_advantages[i]

        # Compute returns
        values = np.array([step['value'] for step in trajectory])
        returns = advantages + values

        return torch.FloatTensor(advantages), torch.FloatTensor(returns)
    
    def update_parameters(self, trajectory_data, advantages, returns):
        experiences = trajectory_data['experiences']
        all_old_log_probs = trajectory_data['old_log_probs']  # shape (N,)
        dataset_size = trajectory_data['size']

        # 1) Move shared tensors to the target device once per update.
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        all_old_log_probs = all_old_log_probs.to(self.device)

        # Normalize advantages on device
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0

        # ====== 2) PPO epochs ======
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(dataset_size, device=self.device)  # Keep the permutation on the current device.

            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_idx = indices[start:end]  # tensor, NOT list

                # 3) Assemble a mini-batch from experiences.
                # Shapes are fixed, so the tensors can be stacked directly.
                batch_steps = [experiences[i] for i in batch_idx.tolist()]  
                # batch_idx.tolist() is only used to index the Python list.
                # The important part is that the forward pass happens in one batch.

                order_features = torch.stack([s['order_features'] for s in batch_steps]).to(self.device)
                courier_features = torch.stack([s['courier_features'] for s in batch_steps]).to(self.device)
                trajectory_features = torch.stack([s['trajectory_features'] for s in batch_steps]).to(self.device)
                route_features = torch.stack([s['route_features'] for s in batch_steps]).to(self.device)
                pair_x = torch.stack([s['order_courier_pair_feature'] for s in batch_steps]).to(self.device)

                actions = torch.tensor([s['action'] for s in batch_steps], device=self.device, dtype=torch.long)

                agent_id = torch.tensor([s['agent_id'] for s in batch_steps], device=self.device, dtype=torch.long)
                dispatch_round = torch.tensor([s['dispatch_round'] for s in batch_steps], device=self.device, dtype=torch.long)
                current_time = torch.tensor([s['time'] for s in batch_steps], device=self.device, dtype=torch.float32)

                # targets
                batch_old_log_probs = all_old_log_probs[batch_idx]     # (B,)
                batch_advantages = advantages[batch_idx]               # (B,)
                batch_returns = returns[batch_idx]                     # (B,)

                # 4) Single actor forward pass for the whole mini-batch.
                logits = self.actor(order_features, courier_features,
                                    trajectory_features, route_features, pair_x)

                # Reshape logits to (B, A).
                # This handles the legacy cases where shapes were (B,1,1,A) or (B,1,A).
                while logits.dim() > 2:
                    logits = logits.squeeze(1)
                # logits should now be (B, A).

                # 5) Compute log-probabilities and entropy in batch form.
                logp_all = torch.log_softmax(logits, dim=-1)                 # (B, A)
                new_log_probs = logp_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

                p_all = torch.softmax(logits, dim=-1)                        # (B, A)
                entropy = -(p_all * logp_all).sum(dim=-1)                    # (B,)

                # 6) Single critic forward pass for the whole mini-batch.
                values = self.critic(
                    agent_id=agent_id,
                    order_features=order_features,
                    courier_features=courier_features,
                    trajectory_features=trajectory_features,
                    route_features=route_features,
                    order_courier_pair_x=pair_x,
                    dispatch_count=dispatch_round,
                    current_time=current_time
                ).squeeze(-1)  # (B,)

                # 7) PPO losses.
                ratio = torch.exp(new_log_probs - batch_old_log_probs)       # (B,)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.value_coef * (values - batch_returns).pow(2).mean()
                entropy_loss = -self.entropy_coef * entropy.mean()

                loss = actor_loss + critic_loss + entropy_loss

                # 8) Backward pass and optimizer step.
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)

                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy_loss.item()
                num_updates += 1
        print(f"PPO updates performed: {num_updates}")
        print(f"Average Actor Loss: {total_actor_loss / max(1, num_updates):.4f}, "
              f"Critic Loss: {total_critic_loss / max(1, num_updates):.4f}, "
              f"Entropy Loss: {total_entropy_loss / max(1, num_updates):.4f}")
        return {
            'actor_loss': total_actor_loss / max(1, num_updates),
            'critic_loss': total_critic_loss / max(1, num_updates),
            'entropy_loss': total_entropy_loss / max(1, num_updates),
        }
