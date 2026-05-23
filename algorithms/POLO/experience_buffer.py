import torch
import numpy as np


class ExperienceBuffer:
    """Store and manage trajectory experiences for PPO training.

    Key design:
    - Each agent (region + direction) has its own trajectory
    - Multiple dispatch rounds within same time step are ordered by dispatch_round
    - GAE is computed per-agent then combined
    """

    def __init__(self, gamma=0.99, gae_lambda=0.95):

        self.gamma = gamma #Discount factor for returns
        self.gae_lambda = gae_lambda #GAE lambda parameter
        self.trajectory = []  # All experiences in insertion order
        self.agent_trajectories = {}  # agent_id -> list of (global_idx, experience)

        self._reward = []
        self._value = []
        self._log_prob = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def store_dispatch_experiences(self, dispatch_experiences):

        # total_dispatches = len(dispatch_experiences)
        # total_reward = sum(exp['reward'] for exp in dispatch_experiences)
        

        # Add experiences to trajectory and index by agent
        for experience in dispatch_experiences:
            # experience['done'] = done
            global_idx = len(self.trajectory)
            self.trajectory.append(experience)

            # agent index
            agent_id = experience['agent_id']
            self.agent_trajectories.setdefault(agent_id, []).append(global_idx)  # 只存 idx 更轻

            r = experience['reward']
            v = experience['value']
            lp = experience['log_prob']

            self._reward.append(r)
            self._value.append(v)
            self._log_prob.append(lp)

        # avg_reward = total_reward / total_dispatches if total_dispatches > 0 else 0.0

        # return total_dispatches, avg_reward

    def compute_advantages_and_returns(self):
        """Compute GAE advantages and returns per agent.

        GAE is computed for each agent's trajectory independently,
        sorted by (time, dispatch_round) for proper temporal ordering.

        Returns:
            advantages: (N,) tensor in original trajectory order
            returns: (N,) tensor in original trajectory order
            trajectory_data: Dict with extracted trajectory data for training
        """
        if len(self.trajectory) == 0:
            return None, None, None
        N = len(self.trajectory)
        # Initialize advantages array
        advantages_all = torch.zeros(N,device=self.device, dtype=torch.float32)
        reward_all = torch.tensor(self._reward, device=self.device, dtype=torch.float32)
        value_all = torch.tensor(self._value, device=self.device, dtype=torch.float32)
        log_prob_all = torch.tensor(self._log_prob, device=self.device, dtype=torch.float32)

        # Compute GAE for each agent independently
        for agent_id, idx_list in self.agent_trajectories.items():

            idx = torch.tensor(idx_list, device=self.device, dtype=torch.long)
            # Extract data
            r = reward_all.index_select(0, idx)
            v = value_all.index_select(0, idx)
            d = torch.zeros_like(r, device=self.device, dtype=torch.float32)  #

            adv = self._compute_agent_gae(r, v, d)

            # Put advantages back into original order
            advantages_all.index_copy_(0, idx, adv)

        # Compute returns = advantages + values
        # values = np.array([step['value'] for step in self.trajectory])
        returns = advantages_all + value_all

        # Convert to tensors
        # advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # Extract trajectory data
        trajectory_data = {
            'experiences': self.trajectory,
            'old_log_probs': log_prob_all,
            'advantages': advantages_all,
            'returns': returns,
            'size': N
        }

        return advantages_all, returns, trajectory_data

    def _compute_agent_gae(self, rewards, values, dones):

        T = rewards.shape[0]
        adv = torch.zeros(T, device=rewards.device, dtype=rewards.dtype)

        last_gae = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
        gamma = self.gamma
        lam = self.gae_lambda

        for t in range(T - 1, -1, -1):
            next_value = values[t + 1] if t < T - 1 else torch.zeros((), device=rewards.device, dtype=rewards.dtype)
            next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            adv[t] = last_gae

        return adv

    def clear(self):
        """Clear trajectory buffer."""
        self.trajectory = []
        self.agent_trajectories = {}
        self._value = []
        self._reward = []
        self._log_prob = []

    def __len__(self):
        """Return current trajectory size."""
        return len(self.trajectory)

    def is_empty(self):
        """Check if buffer is empty."""
        return len(self.trajectory) == 0
