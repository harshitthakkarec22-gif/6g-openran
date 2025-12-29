"""
Proximal Policy Optimization (PPO) for Network Scheduling in 6G

Based on research: "AI-Native Network Scheduling for 6G" 
IEEE Journal on Selected Areas in Communications, 2024

Implements PPO with actor-critic architecture for:
- Continuous action space for power allocation
- Multi-objective optimization (throughput, energy, fairness)
- Real-time scheduling decisions
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO scheduler"""
    state_dim: int
    action_dim: int
    hidden_dims: List[int]
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    c1: float = 1.0  # Value loss coefficient
    c2: float = 0.01  # Entropy coefficient
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    batch_size: int = 64


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO
    
    Actor outputs action distribution (mean and std for continuous actions)
    Critic outputs value function estimate
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        """
        Initialize Actor-Critic network
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer dimensions
        """
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        shared_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*shared_layers)
        
        # Actor head (policy network)
        self.actor_mean = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[-1], action_dim),
            nn.Tanh()  # Bound actions to [-1, 1]
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: State tensor
            
        Returns:
            tuple: (action_mean, value)
        """
        features = self.shared_net(state)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value
    
    def get_action_and_value(self, state: torch.Tensor, 
                            action: Optional[torch.Tensor] = None):
        """
        Get action, log probability, entropy, and value
        
        Args:
            state: State tensor
            action: Action tensor (if None, sample new action)
            
        Returns:
            tuple: (action, log_prob, entropy, value)
        """
        action_mean, value = self.forward(state)
        action_std = torch.exp(self.actor_logstd)
        
        # Create normal distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value.squeeze(-1)


class PPOScheduler:
    """
    PPO-based scheduler for 6G networks
    
    Performs real-time scheduling with multi-objective optimization:
    - Maximize throughput
    - Minimize energy consumption
    - Ensure fairness across users
    - Maintain QoS requirements
    """
    
    def __init__(self, config: PPOConfig):
        """
        Initialize PPO scheduler
        
        Args:
            config: PPO configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize actor-critic network
        self.actor_critic = ActorCritic(
            config.state_dim,
            config.action_dim,
            config.hidden_dims
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=config.learning_rate
        )
        
        # Training storage
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # Metrics
        self.episodes = 0
        self.total_steps = 0
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        
        logger.info(f"PPO Scheduler initialized on {self.device}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """
        Select scheduling action
        
        Args:
            state: Current state
            deterministic: Use deterministic policy (for evaluation)
            
        Returns:
            tuple: (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if deterministic:
                action_mean, value = self.actor_critic(state_tensor)
                action = action_mean
                log_prob = torch.tensor(0.0)
            else:
                action, log_prob, _, value = \
                    self.actor_critic.get_action_and_value(state_tensor)
            
            return action.cpu().numpy()[0], log_prob.item(), value.item()
    
    def store_transition(self, state: np.ndarray, action: np.ndarray,
                        log_prob: float, reward: float, value: float, done: bool):
        """Store transition for training"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def update(self):
        """Update policy using collected experience"""
        if len(self.states) < self.config.batch_size:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Compute advantages using GAE
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + values
        
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / \
                          (advantages_tensor.std() + 1e-8)
        
        # PPO update for multiple epochs
        for epoch in range(self.config.update_epochs):
            # Get current policy outputs
            _, new_log_probs, entropy, new_values = \
                self.actor_critic.get_action_and_value(states, actions)
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(
                ratio,
                1 - self.config.clip_epsilon,
                1 + self.config.clip_epsilon
            ) * advantages_tensor
            
            # Policy loss
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(new_values, returns_tensor)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = (
                policy_loss +
                self.config.c1 * value_loss +
                self.config.c2 * entropy_loss
            )
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            
            # Track metrics
            self.policy_losses.append(policy_loss.item())
            self.value_losses.append(value_loss.item())
            self.entropies.append(-entropy_loss.item())
        
        # Clear buffers
        self.clear_buffers()
        self.episodes += 1
    
    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray,
                     dones: np.ndarray) -> np.ndarray:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags
            
        Returns:
            np.ndarray: Advantages
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = \
                delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_advantage
        
        return advantages
    
    def clear_buffers(self):
        """Clear experience buffers"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def schedule(self, ran_state: Dict) -> Dict[str, Any]:
        """
        Perform scheduling decision
        
        Args:
            ran_state: Current RAN state
            
        Returns:
            dict: Scheduling decisions (power allocation, resource blocks, MCS)
        """
        # Extract state features
        state = self._extract_state_features(ran_state)
        
        # Select action (deterministic for deployment)
        action, _, _ = self.select_action(state, deterministic=True)
        
        # Decode action to scheduling decision
        scheduling_decision = self._decode_action(action, ran_state)
        
        return scheduling_decision
    
    def _extract_state_features(self, ran_state: Dict) -> np.ndarray:
        """Extract state features from RAN state"""
        features = []
        
        users = ran_state.get('users', [])
        max_users = 50
        
        for i in range(max_users):
            if i < len(users):
                user = users[i]
                features.extend([
                    user.get('cqi', 0) / 15.0,
                    user.get('buffer_occupancy', 0) / 1000.0,
                    user.get('power_headroom', 0) / 23.0,
                    user.get('throughput', 0) / 1e9,
                    user.get('energy_consumed', 0) / 1000.0,
                    user.get('qos_5qi', 9) / 100.0
                ])
            else:
                features.extend([0, 0, 0, 0, 0, 0])
        
        # Global features
        features.extend([
            ran_state.get('total_power_budget', 100) / 100.0,
            ran_state.get('current_power_usage', 0) / 100.0,
            len(users) / max_users
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _decode_action(self, action: np.ndarray, ran_state: Dict) -> Dict:
        """Decode continuous action to scheduling decision"""
        users = ran_state.get('users', [])
        
        if not users:
            return {'power_allocation': {}, 'mcs': {}, 'prbs': {}}
        
        # Action components:
        # - Power allocation per user (continuous [0, 1])
        # - MCS selection per user (continuous [0, 1] mapped to MCS index)
        # - PRB allocation weights (continuous [0, 1])
        
        num_users = len(users)
        action_per_user = len(action) // max(num_users, 1)
        
        power_allocation = {}
        mcs_selection = {}
        prb_allocation = {}
        
        total_power = ran_state.get('total_power_budget', 100)
        
        for i, user in enumerate(users):
            if i * action_per_user >= len(action):
                break
            
            user_action = action[i * action_per_user:(i + 1) * action_per_user]
            
            # Power allocation (scaled to [0, max_power])
            power_allocation[user['id']] = \
                float((user_action[0] + 1) / 2 * total_power / num_users)
            
            # MCS selection (map to MCS index 0-28)
            mcs_selection[user['id']] = \
                int((user_action[1] + 1) / 2 * 28)
            
            # PRB allocation weight
            if len(user_action) > 2:
                prb_allocation[user['id']] = float(user_action[2] + 1) / 2
        
        return {
            'power_allocation': power_allocation,
            'mcs': mcs_selection,
            'prb_weights': prb_allocation
        }
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episodes': self.episodes,
            'total_steps': self.total_steps,
            'config': self.config
        }, filepath)
        logger.info(f"PPO model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episodes = checkpoint['episodes']
        self.total_steps = checkpoint['total_steps']
        logger.info(f"PPO model loaded from {filepath}")
    
    def get_metrics(self) -> Dict:
        """Get training metrics"""
        recent_policy_losses = self.policy_losses[-100:]
        recent_value_losses = self.value_losses[-100:]
        recent_entropies = self.entropies[-100:]
        
        return {
            'episodes': self.episodes,
            'total_steps': self.total_steps,
            'avg_policy_loss': np.mean(recent_policy_losses) if recent_policy_losses else 0,
            'avg_value_loss': np.mean(recent_value_losses) if recent_value_losses else 0,
            'avg_entropy': np.mean(recent_entropies) if recent_entropies else 0
        }
