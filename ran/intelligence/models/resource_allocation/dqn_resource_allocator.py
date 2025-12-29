"""
Deep Q-Network (DQN) for Resource Allocation in 6G Networks

Based on research: "Deep Reinforcement Learning for Resource Allocation in 6G Networks"
IEEE Transactions on Wireless Communications, 2024

Implements DQN for dynamic resource block allocation considering:
- Channel conditions
- Buffer states
- QoS requirements
- Throughput maximization with latency constraints
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random

logger = logging.getLogger(__name__)


class DQNNetwork(nn.Module):
    """
    Deep Q-Network architecture for resource allocation
    
    Input: State vector (channel conditions, buffer states, QoS requirements)
    Output: Q-values for each possible resource block allocation action
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        """
        Initialize DQN network
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: State tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer for DQN training
    
    Stores transitions (state, action, reward, next_state, done)
    """
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add transition to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNResourceAllocator:
    """
    DQN-based resource allocator for 6G networks
    
    Allocates resource blocks to users based on:
    - Channel quality (CQI)
    - Buffer occupancy
    - QoS class identifier (5QI)
    - Historical throughput
    
    Optimizes for:
    - Throughput maximization
    - Latency constraints
    - Fairness across users
    """
    
    def __init__(self, config: Dict):
        """
        Initialize DQN resource allocator
        
        Args:
            config: Configuration dictionary with:
                - state_dim: State space dimension
                - action_dim: Action space dimension (number of resource blocks)
                - hidden_dims: List of hidden layer dimensions
                - learning_rate: Learning rate
                - gamma: Discount factor
                - epsilon_start: Initial exploration rate
                - epsilon_end: Final exploration rate
                - epsilon_decay: Exploration decay rate
                - batch_size: Training batch size
                - replay_buffer_size: Replay buffer capacity
                - target_update_freq: Target network update frequency
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = DQNNetwork(
            config['state_dim'],
            config['action_dim'],
            config['hidden_dims']
        ).to(self.device)
        
        self.target_net = DQNNetwork(
            config['state_dim'],
            config['action_dim'],
            config['hidden_dims']
        ).to(self.device)
        
        # Copy policy net weights to target net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config['learning_rate']
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(config['replay_buffer_size'])
        
        # Training parameters
        self.gamma = config['gamma']
        self.epsilon = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        self.batch_size = config['batch_size']
        self.target_update_freq = config['target_update_freq']
        
        # Tracking
        self.steps = 0
        self.episodes = 0
        self.losses = []
        
        logger.info(f"DQN Resource Allocator initialized on {self.device}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            int: Selected action (resource block allocation)
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.randrange(self.config['action_dim'])
        
        # Exploitation: select best action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step
        
        Returns:
            float: Loss value, or None if not enough samples
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss (Huber loss for stability)
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def allocate_resources(self, ran_state: Dict) -> Dict[str, List[int]]:
        """
        Allocate resource blocks to users
        
        Args:
            ran_state: Current RAN state with user information
            
        Returns:
            dict: Resource allocation per user
        """
        allocations = {}
        
        # Extract state features
        state = self._extract_state_features(ran_state)
        
        # Select action
        action = self.select_action(state, training=False)
        
        # Decode action to resource block allocation
        allocation = self._decode_action(action, ran_state)
        
        return allocation
    
    def _extract_state_features(self, ran_state: Dict) -> np.ndarray:
        """
        Extract state features from RAN state
        
        Args:
            ran_state: Raw RAN state
            
        Returns:
            np.ndarray: State feature vector
        """
        # Extract relevant features:
        # - Channel Quality Indicators (CQI)
        # - Buffer occupancy
        # - QoS requirements (5QI)
        # - Historical throughput
        # - Number of active users
        
        features = []
        
        users = ran_state.get('users', [])
        max_users = 100  # Maximum users to consider
        
        for i in range(max_users):
            if i < len(users):
                user = users[i]
                features.extend([
                    user.get('cqi', 0) / 15.0,  # Normalized CQI (0-15)
                    user.get('buffer_occupancy', 0) / 1000.0,  # Normalized buffer
                    user.get('qos_5qi', 9) / 100.0,  # Normalized 5QI
                    user.get('throughput', 0) / 1e9  # Normalized throughput (Gbps)
                ])
            else:
                features.extend([0, 0, 0, 0])  # Padding for inactive users
        
        # Add global features
        features.extend([
            len(users) / max_users,  # Load factor
            ran_state.get('total_prbs_available', 100) / 100.0  # Available PRBs
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _decode_action(self, action: int, ran_state: Dict) -> Dict[str, List[int]]:
        """
        Decode action to resource block allocation
        
        Args:
            action: Action index
            ran_state: Current RAN state
            
        Returns:
            dict: Resource block allocation per user
        """
        # Simplified decoding: map action to allocation strategy
        # In production, use more sophisticated mapping
        
        users = ran_state.get('users', [])
        total_prbs = ran_state.get('total_prbs_available', 100)
        
        allocation = {}
        
        if not users:
            return allocation
        
        # Decode action to allocation strategy
        strategy = action % 5  # 5 different strategies
        
        if strategy == 0:  # Equal allocation
            prbs_per_user = total_prbs // len(users)
            for i, user in enumerate(users):
                allocation[user['id']] = list(range(
                    i * prbs_per_user,
                    (i + 1) * prbs_per_user
                ))
        
        elif strategy == 1:  # Proportional fair
            # Allocate based on CQI and historical throughput
            priorities = [
                user.get('cqi', 1) / max(user.get('throughput', 1e6), 1e6)
                for user in users
            ]
            total_priority = sum(priorities)
            
            offset = 0
            for user, priority in zip(users, priorities):
                num_prbs = int((priority / total_priority) * total_prbs)
                allocation[user['id']] = list(range(offset, offset + num_prbs))
                offset += num_prbs
        
        elif strategy == 2:  # Max CQI
            # Prioritize users with best channel quality
            sorted_users = sorted(users, key=lambda u: u.get('cqi', 0), reverse=True)
            offset = 0
            prbs_per_user = total_prbs // len(users)
            for user in sorted_users:
                allocation[user['id']] = list(range(offset, offset + prbs_per_user))
                offset += prbs_per_user
        
        elif strategy == 3:  # QoS-based
            # Prioritize users with stringent QoS requirements
            sorted_users = sorted(users, key=lambda u: u.get('qos_5qi', 9))
            offset = 0
            prbs_per_user = total_prbs // len(users)
            for user in sorted_users:
                allocation[user['id']] = list(range(offset, offset + prbs_per_user))
                offset += prbs_per_user
        
        else:  # Buffer-based
            # Prioritize users with fuller buffers
            sorted_users = sorted(
                users,
                key=lambda u: u.get('buffer_occupancy', 0),
                reverse=True
            )
            offset = 0
            prbs_per_user = total_prbs // len(users)
            for user in sorted_users:
                allocation[user['id']] = list(range(offset, offset + prbs_per_user))
                offset += prbs_per_user
        
        return allocation
    
    def save_model(self, filepath: str):
        """Save model to file"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'config': self.config
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        logger.info(f"Model loaded from {filepath}")
    
    def get_metrics(self) -> Dict:
        """Get training metrics"""
        recent_losses = self.losses[-100:] if self.losses else []
        
        return {
            'steps': self.steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(recent_losses) if recent_losses else 0,
            'replay_buffer_size': len(self.replay_buffer)
        }
