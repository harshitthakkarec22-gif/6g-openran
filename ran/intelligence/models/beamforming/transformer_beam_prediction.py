"""
Transformer-based Beam Prediction for Massive MIMO in 6G

Based on research: "Attention Mechanisms for Massive MIMO Beamforming in 6G"
IEEE Transactions on Communications, 2024

Implements transformer architecture for:
- Self-attention for temporal beam patterns
- Multi-head attention for spatial correlation
- Sub-millisecond inference time for real-time beamforming
- Support for massive MIMO (64x64, 128x128 arrays)
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import math

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for spatial-temporal beam prediction
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            tuple: (output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.W_o(context)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Forward pass"""
        # Multi-head attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights


class TransformerBeamPredictor(nn.Module):
    """
    Transformer-based beam predictor for massive MIMO
    
    Predicts optimal beamforming vectors based on:
    - Historical beam patterns
    - User mobility patterns
    - Channel state information
    """
    
    def __init__(self, config: Dict):
        """
        Initialize transformer beam predictor
        
        Args:
            config: Configuration with:
                - input_dim: Input feature dimension
                - d_model: Model dimension
                - num_heads: Number of attention heads
                - num_layers: Number of transformer blocks
                - d_ff: Feed-forward dimension
                - num_antennas: Number of antenna elements
                - dropout: Dropout rate
        """
        super(TransformerBeamPredictor, self).__init__()
        
        self.config = config
        
        # Input embedding
        self.input_projection = nn.Linear(config['input_dim'], config['d_model'])
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config['d_model'])
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                config['d_model'],
                config['num_heads'],
                config['d_ff'],
                config['dropout']
            )
            for _ in range(config['num_layers'])
        ])
        
        # Output projection to beam vector
        self.beam_projection = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_model'] // 2', config['num_antennas'] * 2)  # Complex beam vector (real + imag)
        )
        
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            tuple: (beam_vectors, attention_weights)
        """
        # Input projection and positional encoding
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x, mask)
            attention_weights.append(attn_weights)
        
        # Take last time step
        x = x[:, -1, :]
        
        # Project to beam vector
        beam_output = self.beam_projection(x)
        
        # Reshape to complex beam vector
        batch_size = beam_output.size(0)
        num_antennas = self.config['num_antennas']
        
        real_part = beam_output[:, :num_antennas]
        imag_part = beam_output[:, num_antennas:]
        
        # Combine real and imaginary parts
        beam_vector = torch.complex(real_part, imag_part)
        
        # Normalize beam vector
        beam_vector = beam_vector / (torch.abs(beam_vector).sum(dim=1, keepdim=True) + 1e-8)
        
        return beam_vector, attention_weights


class BeamPredictionSystem:
    """
    Complete beam prediction system for 6G networks
    
    Provides real-time beam prediction with sub-millisecond latency
    """
    
    def __init__(self, config: Dict):
        """
        Initialize beam prediction system
        
        Args:
            config: System configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = TransformerBeamPredictor(config).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('scheduler_t0', 10),
            T_mult=2
        )
        
        # Metrics
        self.train_losses = []
        self.inference_times = []
        
        logger.info(f"Transformer Beam Predictor initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def predict_beam(self, channel_history: List[Dict], 
                    user_position: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict optimal beam vector
        
        Args:
            channel_history: Historical channel measurements
            user_position: Optional user position (x, y, z)
            
        Returns:
            np.ndarray: Predicted beam vector (complex)
        """
        import time
        start_time = time.time()
        
        self.model.eval()
        
        # Extract features
        features = self._extract_features(channel_history, user_position)
        
        if features is None or len(features) == 0:
            # Return default beam
            num_antennas = self.config['num_antennas']
            return np.ones(num_antennas, dtype=np.complex64) / np.sqrt(num_antennas)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            beam_vector, _ = self.model(features_tensor)
            beam_vector = beam_vector.cpu().numpy()[0]
        
        # Track inference time
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        
        if inference_time > 1.0:
            logger.warning(f"Inference time exceeded 1ms: {inference_time:.2f}ms")
        
        return beam_vector
    
    def _extract_features(self, channel_history: List[Dict],
                         user_position: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Extract features from channel history
        
        Args:
            channel_history: Historical channel measurements
            user_position: User position
            
        Returns:
            np.ndarray: Feature matrix
        """
        if not channel_history:
            return None
        
        features_list = []
        
        for measurement in channel_history[-self.config.get('sequence_length', 10):]:
            feature_vec = []
            
            # Channel quality indicators
            feature_vec.extend([
                measurement.get('rsrp', -100) / 100.0,  # Normalize RSRP
                measurement.get('rsrq', -20) / 20.0,  # Normalize RSRQ
                measurement.get('sinr', 0) / 30.0,  # Normalize SINR
                measurement.get('cqi', 0) / 15.0  # Normalize CQI
            ])
            
            # Angle of arrival/departure
            feature_vec.extend([
                np.sin(measurement.get('aoa', 0)),
                np.cos(measurement.get('aoa', 0)),
                np.sin(measurement.get('aod', 0)),
                np.cos(measurement.get('aod', 0))
            ])
            
            # Doppler information (mobility indicator)
            feature_vec.append(measurement.get('doppler_shift', 0) / 1000.0)
            
            # User position if available
            if user_position is not None:
                feature_vec.extend(user_position / 1000.0)  # Normalize to km
            else:
                feature_vec.extend([0, 0, 0])
            
            features_list.append(feature_vec)
        
        return np.array(features_list, dtype=np.float32)
    
    def train_step(self, batch_data: Dict) -> float:
        """
        Training step
        
        Args:
            batch_data: Batch of training data
            
        Returns:
            float: Training loss
        """
        self.model.train()
        
        features = torch.FloatTensor(batch_data['features']).to(self.device)
        target_beams = torch.from_numpy(batch_data['target_beams']).to(self.device)
        
        # Forward pass
        predicted_beams, _ = self.model(features)
        
        # Compute loss (beam alignment error)
        loss = self._beam_alignment_loss(predicted_beams, target_beams)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        loss_value = loss.item()
        self.train_losses.append(loss_value)
        
        return loss_value
    
    def _beam_alignment_loss(self, predicted: torch.Tensor,
                            target: torch.Tensor) -> torch.Tensor:
        """
        Compute beam alignment loss
        
        Measures how well the predicted beam aligns with target beam
        Uses negative of absolute value of inner product (maximize alignment)
        """
        # Compute inner product between predicted and target beams
        inner_product = torch.sum(predicted.conj() * target, dim=1)
        
        # Loss is negative absolute value (we want to maximize |<p, t>|)
        loss = -torch.mean(torch.abs(inner_product))
        
        return loss
    
    def compute_beam_gain(self, beam_vector: np.ndarray,
                         target_direction: np.ndarray) -> float:
        """
        Compute beam gain in target direction
        
        Args:
            beam_vector: Beam vector
            target_direction: Target direction (azimuth, elevation)
            
        Returns:
            float: Beam gain in dB
        """
        # Simplified beam gain calculation
        # In production, use actual array response vector
        
        azimuth, elevation = target_direction
        num_antennas = len(beam_vector)
        
        # Create array response vector for target direction
        array_response = np.exp(
            1j * 2 * np.pi * np.arange(num_antennas) * 
            np.sin(azimuth) * np.cos(elevation)
        )
        
        # Compute beam gain
        gain = np.abs(np.dot(beam_vector.conj(), array_response)) ** 2
        gain_db = 10 * np.log10(gain + 1e-10)
        
        return gain_db
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        logger.info(f"Model loaded from {filepath}")
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        recent_losses = self.train_losses[-100:]
        recent_times = self.inference_times[-100:]
        
        return {
            'avg_train_loss': np.mean(recent_losses) if recent_losses else 0,
            'avg_inference_time_ms': np.mean(recent_times) if recent_times else 0,
            'max_inference_time_ms': np.max(recent_times) if recent_times else 0,
            'p95_inference_time_ms': np.percentile(recent_times, 95) if recent_times else 0,
            'realtime_capable': np.percentile(recent_times, 95) < 1.0 if recent_times else False
        }
