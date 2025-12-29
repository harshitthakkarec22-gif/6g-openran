"""
LSTM Network for Traffic Prediction in 6G Networks

Based on research: "Long Short-Term Memory Networks for Traffic Forecasting in 6G"
IEEE Transactions on Vehicular Technology, 2024

Implements bidirectional LSTM for:
- 10-100 second ahead traffic prediction
- Multi-variate time series (per-slice traffic)
- Real-time inference for proactive resource allocation
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque

logger = logging.getLogger(__name__)


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM network for traffic prediction
    
    Uses both past and future context for improved predictions
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 output_dim: int, dropout: float = 0.2):
        """
        Initialize Bidirectional LSTM
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            output_dim: Output dimension (prediction horizon)
            dropout: Dropout rate
        """
        super(BidirectionalLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            tuple: (predictions, attention_weights)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        
        # Attention mechanism
        attention_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_dim*2)
        
        # Output prediction
        output = self.fc(context)  # (batch, output_dim)
        
        return output, attention_weights


class LSTMTrafficPredictor:
    """
    LSTM-based traffic predictor for 6G networks
    
    Predicts future traffic load for:
    - Different network slices (eMBB, URLLC, mMTC)
    - Individual base stations
    - Aggregated cell traffic
    
    Supports multi-step ahead prediction (10-100 seconds)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize LSTM traffic predictor
        
        Args:
            config: Configuration dictionary with:
                - input_dim: Input feature dimension
                - hidden_dim: LSTM hidden dimension
                - num_layers: Number of LSTM layers
                - output_dim: Prediction horizon steps
                - sequence_length: Input sequence length
                - learning_rate: Learning rate
                - dropout: Dropout rate
                - num_slices: Number of network slices
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = BidirectionalLSTM(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            output_dim=config['output_dim'],
            dropout=config.get('dropout', 0.2)
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Data buffers
        self.sequence_length = config['sequence_length']
        self.traffic_history = {
            'embb': deque(maxlen=1000),
            'urllc': deque(maxlen=1000),
            'mmtc': deque(maxlen=1000)
        }
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        
        logger.info(f"LSTM Traffic Predictor initialized on {self.device}")
    
    def preprocess_data(self, traffic_data: Dict) -> np.ndarray:
        """
        Preprocess traffic data for model input
        
        Args:
            traffic_data: Raw traffic data with per-slice metrics
            
        Returns:
            np.ndarray: Preprocessed features
        """
        features = []
        
        # Extract traffic metrics per slice
        for slice_type in ['embb', 'urllc', 'mmtc']:
            slice_data = traffic_data.get(slice_type, {})
            features.extend([
                slice_data.get('throughput', 0) / 1e9,  # Normalize to Gbps
                slice_data.get('active_users', 0) / 1000.0,
                slice_data.get('packet_rate', 0) / 1e6,  # Normalize to Mpps
                slice_data.get('buffer_occupancy', 0) / 100.0,  # Percentage
                slice_data.get('latency_ms', 0) / 100.0
            ])
        
        # Add temporal features
        features.extend([
            traffic_data.get('hour_of_day', 0) / 24.0,
            traffic_data.get('day_of_week', 0) / 7.0,
            traffic_data.get('is_peak_hour', 0)
        ])
        
        return np.array(features, dtype=np.float32)
    
    def collect_traffic_data(self, traffic_data: Dict):
        """
        Collect traffic data for training
        
        Args:
            traffic_data: Current traffic measurements
        """
        preprocessed = self.preprocess_data(traffic_data)
        
        # Store in slice-specific buffers
        for slice_type in ['embb', 'urllc', 'mmtc']:
            if slice_type in traffic_data:
                self.traffic_history[slice_type].append(preprocessed)
    
    def create_sequences(self, data: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences for training
        
        Args:
            data: List of traffic data points
            
        Returns:
            tuple: (input_sequences, target_sequences)
        """
        if len(data) < self.sequence_length + self.config['output_dim']:
            return np.array([]), np.array([])
        
        sequences = []
        targets = []
        
        data_array = np.array(data)
        
        for i in range(len(data) - self.sequence_length - self.config['output_dim'] + 1):
            seq = data_array[i:i + self.sequence_length]
            target = data_array[
                i + self.sequence_length:i + self.sequence_length + self.config['output_dim'],
                0  # Predict only throughput
            ]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """
        Perform one training step
        
        Args:
            batch_size: Training batch size
            
        Returns:
            float: Training loss, or None if insufficient data
        """
        # Aggregate data from all slices
        all_data = []
        for slice_data in self.traffic_history.values():
            all_data.extend(list(slice_data))
        
        if len(all_data) < self.sequence_length + self.config['output_dim']:
            return None
        
        # Create sequences
        X, y = self.create_sequences(all_data)
        
        if len(X) == 0:
            return None
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Training loop
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(X), batch_size):
            batch_X = X_tensor[i:i + batch_size]
            batch_y = y_tensor[i:i + batch_size]
            
            # Forward pass
            predictions, attention_weights = self.model(batch_X)
            
            # Compute loss (MSE + MAE for robustness)
            mse_loss = nn.MSELoss()(predictions, batch_y)
            mae_loss = nn.L1Loss()(predictions, batch_y)
            loss = mse_loss + 0.1 * mae_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        
        # Update learning rate
        self.scheduler.step(avg_loss)
        
        return avg_loss
    
    def predict(self, recent_traffic: List[Dict], 
                prediction_horizon_s: int = 60) -> Dict[str, np.ndarray]:
        """
        Predict future traffic
        
        Args:
            recent_traffic: Recent traffic measurements
            prediction_horizon_s: Prediction horizon in seconds
            
        Returns:
            dict: Predicted traffic per slice
        """
        self.model.eval()
        
        # Preprocess recent traffic
        preprocessed = [self.preprocess_data(t) for t in recent_traffic]
        
        if len(preprocessed) < self.sequence_length:
            logger.warning("Insufficient data for prediction")
            return {'embb': np.array([]), 'urllc': np.array([]), 'mmtc': np.array([])}
        
        # Take last sequence_length samples
        sequence = np.array(preprocessed[-self.sequence_length:])
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions, attention_weights = self.model(sequence_tensor)
            predictions = predictions.cpu().numpy()[0]
        
        # Store prediction
        self.predictions.append(predictions)
        
        # Separate predictions by slice (simplified)
        # In production, train separate models per slice
        return {
            'embb': predictions * 1e9,  # Convert back to bps
            'urllc': predictions * 0.1 * 1e9,  # URLLC typically lower volume
            'mmtc': predictions * 0.05 * 1e9  # mMTC lowest volume
        }
    
    def predict_slice_traffic(self, slice_type: str,
                             recent_data: List[Dict]) -> np.ndarray:
        """
        Predict traffic for specific slice
        
        Args:
            slice_type: Slice type ('embb', 'urllc', 'mmtc')
            recent_data: Recent traffic data
            
        Returns:
            np.ndarray: Predicted traffic values
        """
        predictions = self.predict(recent_data)
        return predictions.get(slice_type, np.array([]))
    
    def evaluate(self, test_data: List[Dict]) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            test_data: Test traffic data
            
        Returns:
            dict: Evaluation metrics (MSE, MAE, RMSE, MAPE)
        """
        self.model.eval()
        
        # Preprocess test data
        preprocessed = [self.preprocess_data(t) for t in test_data]
        X, y = self.create_sequences(preprocessed)
        
        if len(X) == 0:
            return {'mse': 0, 'mae': 0, 'rmse': 0, 'mape': 0}
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        with torch.no_grad():
            predictions, _ = self.model(X_tensor)
            
            # Compute metrics
            mse = nn.MSELoss()(predictions, y_tensor).item()
            mae = nn.L1Loss()(predictions, y_tensor).item()
            rmse = np.sqrt(mse)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = torch.mean(
                torch.abs((y_tensor - predictions) / (y_tensor + 1e-8))
            ).item() * 100
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"Model loaded from {filepath}")
    
    def get_metrics(self) -> Dict:
        """Get training metrics"""
        recent_losses = self.train_losses[-100:]
        
        return {
            'avg_train_loss': np.mean(recent_losses) if recent_losses else 0,
            'min_train_loss': np.min(recent_losses) if recent_losses else 0,
            'data_points_embb': len(self.traffic_history['embb']),
            'data_points_urllc': len(self.traffic_history['urllc']),
            'data_points_mmtc': len(self.traffic_history['mmtc']),
            'num_predictions': len(self.predictions)
        }
