"""
LSTM baseline for hierarchical forecasting.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Tuple
from .base import BaselineModel


class LSTMBaseline(BaselineModel):
    """
    LSTM baseline for hierarchical forecasting.
    
    This baseline uses Long Short-Term Memory networks to capture
    temporal dependencies in the time series data.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, learning_rate: float = 0.001,
                 epochs: int = 120, batch_size: int = 32, device: str = 'auto'):
        """
        Initialize LSTM baseline.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: Device to use ('cpu', 'cuda', 'mps', or 'auto')
        """
        super().__init__("LSTM")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize model
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        self.scaler = StandardScaler()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray, 
                          sequence_length: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequences for LSTM training.
        
        Args:
            X: Input features
            y: Target values
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (sequences, targets)
        """
        sequences = []
        targets = []
        
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X[i:i+sequence_length])
            targets.append(y[i+sequence_length-1])
        
        return (torch.FloatTensor(sequences).to(self.device),
                torch.FloatTensor(targets).to(self.device))
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            hierarchy: Optional[Dict] = None, 
            sequence_length: int = 10, **kwargs) -> 'LSTMBaseline':
        """
        Fit the LSTM model.
        
        Args:
            X: Input features [n_samples, n_features]
            y: Target values [n_samples]
            hierarchy: Ignored for this baseline
            sequence_length: Length of input sequences
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare sequences
        X_seq, y_seq = self._prepare_sequences(X_scaled, y, sequence_length)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X_seq), self.batch_size):
                batch_X = X_seq[i:i+self.batch_size]
                batch_y = y_seq[i:i+self.batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        self.sequence_length = sequence_length
        return self
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions using the fitted LSTM model.
        
        Args:
            X: Input features [n_samples, n_features]
            **kwargs: Additional arguments
            
        Returns:
            Predictions [n_samples]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Normalize features
        X_scaled = self.scaler.transform(X)
        
        # Prepare sequences
        X_seq = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            X_seq.append(X_scaled[i:i+self.sequence_length])
        
        if not X_seq:
            # If not enough data for sequences, return zeros
            return np.zeros(len(X))
        
        X_seq = torch.FloatTensor(X_seq).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_seq).cpu().numpy().squeeze()
        
        # Pad predictions to match input length
        if len(predictions.shape) == 0:
            predictions = np.array([predictions])
        
        # Pad beginning with zeros
        full_predictions = np.zeros(len(X))
        full_predictions[self.sequence_length-1:] = predictions
        
        return full_predictions
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        })
        return params


class LSTMModel(nn.Module):
    """
    LSTM neural network model.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.1):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LSTM model.
        
        Args:
            x: Input sequences [batch_size, sequence_length, input_size]
            
        Returns:
            Output predictions [batch_size, 1]
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and final linear layer
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output


class MultiEntityLSTM(BaselineModel):
    """
    LSTM with separate models for different entity types.
    
    This baseline trains separate LSTM models for different hierarchy levels
    or entity types to capture level-specific temporal patterns.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_layers: int = 2, learning_rate: float = 0.001,
                 epochs: int = 120, device: str = 'auto'):
        """
        Initialize Multi-Entity LSTM.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            learning_rate: Learning rate
            epochs: Training epochs
            device: Device to use
        """
        super().__init__("MultiEntityLSTM")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.level_models = {}
        self.level_scalers = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            hierarchy: Optional[Dict] = None,
            entity_levels: Optional[np.ndarray] = None, 
            sequence_length: int = 10, **kwargs) -> 'MultiEntityLSTM':
        """
        Fit separate LSTM models for each entity level.
        
        Args:
            X: Input features
            y: Target values
            hierarchy: Hierarchy structure
            entity_levels: Level indicator for each sample
            sequence_length: Length of sequences
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        if entity_levels is None:
            entity_levels = np.zeros(len(X))
        
        # Train separate model for each level
        for level in np.unique(entity_levels):
            level_mask = entity_levels == level
            X_level = X[level_mask]
            y_level = y[level_mask]
            
            if len(X_level) >= sequence_length:
                # Create and train LSTM for this level
                model = LSTMBaseline(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    learning_rate=self.learning_rate,
                    epochs=self.epochs,
                    device=self.device
                )
                
                model.fit(X_level, y_level, sequence_length=sequence_length)
                self.level_models[level] = model
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, entity_levels: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """
        Make predictions using level-specific LSTM models.
        
        Args:
            X: Input features
            entity_levels: Level indicator for each sample
            **kwargs: Additional arguments
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if entity_levels is None:
            entity_levels = np.zeros(len(X))
        
        predictions = np.zeros(len(X))
        
        for level in np.unique(entity_levels):
            level_mask = entity_levels == level
            
            if level in self.level_models and np.any(level_mask):
                X_level = X[level_mask]
                level_preds = self.level_models[level].predict(X_level)
                predictions[level_mask] = level_preds
        
        return predictions
