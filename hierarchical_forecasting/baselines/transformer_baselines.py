"""Transformer-based baselines for hierarchical forecasting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaselineModel


@dataclass
class _TransformerShape:
    sequence_length: int
    token_dim: int
    original_dim: int


class TransformerEncoderRegressor(nn.Module):
    """Simple Transformer encoder for regression on flattened sequences."""

    def __init__(self, token_dim: int, sequence_length: int, d_model: int, nhead: int,
                 num_layers: int, dropout: float) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.token_dim = token_dim
        self.input_projection = nn.Linear(token_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.Linear(sequence_length * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(tokens)
        encoded = self.encoder(x)
        flattened = encoded.reshape(encoded.size(0), -1)
        return self.output_head(flattened)


class TemporalTransformerBaseline(BaselineModel):
    """Transformer encoder baseline inspired by PatchTST/TFT style architectures."""

    def __init__(
        self,
        sequence_length: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(name="TemporalTransformer")
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.model: Optional[TransformerEncoderRegressor] = None
        self.shape_info: Optional[_TransformerShape] = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _reshape(self, X: np.ndarray) -> Tuple[np.ndarray, _TransformerShape]:
        n_samples, n_features = X.shape
        token_dim = math.ceil(n_features / self.sequence_length)
        padded_dim = token_dim * self.sequence_length
        if padded_dim != n_features:
            padded = np.zeros((n_samples, padded_dim), dtype=np.float32)
            padded[:, :n_features] = X
        else:
            padded = X.astype(np.float32, copy=False)
        tokens = padded.reshape(n_samples, self.sequence_length, token_dim)
        shape = _TransformerShape(self.sequence_length, token_dim, n_features)
        return tokens, shape

    def _prepare_dataset(self, X: np.ndarray, y: np.ndarray) -> TensorDataset:
        tokens, shape = self._reshape(X)
        if self.shape_info is None:
            self.shape_info = shape
        X_tensor = torch.from_numpy(tokens.astype(np.float32))
        y_tensor = torch.from_numpy(y.reshape(-1, 1).astype(np.float32))
        return TensorDataset(X_tensor, y_tensor)

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray,
            hierarchy: Optional[dict] = None, **kwargs) -> 'TemporalTransformerBaseline':
        dataset = self._prepare_dataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = TransformerEncoderRegressor(
            token_dim=self.shape_info.token_dim,
            sequence_length=self.shape_info.sequence_length,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            total = 0
            for batch_tokens, batch_targets in dataloader:
                batch_tokens = batch_tokens.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                preds = self.model(batch_tokens)
                loss = criterion(preds, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                batch_size = batch_tokens.size(0)
                epoch_loss += loss.item() * batch_size
                total += batch_size

            scheduler.step()
            avg_loss = epoch_loss / max(total, 1)
            if (epoch + 1) % max(self.epochs // 5, 1) == 0:
                print(f"[TemporalTransformer] Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.6f}")

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted or self.model is None or self.shape_info is None:
            raise ValueError("Model must be fitted before calling predict().")

        tokens, _ = self._reshape(X)
        tokens_tensor = torch.from_numpy(tokens.astype(np.float32)).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(tokens_tensor).cpu().numpy().flatten()
        return preds


class PretrainedTransformerBaseline(TemporalTransformerBaseline):
    """Baseline that loads or freezes a pretrained transformer encoder."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        freeze_encoder: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.freeze_encoder = freeze_encoder
        self.name = "FoundationTransformer"

    def fit(self, X: np.ndarray, y: np.ndarray,
            hierarchy: Optional[dict] = None, **kwargs) -> 'PretrainedTransformerBaseline':
        dataset = self._prepare_dataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = TransformerEncoderRegressor(
            token_dim=self.shape_info.token_dim,
            sequence_length=self.shape_info.sequence_length,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        if self.checkpoint_path:
            try:
                state_dict = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained weights from {self.checkpoint_path}")
            except FileNotFoundError:
                print(f"Warning: checkpoint {self.checkpoint_path} not found. Training from scratch.")

        if self.freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            total = 0
            for batch_tokens, batch_targets in dataloader:
                batch_tokens = batch_tokens.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                preds = self.model(batch_tokens)
                loss = criterion(preds, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

                batch_size = batch_tokens.size(0)
                epoch_loss += loss.item() * batch_size
                total += batch_size

            scheduler.step()
            avg_loss = epoch_loss / max(total, 1)
            if (epoch + 1) % max(self.epochs // 5, 1) == 0:
                print(f"[FoundationTransformer] Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.6f}")

        self.is_fitted = True
        return self

