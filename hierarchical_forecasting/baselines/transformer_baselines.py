"""Transformer-style baselines for hierarchical forecasting.

This module provides two high-capacity baselines:

* :class:`PatchTSTBaseline` – optionally loads an official PatchTST checkpoint,
  but can also be trained or fine-tuned directly on local features.
* :class:`TimesNetBaseline` – a PyTorch re-implementation of TimesNet adapted
  from the authors' public repository (https://github.com/thuml/Time-Series-Library).

The goal is to expose strong transformer/foundation-style baselines while
keeping training logic inside this project simple and reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .base import BaselineModel


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _sinusoidal_embedding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class PatchEmbedding1D(nn.Module):
    """Map 1D sequences into overlapping patch embeddings."""

    def __init__(self, seq_len: int, patch_len: int, stride: int, d_model: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        effective_len = max(seq_len, patch_len)
        if effective_len > patch_len and stride > 0:
            remainder = (effective_len - patch_len) % stride
            if remainder:
                effective_len += stride - remainder
        self.effective_len = effective_len
        patch_count = 1 if stride <= 0 else 1 + max(0, (effective_len - patch_len) // stride)
        self.proj = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.randn(1, patch_count, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len]
        if x.dim() != 2:
            raise ValueError("PatchEmbedding1D expects input shape [batch, seq_len].")

        seq_len = x.size(1)
        if seq_len < self.patch_len:
            pad = self.patch_len - seq_len
            x = F.pad(x, (0, pad))
            seq_len = self.patch_len

        if seq_len < self.effective_len:
            x = F.pad(x, (0, self.effective_len - seq_len))

        patches = x.unfold(dimension=1, size=self.patch_len, step=max(self.stride, 1))
        patches = patches.contiguous()  # [batch, num_patches, patch_len]
        embeddings = self.proj(patches)
        if embeddings.size(1) != self.position.size(1):
            pos = F.interpolate(
                self.position.detach().permute(0, 2, 1),
                size=embeddings.size(1),
                mode='linear',
                align_corners=False,
            ).permute(0, 2, 1)
        else:
            pos = self.position
        return embeddings + pos


class PatchTSTRegressor(nn.Module):
    """Lightweight PatchTST-style regressor for univariate forecasting."""

    def __init__(
        self,
        input_dim: int,
        patch_len: int,
        stride: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding1D(input_dim, patch_len, stride, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        pooled = self.norm(pooled)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# PatchTST wrapper
# ---------------------------------------------------------------------------


class PatchTSTBaseline(BaselineModel):
    """PatchTST baseline with optional checkpoint loading and fine-tuning."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        epochs: int = 120,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        allow_training: bool = True,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(name="PatchTST")
        self.checkpoint_path = checkpoint_path
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.allow_training = allow_training
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.model: Optional[PatchTSTRegressor] = None
        self._seq_len: Optional[int] = None

    def _prepare_dataset(self, X: np.ndarray, y: np.ndarray) -> TensorDataset:
        X_tensor = torch.from_numpy(X.astype(np.float32))
        y_tensor = torch.from_numpy(y.astype(np.float32))
        return TensorDataset(X_tensor, y_tensor)

    def _build_model(self, seq_len: int) -> PatchTSTRegressor:
        return PatchTSTRegressor(
            input_dim=seq_len,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray,
            hierarchy: Optional[dict] = None, **kwargs) -> 'PatchTSTBaseline':
        if X.ndim != 2:
            raise ValueError("PatchTSTBaseline expects 2D input [n_samples, seq_len].")

        self._seq_len = X.shape[1]

        # If a checkpoint is provided, try to load it first.
        if self.checkpoint_path:
            try:
                module = torch.jit.load(self.checkpoint_path, map_location=self.device)
                self.model = module
                self.model.eval()
                self.is_fitted = True
                if not self.allow_training:
                    return self
            except (RuntimeError, FileNotFoundError):
                try:
                    state = torch.load(self.checkpoint_path, map_location=self.device)
                    if isinstance(state, torch.nn.Module):
                        self.model = state.to(self.device)
                        self.model.eval()
                        self.is_fitted = True
                        if not self.allow_training:
                            return self
                    else:
                        raise RuntimeError
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f"PatchTST checkpoint not found at {self.checkpoint_path}."
                    ) from exc
                except RuntimeError:
                    raise RuntimeError(
                        "Unable to load PatchTST checkpoint. Provide a TorchScript module or nn.Module."
                    )

        # Train (or fine-tune) locally if allowed.
        if not self.allow_training and self.model is None:
            raise RuntimeError(
                "No valid checkpoint provided for PatchTSTBaseline and training is disabled."
            )

        if self.model is None or self.allow_training:
            self.model = self._build_model(self._seq_len)

            dataset = self._prepare_dataset(X, y)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
            criterion = nn.MSELoss()

            self.model.train()
            for epoch in range(self.epochs):
                total_loss = 0.0
                total = 0
                for features, targets in loader:
                    features = features.to(self.device)
                    targets = targets.to(self.device).unsqueeze(-1)

                    optimizer.zero_grad()
                    preds = self.model(features)
                    loss = criterion(preds, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                    batch_size = features.size(0)
                    total_loss += loss.item() * batch_size
                    total += batch_size

                if (epoch + 1) % max(self.epochs // 5, 1) == 0:
                    avg_loss = total_loss / max(total, 1)
                    print(f"[PatchTST] Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.6f}")

        self.model.eval()
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise ValueError("PatchTSTBaseline must be fitted before prediction.")

        tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(tensor)
        return preds.detach().cpu().numpy().reshape(-1)


# ---------------------------------------------------------------------------
# TimesNet components (adapted from https://github.com/thuml/Time-Series-Library)
# ---------------------------------------------------------------------------


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model).float()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int) -> None:
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.token_conv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=padding,
                                    padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.token_conv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 6) -> None:
        super().__init__()
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            for i in range(num_kernels)
        ])
        for conv in self.kernels:
            nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [conv(x) for conv in self.kernels]
        return torch.stack(outputs, dim=-1).mean(dim=-1)


def _fft_top_k(x: torch.Tensor, k: int) -> tuple[list[int], torch.Tensor]:
    xf = torch.fft.rfft(x, dim=1)
    amplitude = xf.abs().mean(dim=0).mean(dim=-1)
    amplitude[0] = 0
    top_k = min(k, amplitude.shape[0])
    top_indices = torch.topk(amplitude, top_k, dim=0).indices
    periods = (x.shape[1] // top_indices.cpu().numpy()).tolist()
    return periods, xf.abs().mean(dim=-1)[:, top_indices]


class TimesBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, seq_len: int, pred_len: int,
                 top_k: int = 5, num_kernels: int = 6) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k
        self.conv = nn.Sequential(
            InceptionBlock(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlock(d_ff, d_model, num_kernels=num_kernels),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, channels = x.shape
        extended_len = self.seq_len + self.pred_len
        periods, period_weights = _fft_top_k(x, self.top_k)
        if not periods:
            return self.norm(x)

        aggregated = torch.zeros_like(x)
        for idx, period in enumerate(periods):
            length = extended_len if extended_len % period == 0 else (extended_len // period + 1) * period
            pad_len = length - extended_len
            padded = torch.zeros((bsz, pad_len, channels), device=x.device)
            series = torch.cat([x, padded], dim=1)
            series = series.reshape(bsz, length // period, period, channels)
            series = series.permute(0, 3, 1, 2).contiguous()
            series = self.conv(series)
            series = series.permute(0, 2, 3, 1).reshape(bsz, length, channels)
            aggregated = aggregated + series[:, :extended_len, :]

        weight = F.softmax(period_weights, dim=1)
        weight = weight.unsqueeze(1).unsqueeze(1)
        aggregated = aggregated * weight.sum(dim=-1)
        return self.norm(aggregated[:, :x.shape[1], :] + x)


@dataclass
class TimesNetConfig:
    seq_len: int
    pred_len: int
    d_model: int
    d_ff: int
    top_k: int
    num_kernels: int
    e_layers: int
    dropout: float


class TimesNetForecaster(nn.Module):
    def __init__(self, cfg: TimesNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = DataEmbedding(1, cfg.d_model, dropout=cfg.dropout)
        self.blocks = nn.ModuleList([
            TimesBlock(cfg.d_model, cfg.d_ff, cfg.seq_len, cfg.pred_len,
                       top_k=cfg.top_k, num_kernels=cfg.num_kernels)
            for _ in range(cfg.e_layers)
        ])
        self.predict_linear = nn.Linear(cfg.seq_len, cfg.seq_len + cfg.pred_len)
        self.projection = nn.Linear(cfg.d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        means = x.mean(dim=1, keepdim=True)
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = (x - means) / stdev

        out = self.embedding(x)
        out = self.predict_linear(out.permute(0, 2, 1)).permute(0, 2, 1)
        for block in self.blocks:
            out = block(out)
        out = self.projection(out)
        out = out * stdev[:, 0:1, :] + means[:, 0:1, :]
        return out[:, -self.cfg.pred_len:, :]


class TimesNetBaseline(BaselineModel):
    """TimesNet baseline using mini-batch training on local features."""

    def __init__(
        self,
        d_model: int = 128,
        d_ff: int = 256,
        top_k: int = 5,
        num_kernels: int = 6,
        e_layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 120,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(name="TimesNet")
        self.d_model = d_model
        self.d_ff = d_ff
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.e_layers = e_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.model: Optional[TimesNetForecaster] = None
        self.seq_len: Optional[int] = None
        self.feature_scaler: Optional[StandardScaler] = None
        self.target_scaler: Optional[StandardScaler] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            hierarchy: Optional[dict] = None, **kwargs) -> 'TimesNetBaseline':
        if X.ndim != 2:
            raise ValueError("TimesNetBaseline expects 2D input [n_samples, seq_len].")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        X_scaled = self.feature_scaler.fit_transform(X).astype(np.float32)
        y_scaled = self.target_scaler.fit_transform(y).astype(np.float32).squeeze(-1)

        self.seq_len = X_scaled.shape[1]
        cfg = TimesNetConfig(
            seq_len=self.seq_len,
            pred_len=1,
            d_model=self.d_model,
            d_ff=self.d_ff,
            top_k=self.top_k,
            num_kernels=self.num_kernels,
            e_layers=self.e_layers,
            dropout=self.dropout,
        )

        self.model = TimesNetForecaster(cfg).to(self.device)

        dataset = TensorDataset(
            torch.from_numpy(X_scaled),
            torch.from_numpy(y_scaled),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            total = 0
            for features, targets in loader:
                features = features.to(self.device).unsqueeze(-1)
                targets = targets.to(self.device).unsqueeze(-1).unsqueeze(-1)

                optimizer.zero_grad()
                preds = self.model(features)
                loss = criterion(preds, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                batch_size = features.size(0)
                total_loss += loss.item() * batch_size
                total += batch_size

            if (epoch + 1) % max(self.epochs // 5, 1) == 0:
                avg_loss = total_loss / max(total, 1)
                print(f"[TimesNet] Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.6f}")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise ValueError("TimesNetBaseline must be fitted before prediction.")

        X = np.asarray(X, dtype=np.float32)
        if self.feature_scaler is not None:
            X_scaled = self.feature_scaler.transform(X).astype(np.float32)
        else:
            X_scaled = X.astype(np.float32)

        features = torch.from_numpy(X_scaled).to(self.device).unsqueeze(-1)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(features)
        outputs = preds.squeeze(-1).cpu().numpy().reshape(-1, 1)
        if self.target_scaler is not None:
            outputs = self.target_scaler.inverse_transform(outputs)
        return outputs.ravel()
