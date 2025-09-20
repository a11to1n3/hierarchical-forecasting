"""Transformer-style baselines for hierarchical forecasting.

This module provides two high-capacity baselines:

* :class:`PatchTSTBaseline` – a thin wrapper that loads an officially trained
  PatchTST TorchScript or nn.Module checkpoint and uses it in evaluation mode.
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


# ---------------------------------------------------------------------------
# PatchTST wrapper
# ---------------------------------------------------------------------------


class PatchTSTBaseline(BaselineModel):
    """Baseline that relies on an externally trained PatchTST checkpoint.

    The checkpoint must be supplied as a TorchScript module (``torch.jit``) or
    as a serialized ``nn.Module``. No fine-tuning is performed inside this
    project to keep the evaluation faithful to the published model.
    """

    def __init__(self, checkpoint_path: str, device: Optional[str] = None) -> None:
        super().__init__(name="PatchTST")
        if not checkpoint_path:
            raise ValueError("`checkpoint_path` must point to a valid PatchTST checkpoint.")
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model: Optional[torch.nn.Module] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            hierarchy: Optional[dict] = None, **kwargs) -> 'PatchTSTBaseline':
        if self.model is None:
            try:
                module = torch.jit.load(self.checkpoint_path, map_location=self.device)
                self.model = module
            except (RuntimeError, FileNotFoundError):
                try:
                    state_obj = torch.load(self.checkpoint_path, map_location=self.device)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f"PatchTST checkpoint not found at {self.checkpoint_path}."
                    ) from exc

                if isinstance(state_obj, torch.nn.Module):
                    self.model = state_obj
                else:
                    raise RuntimeError(
                        "PatchTSTBaseline expects a TorchScript module or serialized nn.Module."
                        " Export the official model and pass its path via `checkpoint_path`."
                    )

                self.model.to(self.device)

            self.model.eval()

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise ValueError("PatchTSTBaseline must load a checkpoint before prediction.")

        tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
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
        epochs: int = 50,
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

    def fit(self, X: np.ndarray, y: np.ndarray,
            hierarchy: Optional[dict] = None, **kwargs) -> 'TimesNetBaseline':
        if X.ndim != 2:
            raise ValueError("TimesNetBaseline expects 2D input [n_samples, seq_len].")

        self.seq_len = X.shape[1]
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
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
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

        features = torch.from_numpy(X.astype(np.float32)).to(self.device).unsqueeze(-1)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(features)
        return preds.squeeze(-1).cpu().numpy().reshape(-1)
