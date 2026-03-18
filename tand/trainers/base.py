from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any


class BaseTrainer:
    """Base interface for different training regimes.

    Subclasses should implement:
    - build_model(dataset) -> nn.Module
    - train_epoch(model, dataloader, optimizer) -> (loss, det_loss, cls_loss)
    - evaluate(model, dataloader) -> metrics dict (must include 'overall_f1')
    """

    def __init__(self, config: Dict[str, Any], device: torch.device, logger):
        self.config = config
        self.device = device
        self.logger = logger

    # API methods to be implemented by subclasses
    def build_model(self, train_dataset) -> nn.Module:  # noqa: ANN001
        raise NotImplementedError

    def train_epoch(self, model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> Tuple[float, float, float]:
        raise NotImplementedError

    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, Any]:
        raise NotImplementedError
