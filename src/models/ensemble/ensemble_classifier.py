#!/usr/bin/env python3
"""
Ensemble classifier for combining multiple models.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EnsembleClassifier(nn.Module):
    """
    Ensemble classifier that combines predictions from multiple models.

    Supports different combination strategies:
    - Simple averaging
    - Weighted averaging
    - Majority voting (for discrete predictions)
    """

    def __init__(
        self,
        models: Dict[str, nn.Module],
        weights: Optional[Dict[str, float]] = None,
        combination_strategy: str = "average",
    ):
        """
        Initialize ensemble classifier.

        Args:
            models: Dictionary of {name: model} pairs
            weights: Optional weights for weighted averaging
            combination_strategy: How to combine predictions ('average', 'weighted', 'vote')
        """
        super().__init__()

        self.models = nn.ModuleDict(models)
        self.combination_strategy = combination_strategy

        # Set up weights
        if weights is None:
            # Equal weights
            self.weights = {name: 1.0 / len(models) for name in models.keys()}
        else:
            # Normalize weights
            total_weight = sum(weights.values())
            self.weights = {name: w / total_weight for name, w in weights.items()}

        logger.info(
            f"Created ensemble with {len(models)} models: {list(models.keys())}"
        )
        logger.info(f"Combination strategy: {combination_strategy}")
        logger.info(f"Model weights: {self.weights}")

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through ensemble.

        Args:
            inputs: Dictionary of {model_name: input_tensor} pairs

        Returns:
            Combined ensemble predictions
        """
        predictions = {}

        # Get predictions from each model
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                pred = model(inputs[name])
                predictions[name] = torch.sigmoid(pred)  # Convert to probabilities

        # Combine predictions
        if self.combination_strategy == "average":
            return self._average_predictions(predictions)
        elif self.combination_strategy == "weighted":
            return self._weighted_average_predictions(predictions)
        elif self.combination_strategy == "vote":
            return self._majority_vote_predictions(predictions)
        else:
            raise ValueError(
                f"Unknown combination strategy: {self.combination_strategy}"
            )

    def _average_predictions(
        self, predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Simple averaging of predictions."""
        pred_tensors = list(predictions.values())
        return torch.mean(torch.stack(pred_tensors), dim=0)

    def _weighted_average_predictions(
        self, predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Weighted averaging of predictions."""
        weighted_preds = []
        for name, pred in predictions.items():
            weighted_preds.append(pred * self.weights[name])
        return torch.sum(torch.stack(weighted_preds), dim=0)

    def _majority_vote_predictions(
        self, predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Majority voting on discrete predictions."""
        votes = []
        for pred in predictions.values():
            votes.append((pred >= 0.5).float())

        # Average votes (will be 0.5 for ties)
        vote_average = torch.mean(torch.stack(votes), dim=0)
        return vote_average

    def predict_individual(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Get individual model predictions for analysis."""
        predictions = {}

        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                pred = model(inputs[name])
                predictions[name] = torch.sigmoid(pred)

        return predictions

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the ensemble."""
        return {
            "num_models": len(self.models),
            "model_names": list(self.models.keys()),
            "weights": self.weights,
            "combination_strategy": self.combination_strategy,
        }
